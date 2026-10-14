"""Turning the CSR corpus into model batches for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import OOV_ID, PAD_ID, POSITIVE_RATIO, Corpus

FIELDS = ("item", "ratio_bucket", "liked", "disliked", "organic", "duration_bucket")


def period_bounds(corpus: Corpus, u: int, start_ts: int, end_ts: int | None) -> tuple[int, int]:
    """Absolute index range of user u's events with start_ts <= ts < end_ts."""
    sl = corpus.user_slice(u)
    ts = corpus.ts[sl]
    lo = sl.start + int(np.searchsorted(ts, start_ts, side="left"))
    hi = sl.stop if end_ts is None else sl.start + int(np.searchsorted(ts, end_ts, side="left"))
    return lo, hi


def gather(corpus: Corpus, index_rows: list[np.ndarray], length: int) -> dict:
    """Right-padded batch of event fields from absolute event indices."""
    batch = {f: np.zeros((len(index_rows), length), dtype=np.int64) for f in FIELDS}
    mask = np.zeros((len(index_rows), length), dtype=np.int64)
    for r, idx in enumerate(index_rows):
        n = len(idx)
        for f in FIELDS:
            batch[f][r, :n] = getattr(corpus, f)[idx]
        mask[r, :n] = 1
    out = {f: torch.from_numpy(v) for f, v in batch.items()}
    out["attention_mask"] = torch.from_numpy(mask)
    return out


class TrainWindows:
    """Non-overlapping windows of L+1 events from each user's train period.

    Only events strictly before `val_start` are used, so no validation or test
    event can appear as an input or a target.
    """

    def __init__(self, corpus: Corpus, seq_len: int, min_len: int = 8):
        self.corpus = corpus
        self.seq_len = seq_len
        starts = []
        for u in range(corpus.n_users):
            lo, hi = period_bounds(corpus, u, 0, corpus.val_start)
            # Walk backwards so the most recent history always forms a full window.
            for end in range(hi, lo, -seq_len):
                begin = max(lo, end - seq_len - 1)
                if end - begin >= min_len:
                    starts.append((begin, end))
        self.windows = np.asarray(starts, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.windows)

    def batch(self, ids) -> dict:
        rows = [np.arange(b, e) for b, e in self.windows[ids]]
        full = gather(self.corpus, rows, self.seq_len + 1)
        inputs = {k: v[:, :-1] for k, v in full.items()}
        targets = full["item"][:, 1:].clone()
        valid = full["attention_mask"][:, 1:].bool() & (targets != OOV_ID) & (targets != PAD_ID)
        inputs["targets"] = targets
        inputs["target_mask"] = valid
        return inputs


@dataclass
class EvalChunk:
    """One forward pass: inputs plus which positions are scored for which target."""
    batch: dict
    rows: np.ndarray       # row in batch of each scored position
    cols: np.ndarray       # column whose hidden state predicts the target
    target_idx: np.ndarray  # absolute corpus index of the target event


def next_item_chunks(corpus: Corpus, users, start_ts: int, end_ts: int | None,
                     seq_len: int, batch_size: int):
    """Teacher-forced next-item evaluation over a period.

    Every listen in [start_ts, end_ts) is a target, predicted from the true
    history before it (at most seq_len events). Windows advance by seq_len//2,
    so each target sees at least half a window of context when it exists.
    """
    stride = max(1, seq_len // 2)
    pending = []
    for u in users:
        sl = corpus.user_slice(u)
        lo, hi = period_bounds(corpus, u, start_ts, end_ts)
        first = max(lo, sl.start + 1)  # the very first event has no history
        for chunk_start in range(first, hi, stride):
            chunk_end = min(chunk_start + stride, hi)
            win_begin = max(sl.start, chunk_end - 1 - seq_len)
            idx = np.arange(win_begin, chunk_end - 1)
            targets = np.arange(chunk_start, chunk_end)
            pending.append((idx, targets - 1 - win_begin, targets))
            if len(pending) == batch_size:
                yield _make_chunk(corpus, pending, seq_len)
                pending = []
    if pending:
        yield _make_chunk(corpus, pending, seq_len)


def _make_chunk(corpus, pending, seq_len) -> EvalChunk:
    batch = gather(corpus, [p[0] for p in pending], seq_len)
    rows = np.concatenate([np.full(len(p[2]), r) for r, p in enumerate(pending)])
    cols = np.concatenate([p[1] for p in pending])
    target_idx = np.concatenate([p[2] for p in pending])
    return EvalChunk(batch, rows, cols, target_idx)


def users_with_period_events(corpus: Corpus, start_ts: int, end_ts: int | None) -> np.ndarray:
    users = []
    for u in range(corpus.n_users):
        lo, hi = period_bounds(corpus, u, start_ts, end_ts)
        if hi > lo and lo > corpus.user_offsets[u]:
            users.append(u)
    return np.asarray(users, dtype=np.int64)


def state_batches(corpus: Corpus, users, start_ts: int, seq_len: int, batch_size: int):
    """For user-level evaluation: the last seq_len events before start_ts.

    Yields (batch, users, last_col) where last_col is the position whose hidden
    state is the user state at the start of the period.
    """
    for i in range(0, len(users), batch_size):
        chunk = users[i:i + batch_size]
        rows, last = [], []
        for u in chunk:
            sl = corpus.user_slice(u)
            lo, _ = period_bounds(corpus, u, start_ts, None)
            idx = np.arange(max(sl.start, lo - seq_len), lo)
            rows.append(idx)
            last.append(len(idx) - 1)
        yield gather(corpus, rows, seq_len), np.asarray(chunk), np.asarray(last)


def relevant_items(corpus: Corpus, u: int, start_ts: int, end_ts: int | None) -> np.ndarray:
    """Distinct in-vocabulary items the user listened to positively in the period."""
    lo, hi = period_bounds(corpus, u, start_ts, end_ts)
    items = corpus.item[lo:hi]
    pos = (corpus.ratio[lo:hi] >= POSITIVE_RATIO) | (corpus.liked[lo:hi] == 1)
    keep = pos & (items != OOV_ID)
    return np.unique(items[keep])
