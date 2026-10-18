"""Shared evaluation protocol, used unchanged for the base, RL and baseline models."""

from __future__ import annotations

import contextlib
import logging

import numpy as np
import torch

from .batching import (next_item_chunks, relevant_items, state_batches,
                       users_with_period_events)
from .data import N_SPECIAL, OOV_ID, POSITIVE_RATIO, Corpus
from .metrics import behaviour_stats, event_metrics, target_ranks, user_metrics

logger = logging.getLogger(__name__)

RANK_CHUNK = 2048


def autocast(device: torch.device, enabled: bool = True):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def item_popularity(corpus: Corpus) -> np.ndarray:
    """Train-period listen count per vocabulary id."""
    train = corpus.ts < corpus.val_start
    return np.bincount(corpus.item[train], minlength=corpus.n_items).astype(np.float64)


def sample_users(users: np.ndarray, max_users: int | None, seed: int = 0) -> np.ndarray:
    if max_users is None or len(users) <= max_users:
        return users
    return np.sort(np.random.default_rng(seed).choice(users, max_users, replace=False))


@torch.no_grad()
def evaluate_next_item(model, corpus: Corpus, start_ts: int, end_ts: int | None, *, seq_len: int,
                       batch_size: int, device, amp: bool = True, max_users=None) -> dict:
    """Teacher-forced next-item metrics over every listen in the period.

    Slices: all targets; non-organic (recommendation-driven) targets; positive
    targets (played >= 50% or liked); in-vocabulary targets only.
    """
    model.eval()
    users = sample_users(users_with_period_events(corpus, start_ts, end_ts), max_users)
    ranks, idx_all = [], []
    for chunk in next_item_chunks(corpus, users, start_ts, end_ts, seq_len, batch_size):
        batch = to_device(chunk.batch, device)
        with autocast(device, amp):
            h = model.hidden(batch, attention_mask=batch["attention_mask"])
        rows = torch.as_tensor(chunk.rows, device=device)
        cols = torch.as_tensor(chunk.cols, device=device)
        states = h[rows, cols].float()
        targets = torch.as_tensor(corpus.item[chunk.target_idx], device=device).long()
        # Rank in slices: [positions x vocabulary] logits are the memory peak.
        r = torch.cat([target_ranks(model.logits(states[i:i + RANK_CHUNK]),
                                    targets[i:i + RANK_CHUNK])
                       for i in range(0, len(targets), RANK_CHUNK)]).float().cpu().numpy()
        r[corpus.item[chunk.target_idx] == OOV_ID] = np.inf
        ranks.append(r)
        idx_all.append(chunk.target_idx)

    if not ranks:
        return {}
    ranks = np.concatenate(ranks)
    idx = np.concatenate(idx_all)
    positive = (corpus.ratio[idx] >= POSITIVE_RATIO) | (corpus.liked[idx] == 1)
    slices = {
        "all": np.ones_like(positive),
        "nonorganic": corpus.organic[idx] == 0,
        "positive": positive,
        "in_vocab": corpus.item[idx] != OOV_ID,
    }
    out = {name: event_metrics(ranks[mask]) for name, mask in slices.items()}
    out["n_users"] = int(len(users))
    return out


@torch.no_grad()
def recommend_topk(model, corpus: Corpus, users: np.ndarray, start_ts: int, *, seq_len: int,
                   batch_size: int, device, amp: bool = True, k: int = 100) -> np.ndarray:
    """Top-k item ids per user from the state at `start_ts` (true history only)."""
    model.eval()
    out = []
    for batch, _, last in state_batches(corpus, users, start_ts, seq_len, batch_size):
        batch = to_device(batch, device)
        with autocast(device, amp):
            h = model.hidden(batch, attention_mask=batch["attention_mask"])
        state = h[torch.arange(len(last), device=device), torch.as_tensor(last, device=device)]
        logits = model.logits(state.float())
        out.append(torch.topk(logits, k, dim=1).indices.cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, k), dtype=np.int64)


def evaluate_user_level(model, corpus: Corpus, start_ts: int, end_ts: int | None, *, seq_len: int,
                        batch_size: int, device, amp: bool = True, k: int = 100,
                        max_users=None, topk_fn=None) -> tuple[dict, np.ndarray, np.ndarray]:
    """Recall/NDCG/MAP@K of one list per user vs. the period's positive items.

    `topk_fn(users) -> [n, k]` lets the RL and multi-step variants be scored
    through exactly the same code path. Returns (metrics, users, topk).
    """
    users = users_with_period_events(corpus, start_ts, end_ts)
    relevant = [relevant_items(corpus, u, start_ts, end_ts) for u in users]
    keep = np.asarray([r.size > 0 for r in relevant])
    users = users[keep]
    relevant = [r for r, k_ in zip(relevant, keep) if k_]
    chosen = sample_users(np.arange(len(users)), max_users)
    users = users[chosen]
    relevant = [relevant[i] for i in chosen]

    if topk_fn is None:
        topk = recommend_topk(model, corpus, users, start_ts, seq_len=seq_len,
                              batch_size=batch_size, device=device, amp=amp, k=k)
    else:
        topk = topk_fn(users)

    seen = []
    for u in users:
        sl = corpus.user_slice(u)
        hist = corpus.item[sl][corpus.ts[sl] < start_ts]
        seen.append(np.unique(hist[hist >= N_SPECIAL]))

    metrics = user_metrics(topk, relevant, ks=(10, min(100, k)))
    metrics["behaviour@10"] = behaviour_stats(
        topk, item_popularity=item_popularity(corpus), item_length=corpus.item_length,
        item_artist=corpus.item_artist, n_catalog=corpus.n_items, seen=seen, k=10,
    )
    return metrics, users, topk
