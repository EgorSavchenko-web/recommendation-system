"""Event table -> per-user token streams with a leakage-free temporal split.

One token per listen. A token carries the item and the feedback the user gave
to it (listened-ratio bucket, like, dislike), the organic flag and a track
duration bucket. Likes/dislikes are separate events in Yambda; each is
attached to the listen it refers to (see `attach_reactions`).

The corpus is stored CSR-style: flat per-event arrays sorted by
(uid, timestamp, file order) plus `user_offsets`, so a user's stream is
`arrays[off[u]:off[u+1]]`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Timestamps are 5-second bins.
BINS_PER_DAY = 86_400 // 5

PAD_ID = 0
OOV_ID = 1
N_SPECIAL = 2

# Ratio bucket: played_ratio_pct clipped to 100, // 10 -> 0..10; 11 = hidden.
N_RATIO_BUCKETS = 11
RATIO_UNKNOWN = N_RATIO_BUCKETS
# Like / dislike flags: 0, 1; 2 = hidden.
FLAG_UNKNOWN = 2
# Track duration edges in seconds -> 7 buckets.
DURATION_EDGES = np.array([60, 120, 180, 240, 300, 420], dtype=np.int64)
N_DURATION_BUCKETS = len(DURATION_EDGES) + 1

# A listen counts as positive at or above this played share (Yambda's own
# "listened" threshold is 50%).
POSITIVE_RATIO = 0.5


@dataclass
class DataConfig:
    vocab_size: int = 100_000
    val_days: int = 7
    test_days: int = 7


@dataclass
class Corpus:
    """Flat event arrays + CSR user offsets + vocabulary metadata."""
    item: np.ndarray          # int32 vocab id (OOV_ID for rare items)
    raw_item: np.ndarray      # uint32 original item id
    ts: np.ndarray            # int64 5s-bin timestamp
    ratio: np.ndarray         # float32 played share, clipped to [0, 1]
    ratio_bucket: np.ndarray  # int8
    liked: np.ndarray         # int8 0/1
    disliked: np.ndarray      # int8 0/1
    organic: np.ndarray       # int8 0/1
    duration_bucket: np.ndarray  # int8
    user_offsets: np.ndarray  # int64, len n_users + 1
    uids: np.ndarray          # uint32 original uid per CSR row
    vocab_raw_ids: np.ndarray  # uint32, raw id of vocab id k at [k - N_SPECIAL]
    item_length: np.ndarray   # float32 median track length (s) per vocab id
    item_artist: np.ndarray   # int64 artist id per vocab id (-1 unknown)
    val_start: int
    test_start: int
    stats: dict = field(default_factory=dict)

    ARRAYS = ("item", "raw_item", "ts", "ratio", "ratio_bucket", "liked", "disliked",
              "organic", "duration_bucket", "user_offsets", "uids", "vocab_raw_ids",
              "item_length", "item_artist")

    @property
    def n_users(self) -> int:
        return len(self.user_offsets) - 1

    @property
    def n_items(self) -> int:
        """Size of the item embedding table, specials included."""
        return len(self.vocab_raw_ids) + N_SPECIAL

    def user_slice(self, u: int) -> slice:
        return slice(int(self.user_offsets[u]), int(self.user_offsets[u + 1]))

    def save(self, path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "corpus.npz", **{k: getattr(self, k) for k in self.ARRAYS})
        meta = {"val_start": self.val_start, "test_start": self.test_start, "stats": self.stats}
        (path / "corpus_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @staticmethod
    def load(path) -> "Corpus":
        path = Path(path)
        arrays = np.load(path / "corpus.npz")
        meta = json.loads((path / "corpus_meta.json").read_text(encoding="utf-8"))
        return Corpus(**{k: arrays[k] for k in Corpus.ARRAYS},
                      val_start=meta["val_start"], test_start=meta["test_start"],
                      stats=meta["stats"])


def duration_bucket(seconds) -> np.ndarray:
    return np.searchsorted(DURATION_EDGES, np.asarray(seconds), side="right").astype(np.int8)


def ratio_bucket(ratio_pct) -> np.ndarray:
    return (np.minimum(np.asarray(ratio_pct), 100) // 10).astype(np.int8)


def split_points(max_ts: int, cfg: DataConfig) -> tuple[int, int]:
    """Global temporal cut-offs: [.., val_start) train, [val_start, test_start) val, rest test."""
    test_start = int(max_ts) - cfg.test_days * BINS_PER_DAY + 1
    val_start = test_start - cfg.val_days * BINS_PER_DAY
    if val_start <= 0:
        raise ValueError("Validation + test windows exceed the data span.")
    return val_start, test_start


def load_events(raw_dir) -> pl.DataFrame:
    path = Path(raw_dir) / "multi_event.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    # File order is (uid, timestamp); `order` keeps it as the deterministic
    # tie-break for events sharing a 5-second bin.
    columns = ["uid", "timestamp", "item_id", "is_organic", "event_type",
               "played_ratio_pct", "track_length_seconds"]
    return pl.read_parquet(path, columns=columns).with_row_index("order")


# A reaction is attributed to a listen of the same track at most this far
# away (10 min). Farther likes are library actions, not feedback on a play.
REACTION_WINDOW_BINS = 120


def attach_reactions(listens: pl.DataFrame, reactions: pl.DataFrame,
                     window: int = REACTION_WINDOW_BINS) -> np.ndarray:
    """Index (into `listens`) of the listen each reaction refers to.

    Measured on Yambda-50M, a like is as often logged shortly *before* the
    listen of the same track as after it (the listen timestamp looks like the
    end of playback), so both directions are searched and the nearest listen
    within `window` wins.

    Leakage rule: a reaction found *after* its listen is kept only if it
    happened no later than that user's next listen; otherwise it would reach
    back into a context where it had not happened yet. Same 5-second bin as
    the next listen is accepted, since order inside a bin is unknowable. A
    reaction *before* its listen is in the past by construction.
    Unmatched reactions are dropped and counted by the caller.
    """
    if reactions.height == 0:
        return np.empty(0, dtype=np.int64)

    left = (reactions.select("uid", "item_id", "timestamp")
            .with_row_index("rid").sort("uid", "item_id", "timestamp"))
    # Only (user, item) pairs that have a reaction can match; keeps the sort small.
    pairs = left.select("uid", "item_id").unique()
    right = (listens.select("uid", "item_id", "timestamp", "listen_idx", "next_ts")
             .join(pairs, on=["uid", "item_id"], how="semi")
             .rename({"timestamp": "listen_ts"})
             .with_columns(pl.col("listen_ts").alias("timestamp"))
             .sort("uid", "item_id", "timestamp"))

    back = (left.join_asof(right, on="timestamp", by=["uid", "item_id"], strategy="backward")
            .filter(pl.col("listen_idx").is_not_null()
                    & (pl.col("timestamp") - pl.col("listen_ts") <= window)
                    & (pl.col("next_ts").is_null() | (pl.col("timestamp") <= pl.col("next_ts"))))
            .select("rid", "listen_idx", (pl.col("timestamp") - pl.col("listen_ts")).alias("gap")))
    fwd = (left.join_asof(right, on="timestamp", by=["uid", "item_id"], strategy="forward")
           .filter(pl.col("listen_idx").is_not_null()
                   & (pl.col("listen_ts") - pl.col("timestamp") <= window))
           .select("rid", "listen_idx", (pl.col("listen_ts") - pl.col("timestamp")).alias("gap")))

    best = (pl.concat([back, fwd]).sort("rid", "gap", "listen_idx")
            .group_by("rid", maintain_order=True).first())
    return best["listen_idx"].unique().to_numpy().astype(np.int64)


def build_vocab(listens: pl.DataFrame, val_start: int, size: int) -> np.ndarray:
    """Top-`size` items by train-period listen count; ties broken by item id."""
    counts = (listens.filter(pl.col("timestamp") < val_start)
              .group_by("item_id").len()
              .sort(["len", "item_id"], descending=[True, False])
              .head(size))
    return counts["item_id"].to_numpy().astype(np.uint32)


def build_corpus(raw_dir, cfg: DataConfig | None = None) -> Corpus:
    cfg = cfg or DataConfig()
    events = load_events(raw_dir)
    val_start, test_start = split_points(events["timestamp"].max(), cfg)

    likes = events.filter(pl.col("event_type") == "like").select("uid", "item_id", "timestamp")
    dislikes = events.filter(pl.col("event_type") == "dislike").select("uid", "item_id", "timestamp")
    # The file is already sorted by (uid, timestamp); the filter keeps that
    # order, and `order` breaks ties identically to a full sort.
    listens = (events.filter(pl.col("event_type") == "listen").drop("event_type")
               .sort("uid", "timestamp", "order")
               .drop("order")
               .with_row_index("listen_idx"))
    del events
    same_user = pl.col("uid").shift(-1) == pl.col("uid")
    listens = listens.with_columns(
        pl.when(same_user).then(pl.col("timestamp").shift(-1)).otherwise(None).alias("next_ts")
    )
    n = listens.height
    logger.info("Listens: %d; val_start=%d test_start=%d", n, val_start, test_start)

    liked = np.zeros(n, dtype=np.int8)
    disliked = np.zeros(n, dtype=np.int8)
    like_idx = attach_reactions(listens, likes)
    dislike_idx = attach_reactions(listens, dislikes)
    liked[like_idx] = 1
    disliked[dislike_idx] = 1
    n_likes, n_dislikes = likes.height, dislikes.height
    del likes, dislikes
    listens = listens.drop("next_ts")

    vocab = build_vocab(listens, val_start, cfg.vocab_size)
    raw_item = listens["item_id"].to_numpy().astype(np.uint32)
    # Map raw id -> vocab id via a sorted lookup.
    order = np.argsort(vocab)
    pos = np.searchsorted(vocab[order], raw_item)
    pos = np.clip(pos, 0, len(vocab) - 1)
    hit = vocab[order][pos] == raw_item
    item = np.where(hit, order[pos] + N_SPECIAL, OOV_ID).astype(np.int32)

    ratio_pct = listens["played_ratio_pct"].fill_null(0).to_numpy()
    length = listens["track_length_seconds"].fill_null(0).to_numpy()
    ts = listens["timestamp"].to_numpy().astype(np.int64)
    uid = listens["uid"].to_numpy()

    # Per-vocab median length, from the train period only.
    train = ts < val_start
    item_length = np.zeros(len(vocab) + N_SPECIAL, dtype=np.float32)
    med = (pl.DataFrame({"item": item[train], "length": length[train]})
           .filter(pl.col("item") >= N_SPECIAL)
           .group_by("item").agg(pl.col("length").median()))
    item_length[med["item"].to_numpy()] = med["length"].to_numpy()

    boundaries = np.flatnonzero(np.diff(uid.astype(np.int64))) + 1
    user_offsets = np.concatenate([[0], boundaries, [n]]).astype(np.int64)

    stats = {
        "n_listens": n,
        "n_users": len(user_offsets) - 1,
        "liked_listens": int(len(like_idx)), "likes_total": n_likes,
        "disliked_listens": int(len(dislike_idx)), "dislikes_total": n_dislikes,
        "vocab_size": int(len(vocab)),
        "oov_share_train": float(np.mean(item[train] == OOV_ID)),
        "oov_share_test": float(np.mean(item[ts >= test_start] == OOV_ID)),
        "n_test_listens": int(np.sum(ts >= test_start)),
        "n_val_listens": int(np.sum((ts >= val_start) & (ts < test_start))),
    }
    logger.info("Corpus stats: %s", stats)

    return Corpus(
        item=item,
        raw_item=raw_item,
        ts=ts,
        ratio=np.clip(ratio_pct / 100.0, 0.0, 1.0).astype(np.float32),
        ratio_bucket=ratio_bucket(ratio_pct),
        liked=liked,
        disliked=disliked,
        organic=listens["is_organic"].to_numpy().astype(np.int8),
        duration_bucket=duration_bucket(length),
        user_offsets=user_offsets,
        uids=uid[user_offsets[:-1]].astype(np.uint32),
        vocab_raw_ids=vocab,
        item_length=item_length,
        item_artist=np.full(len(vocab) + N_SPECIAL, -1, dtype=np.int64),
        val_start=val_start,
        test_start=test_start,
        stats=stats,
    )


def attach_artists(corpus: Corpus, raw_dir) -> None:
    """Fill `item_artist` (first artist id per item); used only for analysis."""
    path = Path(raw_dir) / "artist_item_mapping.parquet"
    if not path.exists():
        logger.warning("No artist mapping at %s; diversity by artist unavailable.", path)
        return
    mapping = (pl.read_parquet(path).group_by("item_id").agg(pl.col("artist_id").min()))
    lookup = dict(zip(mapping["item_id"].to_list(), mapping["artist_id"].to_list()))
    for k, raw in enumerate(corpus.vocab_raw_ids.tolist()):
        corpus.item_artist[k + N_SPECIAL] = lookup.get(raw, -1)


def load_or_build(raw_dir, cache_dir, cfg: DataConfig | None = None) -> Corpus:
    """Build once, then reuse the cache for the same config."""
    cfg = cfg or DataConfig()
    cache = Path(cache_dir) / f"v{cfg.vocab_size}_val{cfg.val_days}_test{cfg.test_days}"
    if (cache / "corpus.npz").exists():
        logger.info("Loading cached corpus from %s", cache)
        return Corpus.load(cache)
    corpus = build_corpus(raw_dir, cfg)
    attach_artists(corpus, raw_dir)
    corpus.stats["config"] = asdict(cfg)
    corpus.save(cache)
    return corpus
