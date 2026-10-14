import numpy as np
import polars as pl
import pytest

from recsys_rl.data import (
    BINS_PER_DAY, N_SPECIAL, OOV_ID, REACTION_WINDOW_BINS, DataConfig,
    attach_reactions, build_corpus, duration_bucket, ratio_bucket, split_points,
)


def _write_events(tmp_path, rows):
    df = pl.DataFrame(rows, schema={
        "uid": pl.UInt32, "timestamp": pl.UInt32, "item_id": pl.UInt32,
        "is_organic": pl.UInt8, "event_type": pl.Utf8,
        "played_ratio_pct": pl.UInt16, "track_length_seconds": pl.UInt32,
    }, orient="row")
    df.write_parquet(tmp_path / "multi_event.parquet")
    return tmp_path


def _listens(rows):
    """rows: (uid, item, ts) -> listens frame as build_corpus prepares it."""
    df = pl.DataFrame(rows, schema={"uid": pl.UInt32, "item_id": pl.UInt32,
                                    "timestamp": pl.Int64}, orient="row")
    df = df.sort("uid", "timestamp").with_row_index("listen_idx")
    same = pl.col("uid").shift(-1) == pl.col("uid")
    return df.with_columns(pl.when(same).then(pl.col("timestamp").shift(-1)).alias("next_ts"))


def _reactions(rows):
    return pl.DataFrame(rows, schema={"uid": pl.UInt32, "item_id": pl.UInt32,
                                      "timestamp": pl.Int64}, orient="row")


def test_split_points_cover_last_days():
    cfg = DataConfig(val_days=7, test_days=7)
    max_ts = 100 * BINS_PER_DAY
    val_start, test_start = split_points(max_ts, cfg)
    assert max_ts - test_start + 1 == 7 * BINS_PER_DAY
    assert test_start - val_start == 7 * BINS_PER_DAY


def test_split_points_reject_short_span():
    with pytest.raises(ValueError):
        split_points(5 * BINS_PER_DAY, DataConfig())


def test_buckets():
    assert ratio_bucket([0, 9, 10, 99, 100, 159]).tolist() == [0, 0, 1, 9, 10, 10]
    assert duration_bucket([30, 60, 61, 1000]).tolist() == [0, 1, 1, 6]


def test_reaction_after_next_listen_is_not_attached():
    # Listen of item 7 at t=100, next listen at t=110; like of 7 at t=115.
    listens = _listens([(1, 7, 100), (1, 8, 110)])
    idx = attach_reactions(listens, _reactions([(1, 7, 115)]))
    assert idx.size == 0, "a later like leaked into an earlier context"


def test_reaction_before_listen_is_attached():
    # Listen timestamps mark the end of play: a like just before belongs to it.
    listens = _listens([(1, 7, 100), (1, 8, 110)])
    idx = attach_reactions(listens, _reactions([(1, 8, 108)]))
    assert idx.tolist() == [1]


def test_reaction_outside_window_is_dropped():
    listens = _listens([(1, 7, 100)])
    far = 100 + REACTION_WINDOW_BINS + 1
    assert attach_reactions(listens, _reactions([(1, 7, far)])).size == 0


def test_reaction_picks_nearest_listen():
    listens = _listens([(1, 7, 100), (1, 9, 104), (1, 7, 106)])
    # Like of 7 at 105: 5 bins after listen 0, 1 bin before listen 2.
    idx = attach_reactions(listens, _reactions([(1, 7, 105)]))
    assert idx.tolist() == [2]


def test_reaction_never_crosses_users():
    listens = _listens([(1, 7, 100), (2, 7, 101)])
    idx = attach_reactions(listens, _reactions([(2, 7, 101)]))
    assert idx.tolist() == [1]


def _corpus_rows():
    day = BINS_PER_DAY
    rows = []
    # Train period: item 10 popular, item 11 once. Test period: item 99 popular.
    for t in range(5):
        rows.append((1, 1 * day + t, 10, 1, "listen", 100, 200))
    rows.append((1, 2 * day, 11, 0, "listen", 30, 90))
    rows.append((1, 2 * day + 1, 11, 0, "like", None, None))
    for t in range(6):
        rows.append((2, 3 * day + t, 10, 0, "listen", 50, 200))
    for t in range(20):
        rows.append((2, 29 * day + t, 99, 0, "listen", 100, 300))
    return rows


def test_corpus_vocab_uses_train_period_only(tmp_path):
    raw = _write_events(tmp_path, _corpus_rows())
    corpus = build_corpus(raw, DataConfig(vocab_size=10, val_days=7, test_days=7))
    vocab = set(corpus.vocab_raw_ids.tolist())
    assert 10 in vocab and 11 in vocab
    assert 99 not in vocab, "a test-only item entered the vocabulary"
    test_mask = corpus.ts >= corpus.test_start
    assert (corpus.item[test_mask] == OOV_ID).all()


def test_corpus_layout(tmp_path):
    raw = _write_events(tmp_path, _corpus_rows())
    corpus = build_corpus(raw, DataConfig(vocab_size=10))
    assert corpus.n_users == 2
    assert corpus.user_offsets[-1] == len(corpus.item)
    for u in range(corpus.n_users):
        ts = corpus.ts[corpus.user_slice(u)]
        assert (np.diff(ts) >= 0).all()
    assert corpus.liked.sum() == 1
    # Vocab ids start after the specials; most frequent item gets the first id.
    assert corpus.vocab_raw_ids[0] == 10
    assert corpus.item[0] == N_SPECIAL
    assert corpus.item_length[N_SPECIAL] == 200
