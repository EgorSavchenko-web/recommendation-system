"""Ranking metrics (event-level and user-level) and recommendation-behaviour statistics."""

from __future__ import annotations

import numpy as np
import torch


def target_ranks(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """1-based rank of each target among all items (ties count against it)."""
    target_logit = logits.gather(1, targets.view(-1, 1))
    return (logits >= target_logit).sum(dim=1)


def event_metrics(ranks: np.ndarray, ks=(10, 100)) -> dict:
    """Hit@K and NDCG@K for single-target next-item prediction.

    A target outside the vocabulary must be passed with rank = np.inf: it
    counts as a miss, so the numbers include the vocabulary-coverage ceiling.
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    if ranks.size == 0:
        return {}
    out = {"n": int(ranks.size)}
    for k in ks:
        hit = ranks <= k
        out[f"hit@{k}"] = float(hit.mean())
        out[f"ndcg@{k}"] = float(np.where(hit, 1.0 / np.log2(ranks + 1.0), 0.0).mean())
    out["mrr"] = float(np.where(np.isfinite(ranks), 1.0 / ranks, 0.0).mean())
    return out


def user_metrics(topk: np.ndarray, relevant: list[np.ndarray], ks=(10, 100)) -> dict:
    """Recall@K, NDCG@K, MAP@K of one ranked list per user against a relevant set.

    topk: [n_users, K_max] item ids, best first. Users with an empty relevant
    set must be filtered out by the caller.
    """
    if len(relevant) != len(topk):
        raise ValueError("topk and relevant must align.")
    out = {"n_users": len(relevant)}
    for k in ks:
        recall, ndcg, ap = [], [], []
        for rec, rel in zip(topk[:, :k], relevant):
            if rel.size == 0:
                raise ValueError("Empty relevant set; filter such users out.")
            hits = np.isin(rec, rel)
            recall.append(hits.sum() / rel.size)
            gains = hits / np.log2(np.arange(2, len(rec) + 2))
            ideal = (1.0 / np.log2(np.arange(2, min(rel.size, k) + 2))).sum()
            ndcg.append(gains.sum() / ideal)
            precision_at = np.cumsum(hits) / np.arange(1, len(rec) + 1)
            ap.append((precision_at * hits).sum() / min(rel.size, k))
        out[f"recall@{k}"] = float(np.mean(recall))
        out[f"ndcg@{k}"] = float(np.mean(ndcg))
        out[f"map@{k}"] = float(np.mean(ap))
    return out


def behaviour_stats(topk: np.ndarray, *, item_popularity: np.ndarray, item_length: np.ndarray,
                    item_artist: np.ndarray, n_catalog: int, seen: list[np.ndarray] | None = None,
                    k: int = 10) -> dict:
    """What the recommended lists look like, independent of relevance.

    popularity / length are averaged over recommended slots; coverage is the
    share of the catalogue that appears in anyone's top-k; `top10_share` is
    the share of all slots taken by the 10 most-recommended items (collapse
    indicator); `pairwise_jaccard` compares random user pairs.
    """
    lists = topk[:, :k]
    flat = lists.ravel()
    counts = np.bincount(flat, minlength=n_catalog)
    sorted_counts = np.sort(counts)[::-1]
    out = {
        "avg_popularity": float(item_popularity[flat].mean()),
        "median_popularity": float(np.median(item_popularity[flat])),
        "avg_track_length_s": float(item_length[flat].mean()),
        "coverage": float((counts > 0).sum() / n_catalog),
        "top10_share": float(sorted_counts[:10].sum() / max(flat.size, 1)),
        "entropy_bits": float(_entropy(counts)),
        "gini": float(_gini(counts)),
        "unique_artists_per_list": float(np.mean([len(np.unique(item_artist[r])) for r in lists])),
    }
    rng = np.random.default_rng(0)
    if len(lists) > 1:
        a = rng.integers(0, len(lists), 2000)
        b = rng.integers(0, len(lists), 2000)
        keep = a != b
        jac = [len(np.intersect1d(lists[i], lists[j])) / len(np.union1d(lists[i], lists[j]))
               for i, j in zip(a[keep], b[keep])]
        out["pairwise_jaccard"] = float(np.mean(jac))
    if seen is not None:
        out["repeat_share"] = float(np.mean([np.isin(r, s).mean() for r, s in zip(lists, seen)]))
    return out


def _entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _gini(counts: np.ndarray) -> float:
    x = np.sort(counts.astype(np.float64))
    n = x.size
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))
