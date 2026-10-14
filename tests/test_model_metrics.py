import numpy as np
import torch
import torch.nn.functional as F

from recsys_rl.data import PAD_ID
from recsys_rl.metrics import behaviour_stats, event_metrics, target_ranks, user_metrics
from recsys_rl.model import ModelConfig, SeqRecommender, chunked_cross_entropy


def _batch(B=2, L=6, n_items=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "item": torch.randint(2, n_items, (B, L), generator=g),
        "ratio_bucket": torch.randint(0, 11, (B, L), generator=g),
        "liked": torch.randint(0, 2, (B, L), generator=g),
        "disliked": torch.zeros(B, L, dtype=torch.long),
        "organic": torch.randint(0, 2, (B, L), generator=g),
        "duration_bucket": torch.randint(0, 7, (B, L), generator=g),
    }


def _model():
    torch.manual_seed(0)
    model = SeqRecommender(ModelConfig(n_items=20, n_positions=8, d_model=16, n_layer=2,
                                       n_head=2, dropout=0.0, feedback_dropout=0.0))
    return model.eval()


def test_model_is_causal():
    """Changing event t (item or its feedback) must not change outputs before t."""
    model = _model()
    a = _batch()
    b = {k: v.clone() for k, v in a.items()}
    b["item"][:, 4] = 3
    b["liked"][:, 4] = 1 - b["liked"][:, 4]
    with torch.no_grad():
        ha, hb = model.hidden(a), model.hidden(b)
    assert torch.allclose(ha[:, :4], hb[:, :4], atol=1e-6)
    assert not torch.allclose(ha[:, 4], hb[:, 4])


def test_special_tokens_are_never_recommended():
    model = _model()
    with torch.no_grad():
        logits = model.logits(model.hidden(_batch()))
    top = logits.topk(5, dim=-1).indices
    assert (top >= 2).all()


def test_chunked_ce_matches_full_ce():
    torch.manual_seed(0)
    h = torch.randn(37, 8, requires_grad=True)
    w = torch.randn(15, 8, requires_grad=True)
    t = torch.randint(1, 15, (37,))
    chunked = chunked_cross_entropy(h, w, t, chunk=10)
    logits = h @ w.t()
    logits[:, PAD_ID] = float("-inf")
    full = F.cross_entropy(logits, t)
    assert torch.allclose(chunked, full, atol=1e-5)
    g1 = torch.autograd.grad(chunked, (h, w))
    g2 = torch.autograd.grad(F.cross_entropy((h @ w.t()).masked_fill(
        torch.arange(15) == PAD_ID, float("-inf")), t), (h, w))
    for a, b in zip(g1, g2):
        assert torch.allclose(a, b, atol=1e-5)


def test_target_ranks():
    logits = torch.tensor([[0.1, 0.9, 0.5], [0.3, 0.2, 0.1]])
    assert target_ranks(logits, torch.tensor([2, 0])).tolist() == [2, 1]


def test_event_metrics_counts_oov_as_miss():
    m = event_metrics(np.array([1.0, np.inf]), ks=(10,))
    assert m["hit@10"] == 0.5
    assert np.isclose(m["ndcg@10"], 0.5)
    assert np.isclose(m["mrr"], 0.5)


def test_user_metrics_perfect_and_empty_hits():
    topk = np.array([[5, 6, 7], [1, 2, 3]])
    rel = [np.array([5, 6]), np.array([9])]
    m = user_metrics(topk, rel, ks=(3,))
    assert np.isclose(m["recall@3"], 0.5)
    assert np.isclose(m["ndcg@3"], 0.5)
    assert np.isclose(m["map@3"], 0.5)


def test_behaviour_detects_collapse():
    same = np.tile(np.arange(2, 12), (50, 1))
    stats = behaviour_stats(same, item_popularity=np.ones(100), item_length=np.ones(100),
                            item_artist=np.arange(100), n_catalog=100)
    assert stats["pairwise_jaccard"] == 1.0
    assert np.isclose(stats["top10_share"], 1.0)
