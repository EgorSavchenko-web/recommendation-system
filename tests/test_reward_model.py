import numpy as np
import torch

from recsys_rl.model import ModelConfig, SeqRecommender
from recsys_rl.reward_model import (SESSION_GAP_BINS, RewardConfig, RewardModel,
                                    continue_labels, reward_from_heads)


def _rm():
    torch.manual_seed(0)
    mcfg = ModelConfig(n_items=20, n_positions=8, d_model=16, n_layer=1, n_head=2,
                       dropout=0.0, feedback_dropout=0.0)
    rm = RewardModel(RewardConfig(model=mcfg.to_dict()), np.full(20, 200.0))
    rm.init_from_recommender(SeqRecommender(mcfg))
    return rm.eval()


def test_score_single_and_grouped_candidates_agree():
    rm = _rm()
    h = torch.randn(3, 16)
    items = torch.tensor([[2, 5], [7, 7], [9, 3]])
    grouped = rm.score(h, items)
    single = rm.score(h, items[:, 1])
    assert grouped.shape == (3, 2, 3)
    assert torch.allclose(grouped[:, 1], single, atol=1e-6)


def test_init_copies_recommender_item_table():
    mcfg = ModelConfig(n_items=20, n_positions=8, d_model=16, n_layer=1, n_head=2)
    base = SeqRecommender(mcfg)
    rm = RewardModel(RewardConfig(model=mcfg.to_dict()), np.full(20, 200.0))
    rm.init_from_recommender(base)
    assert torch.equal(rm.cand_item.weight, base.embed.item.weight)


def test_continue_labels_respect_users_and_horizon():
    offsets = np.array([0, 3, 5])
    ts = np.array([0, 10, 10 + SESSION_GAP_BINS + 1, 0, 50], dtype=np.int64)
    label, valid = continue_labels(offsets, ts, horizon_ts=40)
    # user 0: 0->10 continues; 10->371 is a new session but 371 >= horizon -> invalid;
    # last event of each user has no label.
    assert valid.tolist() == [True, False, False, False, False]
    assert label.tolist()[0] == 1
    label2, valid2 = continue_labels(offsets, ts, horizon_ts=10**9)
    assert valid2.tolist() == [True, True, False, True, False]
    assert label2.tolist() == [1, 0, 0, 1, 0]


def test_rewards():
    probs = torch.tensor([[0.5, 0.8, 0.9]])
    items = torch.tensor([3])
    length = torch.full((5,), 150.0)
    mean = torch.full((5,), 0.6)
    assert torch.isclose(reward_from_heads("paper", probs, items, length, mean), torch.tensor(0.58))
    assert torch.isclose(reward_from_heads("seconds", probs, items, length, mean), torch.tensor(0.4))
    assert torch.isclose(reward_from_heads("relative", probs, items, length, mean), torch.tensor(0.2))
