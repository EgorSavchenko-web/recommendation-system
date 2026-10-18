import numpy as np
import torch
import torch.nn as nn

from recsys_rl.model import ModelConfig, SeqRecommender
from recsys_rl.stage_rl import RLConfig, rl_update


class OneItemEnsemble(nn.Module):
    """Stand-in RM ensemble: P(like) = 1 only for one target item."""

    def __init__(self, target: int):
        super().__init__()
        self.target = target

    def probs(self, batch, positions, items):
        like = (items == self.target).float()
        return torch.stack([like, torch.zeros_like(like), torch.zeros_like(like)], -1).unsqueeze(0)


class Tables:
    def __init__(self, n):
        self.item_length = torch.full((n,), 200.0)
        self.item_mean_ratio = torch.zeros(n)
        self.duration_bucket = torch.zeros(n, dtype=torch.long)


def _setup(n_items=12):
    torch.manual_seed(0)
    cfg = ModelConfig(n_items=n_items, n_positions=16, d_model=16, n_layer=1, n_head=2,
                      dropout=0.0, feedback_dropout=0.0)
    policy, reference = SeqRecommender(cfg), SeqRecommender(cfg)
    reference.load_state_dict(policy.state_dict())
    batch = {k: torch.zeros(4, 6, dtype=torch.long) for k in
             ("ratio_bucket", "liked", "disliked", "organic", "duration_bucket")}
    batch["item"] = torch.randint(2, n_items, (4, 6))
    return policy.eval(), reference.eval(), batch, Tables(n_items)


def _prob(policy, batch, item):
    with torch.no_grad():
        return torch.softmax(policy.logits(policy.hidden(batch)[:, -1]), -1)[:, item].mean().item()


def test_policy_gradient_moves_mass_to_rewarded_item():
    policy, reference, batch, tables = _setup()
    target = 7
    before = _prob(policy, batch, target)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    cfg = RLConfig(reward="like", beta=0.0, group=8, horizon=1)
    for _ in range(40):
        rl_update(policy, reference, opt, batch, cfg, OneItemEnsemble(target), tables,
                  torch.device("cpu"), False)
    assert _prob(policy, batch, target) > before + 0.2


def test_strong_kl_keeps_policy_close_to_reference():
    policy, reference, batch, tables = _setup()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    free, tied = [], []
    for beta, store in ((0.0, free), (50.0, tied)):
        policy.load_state_dict(reference.state_dict())
        opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
        cfg = RLConfig(reward="like", beta=beta, group=8, horizon=2, feedback="unknown")
        for _ in range(30):
            stats = rl_update(policy, reference, opt, batch, cfg, OneItemEnsemble(7), tables,
                              torch.device("cpu"), False)
        store.append(stats["kl"])
    assert tied[0] < free[0]
    assert np.isfinite(free[0]) and np.isfinite(tied[0])
