"""User-feedback (reward) model and the reward definitions built on it.

The RM reads the history with the same GPT-2 backbone as the policy and
scores a candidate next track: P(like), expected listened ratio, and
P(session continues after it). One forward pass over the history scores any
number of candidates, which is what RL with sampled groups needs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import N_DURATION_BUCKETS, DURATION_EDGES
from .model import ModelConfig, SeqRecommender

HEADS = ("like", "ratio", "continue")
# Paper weighting (Matveev et al. 2026, like : ratio = 2 : 8); continue added.
LOSS_WEIGHTS = {"like": 0.2, "ratio": 0.6, "continue": 0.2}

# A session continues if the next listen starts within 30 minutes.
SESSION_GAP_BINS = 360


@dataclass
class RewardConfig:
    model: dict
    head_hidden: int = 256
    user_half: str = "A"
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class RewardModel(nn.Module):
    def __init__(self, cfg: RewardConfig, item_length: np.ndarray):
        super().__init__()
        self.cfg = cfg
        mcfg = ModelConfig(**cfg.model)
        self.backbone = SeqRecommender(mcfg)
        d = mcfg.d_model
        self.cand_item = nn.Embedding(mcfg.n_items, d)
        self.cand_duration = nn.Embedding(N_DURATION_BUCKETS, d)
        self.head = nn.Sequential(nn.Linear(3 * d, cfg.head_hidden), nn.GELU(),
                                  nn.Linear(cfg.head_hidden, len(HEADS)))
        # Candidate duration comes from the item table, never from the event,
        # so training and RL scoring see the same input.
        lengths = torch.as_tensor(item_length, dtype=torch.float32)
        edges = torch.as_tensor(DURATION_EDGES, dtype=torch.float32)
        self.register_buffer("item_duration_bucket",
                             torch.bucketize(lengths, edges, right=True), persistent=False)

    def init_from_recommender(self, recommender: SeqRecommender) -> None:
        self.backbone.load_state_dict(recommender.state_dict())
        self.cand_item.weight.data.copy_(recommender.embed.item.weight.data)

    def history(self, batch: dict, **kw) -> torch.Tensor:
        return self.backbone.hidden(batch, attention_mask=batch.get("attention_mask"), **kw)

    def score(self, h: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """h: [N, d], items: [N] or [N, G] -> logits [..., 3] in HEADS order."""
        if items.dim() == 2:
            h = h.unsqueeze(1).expand(-1, items.shape[1], -1)
        c = self.cand_item(items) + self.cand_duration(self.item_duration_bucket[items])
        return self.head(torch.cat([h, c, h * c], dim=-1))


def rm_loss(logits: torch.Tensor, like: torch.Tensor, ratio: torch.Tensor,
            cont: torch.Tensor, cont_mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
    logits = logits.float()
    parts = {
        "like": F.binary_cross_entropy_with_logits(logits[:, 0], like.float()),
        # Soft target in [0, 1]: BCE with logits is a proper loss for a share.
        "ratio": F.binary_cross_entropy_with_logits(logits[:, 1], ratio.float()),
    }
    if cont_mask.any():
        parts["continue"] = F.binary_cross_entropy_with_logits(
            logits[cont_mask, 2], cont[cont_mask].float())
    else:
        parts["continue"] = logits.new_zeros(())
    total = sum(LOSS_WEIGHTS[k] * v for k, v in parts.items())
    return total, {k: float(v.detach()) for k, v in parts.items()}


def continue_labels(user_offsets: np.ndarray, ts: np.ndarray, horizon_ts: int):
    """(label, valid) per event: does the user's next listen follow within the gap?

    Valid only when the next listen is before `horizon_ts`, so a training
    label never peeks into the validation/test period. A user's last event has
    no label.
    """
    n = len(ts)
    nxt = np.empty(n, dtype=np.int64)
    nxt[:-1] = ts[1:]
    nxt[-1] = -1
    last = np.zeros(n, dtype=bool)
    last[user_offsets[1:] - 1] = True
    valid = ~last & (nxt < horizon_ts)
    label = valid & (nxt - ts <= SESSION_GAP_BINS)
    return label.astype(np.int8), valid


# ---- rewards ---------------------------------------------------------------

REWARDS = ("paper", "ratio", "seconds", "like", "relative", "continue")


def reward_from_heads(name: str, probs: torch.Tensor, items: torch.Tensor,
                      item_length: torch.Tensor, item_mean_ratio: torch.Tensor) -> torch.Tensor:
    """Scalar reward per candidate from RM probabilities [..., 3].

    paper    : 1{like} + 0.1 * ratio (Matveev et al. 2026)
    ratio    : expected listened share
    seconds  : expected listened time, in units of 5 minutes
    like     : P(like)
    relative : expected share minus the item's train-period mean share
    continue : P(session continues after this track)
    """
    like, ratio, cont = probs[..., 0], probs[..., 1], probs[..., 2]
    if name == "paper":
        return like + 0.1 * ratio
    if name == "ratio":
        return ratio
    if name == "seconds":
        return ratio * item_length[items] / 300.0
    if name == "like":
        return like
    if name == "relative":
        return ratio - item_mean_ratio[items]
    if name == "continue":
        return cont
    raise ValueError(f"Unknown reward '{name}'. Choose from {REWARDS}.")


def save_rm(path, model: RewardModel, extra: dict | None = None) -> None:
    torch.save({"reward_config": model.cfg.to_dict(), "state_dict": model.state_dict(),
                "extra": extra or {}}, path)


def load_rm(path, item_length: np.ndarray, device="cpu") -> RewardModel:
    bundle = torch.load(path, map_location=device, weights_only=False)
    model = RewardModel(RewardConfig(**bundle["reward_config"]), item_length)
    model.load_state_dict(bundle["state_dict"])
    return model.to(device).eval()


class RewardEnsemble(nn.Module):
    """Mean and spread of several RMs' probabilities over the same candidates."""

    def __init__(self, members: list[RewardModel]):
        super().__init__()
        self.members = nn.ModuleList(members)

    @torch.no_grad()
    def probs(self, batch: dict, positions: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """[n_members, N, G, 3] probabilities; positions index (row, col) states."""
        out = []
        for m in self.members:
            h = m.history(batch)[positions[:, 0], positions[:, 1]]
            out.append(torch.sigmoid(m.score(h, items).float()))
        return torch.stack(out)
