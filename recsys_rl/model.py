"""GPT-2 sequential recommender over event tokens.

Input token = sum of embeddings of the item, its feedback (ratio bucket,
reaction), organic flag and duration bucket; GPT-2 adds positions. Output
logits are tied to the item embedding table, so token ids are item ids.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import GPT2Config, GPT2Model

from .data import (FLAG_UNKNOWN, N_DURATION_BUCKETS, N_RATIO_BUCKETS, OOV_ID, PAD_ID,
                   RATIO_UNKNOWN)

# Reaction token: 0 none, 1 like, 2 dislike, 3 hidden.
REACTION_UNKNOWN = 3


@dataclass
class ModelConfig:
    n_items: int
    n_positions: int = 512
    d_model: int = 256
    n_layer: int = 4
    n_head: int = 4
    dropout: float = 0.1
    feedback_dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


def reaction_code(liked: torch.Tensor, disliked: torch.Tensor) -> torch.Tensor:
    """Merge like/dislike flags into one categorical; dislike wins a tie."""
    code = liked.long().clamp(max=1)
    code = torch.where(disliked.long() == 1, torch.full_like(code, 2), code)
    unknown = (liked.long() == FLAG_UNKNOWN) | (disliked.long() == FLAG_UNKNOWN)
    return torch.where(unknown, torch.full_like(code, REACTION_UNKNOWN), code)


class EventEmbedding(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.item = nn.Embedding(cfg.n_items, d, padding_idx=PAD_ID)
        self.ratio = nn.Embedding(N_RATIO_BUCKETS + 1, d)
        self.reaction = nn.Embedding(REACTION_UNKNOWN + 1, d)
        self.organic = nn.Embedding(2, d)
        self.duration = nn.Embedding(N_DURATION_BUCKETS, d)
        self.feedback_dropout = cfg.feedback_dropout
        nn.init.normal_(self.item.weight, std=0.02)
        for table in (self.ratio, self.reaction, self.organic, self.duration):
            nn.init.normal_(table.weight, std=0.02)

    def forward(self, batch: dict) -> torch.Tensor:
        ratio = batch["ratio_bucket"].long()
        reaction = reaction_code(batch["liked"], batch["disliked"])
        if self.training and self.feedback_dropout > 0:
            # Teaches the model to act when feedback is unknown, which is the
            # situation for items it generated itself (multi-step rollouts).
            hide = torch.rand(ratio.shape, device=ratio.device) < self.feedback_dropout
            ratio = torch.where(hide, torch.full_like(ratio, RATIO_UNKNOWN), ratio)
            reaction = torch.where(hide, torch.full_like(reaction, REACTION_UNKNOWN), reaction)
        return (self.item(batch["item"].long()) + self.ratio(ratio) + self.reaction(reaction)
                + self.organic(batch["organic"].long()) + self.duration(batch["duration_bucket"].long()))


class SeqRecommender(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = EventEmbedding(cfg)
        gpt_cfg = GPT2Config(
            vocab_size=1,  # unused: inputs arrive as embeddings
            n_positions=cfg.n_positions, n_embd=cfg.d_model, n_layer=cfg.n_layer,
            n_head=cfg.n_head, resid_pdrop=cfg.dropout, embd_pdrop=cfg.dropout,
            attn_pdrop=cfg.dropout, bos_token_id=None, eos_token_id=None,
        )
        self.gpt = GPT2Model(gpt_cfg)

    def hidden(self, batch: dict, attention_mask=None, past_key_values=None,
               position_ids=None, use_cache: bool = False):
        out = self.gpt(inputs_embeds=self.embed(batch), attention_mask=attention_mask,
                       past_key_values=past_key_values, position_ids=position_ids,
                       use_cache=use_cache)
        if use_cache:
            return out.last_hidden_state, out.past_key_values
        return out.last_hidden_state

    def logits(self, h: torch.Tensor, mask_special: bool = True) -> torch.Tensor:
        logits = h @ self.embed.item.weight.t()
        if mask_special:
            logits[..., PAD_ID] = float("-inf")
            logits[..., OOV_ID] = float("-inf")
        return logits


def _chunk_ce_sum(h: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = (h @ weight.t()).float()
    logits[:, PAD_ID] = float("-inf")
    return F.cross_entropy(logits, targets, reduction="sum")


def chunked_cross_entropy(h: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor,
                          chunk: int = 4096) -> torch.Tensor:
    """Mean full-softmax CE without materialising all logits at once.

    Each chunk is checkpointed, so backward recomputes its logits instead of
    keeping [N, V] activations alive.
    """
    if h.shape[0] == 0:
        return h.sum() * 0.0
    total = h.new_zeros((), dtype=torch.float32)
    for start in range(0, h.shape[0], chunk):
        sl = slice(start, start + chunk)
        total = total + checkpoint(_chunk_ce_sum, h[sl], weight, targets[sl], use_reentrant=False)
    return total / h.shape[0]


def save_bundle(path, model: nn.Module, cfg: ModelConfig, extra: dict | None = None) -> None:
    torch.save({"model_config": cfg.to_dict(), "state_dict": model.state_dict(),
                "extra": extra or {}}, path)


def load_recommender(path, device="cpu") -> tuple[SeqRecommender, dict]:
    bundle = torch.load(path, map_location=device, weights_only=False)
    model = SeqRecommender(ModelConfig(**bundle["model_config"]))
    model.load_state_dict(bundle["state_dict"])
    return model.to(device), bundle.get("extra", {})
