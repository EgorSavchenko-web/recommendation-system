"""Stage 3: RL fine-tuning of the GPT-2 recommender against the reward-model ensemble.

Policy gradient with a leave-one-out group baseline (RLOO): for every state G
continuations are sampled, scored by the RM ensemble, and each one's
advantage is its return minus the mean return of the other G-1. The policy is
regularised by the exact KL to the frozen base model, computed over the full
item vocabulary at every generated position.

A state is a real train-period context of fixed length C that ends right
before a recommendation-driven (non-organic) event, i.e. a moment where the
recommender was in control. Rollouts of T > 1 steps append generated items
with either hidden or RM-predicted feedback.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .batching import FIELDS, gather, relevant_items, state_batches, users_with_period_events
from .clearml_io import report_scalars
from .data import FLAG_UNKNOWN, N_SPECIAL, RATIO_UNKNOWN, Corpus, duration_bucket
from .evaluate import (autocast, evaluate_next_item, evaluate_user_level, sample_users,
                       to_device)
from .model import SeqRecommender, save_bundle
from .reward_model import RewardEnsemble, reward_from_heads
from .utils import save_json, set_seed

logger = logging.getLogger("train_rl")


@dataclass
class RLConfig:
    reward: str = "paper"
    beta: float = 0.05            # KL weight
    uncertainty: float = 0.0      # lambda: reward = mean - lambda * std over the ensemble
    group: int = 8                # samples per state
    horizon: int = 1              # rollout length T
    gamma: float = 0.9
    feedback: str = "predicted"   # feedback appended to generated items: predicted | unknown
    states: int = 32              # states per step
    steps: int = 600
    lr: float = 1e-5
    context: int = 256
    seed: int = 0

    @property
    def name(self) -> str:
        unc = f"_u{self.uncertainty:g}" if self.uncertainty else ""
        hor = f"_T{self.horizon}{self.feedback[0]}" if self.horizon > 1 else ""
        return f"{self.reward}_b{self.beta:g}{unc}{hor}_s{self.seed}"


class RewardTables:
    """Item-level tensors the reward definitions need, on device."""

    def __init__(self, corpus: Corpus, device):
        train = corpus.ts < corpus.val_start
        sums = np.bincount(corpus.item[train], weights=corpus.ratio[train], minlength=corpus.n_items)
        counts = np.bincount(corpus.item[train], minlength=corpus.n_items)
        mean_ratio = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
        self.item_mean_ratio = torch.as_tensor(mean_ratio, dtype=torch.float32, device=device)
        self.item_length = torch.as_tensor(corpus.item_length, dtype=torch.float32, device=device)
        self.duration_bucket = torch.as_tensor(duration_bucket(corpus.item_length.astype(np.int64)),
                                               device=device)


def eligible_state_ends(corpus: Corpus, context: int, horizon_ts: int) -> np.ndarray:
    """Absolute indices e such that [e - context, e) is one user's history,
    event e is non-organic and happened before `horizon_ts`."""
    ends = []
    for u in range(corpus.n_users):
        sl = corpus.user_slice(u)
        idx = np.arange(sl.start + context, sl.stop)
        keep = (corpus.organic[idx] == 0) & (corpus.ts[idx] < horizon_ts)
        ends.append(idx[keep])
    return np.concatenate(ends) if ends else np.empty(0, dtype=np.int64)


def context_batch(corpus: Corpus, ends: np.ndarray, context: int) -> dict:
    batch = gather(corpus, [np.arange(e - context, e) for e in ends], context)
    batch.pop("attention_mask")
    return batch


def repeat_rows(batch: dict, times: int) -> dict:
    return {k: v.repeat_interleave(times, dim=0) for k, v in batch.items()}


def append_token(batch: dict, items: torch.Tensor, tables: RewardTables,
                 probs: torch.Tensor | None, feedback: str) -> dict:
    """Append one generated event per row; feedback is hidden or RM-predicted."""
    n = items.shape[0]
    if feedback == "predicted" and probs is not None:
        ratio_b = (probs[:, 1].clamp(0, 1) * 100).long().div(10, rounding_mode="floor").clamp(max=10)
        liked = (probs[:, 0] > 0.5).long()
        disliked = torch.zeros_like(liked)
    else:
        ratio_b = torch.full((n,), RATIO_UNKNOWN, dtype=torch.long, device=items.device)
        liked = torch.full((n,), FLAG_UNKNOWN, dtype=torch.long, device=items.device)
        disliked = liked.clone()
    new = {
        "item": items, "ratio_bucket": ratio_b, "liked": liked, "disliked": disliked,
        "organic": torch.zeros_like(items), "duration_bucket": tables.duration_bucket[items],
    }
    return {k: torch.cat([batch[k], new[k].view(-1, 1).to(batch[k].dtype)], dim=1) for k in FIELDS}


def score(ensemble: RewardEnsemble, batch: dict, items: torch.Tensor, cfg_reward: str,
          tables: RewardTables, device, amp: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Reward per member [M, N, K] and mean probabilities [N, K, 3] for candidates items [N, K]
    placed after the last position of each row."""
    n, length = batch["item"].shape
    positions = torch.stack([torch.arange(n, device=device),
                             torch.full((n,), length - 1, device=device)], dim=1)
    with autocast(device, amp):
        probs = ensemble.probs(batch, positions, items)
    rewards = reward_from_heads(cfg_reward, probs, items.unsqueeze(0).expand(probs.shape[0], -1, -1),
                                tables.item_length, tables.item_mean_ratio)
    return rewards, probs.mean(0)


def combine(rewards: torch.Tensor, uncertainty: float) -> torch.Tensor:
    if rewards.shape[0] > 1 and uncertainty:
        return rewards.mean(0) - uncertainty * rewards.std(0)
    return rewards.mean(0)


@torch.no_grad()
def rollout(policy, proxy, batch, cfg: RLConfig, tables, device, amp):
    """Sample cfg.group trajectories of cfg.horizon steps for each context row."""
    seqs = repeat_rows(batch, cfg.group)
    actions, rewards = [], []
    for _ in range(cfg.horizon):
        with autocast(device, amp):
            h = policy.hidden(seqs)[:, -1]
        logits = policy.logits(h.float())
        a = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(1)
        r_members, probs = score(proxy, seqs, a.view(-1, 1), cfg.reward, tables, device, amp)
        rewards.append(combine(r_members, cfg.uncertainty).squeeze(1))
        actions.append(a)
        seqs = append_token(seqs, a, tables, probs[:, 0], cfg.feedback)
    return seqs, torch.stack(actions, 1), torch.stack(rewards, 1)


def rl_update(policy, reference, optimizer, batch, cfg: RLConfig, proxy, tables, device, amp) -> dict:
    policy.eval()  # sampling without dropout
    seqs, actions, rewards = rollout(policy, proxy, batch, cfg, tables, device, amp)
    discount = cfg.gamma ** torch.arange(cfg.horizon, device=device, dtype=torch.float32)
    returns = (rewards * discount).sum(1).view(-1, cfg.group)
    # Leave-one-out baseline inside each group.
    baseline = (returns.sum(1, keepdim=True) - returns) / (cfg.group - 1)
    advantage = (returns - baseline).view(-1)

    # Dropout stays off: the gradient must be taken under the same policy that sampled.
    policy.embed.feedback_dropout = 0.0
    inputs = {k: v[:, :-1] for k, v in seqs.items()}  # positions predicting each action
    C = batch["item"].shape[1]
    with autocast(device, amp):
        h = policy.hidden(inputs)[:, C - 1:]
        with torch.no_grad():
            h_ref = reference.hidden(inputs)[:, C - 1:]
    # Special tokens have probability 0 under both models; dropping their
    # columns keeps every log-prob finite, so KL and its gradient stay finite.
    logp = torch.log_softmax(policy.logits(h.float(), mask_special=False)[..., N_SPECIAL:], dim=-1)
    logq = torch.log_softmax(reference.logits(h_ref.float(), mask_special=False)[..., N_SPECIAL:],
                             dim=-1)
    logp_a = logp.gather(-1, (actions - N_SPECIAL).unsqueeze(-1)).squeeze(-1)  # [BG, T]
    kl = (logp.exp() * (logp - logq)).sum(-1)                                   # [BG, T]

    pg_loss = -(advantage.unsqueeze(1).detach() * logp_a).sum(1).mean()
    kl_loss = kl.sum(1).mean()
    loss = pg_loss + cfg.beta * kl_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    entropy = -(logp.detach().exp() * logp.detach()).sum(-1).mean()
    return {"loss": loss.item(), "pg_loss": pg_loss.item(), "kl": kl.mean().item(),
            "reward": rewards.mean().item(), "return": returns.mean().item(),
            "entropy": entropy.item(), "grad_norm": gnorm.item(),
            "unique_actions": int(actions.unique().numel())}


# ---- evaluation -------------------------------------------------------------

@torch.no_grad()
def slate_rewards(corpus, users, start_ts, topk, ensembles: dict, cfg_reward: str, tables,
                  context: int, device, amp, k: int = 10, batch_size: int = 128) -> dict:
    """Mean reward of each user's top-k list under every RM ensemble (proxy, gold),
    for the optimised reward and for the paper reward."""
    out = {f"{name}_{r}": [] for name in ensembles for r in {cfg_reward, "paper"}}
    out.update({f"{name}_disagreement": [] for name in ensembles})
    pos = {u: i for i, u in enumerate(users)}
    for batch, us, last in state_batches(corpus, users, start_ts, context, batch_size):
        batch = to_device(batch, device)
        batch.pop("attention_mask")
        items = torch.as_tensor(topk[[pos[u] for u in us], :k], device=device).long()
        positions = torch.stack([torch.arange(len(us), device=device),
                                 torch.as_tensor(last, device=device)], dim=1)
        for name, ens in ensembles.items():
            with autocast(device, amp):
                probs = ens.probs(batch, positions, items)
            for r in {cfg_reward, "paper"}:
                vals = reward_from_heads(r, probs, items.unsqueeze(0).expand(probs.shape[0], -1, -1),
                                         tables.item_length, tables.item_mean_ratio)
                out[f"{name}_{r}"].append(vals.mean(0).mean(1).cpu().numpy())
            out[f"{name}_disagreement"].append(probs[..., 1].std(0).mean(1).cpu().numpy()
                                               if probs.shape[0] > 1 else np.zeros(len(us)))
    return {k_: float(np.concatenate(v).mean()) for k_, v in out.items() if v}


@torch.no_grad()
def realised_feedback(corpus: Corpus, users, start_ts, topk, k: int = 10) -> dict:
    """Ground truth where it exists: of the recommended items the user actually
    played in the period, how fully were they played and how often liked."""
    ratios, likes, hits = [], [], 0
    for row, u in zip(topk[:, :k], users):
        sl = corpus.user_slice(u)
        mask = corpus.ts[sl] >= start_ts
        items = corpus.item[sl][mask]
        hit = np.isin(items, row)
        hits += int(np.isin(row, items).sum())
        ratios.append(corpus.ratio[sl][mask][hit])
        likes.append(corpus.liked[sl][mask][hit])
    r = np.concatenate(ratios) if ratios else np.empty(0)
    lk = np.concatenate(likes) if likes else np.empty(0)
    return {"played_hits": int(hits), "played_events": int(r.size),
            "mean_ratio_when_played": float(r.mean()) if r.size else float("nan"),
            "like_rate_when_played": float(lk.mean()) if lk.size else float("nan")}


@torch.no_grad()
def multistep_eval(policy, proxy, gold, corpus, users, start_ts, cfg: RLConfig, tables,
                   device, amp, steps: int = 5, feedback: str = "predicted",
                   batch_size: int = 128) -> list[dict]:
    """Greedy T-step generation from the true state at start_ts.

    Per step: proxy and gold reward of the generated item, ensemble
    disagreement (an out-of-distribution signal for the RM), and whether the
    item is among the user's actual positive items of the period.
    """
    users = np.asarray([u for u in users
                        if corpus.user_offsets[u + 1] - corpus.user_offsets[u] > cfg.context])
    per_step = [{"proxy": [], "gold": [], "disagreement": [], "precision": [], "repeat": []}
                for _ in range(steps)]
    for i in range(0, len(users), batch_size):
        us = users[i:i + batch_size]
        ends = []
        for u in us:
            sl = corpus.user_slice(u)
            ends.append(sl.start + int(np.searchsorted(corpus.ts[sl], start_ts)))
        ends = np.asarray(ends)
        ok = ends - corpus.user_offsets[us] >= cfg.context
        us, ends = us[ok], ends[ok]
        if len(us) == 0:
            continue
        seqs = to_device(context_batch(corpus, ends, cfg.context), device)
        rel = [relevant_items(corpus, u, start_ts, None) for u in us]
        hist = [set(corpus.item[corpus.user_offsets[u]:e].tolist()) for u, e in zip(us, ends)]
        for t in range(steps):
            with autocast(device, amp):
                h = policy.hidden(seqs)[:, -1]
            a = policy.logits(h.float()).argmax(-1)
            r_p, probs = score(proxy, seqs, a.view(-1, 1), cfg.reward, tables, device, amp)
            r_g, _ = score(gold, seqs, a.view(-1, 1), cfg.reward, tables, device, amp)
            a_np = a.cpu().numpy()
            per_step[t]["proxy"].append(r_p.mean(0).squeeze(1).cpu().numpy())
            per_step[t]["gold"].append(r_g.mean(0).squeeze(1).cpu().numpy())
            per_step[t]["disagreement"].append(r_p.std(0).squeeze(1).cpu().numpy()
                                               if r_p.shape[0] > 1 else np.zeros(len(us)))
            per_step[t]["precision"].append(np.array([x in r for x, r in zip(a_np, rel)], float))
            per_step[t]["repeat"].append(np.array([x in s for x, s in zip(a_np, hist)], float))
            seqs = append_token(seqs, a, tables, probs[:, 0], feedback)
    return [{k: float(np.concatenate(v).mean()) for k, v in step.items() if v} for step in per_step]


def evaluate_policy(policy, corpus, proxy, gold, cfg: RLConfig, tables, device, amp, *,
                    eval_users: int | None, batch_size: int) -> dict:
    ensembles = {"proxy": proxy, "gold": gold}
    kw = dict(seq_len=policy.cfg.n_positions, batch_size=batch_size, device=device, amp=amp)
    out = {}
    for split, (start, end) in {"val": (corpus.val_start, corpus.test_start),
                                "test": (corpus.test_start, None)}.items():
        user, users, topk = evaluate_user_level(policy, corpus, start, end, max_users=eval_users, **kw)
        out[split] = {
            "user_level": user,
            "next_item": evaluate_next_item(policy, corpus, start, end, max_users=eval_users, **kw),
            "slate_reward@10": slate_rewards(corpus, users, start, topk, ensembles, cfg.reward,
                                             tables, cfg.context, device, amp),
            "realised@10": realised_feedback(corpus, users, start, topk),
        }
        out[split]["_topk"] = (users, topk)
    test_users = sample_users(users_with_period_events(corpus, corpus.test_start, None),
                              min(eval_users or 2000, 2000), seed=1)
    out["multistep_test"] = {
        fb: multistep_eval(policy, proxy, gold, corpus, test_users, corpus.test_start, cfg,
                           tables, device, amp, feedback=fb)
        for fb in ("predicted", "unknown")
    }
    return out


def train_config(cfg: RLConfig, base: SeqRecommender, base_state: dict, corpus: Corpus,
                 state_ends: np.ndarray, proxy, gold, tables, device, amp, task, out_dir: Path,
                 eval_users, eval_batch_size: int, log_every: int = 25) -> dict:
    set_seed(cfg.seed)
    policy = SeqRecommender(base.cfg).to(device)
    policy.load_state_dict(base_state)
    policy.eval()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=0.0)
    rng = np.random.default_rng(cfg.seed)
    history = []
    t0 = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        ends = rng.choice(state_ends, cfg.states, replace=False)
        batch = to_device(context_batch(corpus, ends, cfg.context), device)
        stats = rl_update(policy, base, optimizer, batch, cfg, proxy, tables, device, amp)
        if step % log_every == 0 or step == cfg.steps:
            stats["step"] = step
            history.append(stats)
            report_scalars(task, f"rl/{cfg.name}", stats, step)
            logger.info("[%s] step %d %s (%.0fs)", cfg.name, step,
                        {k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()},
                        time.perf_counter() - t0)

    policy.eval()
    metrics = evaluate_policy(policy, corpus, proxy, gold, cfg, tables, device, amp,
                              eval_users=eval_users, batch_size=eval_batch_size)
    run_dir = out_dir / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        users, topk = metrics[split].pop("_topk")
        np.savez_compressed(run_dir / f"topk_{split}.npz", users=users, topk=topk)
    result = {"config": asdict(cfg), "history": history, **metrics}
    save_json(result, run_dir / "metrics.json")
    save_bundle(run_dir / "policy.pt", policy, policy.cfg, extra={"rl_config": asdict(cfg)})
    _log_summary(cfg.name, result)
    return result


def evaluate_reference(base, corpus, proxy, gold, tables, device, amp, out_dir: Path,
                       eval_users, eval_batch_size, reward: str, context: int) -> dict:
    """The base model through the exact same RL evaluation, as the beta = infinity point."""
    cfg = RLConfig(reward=reward, beta=float("inf"), steps=0, context=context)
    metrics = evaluate_policy(base, corpus, proxy, gold, cfg, tables, device, amp,
                              eval_users=eval_users, batch_size=eval_batch_size)
    for split in ("val", "test"):
        metrics[split].pop("_topk")
    result = {"config": {**asdict(cfg), "beta": "base"}, **metrics}
    save_json(result, out_dir / f"base_{reward}" / "metrics.json")
    _log_summary(f"base_{reward}", result)
    return result


def _log_summary(name: str, r: dict) -> None:
    t = r["test"]
    logger.info("SUMMARY %s | recall@10 %.4f ndcg@10 %.4f | slate %s | beh %s", name,
                t["user_level"]["recall@10"], t["user_level"]["ndcg@10"],
                {k: round(v, 4) for k, v in t["slate_reward@10"].items()},
                {k: round(t["user_level"]["behaviour@10"][k], 3)
                 for k in ("avg_popularity", "avg_track_length_s", "coverage", "top10_share",
                           "pairwise_jaccard", "repeat_share")})


def parse_grid(args) -> list[RLConfig]:
    configs = []
    for reward in args.rewards.split(","):
        for beta in [float(b) for b in args.betas.split(",")]:
            for unc in [float(u) for u in args.uncertainty.split(",")]:
                for seed in [int(s) for s in args.seeds.split(",")]:
                    configs.append(RLConfig(
                        reward=reward, beta=beta, uncertainty=unc, group=args.group,
                        horizon=args.horizon, gamma=args.gamma, feedback=args.feedback,
                        states=args.states, steps=args.steps, lr=args.lr, context=args.context,
                        seed=seed))
    return configs


def run(args, task, data_dir: Path) -> dict:
    from .clearml_io import fetch_artifact, resolve_checkpoint
    from .data import DataConfig, load_or_build
    from .model import load_recommender
    from .reward_model import load_rm
    from .utils import require_gpu_on_agent, resolve_device

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    require_gpu_on_agent(task, device, args.allow_cpu)
    amp = not args.no_amp

    base, base_extra = load_recommender(resolve_checkpoint(args.base_checkpoint, args.base_task),
                                        device)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    base_state = {k: v.clone() for k, v in base.state_dict().items()}
    corpus = load_or_build(data_dir, args.cache_dir,
                           DataConfig(vocab_size=base_extra.get("args", {}).get("vocab_size", 100_000)))

    def rm_path(spec: str) -> Path:
        half, seed = spec.split(":")
        name = f"rm_{half}_s{seed}.pt"
        if args.rm_dir:
            return Path(args.rm_dir) / name
        return fetch_artifact(args.rm_task, f"outputs/{name}")

    proxy = RewardEnsemble([load_rm(rm_path(s), corpus.item_length, device)
                            for s in args.proxy_members.split(",")]).eval()
    gold = RewardEnsemble([load_rm(rm_path(s), corpus.item_length, device)
                           for s in args.gold_members.split(",")]).eval()
    for p in list(proxy.parameters()) + list(gold.parameters()):
        p.requires_grad_(False)
    tables = RewardTables(corpus, device)

    state_ends = eligible_state_ends(corpus, args.context, corpus.val_start)
    logger.info("Eligible RL states: %d", len(state_ends))
    eval_users = 300 if args.smoke else args.eval_users

    results = {}
    configs = parse_grid(args)
    if args.eval_base:
        for reward in sorted({c.reward for c in configs}):
            results[f"base_{reward}"] = evaluate_reference(
                base, corpus, proxy, gold, tables, device, amp, out, eval_users,
                args.eval_batch_size, reward, args.context)
    for cfg in configs:
        r = train_config(cfg, base, base_state, corpus, state_ends, proxy, gold, tables, device,
                         amp, task, out, eval_users, args.eval_batch_size)
        results[cfg.name] = {k: v for k, v in r.items() if k != "history"}
        save_json(results, out / "summary.json")
    return results
