"""Stage 2 training logic; `scripts/train_reward.py` is the launcher."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

from .batching import TrainWindows, next_item_chunks, users_with_period_events
from .clearml_io import report_scalars, resolve_checkpoint
from .data import OOV_ID, DataConfig, load_or_build
from .evaluate import autocast, sample_users, to_device
from .model import load_recommender
from .reward_model import (HEADS, RewardConfig, RewardModel, continue_labels, rm_loss,
                           save_rm)
from .utils import (cosine_with_warmup, param_groups, require_gpu_on_agent, resolve_device,
                    save_json, set_seed)

logger = logging.getLogger("train_reward")
HALF_SEED = 12345


def user_halves(n_users: int) -> np.ndarray:
    """0 for half A, 1 for half B; fixed permutation, independent of run seeds."""
    half = np.zeros(n_users, dtype=np.int8)
    half[np.random.default_rng(HALF_SEED).permutation(n_users)[n_users // 2:]] = 1
    return half


def window_users(corpus, windows: TrainWindows) -> np.ndarray:
    return np.searchsorted(corpus.user_offsets, windows.windows[:, 0], side="right") - 1


def labels_for(corpus, idx: np.ndarray, cont, cont_valid) -> dict:
    return {
        "like": torch.from_numpy(corpus.liked[idx].astype(np.int64)),
        "ratio": torch.from_numpy(corpus.ratio[idx]),
        "cont": torch.from_numpy(cont[idx].astype(np.int64)),
        "cont_mask": torch.from_numpy(cont_valid[idx]),
    }


@torch.no_grad()
def evaluate_rm(rm, corpus, start, end, users, *, seq_len, batch_size, device, amp) -> dict:
    """Held-out-time quality of each head on the real next track."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    rm.eval()
    cont, valid = continue_labels(corpus.user_offsets, corpus.ts, np.iinfo(np.int64).max)
    probs, idx_all = [], []
    for chunk in next_item_chunks(corpus, users, start, end, seq_len, batch_size):
        keep = corpus.item[chunk.target_idx] != OOV_ID
        if not keep.any():
            continue
        batch = to_device(chunk.batch, device)
        with autocast(device, amp):
            h = rm.history(batch)
        rows = torch.as_tensor(chunk.rows[keep], device=device)
        cols = torch.as_tensor(chunk.cols[keep], device=device)
        items = torch.as_tensor(corpus.item[chunk.target_idx[keep]], device=device).long()
        probs.append(torch.sigmoid(rm.score(h[rows, cols].float(), items).float()).cpu().numpy())
        idx_all.append(chunk.target_idx[keep])
    p = np.concatenate(probs)
    idx = np.concatenate(idx_all)
    like, ratio = corpus.liked[idx], corpus.ratio[idx]
    out = {
        "n": int(len(idx)),
        "like_base_rate": float(like.mean()),
        "ratio_mae": float(np.abs(p[:, 1] - ratio).mean()),
        "ratio_mae_mean_baseline": float(np.abs(ratio.mean() - ratio).mean()),
        "ratio_corr": float(np.corrcoef(p[:, 1], ratio)[0, 1]),
        "skip_auc": float(roc_auc_score(ratio < 0.5, -p[:, 1])),
        "reward_mae": float(np.abs((p[:, 0] + 0.1 * p[:, 1]) - (like + 0.1 * ratio)).mean()),
    }
    if 0 < like.sum() < len(like):
        out["like_roc_auc"] = float(roc_auc_score(like, p[:, 0]))
        out["like_pr_auc"] = float(average_precision_score(like, p[:, 0]))
    v = valid[idx]
    if v.any() and 0 < cont[idx][v].sum() < v.sum():
        out["continue_auc"] = float(roc_auc_score(cont[idx][v], p[v, 2]))
        out["continue_base_rate"] = float(cont[idx][v].mean())
    # Calibration of the ratio head in 10 bins of predicted value.
    bins = np.clip((p[:, 1] * 10).astype(int), 0, 9)
    out["ratio_calibration"] = [
        {"bin": b, "pred": float(p[bins == b, 1].mean()), "true": float(ratio[bins == b].mean()),
         "n": int((bins == b).sum())} for b in range(10) if (bins == b).any()
    ]
    return out


def train_member(half: str, seed: int, args, corpus, base, windows, win_half, cont, cont_valid,
                 device, task, out: Path) -> dict:
    set_seed(seed)
    amp = not args.no_amp
    cfg = RewardConfig(model=base.cfg.to_dict(), user_half=half, seed=seed)
    rm = RewardModel(cfg, corpus.item_length).to(device)
    rm.init_from_recommender(base)
    rm.backbone.embed.feedback_dropout = 0.0

    ids_all = np.flatnonzero(win_half == (0 if half == "A" else 1))
    steps_per_epoch = len(ids_all) // args.batch_size
    total = min(60, steps_per_epoch) if args.smoke else steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(param_groups(rm, args.weight_decay), lr=args.lr)
    sched = cosine_with_warmup(opt, args.warmup_steps, total)

    halves = user_halves(corpus.n_users)
    val_users = users_with_period_events(corpus, corpus.val_start, corpus.test_start)
    val_users = sample_users(val_users[halves[val_users] == (0 if half == "A" else 1)],
                             args.val_users, seed=seed)
    ev = dict(seq_len=base.cfg.n_positions, batch_size=args.eval_batch_size, device=device, amp=amp)

    name = f"rm_{half}_s{seed}"
    best, stale, step, history = np.inf, 0, 0, []
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    while step < total and stale < args.patience:
        order = rng.permutation(ids_all)
        for b in range(steps_per_epoch):
            if step >= total or stale >= args.patience:
                break
            wid = order[b * args.batch_size:(b + 1) * args.batch_size]
            raw = windows.batch(wid)
            begins = windows.windows[wid, 0]
            L = raw["targets"].shape[1]
            target_idx = begins[:, None] + 1 + np.arange(L)[None, :]
            sel_np = raw["target_mask"].numpy()
            lab = labels_for(corpus, target_idx[sel_np], cont, cont_valid)
            batch = to_device(raw, device)
            rm.train()
            with autocast(device, amp):
                h = rm.history(batch)
                sel = batch["target_mask"]
                logits = rm.score(h[sel], batch["targets"][sel])
            lab = to_device(lab, device)
            loss, parts = rm_loss(logits, lab["like"], lab["ratio"], lab["cont"], lab["cont_mask"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                logger.info("%s step %d/%d loss %.4f %s (%.0fs)", name, step, total, loss.item(),
                            parts, time.perf_counter() - t0)
                report_scalars(task, f"{name}/train", {"loss": loss.item(), **parts}, step)
            if step % args.eval_every == 0 or step == total:
                val = evaluate_rm(rm, corpus, corpus.val_start, corpus.test_start, val_users, **ev)
                # Selection on the two dense heads; like is too sparse to steer it.
                score = val["ratio_mae"] - val.get("continue_auc", 0.0) * 0.1
                history.append({"step": step, **{k: v for k, v in val.items()
                                                 if not isinstance(v, list)}})
                report_scalars(task, f"{name}/val", history[-1], step)
                logger.info("%s val step %d: %s", name, step, history[-1])
                if score < best:
                    best, stale = score, 0
                    save_rm(out / f"{name}.pt", rm, extra={"step": step, "val": val})
                else:
                    stale += 1

    bundle = torch.load(out / f"{name}.pt", map_location=device, weights_only=False)
    rm.load_state_dict(bundle["state_dict"])
    test_users = users_with_period_events(corpus, corpus.test_start, None)
    if args.smoke:
        test_users = test_users[:100]
    result = {"history": history, "selected_step": bundle["extra"]["step"]}
    for label, mask in (("own_half", halves[test_users] == (0 if half == "A" else 1)),
                        ("other_half", halves[test_users] != (0 if half == "A" else 1))):
        result[f"test_{label}"] = evaluate_rm(rm, corpus, corpus.test_start, None,
                                              test_users[mask], **ev)
        logger.info("%s test %s: %s", name, label, {k: v for k, v in result[f"test_{label}"].items()
                                                     if not isinstance(v, list)})
    return result


def run(args, task, data_dir: Path) -> dict:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    require_gpu_on_agent(task, device, args.allow_cpu)
    base_path = resolve_checkpoint(args.base_checkpoint, args.base_task)
    base, base_extra = load_recommender(base_path, device)
    base_args = base_extra.get("args", {})
    corpus = load_or_build(data_dir, args.cache_dir,
                           DataConfig(vocab_size=base_args.get("vocab_size", 100_000)))
    if corpus.n_items != base.cfg.n_items:
        raise RuntimeError("Corpus vocabulary does not match the base model.")

    windows = TrainWindows(corpus, base.cfg.n_positions)
    win_half = user_halves(corpus.n_users)[window_users(corpus, windows)]
    cont, cont_valid = continue_labels(corpus.user_offsets, corpus.ts, corpus.val_start)
    logger.info("Windows: %d (half A %d); continue label valid on %.1f%% of events",
                len(windows), int((win_half == 0).sum()), 100 * cont_valid.mean())

    results = {"heads": HEADS, "members": {}}
    for spec in args.members.split(","):
        half, seed = spec.split(":")
        if half not in ("A", "B"):
            raise ValueError(f"Bad member '{spec}': half must be A or B.")
        results["members"][spec] = train_member(half, int(seed), args, corpus, base, windows,
                                                win_half, cont, cont_valid, device, task, out)
        save_json(results, out / "metrics.json")
    return results
