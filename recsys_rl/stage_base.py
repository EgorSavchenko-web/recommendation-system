"""Stage 1 training logic; `scripts/train_base.py` is the launcher."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

from .batching import TrainWindows, period_bounds, users_with_period_events
from .clearml_io import report_scalars
from .data import N_SPECIAL, DataConfig, load_or_build
from .evaluate import (autocast, evaluate_next_item, evaluate_user_level, item_popularity,
                       to_device)
from .metrics import event_metrics
from .model import ModelConfig, SeqRecommender, chunked_cross_entropy, save_bundle
from .utils import (cosine_with_warmup, param_groups, require_gpu_on_agent, resolve_device,
                    save_json, set_seed)

logger = logging.getLogger("train_base")


def mostpop_metrics(corpus, start_ts, end_ts, args, device) -> dict:
    """Same protocol, everyone gets the train-period popularity ranking."""
    pop = item_popularity(corpus)
    pop[:N_SPECIAL] = -1
    order = np.argsort(-pop, kind="stable")
    rank_of = np.empty_like(order)
    rank_of[order] = np.arange(1, len(order) + 1)

    class _Pop:
        """Duck-typed model for the user-level path."""

    users = users_with_period_events(corpus, start_ts, end_ts)
    idx = np.concatenate([np.arange(*_period(corpus, u, start_ts, end_ts)) for u in users])
    ranks = rank_of[corpus.item[idx]].astype(np.float64)
    ranks[corpus.item[idx] < N_SPECIAL] = np.inf
    event = {"all": event_metrics(ranks)}
    user, _, _ = evaluate_user_level(
        _Pop(), corpus, start_ts, end_ts, seq_len=args.seq_len, batch_size=args.eval_batch_size,
        device=device, topk_fn=lambda us: np.tile(order[:100], (len(us), 1)),
    )
    return {"next_item": event, "user_level": user}


def _period(corpus, u, start_ts, end_ts):
    lo, hi = period_bounds(corpus, u, start_ts, end_ts)
    return max(lo, int(corpus.user_offsets[u]) + 1), hi


def train(args, task, data_dir: Path) -> dict:
    set_seed(args.seed)
    device = resolve_device(args.device)
    require_gpu_on_agent(task, device, args.allow_cpu)
    amp = not args.no_amp
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    corpus = load_or_build(data_dir, args.cache_dir, DataConfig(vocab_size=args.vocab_size))
    windows = TrainWindows(corpus, args.seq_len)
    logger.info("Train windows: %d (seq_len %d)", len(windows), args.seq_len)

    cfg = ModelConfig(n_items=corpus.n_items, n_positions=args.seq_len, d_model=args.d_model,
                      n_layer=args.layers, n_head=args.heads, dropout=args.dropout,
                      feedback_dropout=args.feedback_dropout)
    model = SeqRecommender(cfg).to(device)
    logger.info("Parameters: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)

    steps_per_epoch = len(windows) // args.batch_size
    total = steps_per_epoch * args.epochs
    if args.smoke:
        total = min(total, 60)
    optimizer = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr,
                                  betas=(0.9, 0.98))
    scheduler = cosine_with_warmup(optimizer, args.warmup_steps, total)

    val_kw = dict(seq_len=args.seq_len, batch_size=args.eval_batch_size, device=device, amp=amp)
    best, stale, step, history = -1.0, 0, 0, []
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()

    for epoch in range(args.epochs):
        order = rng.permutation(len(windows))
        model.train()
        for b in range(steps_per_epoch):
            if step >= total:
                break
            batch = to_device(windows.batch(order[b * args.batch_size:(b + 1) * args.batch_size]),
                              device)
            with autocast(device, amp):
                h = model.hidden(batch, attention_mask=batch["attention_mask"])
                sel = batch["target_mask"]
                loss = chunked_cross_entropy(h[sel], model.embed.item.weight,
                                             batch["targets"][sel])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % 50 == 0:
                logger.info("epoch %d step %d/%d loss %.4f gnorm %.2f lr %.2e (%.0fs)", epoch, step,
                            total, loss.item(), grad_norm.item(), scheduler.get_last_lr()[0],
                            time.perf_counter() - t0)
                report_scalars(task, "train", {"loss": loss.item(), "grad_norm": grad_norm.item(),
                                               "lr": scheduler.get_last_lr()[0]}, step)

            if step % args.eval_every == 0 or step == total:
                val = evaluate_next_item(model, corpus, corpus.val_start, corpus.test_start,
                                         max_users=args.val_users, **val_kw)
                score = val["all"]["ndcg@10"]
                history.append({"step": step, "epoch": epoch, "loss": loss.item(),
                                "val": val["all"]})
                report_scalars(task, "val", val["all"], step)
                logger.info("val step %d: %s", step, val["all"])
                if score > best:
                    best, stale = score, 0
                    save_bundle(out / "model.pt", model, cfg,
                                extra={"step": step, "val_ndcg@10": score, "args": vars(args)})
                else:
                    stale += 1
                model.train()
                if stale >= args.patience:
                    logger.info("Early stop at step %d.", step)
                    break
        if stale >= args.patience or step >= total:
            break

    bundle = torch.load(out / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(bundle["state_dict"])
    logger.info("Best checkpoint: step %s, val NDCG@10 %.5f", bundle["extra"]["step"], best)

    eval_users = 200 if args.smoke else None
    results = {"selected_step": bundle["extra"]["step"], "history": history,
               "corpus_stats": corpus.stats}
    for name, (start, end) in {"val": (corpus.val_start, corpus.test_start),
                               "test": (corpus.test_start, None)}.items():
        nxt = evaluate_next_item(model, corpus, start, end, max_users=eval_users, **val_kw)
        usr, users, topk = evaluate_user_level(model, corpus, start, end, max_users=eval_users,
                                               **val_kw)
        results[name] = {"next_item": nxt, "user_level": usr}
        results[f"mostpop_{name}"] = mostpop_metrics(corpus, start, end, args, device)
        np.savez_compressed(out / f"topk_{name}.npz", users=users, topk=topk)
        logger.info("%s next-item: %s", name, nxt["all"])
        logger.info("%s user-level: %s", name, {k: v for k, v in usr.items() if "@" in k})

    save_json(results, out / "metrics.json")
    return results

