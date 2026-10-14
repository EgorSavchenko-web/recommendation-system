"""Stage 1: GPT-2 next-item recommender (supervised), plus the MostPop baseline.

    python scripts/train_base.py --data-dir data/yambda-50m             # local
    python scripts/train_base.py --clearml --queue <queue>               # remote
    python scripts/train_base.py --data-dir data/yambda-50m --smoke      # tiny CPU run

Model selection uses validation NDCG@10 (teacher-forced next item); the test
week is scored once, with the selected checkpoint.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys_rl.batching import TrainWindows  # noqa: E402
from recsys_rl.clearml_io import (add_clearml_arguments, clearml_start, report_scalars,  # noqa: E402
                                  running_on_agent, upload_outputs)
from recsys_rl.data import N_SPECIAL, DataConfig, load_or_build  # noqa: E402
from recsys_rl.evaluate import (autocast, evaluate_next_item, evaluate_user_level,  # noqa: E402
                                item_popularity, to_device)
from recsys_rl.metrics import event_metrics  # noqa: E402
from recsys_rl.model import ModelConfig, SeqRecommender, chunked_cross_entropy, save_bundle  # noqa: E402
from recsys_rl.utils import (cosine_with_warmup, param_groups, require_gpu_on_agent,  # noqa: E402
                             resolve_device, save_json, set_seed, setup_logging)

logger = logging.getLogger("train_base")
TASK_NAME = "stage1_base_gpt2"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 1: GPT-2 next-item recommender.")
    p.add_argument("--data-dir", default="./data/yambda-50m")
    p.add_argument("--cache-dir", default="./data/cache")
    p.add_argument("--output-dir", default="./outputs/base")
    p.add_argument("--vocab-size", type=int, default=100_000)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--feedback-dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=1000, help="Steps between validations.")
    p.add_argument("--val-users", type=int, default=2000,
                   help="Users sampled for in-training validation (final eval uses all).")
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--patience", type=int, default=4, help="Validations without improvement.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny model, few steps, few users: checks the pipeline end to end.")
    add_clearml_arguments(p)
    return p


def apply_smoke(args) -> None:
    args.seq_len, args.d_model, args.layers, args.heads = 64, 32, 1, 2
    args.batch_size, args.epochs, args.eval_every, args.warmup_steps = 8, 1, 20, 5
    args.val_users = 50


def mostpop_metrics(corpus, start_ts, end_ts, args, device) -> dict:
    """Same protocol, everyone gets the train-period popularity ranking."""
    pop = item_popularity(corpus)
    pop[:N_SPECIAL] = -1
    order = np.argsort(-pop, kind="stable")
    rank_of = np.empty_like(order)
    rank_of[order] = np.arange(1, len(order) + 1)

    class _Pop:
        """Duck-typed model for the user-level path."""

    from recsys_rl.batching import users_with_period_events
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
    from recsys_rl.batching import period_bounds
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


def main() -> None:
    raw = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(raw)
    if args.smoke:
        apply_smoke(args)

    task, data_dir = None, Path(args.data_dir)
    if args.clearml or running_on_agent():
        task, args, data_dir = clearml_start(TASK_NAME, raw, parser, args)
        if args.smoke:
            apply_smoke(args)

    setup_logging(Path(args.output_dir) / "run.log")
    train(args, task, data_dir)
    upload_outputs(task, args.output_dir)


if __name__ == "__main__":
    main()
