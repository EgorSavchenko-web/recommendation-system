"""Stage 1: GPT-2 next-item recommender (supervised), plus the MostPop baseline.

    python scripts/train_base.py --data-dir data/yambda-50m             # local
    python scripts/train_base.py --clearml --queue <queue>               # remote
    python scripts/train_base.py --data-dir data/yambda-50m --smoke      # tiny CPU run

Model selection uses validation NDCG@10 (teacher-forced next item); the test
week is scored once, with the selected checkpoint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Only stdlib + clearml are needed to enqueue a run; the training stack is
# imported after the hand-off, on the machine that actually trains.
from recsys_rl.clearml_io import (add_clearml_arguments, clearml_start,  # noqa: E402
                                  running_on_agent, upload_outputs)

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

    from recsys_rl.stage_base import train
    from recsys_rl.utils import setup_logging

    setup_logging(Path(args.output_dir) / "run.log")
    train(args, task, data_dir)
    upload_outputs(task, args.output_dir)


if __name__ == "__main__":
    main()
