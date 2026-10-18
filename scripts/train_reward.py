"""Stage 2: reward-model ensemble on user half A plus an independent "gold" RM on half B.

    python scripts/train_reward.py --base-task <stage-1 task id> --clearml --queue <q>
    python scripts/train_reward.py --base-checkpoint outputs/base/model.pt --data-dir ...

Members are given as HALF:SEED. The RL stage optimises against the half-A
ensemble; the half-B model never sees RL and serves as the independent judge
used to detect reward hacking (proxy reward up, gold reward down).
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

TASK_NAME = "stage2_reward_models"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 2: reward models.")
    p.add_argument("--data-dir", default="./data/yambda-50m")
    p.add_argument("--cache-dir", default="./data/cache")
    p.add_argument("--output-dir", default="./outputs/reward")
    p.add_argument("--base-checkpoint", default=None, help="Local stage-1 model.pt.")
    p.add_argument("--base-task", default=None, help="ClearML id of the stage-1 task.")
    p.add_argument("--members", default="A:0,A:1,A:2,B:0", help="Comma-separated HALF:SEED.")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--val-users", type=int, default=2000)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--smoke", action="store_true")
    add_clearml_arguments(p)
    return p


def main() -> None:
    raw = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(raw)

    task, data_dir = None, Path(args.data_dir)
    if args.clearml or running_on_agent():
        task, args, data_dir = clearml_start(TASK_NAME, raw, parser, args)

    # Checked after the hand-off: on the agent argv arrives via the task.
    if not (args.base_checkpoint or args.base_task):
        parser.error("give --base-task or --base-checkpoint")

    from recsys_rl.stage_reward import run
    from recsys_rl.utils import setup_logging

    setup_logging(Path(args.output_dir) / "run.log")
    run(args, task, data_dir)
    upload_outputs(task, args.output_dir)


if __name__ == "__main__":
    main()
