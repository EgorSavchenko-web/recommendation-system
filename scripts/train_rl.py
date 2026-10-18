"""Stage 3: RL fine-tuning against the reward-model ensemble, over a grid of settings.

    python scripts/train_rl.py --base-task <stage-1 id> --rm-task <stage-2 id> \
        --rewards paper --betas 0,0.01,0.05,0.2,1 --clearml --queue <q>

Every combination of --rewards x --betas x --uncertainty x --seeds is trained
from the same base checkpoint and evaluated with the shared protocol; with
--eval-base the untouched base model is scored the same way first.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Only stdlib + clearml are needed to enqueue a run.
from recsys_rl.clearml_io import (add_clearml_arguments, clearml_start,  # noqa: E402
                                  running_on_agent, upload_outputs)

TASK_NAME = "stage3_rl"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 3: RL fine-tuning.")
    p.add_argument("--data-dir", default="./data/yambda-50m")
    p.add_argument("--cache-dir", default="./data/cache")
    p.add_argument("--output-dir", default="./outputs/rl")
    p.add_argument("--base-checkpoint", default=None)
    p.add_argument("--base-task", default=None)
    p.add_argument("--rm-dir", default=None, help="Local folder with rm_<half>_s<seed>.pt files.")
    p.add_argument("--rm-task", default=None, help="ClearML id of the stage-2 task.")
    p.add_argument("--proxy-members", default="A:0,A:1,A:2")
    p.add_argument("--gold-members", default="B:0")
    p.add_argument("--rewards", default="paper",
                   help="Comma list of: paper, ratio, seconds, like, relative, continue.")
    p.add_argument("--betas", default="0.05", help="Comma list of KL weights.")
    p.add_argument("--uncertainty", default="0", help="Comma list of ensemble-std penalties.")
    p.add_argument("--seeds", default="0")
    p.add_argument("--horizon", type=int, default=1, help="Rollout length T.")
    p.add_argument("--feedback", default="predicted", choices=["predicted", "unknown"],
                   help="Feedback attached to generated items when T > 1.")
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--states", type=int, default=32)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--eval-users", type=int, default=None,
                   help="Subsample users in evaluation (default: all).")
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--eval-base", action="store_true",
                   help="Also score the base model with the same evaluation.")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--smoke", action="store_true")
    add_clearml_arguments(p)
    return p


def apply_smoke(args) -> None:
    args.steps, args.states, args.group, args.context = 6, 4, 4, 32


def main() -> None:
    raw = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(raw)
    if not (args.base_checkpoint or args.base_task):
        parser.error("give --base-task or --base-checkpoint")
    if not (args.rm_dir or args.rm_task):
        parser.error("give --rm-task or --rm-dir")
    if args.smoke:
        apply_smoke(args)

    task, data_dir = None, Path(args.data_dir)
    if args.clearml or running_on_agent():
        task, args, data_dir = clearml_start(TASK_NAME, raw, parser, args)
        if args.smoke:
            apply_smoke(args)

    from recsys_rl.stage_rl import run
    from recsys_rl.utils import setup_logging

    setup_logging(Path(args.output_dir) / "run.log")
    run(args, task, data_dir)
    upload_outputs(task, args.output_dir)


if __name__ == "__main__":
    main()
