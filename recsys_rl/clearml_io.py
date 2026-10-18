"""Local/remote switch for ClearML, shared by every entry point.

No infrastructure identifier lives in the source: project, default queue and
dataset id come from the environment. A local run needs none of them.
"""

from __future__ import annotations

import logging
import os
import shlex
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_NAME = os.environ.get("YAMBDA_CLEARML_PROJECT", "TLab-YambdaRL")
QUEUE_NAME = os.environ.get("YAMBDA_CLEARML_QUEUE", "")
DATASET_ID = os.environ.get("YAMBDA_DATASET_ID", "")
# The agent's default image (cuda 12.9.1) ships expired apt keys, so apt cannot
# install python/git inside it. Same CUDA/Ubuntu line, newer build.
DOCKER_IMAGE = os.environ.get("YAMBDA_CLEARML_DOCKER", "nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04")

PACKAGES = [
    "clearml>=2.1",
    "boto3>=1.9",
    "torch==2.10.0",
    "numpy>=1.26",
    "polars>=1.0",
    "pyarrow>=14",
    "transformers>=4.40",
    "scikit-learn>=1.3",
    "tqdm>=4.66",
]


def running_on_agent() -> bool:
    return bool(os.environ.get("CLEARML_TASK_ID"))


def add_clearml_arguments(parser) -> None:
    parser.add_argument("--clearml", action="store_true",
                        help="Run on a ClearML agent instead of this machine.")
    parser.add_argument("--queue", default=QUEUE_NAME,
                        help="ClearML queue (default $YAMBDA_CLEARML_QUEUE).")
    parser.add_argument("--docker", default=DOCKER_IMAGE,
                        help="Base docker image for the agent (default $YAMBDA_CLEARML_DOCKER "
                             "or a cuda 12.9 / ubuntu 24.04 runtime image).")


def _transfer_launch_state(task, raw, parser, args):
    """Carry the command line and the dataset id to the agent as task parameters.

    The agent has neither our argv nor our environment variables; on the agent
    `connect` overwrites the payload with the values stored at launch.
    Returns (args, dataset_id).
    """
    on_agent = running_on_agent()
    payload = {
        "argv": "" if on_agent else " ".join(shlex.quote(a) for a in raw),
        "dataset_id": "" if on_agent else DATASET_ID,
    }
    # Own section, so it never mixes with ClearML's argparse bookkeeping.
    task.connect(payload, name="Launch")
    if not payload["dataset_id"]:
        raise RuntimeError("No dataset id: set YAMBDA_DATASET_ID before launching.")
    if not on_agent:
        return args, payload["dataset_id"]
    if not payload["argv"]:
        raise RuntimeError("No command line stored on the task; re-launch from a current checkout.")
    print(f"[clearml] restored argv: {payload['argv']}", flush=True)
    return parser.parse_args(shlex.split(payload["argv"])), payload["dataset_id"]


def clearml_start(task_name: str, raw, parser, args, task_type: str = "training"):
    """Create the task, send it to the queue, and on the agent fetch the dataset.

    Returns (task, args, raw_data_dir).
    """
    from clearml import Dataset, Task

    if not running_on_agent() and not DATASET_ID:
        raise RuntimeError("Set YAMBDA_DATASET_ID for a remote run.")

    task = Task.init(project_name=PROJECT_NAME, task_name=task_name,
                     task_type=getattr(Task.TaskTypes, task_type),
                     reuse_last_task_id=False,
                     # ClearML's argparse hook re-binds parse_args on the agent to the
                     # values it captured; ours are restored explicitly instead.
                     auto_connect_arg_parser=False)
    args, dataset_id = _transfer_launch_state(task, raw, parser, args)

    if not running_on_agent():
        if not args.queue:
            raise RuntimeError("No queue given: pass --queue or set YAMBDA_CLEARML_QUEUE.")
        os.environ["CLEARML_APT_INSTALL"] = ""
        if args.docker:
            task.set_base_docker(args.docker)
        # set_packages replaces auto-detection, so every dependency is listed.
        task.set_packages(list(PACKAGES))
        task.execute_remotely(queue_name=args.queue, exit_process=True)

    data_dir = Dataset.get(dataset_id=dataset_id, alias="yambda-50m").get_local_copy()
    logger.info("Dataset materialised at %s", data_dir)
    return task, args, Path(data_dir)


def fetch_artifact(task_id: str, name: str) -> Path:
    """Local copy of an artifact produced by an earlier task (e.g. a checkpoint)."""
    from clearml import Task

    source = Task.get_task(task_id=task_id)
    if name not in source.artifacts:
        raise KeyError(f"Task {task_id} has no artifact '{name}'. "
                       f"Available: {sorted(source.artifacts)}")
    return Path(source.artifacts[name].get_local_copy())


def resolve_checkpoint(local_path: str | None, task_id: str | None,
                       artifact: str = "outputs/model.pt") -> Path:
    """A checkpoint from a local path, or from a previous ClearML task."""
    if local_path:
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    if task_id:
        return fetch_artifact(task_id, artifact)
    raise ValueError("Give either a local checkpoint path or the id of the task that produced it.")


def upload_outputs(task, output_dir) -> int:
    """Every produced file becomes a separate artifact `outputs/<relative path>`."""
    output_dir = Path(output_dir)
    if task is None or not output_dir.is_dir():
        return 0
    uploaded = 0
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            task.upload_artifact(name=f"outputs/{path.relative_to(output_dir).as_posix()}",
                                 artifact_object=str(path), wait_on_upload=True)
            uploaded += 1
    logger.info("Uploaded %d artifacts.", uploaded)
    return uploaded


def report_scalars(task, title: str, values: dict, iteration: int) -> None:
    if task is None:
        return
    reporter = task.get_logger()
    for key, value in values.items():
        if isinstance(value, (int, float)):
            reporter.report_scalar(title=title, series=key, value=float(value), iteration=iteration)
