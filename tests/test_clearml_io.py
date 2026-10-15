import argparse

import pytest

from recsys_rl import clearml_io


class FakeTask:
    """connect() mimics ClearML: on the agent, stored values overwrite the payload."""

    def __init__(self, stored=None):
        self.stored = stored

    def connect(self, payload, name=None):
        if self.stored is not None:
            payload.update(self.stored)
        else:
            self.stored = dict(payload)
        return payload


def _parser():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1)
    clearml_io.add_clearml_arguments(p)
    return p


def test_launch_state_round_trip(monkeypatch):
    parser = _parser()
    raw = ["--epochs", "7", "--clearml", "--queue", "q1"]
    monkeypatch.setattr(clearml_io, "DATASET_ID", "abc123")
    monkeypatch.delenv("CLEARML_TASK_ID", raising=False)
    local = FakeTask()
    args, ds = clearml_io._transfer_launch_state(local, raw, parser, parser.parse_args(raw))
    assert ds == "abc123"

    # On the agent: no env var, no argv - both must come from the task.
    monkeypatch.setattr(clearml_io, "DATASET_ID", "")
    monkeypatch.setenv("CLEARML_TASK_ID", "t")
    agent = FakeTask(stored=local.stored)
    args, ds = clearml_io._transfer_launch_state(agent, [], parser, parser.parse_args([]))
    assert ds == "abc123" and args.epochs == 7 and args.queue == "q1"


def test_launch_without_dataset_id_fails(monkeypatch):
    parser = _parser()
    monkeypatch.setattr(clearml_io, "DATASET_ID", "")
    monkeypatch.delenv("CLEARML_TASK_ID", raising=False)
    with pytest.raises(RuntimeError):
        clearml_io._transfer_launch_state(FakeTask(), [], parser, parser.parse_args([]))
