"""Register Yambda-50M as a ClearML Dataset.

Run once on a machine with internet access and a configured ClearML client
(~/clearml.conf). Downloads the flat 50M event table and the item->artist /
item->album mappings from Hugging Face, checks their schema, and uploads them
as one ClearML Dataset. Prints the dataset id at the end; export it as
YAMBDA_DATASET_ID for every training script.

    python scripts/upload_yambda.py --project <clearml-project>
    python scripts/upload_yambda.py --no-upload        # download + check only
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger("yambda.upload")

HF_REPO = "yandex/yambda"

# Published path on the Hub -> flat file name inside the ClearML dataset.
FILES = {
    "flat/50m/multi_event.parquet": "multi_event.parquet",
    "artist_item_mapping.parquet": "artist_item_mapping.parquet",
    "album_item_mapping.parquet": "album_item_mapping.parquet",
}

# Columns the pipeline relies on; a missing one means the upstream schema moved.
REQUIRED_COLUMNS = {
    "multi_event.parquet": {
        "uid", "item_id", "timestamp", "is_organic",
        "event_type", "played_ratio_pct", "track_length_seconds",
    },
    "artist_item_mapping.parquet": {"item_id"},
    "album_item_mapping.parquet": {"item_id"},
}

DEFAULT_PROJECT = os.environ.get("YAMBDA_CLEARML_PROJECT", "TLab-YambdaRL")
DEFAULT_NAME = "yambda-50m-flat"


def download(staging_dir: Path, cache_dir: Path) -> list[Path]:
    """Fetch every file from the Hub and copy it under a flat name."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise SystemExit("Install huggingface_hub: pip install huggingface_hub") from error

    staging_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for remote, local_name in FILES.items():
        target = staging_dir / local_name
        if target.exists() and target.stat().st_size > 0:
            logger.info("Already present, skipping: %s", target)
            paths.append(target)
            continue

        logger.info("Downloading %s ...", remote)
        cached = hf_hub_download(
            repo_id=HF_REPO, repo_type="dataset", filename=remote, cache_dir=str(cache_dir)
        )
        # Copy rather than link: Windows symlinks need admin rights.
        shutil.copyfile(cached, target)
        logger.info("Saved %s (%.1f MB)", target, target.stat().st_size / 2**20)
        paths.append(target)
    return paths


def check_schema(paths: list[Path]) -> dict:
    """Read parquet metadata only; fail if a required column is missing."""
    import pyarrow.parquet as pq

    summary = {}
    for path in paths:
        meta = pq.ParquetFile(path)
        columns = set(meta.schema_arrow.names)
        missing = REQUIRED_COLUMNS.get(path.name, set()) - columns
        if missing:
            raise RuntimeError(f"{path.name}: missing columns {sorted(missing)}; got {sorted(columns)}")

        summary[path.name] = {"rows": meta.metadata.num_rows, "columns": sorted(columns)}
        logger.info("%s: %d rows, columns %s", path.name, meta.metadata.num_rows, sorted(columns))
    return summary


def upload(staging_dir: Path, project: str, name: str, summary: dict) -> str:
    """Create, fill, upload and finalise the ClearML Dataset; return its id."""
    from clearml import Dataset

    dataset = Dataset.create(
        dataset_name=name,
        dataset_project=project,
        dataset_tags=["yambda", "50m", "raw"],
        description=f"Yambda-50M flat multi_event + item mappings from hf.co/datasets/{HF_REPO}",
    )
    dataset.add_files(path=str(staging_dir))
    # Row counts and columns are visible in the UI without downloading anything.
    dataset.set_metadata({k: str(v) for k, v in summary.items()}, metadata_name="schema")
    dataset.upload(show_progress=True)
    dataset.finalize()
    return dataset.id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Register Yambda-50M as a ClearML Dataset.")
    parser.add_argument("--project", default=DEFAULT_PROJECT,
                        help="ClearML project. Defaults to $YAMBDA_CLEARML_PROJECT.")
    parser.add_argument("--name", default=DEFAULT_NAME, help="ClearML dataset name.")
    parser.add_argument("--staging-dir", default="./data/yambda-50m",
                        help="Where the flat files are collected before upload.")
    parser.add_argument("--cache-dir", default="./data/hf-cache",
                        help="Hugging Face download cache.")
    parser.add_argument("--no-upload", action="store_true",
                        help="Download and check the schema only.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    args = build_parser().parse_args()

    staging_dir = Path(args.staging_dir)
    paths = download(staging_dir, Path(args.cache_dir))
    summary = check_schema(paths)

    if args.no_upload:
        logger.info("--no-upload: files are in %s", staging_dir.resolve())
        return

    dataset_id = upload(staging_dir, args.project, args.name, summary)
    logger.info("Done. Dataset id: %s", dataset_id)
    print(f"\nYAMBDA_DATASET_ID={dataset_id}")


if __name__ == "__main__":
    main()
