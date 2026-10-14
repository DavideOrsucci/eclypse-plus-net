"""Collect the results of the Network benchmark from the Ray Tune trial folders.

``main.py`` saves ``results.csv`` when the grid ends. If the run was interrupted, or
was split into several experiments, this script rebuilds the same table from the
``params.json`` and ``result.json`` files that Ray Tune writes in every trial folder.

Usage::

    python merge_results.py [--root ~/eclypse-sim/ray] [--prefix network_grid]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eclypse.utils.defaults import get_default_sim_path


def merge_results(root: Path, prefix: str) -> pd.DataFrame:
    """Read the configuration and the metrics of every completed trial.

    Args:
        root (Path): The folder that contains the experiment folders.
        prefix (str): Prefix of the names of the experiment folders to read.

    Returns:
        pd.DataFrame: One row per completed trial, with the metrics and the
            configuration (columns ``config/<parameter>``), as in ``results.csv``.
    """
    rows, skipped = [], 0
    for experiment in sorted(root.iterdir()):
        if not (experiment.is_dir() and experiment.name.startswith(prefix)):
            continue
        for trial in sorted(p for p in experiment.iterdir() if p.is_dir()):
            result, params = trial / "result.json", trial / "params.json"
            if not (result.exists() and params.exists()) or not result.stat().st_size:
                skipped += 1  # trial not completed
                continue
            metrics = json.loads(result.read_text().strip().splitlines()[-1])
            config = json.loads(params.read_text())
            rows.append(
                {
                    **metrics,
                    **{f"config/{k}": v for k, v in config.items()},
                    "logdir": str(trial),
                }
            )
    print(f"Trials read: {len(rows)}, skipped: {skipped}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=get_default_sim_path() / "ray",
        help="folder that contains the experiment folders",
    )
    parser.add_argument(
        "--prefix", default="network_grid", help="prefix of the experiment folders"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results.csv"), help="output CSV file"
    )
    args = parser.parse_args()

    df = merge_results(args.root.expanduser(), args.prefix)
    if df.empty:
        print(f"No results found in {args.root}")
    else:
        df.to_csv(args.output, index=False)
        print(f"Saved {len(df)} trials to {args.output}")
