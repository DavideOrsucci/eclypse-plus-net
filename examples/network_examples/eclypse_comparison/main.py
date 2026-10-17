"""Compare the execution time of base ECLYPSE and ECLYPSE+NET.

Every scenario of the grid (topology x number of routers x seed) is simulated twice,
once with base ECLYPSE and once with ECLYPSE+NET, with the same update policies on
the infrastructure and on the application. Each simulation runs in a new process, so
that trials do not share caches or memory. The metrics are saved in ``results.csv``
(one row per simulation) and the duration of every phase of every step in
``steps.csv``.

Usage::

    python main.py                 # whole grid, one simulation at a time
    python main.py --workers 4     # four simulations in parallel (noisier times)
    python main.py --smoke         # one small scenario per topology
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from pathlib import Path

from experiment import (
    MODES,
    run_trial,
)
from scenario import TOPOLOGIES

from eclypse.utils.defaults import get_default_sim_path

GRID = {
    "topology": list(TOPOLOGIES),
    "routers": [64, 128, 256, 512],
    "seed": [3997, 151195, 300425],
}
"""Varied parameters: every combination is one scenario, simulated in both modes."""

FIXED = {
    # Traffic: `flows` random host pairs share `packets_per_step` packets per step
    "flows": 100,
    "packets_per_step": 1000,
    "packet_size_bytes": 1500,
    # Simulation: `steps` steps of `step_duration_s` seconds
    "steps": 100,
    "step_duration_s": 0.01,
    # Network
    "max_queue_size": 100,
    "ba_m": 2,  # links of each new router in the Barabasi-Albert model
    "er_degree_factor": 1.5,  # Erdos-Renyi p = er_degree_factor * ln(n) / n
    # Infrastructure policies
    "bandwidth_range": (0.8, 1.2),  # bandwidth, relative to the initial one
    "failure_period": 20,  # steps between two link failures
    "failure_duration": 5,  # steps a failed link stays down
    "failure_fraction": 0.02,  # share of router links that fail each time
    # Application policies
    "load_amplitude": 0.5,  # packet rate between 0.5 and 1.5 times the initial one
    "load_period": 50,  # steps of a daily cycle
    "size_period": 10,  # steps between two changes of the packet sizes
    "size_range": (0.5, 1.0),  # packet size, relative to the initial one
}
"""Parameters shared by all the simulations."""

SMOKE = {
    "routers": [32],
    "seed": [1],
    "flows": 20,
    "packets_per_step": 200,
    "steps": 20,
}
"""Overrides of ``--smoke``."""

PARAMS = ("mode", *GRID)
"""Parameters written in the result files, before the metrics."""


def trials(grid: dict, fixed: dict) -> list[dict]:
    """Return the configuration of every simulation, both modes of a scenario in a row.

    Args:
        grid (dict): The values of the varied parameters.
        fixed (dict): The parameters shared by all the simulations.

    Returns:
        list[dict]: One configuration per simulation.
    """
    keys = list(grid)
    configs = []
    for values in itertools.product(*grid.values()):
        for mode in MODES:
            configs.append(
                {**fixed, **dict(zip(keys, values, strict=True)), "mode": mode}
            )
    return configs


def trial_name(config: dict) -> str:
    """Return the name of the folder of a simulation."""
    return "_".join(str(config[p]) for p in (*GRID, "mode"))


def run_one(config: dict, output: Path) -> tuple[dict, list[dict]]:
    """Run one simulation and prefix its rows with its parameters."""
    metrics, steps = run_trial(config, output / "trials" / trial_name(config))
    params = {p: config[p] for p in PARAMS}
    return {**params, **metrics}, [{**params, **row} for row in steps]


def write_csv(path: Path, rows: list[dict]):
    """Write rows with the same keys to a CSV file."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(args: argparse.Namespace):
    """Run all the simulations and save their results."""
    grid, fixed = dict(GRID), dict(FIXED)
    if args.smoke:
        grid.update({k: v for k, v in SMOKE.items() if k in grid})
        fixed.update({k: v for k, v in SMOKE.items() if k not in grid})
    for key in ("topology", "routers", "seed"):
        if getattr(args, key) is not None:
            grid[key] = getattr(args, key)

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    configs = trials(grid, fixed)
    print(f"{len(configs)} simulations, {args.workers} at a time; results in {output}")

    results, steps = [], []
    start = time.perf_counter()
    # One process per simulation: no caches or memory shared between simulations
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as pool:
        futures = {pool.submit(run_one, c, output): c for c in configs}
        for i, future in enumerate(as_completed(futures), 1):
            config = futures[future]
            try:
                row, rows = future.result()
            except Exception as e:  # report the failure and go on
                print(f"[{i}/{len(configs)}] {trial_name(config)} failed: {e!r}")
                continue
            results.append(row)
            steps.extend(rows)
            print(
                f"[{i}/{len(configs)}] {trial_name(config)}: {row['sim_time_s']:.2f} s"
            )
            # Saved after every simulation, so an interrupted run keeps its results
            write_csv(output / "results.csv", results)
            write_csv(output / "steps.csv", steps)
    print(f"Elapsed time: {time.perf_counter() - start:.1f} s")


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--workers", type=int, default=1, help="simulations run in parallel"
    )
    parser.add_argument(
        "--output",
        default=str(get_default_sim_path() / "eclypse_comparison"),
        help="folder of the results",
    )
    parser.add_argument("--topology", nargs="+", choices=TOPOLOGIES, help="topologies")
    parser.add_argument("--routers", nargs="+", type=int, help="numbers of routers")
    parser.add_argument("--seed", nargs="+", type=int, help="seeds")
    parser.add_argument(
        "--smoke", action="store_true", help="one small scenario per topology"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
