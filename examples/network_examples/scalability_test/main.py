"""Run the Network benchmark as a Ray Tune grid search.

Each trial of the grid defined in ``search_space.py`` runs one simulation in a Ray
worker, writes the ECLYPSE outputs in its trial folder and reports its metrics to Ray
Tune. At the end, the metrics of all the trials are saved in ``results.csv`` in the
experiment folder.

Usage::

    python main.py                   # local Ray instance, all the cores
    python main.py --address auto    # existing Ray cluster
    python main.py --smoke           # one small trial per topology, without Ray
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

from experiment import (
    TOPOLOGIES,
    run_trial,
)

from eclypse.utils.defaults import get_default_sim_path

HERE = Path(__file__).resolve().parent
EXPERIMENT_NAME = "network_grid"


def network_grid(config: dict):
    """Ray Tune trainable: run one trial and report its metrics."""
    from ray import tune  # noqa: PLC0415 (imported in the Ray worker)

    output_dir = Path(tune.get_context().get_trial_dir()) / "output"
    tune.report(run_trial(config, output_dir))


def run_grid(args: argparse.Namespace):
    """Run the whole grid with Ray Tune and save the results."""
    import ray  # noqa: PLC0415 (Ray is needed only for the grid)
    from ray import tune  # noqa: PLC0415
    from search_space import search_space  # noqa: PLC0415 (imports Ray)

    # The worker processes import `experiment` from this folder
    ray.init(
        address=args.address,
        runtime_env={"working_dir": str(HERE), "excludes": ["*.ipynb", "__pycache__"]},
    )

    tuner = tune.Tuner(
        tune.with_resources(network_grid, {"cpu": args.cpus_per_trial}),
        param_space=search_space,
        run_config=tune.RunConfig(
            name=args.name, storage_path=str(Path(args.storage).resolve())
        ),
    )
    start = time.perf_counter()
    results = tuner.fit()
    print(f"Elapsed time: {time.perf_counter() - start:.1f} s")

    if results.errors:
        print(f"{len(results.errors)} trials failed; see their error.txt files.")
    df = results.get_dataframe()
    output = Path(args.storage).resolve() / args.name / "results.csv"
    df.to_csv(output, index=False)
    print(f"Results of {len(df)} trials saved to {output}")


def run_smoke():
    """Run one small trial per topology, without Ray, to check the setup."""
    config = {
        "routers": 64,
        "packets_per_step": 100,
        "seed": 1,
        "flows": 20,
        "packet_size_bytes": 1500,
        "steps": 20,
        "step_duration_s": 0.01,
        "max_queue_size": 100,
        "ba_m": 2,
        "er_degree_factor": 1.5,
        "telemetry": False,
    }
    with tempfile.TemporaryDirectory() as tmp:
        for topology in TOPOLOGIES:
            metrics = run_trial({**config, "topology": topology}, Path(tmp) / topology)
            print(
                f"{topology:>11}: {metrics['n_routers']} routers, "
                f"{metrics['n_hosts']} hosts, {metrics['sim_time_s']:.2f} s, "
                f"{metrics['delivered']}/{metrics['generated']} packets delivered"
            )


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--address",
        default=None,
        help="address of an existing Ray cluster ('auto' to find it); "
        "by default a local Ray instance is started",
    )
    parser.add_argument(
        "--cpus-per-trial", type=float, default=1, help="CPUs reserved for each trial"
    )
    parser.add_argument(
        "--storage",
        default=str(get_default_sim_path() / "ray"),
        help="folder where Ray Tune stores the experiment",
    )
    parser.add_argument("--name", default=EXPERIMENT_NAME, help="experiment name")
    parser.add_argument(
        "--smoke", action="store_true", help="run a quick check without Ray"
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.smoke:
        run_smoke()
    else:
        run_grid(arguments)
