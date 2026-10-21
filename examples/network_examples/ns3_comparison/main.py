"""Compare ECLYPSE+NET with ns-3 on the same networks and the same packets.

Every trial of the grid (topology x routers x step duration x load x seed):

1. simulates the scenario with ECLYPSE+NET, recording every packet it injects, every
   packet it delivers (with its arrival time) and every packet it drops;
2. exports the topology, the forwarding tables and the packets (``scenario.py``);
3. simulates the same packets on the same paths with ns-3 (``eclypse_scenario.cc``);
4. matches the packets of the two simulators by ID.

``results.csv`` has one row per trial; ``trials/<trial>/comparison.csv`` has one row
per packet, with its end-to-end latency in both simulators. Trials run one at a time,
each in a new process, so that their times are not affected by one another.

Usage::

    python main.py --ns3 ~/ns-3-dev            # whole grid
    python main.py --ns3 ~/ns-3-dev --smoke    # one small trial per topology
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

import numpy as np
from eclypse_run import run_eclypse
from ns3_run import (
    find_program,
    run_ns3,
)
from scenario import (
    TOPOLOGIES,
    export_for_ns3,
    write_csv,
)

from eclypse.utils.defaults import get_default_sim_path

GRID = {
    "topology": list(TOPOLOGIES),
    "routers": [16, 32, 64, 128],
    "step_duration_s": [1e-2, 1e-3, 1e-4],
    "packets_per_s": [20_000, 200_000],
    "seed": [3997],
}
"""Varied parameters: every combination is one trial."""

FIXED = {
    # Traffic: `flows` random host pairs share `packets_per_s` packets per second
    # for `traffic_time_s` seconds; then `drain_time_s` seconds without new packets
    "flows": 50,
    "packet_size_bytes": 1500,
    "traffic_time_s": 0.1,
    "drain_time_s": 0.1,
    # Network
    "max_queue_size": 50,
    "ba_m": 2,  # links of each new router in the Barabasi-Albert model
    "er_degree_factor": 1.5,  # Erdos-Renyi p = er_degree_factor * ln(n) / n
}
"""Parameters shared by all the trials."""

SMOKE = {
    "routers": [16],
    "step_duration_s": [1e-3],
    "packets_per_s": [20_000],
    "seed": [1],
    "traffic_time_s": 0.05,
    "drain_time_s": 0.05,
}
"""Overrides of ``--smoke``."""

PARAMS = tuple(GRID)
"""Parameters written in ``results.csv``, before the metrics."""


def trial_name(config: dict) -> str:
    """Return the name of the folder of a trial."""
    return "_".join(
        f"{config[p]:g}" if p == "step_duration_s" else str(config[p]) for p in PARAMS
    )


def latency_stats(prefix: str, values: np.ndarray) -> dict:
    """Return mean and percentiles of latencies in milliseconds."""
    if not len(values):
        return {f"{prefix}_{s}": float("nan") for s in ("mean", "p50", "p95", "p99")}
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_p99": float(np.percentile(values, 99)),
    }


def run_trial(config: dict, output: Path, executable: Path) -> dict:
    """Run one trial in both simulators and return its metrics.

    Args:
        config (dict): The trial configuration.
        output (Path): The folder of the results.
        executable (Path): The ns-3 program.

    Returns:
        dict: The parameters and the metrics of the trial.
    """
    folder = output / "trials" / trial_name(config)
    folder.mkdir(parents=True, exist_ok=True)

    ecl = run_eclypse(config, folder / "eclypse")
    net = ecl["net"]
    start = time.perf_counter()
    export_for_ns3(net, ecl["injected"], folder)
    export_time = time.perf_counter() - start
    ns3 = run_ns3(executable, folder, folder / "ns3")

    # Packet by packet: latency in ms, or empty if not delivered
    rows = []
    ecl_lat, ns3_lat, diff = [], [], []
    for pid, src, dst, _, sent in ecl["injected"]:
        e = ecl["delivered"].get(pid)
        n = ns3["received"].get(pid)
        e_ms = (e[0] - sent) * 1000 if e else None
        n_ms = (n - sent) * 1000 if n is not None else None
        if e_ms is not None:
            ecl_lat.append(e_ms)
        if n_ms is not None:
            ns3_lat.append(n_ms)
        if e_ms is not None and n_ms is not None:
            diff.append(e_ms - n_ms)
        rows.append(
            (
                pid,
                src,
                dst,
                repr(sent),
                "" if e_ms is None else repr(e_ms),
                "" if n_ms is None else repr(n_ms),
                e[1] if e else "",
                ecl["dropped"].get(pid, ""),
            )
        )
    write_csv(
        folder / "comparison.csv",
        [
            "id",
            "src",
            "dst",
            "send_time_s",
            "eclypse_latency_ms",
            "ns3_latency_ms",
            "hops",
            "eclypse_dropped_on",
        ],
        rows,
    )

    diff_a = np.abs(np.array(diff))
    sent = len(ecl["injected"])
    return {
        **{p: config[p] for p in PARAMS},
        # Scenario
        "n_routers": len(net.routers),
        "n_hosts": len(net.hosts),
        "n_links": net.number_of_edges() // 2,
        "n_flows": ecl["n_flows"],
        "steps": ecl["steps"],
        # Packets
        "sent": sent,
        "eclypse_delivered": len(ecl["delivered"]),
        "eclypse_dropped": len(ecl["dropped"]),
        "eclypse_in_transit_at_end": len(net.in_transit),
        "ns3_delivered": int(ns3["received_count"]),
        "ns3_dropped": int(ns3["queue_drops"] + ns3["ip_drops"]),
        "both_delivered": len(diff),
        "hops_mean": float(np.mean([h for _, h in ecl["delivered"].values()]))
        if ecl["delivered"]
        else float("nan"),
        # Latency of the packets delivered by each simulator, and differences
        # on the packets delivered by both
        **latency_stats("eclypse_latency_ms", np.array(ecl_lat)),
        **latency_stats("ns3_latency_ms", np.array(ns3_lat)),
        "latency_diff_mean_ms": float(np.mean(diff)) if diff else float("nan"),
        "latency_absdiff_mean_ms": float(np.mean(diff_a)) if diff else float("nan"),
        "latency_absdiff_p95_ms": float(np.percentile(diff_a, 95))
        if diff
        else float("nan"),
        "latency_absdiff_max_ms": float(np.max(diff_a)) if diff else float("nan"),
        # Execution time
        "eclypse_build_time_s": ecl["build_time_s"],
        "eclypse_sim_time_s": ecl["sim_time_s"],
        "export_time_s": export_time,
        "ns3_setup_time_s": ns3["setup_time_s"],
        "ns3_run_time_s": ns3["run_time_s"],
        "ns3_wall_time_s": ns3["wall_time_s"],
    }


def write_results(path: Path, rows: list[dict]):
    """Write rows with the same keys to a CSV file."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(args: argparse.Namespace):
    """Run all the trials and save their results."""
    executable = find_program(Path(args.ns3).expanduser())
    grid, fixed = dict(GRID), dict(FIXED)
    if args.smoke:
        grid.update({k: v for k, v in SMOKE.items() if k in grid})
        fixed.update({k: v for k, v in SMOKE.items() if k not in grid})
    for key in ("topology", "routers", "step_duration_s", "packets_per_s", "seed"):
        if getattr(args, key) is not None:
            grid[key] = getattr(args, key)

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    configs = [
        {**fixed, **dict(zip(grid, values, strict=True))}
        for values in itertools.product(*grid.values())
    ]
    print(f"{len(configs)} trials; ns-3 program {executable}; results in {output}")

    results = []
    start = time.perf_counter()
    # One process per trial, one trial at a time: comparable execution times
    with ProcessPoolExecutor(1, max_tasks_per_child=1) as pool:
        futures = {pool.submit(run_trial, c, output, executable): c for c in configs}
        for i, future in enumerate(as_completed(futures), 1):
            name = trial_name(futures[future])
            try:
                row = future.result()
            except Exception as e:  # report the failure and go on
                print(f"[{i}/{len(configs)}] {name} failed: {e!r}")
                continue
            results.append(row)
            print(
                f"[{i}/{len(configs)}] {name}: "
                f"ECLYPSE {row['eclypse_sim_time_s']:.2f} s, "
                f"ns-3 {row['ns3_run_time_s']:.2f} s, latency difference "
                f"{row['latency_absdiff_mean_ms']:.3f} ms on average"
            )
            # Saved after every trial, so an interrupted run keeps its results
            write_results(output / "results.csv", results)
    print(f"Elapsed time: {time.perf_counter() - start:.1f} s")


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ns3", required=True, help="ns-3 folder where eclypse_scenario is built"
    )
    parser.add_argument(
        "--output",
        default=str(get_default_sim_path() / "ns3_comparison"),
        help="folder of the results",
    )
    parser.add_argument("--topology", nargs="+", choices=TOPOLOGIES)
    parser.add_argument("--routers", nargs="+", type=int)
    parser.add_argument("--step-duration-s", nargs="+", type=float)
    parser.add_argument("--packets-per-s", nargs="+", type=int)
    parser.add_argument("--seed", nargs="+", type=int)
    parser.add_argument(
        "--smoke", action="store_true", help="one small trial per topology"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
