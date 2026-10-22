"""Stress test of ECLYPSE+NET: scalability and throughput on Fat-Tree networks.

For every Fat-Tree parameter ``k`` and every packet rate, the script simulates:

- base ECLYPSE on the same Fat-Tree, without traffic, as a reference time;
- ECLYPSE+NET, where the first host sends ``rate`` packets per step in total,
  split evenly among all the other hosts.

It measures the execution time of each simulation and the packets dropped by
ECLYPSE+NET, prints a summary and saves it to ``results.csv`` in the results folder.
"""

from __future__ import annotations

import random
import shutil
import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from eclypse.builders.infrastructure.generators import get_fat_tree
from eclypse.builders.network import get_network_fat_tree
from eclypse.graph import Application
from eclypse.network import (
    NetworkApplication,
    PacketGenerationEvent,
    RoutingEvent,
    RoutingMetric,
)
from eclypse.placement.strategies import StaticStrategy
from eclypse.simulation import (
    Simulation,
    SimulationConfig,
)
from eclypse.utils.defaults import get_default_sim_path

if TYPE_CHECKING:
    from pathlib import Path

    from eclypse.graph import Infrastructure

GLOBAL_SEED = 42
"""Seed of the simulations; run ``i`` uses ``GLOBAL_SEED + i``."""

RESULTS_DIR = get_default_sim_path() / "stress_test"
"""Folder of the outputs of the simulations and of ``results.csv``."""

# Benchmark configuration
# k=4 (36 nodes), k=6 (99 nodes), k=8 (208 nodes), k=10 (375 nodes), k=12 (612 nodes)
K_VALUES = [6, 8, 10, 12]
PACKETS_PER_STEP_RATES = [10, 100, 300, 600, 900, 1200, 1500]
PACKET_SIZE_BYTES = 1500
NUM_RUNS = 1
SIM_STEPS = 50
STEP_DURATION_S = 1.0


def seed_everything(seed: int):
    """Seed the global generators used by ECLYPSE and by the packet generation."""
    random.seed(seed)
    np.random.seed(seed)


def results_path(name: str) -> Path:
    """Return an empty folder for the outputs of one simulation."""
    path = RESULTS_DIR / name
    # ECLYPSE writes to a new timestamped folder if the path exists: start clean
    shutil.rmtree(path, ignore_errors=True)
    return path


def run_eclypse_base(infra: Infrastructure, steps: int, seed: int) -> float:
    """Simulate base ECLYPSE on an infrastructure, without traffic.

    Args:
        infra (Infrastructure): The Fat-Tree infrastructure.
        steps (int): The number of steps.
        seed (int): The seed of the simulation.

    Returns:
        float: The execution time, in seconds.
    """
    seed_everything(seed)
    num_nodes = len(infra.nodes)
    app = Application(f"Base_App_{num_nodes}")
    mapping = {}
    for node in infra.nodes():
        infra.nodes[node]["cpu"] = 2
        infra.nodes[node]["ram"] = 4
        if "host" in node:
            service = f"App_{node}"
            app.add_node(service, cpu=1, ram=1)
            mapping[service] = node

    config = SimulationConfig(
        seed=seed,
        max_steps=steps,
        path=results_path(f"base_{num_nodes}_nodes_seed{seed}"),
        include_default_metrics=False,
        log_level="ERROR",
    )
    sim = Simulation(infra, simulation_config=config)
    sim.register(app, placement_strategy=StaticStrategy(mapping))

    start = time.perf_counter()
    sim.start()
    sim.wait()
    return time.perf_counter() - start


def run_eclypse_net(
    k: int,
    total_pkt_per_step: float,
    pkt_size: int,
    steps: int,
    seed: int,
    generation: PacketGenerationEvent | None = None,
    routing: RoutingEvent | None = None,
) -> dict:
    """Simulate ECLYPSE+NET on a Fat-Tree network with one source host.

    Args:
        k (int): The Fat-Tree parameter.
        total_pkt_per_step (float): The packets sent per step by the source host,
            split evenly among the other hosts.
        pkt_size (int): The packet size, in bytes.
        steps (int): The number of steps.
        seed (int): The seed of the simulation.
        generation (PacketGenerationEvent | None): The packet generation event,
            e.g. an instrumented one (see ``stress_test.ipynb``). If None, a new
            :class:`PacketGenerationEvent`. Defaults to None.
        routing (RoutingEvent | None): The routing event. If None, a new
            :class:`RoutingEvent` with steps of :data:`STEP_DURATION_S`. Defaults
            to None.

    Returns:
        dict: The execution time in seconds (``exec_time``) and the number of
            packets dropped because a queue was full (``dropped``).
    """
    seed_everything(seed)
    infra = get_network_fat_tree(
        k,
        infrastructure_id=f"Net_Infra_{k}_{total_pkt_per_step}",
        bandwidth_mbps=1000,
        length_km=1,
        host_cpu=2,
        host_ram=4,
        max_queue_size=100_000,
        router_processing_time=0.0001,
        host_processing_time=0.0001,
        seed=seed,
    )
    num_nodes = len(infra.nodes)
    app = NetworkApplication(f"Net_App_{num_nodes}_{total_pkt_per_step}")
    mapping = {}
    hosts = list(infra.hosts)
    for host in hosts:
        service = f"App_{host}"
        app.add_node(service, cpu=1, ram=1)
        mapping[service] = host

    # The first host is the source; the rate is split evenly among the others.
    # The Poisson generator accepts fractional rates.
    if len(hosts) > 1:
        rate = total_pkt_per_step / (len(hosts) - 1)
        for target in hosts[1:]:
            app.add_edge(
                f"App_{hosts[0]}",
                f"App_{target}",
                packet_size_bytes=pkt_size,
                avg_packets_per_step=rate,
            )

    config = SimulationConfig(
        seed=seed,
        max_steps=steps,
        path=results_path(f"net_{num_nodes}_nodes_{total_pkt_per_step}_pkt_seed{seed}"),
        events=[
            PacketGenerationEvent() if generation is None else generation,
            RoutingEvent(step_duration_s=STEP_DURATION_S)
            if routing is None
            else routing,
            RoutingMetric(),
        ],
        include_default_metrics=False,
        log_level="ERROR",
    )
    sim = Simulation(infra, simulation_config=config)
    sim.register(app, placement_strategy=StaticStrategy(mapping))

    start = time.perf_counter()
    sim.start()
    sim.wait()
    return {
        "exec_time": time.perf_counter() - start,
        "dropped": infra.dropped_packets,
    }


def main():
    """Run the benchmark, print the results and save them to ``results.csv``."""
    results = []
    print("Starting Stress Test Benchmark: Scalability & Throughput...")
    for k in K_VALUES:
        total_nodes = (5 * k**2) // 4 + k**3 // 4  # nodes of a Fat-Tree
        print(f"\n--- Testing Fat-Tree with k={k} ({total_nodes} nodes) ---")

        # Base ECLYPSE: reference time, without traffic
        seeds = [GLOBAL_SEED + i for i in range(NUM_RUNS)]
        base_times = [
            run_eclypse_base(get_fat_tree(k=k, seed=seed), SIM_STEPS, seed)
            for seed in seeds
        ]

        for rate in PACKETS_PER_STEP_RATES:
            runs = [
                run_eclypse_net(k, rate, PACKET_SIZE_BYTES, SIM_STEPS, seed)
                for seed in seeds
            ]
            net_time = float(np.mean([r["exec_time"] for r in runs]))
            dropped = float(np.mean([r["dropped"] for r in runs]))
            results.append(
                {
                    "k_param": k,
                    "Total_Nodes": total_nodes,
                    "Total_Pkt_per_Step": rate,
                    "Packet_Size (Bytes)": PACKET_SIZE_BYTES,
                    "Eclypse Base Time (s)": round(float(np.mean(base_times)), 4),
                    "Eclypse+NET Time (s)": round(net_time, 4),
                    "Avg_Dropped_Packets": round(dropped, 2),
                }
            )
            print(
                f"  Rate: {rate} pkt/step -> Exec: {net_time:.3f}s | "
                f"Drops: {dropped:.1f}"
            )

    df_results = pd.DataFrame(results)
    print(df_results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(RESULTS_DIR / "results.csv", index=False)
    print(f"Results saved to {RESULTS_DIR / 'results.csv'}")


if __name__ == "__main__":
    main()
