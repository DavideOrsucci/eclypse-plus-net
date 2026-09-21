import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GLOBAL_SEED = 42

random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# Import the topology generator
from eclypse.builders.infrastructure.generators import get_fat_tree

# Import of the base Eclypse modules
from eclypse.graph import (
    Application,
    Infrastructure,
)

# Import of the Eclypse+NET modules
from eclypse.network import (
    Network,
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


def run_eclypse_base_shared(base_infra: Infrastructure, steps: int = 10, run_idx: int = 0) -> float:
    """Execute the base simulation using a pre-generated shared topology."""
    num_nodes = len(base_infra.nodes)
    app = Application(f"Base_App_{num_nodes}")
    mapping = {}

    for node in base_infra.nodes():
        base_infra.nodes[node]['cpu'] = 2
        base_infra.nodes[node]['ram'] = 4

        if "host" in node:
            app_name = f"App_{node}"
            app.add_node(app_name, cpu=1, ram=1)
            mapping[app_name] = node

    config = SimulationConfig(
        seed=GLOBAL_SEED,
        max_steps=steps,
        include_default_metrics=False,
        log_level="ERROR"
    )

    sim = Simulation(base_infra, simulation_config=config)
    sim.register(app, placement_strategy=StaticStrategy(mapping))

    start_time = time.perf_counter()
    sim.start()
    sim.wait()
    return time.perf_counter() - start_time


def run_eclypse_net_shared(base_infra: Infrastructure, rate_pps: int, pkt_size: int, steps: int = 10, run_idx: int = 0) -> dict:
    """Execute the Eclypse+NET simulation with specific traffic constraints."""
    num_nodes = len(base_infra.nodes)
    k_param = base_infra.graph.get('k', 'unknown') # Try to extract k if saved in the graph
    infra = Network(f"Net_Infra_{num_nodes}_{rate_pps}")
    app = NetworkApplication(f"Net_App_{num_nodes}_{rate_pps}")
    mapping = {}

    for node in base_infra.nodes():
        if "host" in node:
            infra.add_host(node, processing_time=0.0001, cpu=2, ram=4)
            app_name = f"App_{node}"
            app.add_node(app_name, cpu=1, ram=1)
            mapping[app_name] = node
        else:
            infra.add_router(node, processing_time=0.0001)

    for u, v in base_infra.edges():
        infra.add_edge(u, v, bandwidth_mbps=1000, length_km=1, max_queue_size=100)

    available_hosts = list(infra.hosts)

    # Calculate the number of packets per step (1 step = 1 ms)
    # If rate_pps = 1500, pkt_per_step = 1.5
    step_duration = 0.001
    pkt_per_step = rate_pps * step_duration

    if len(available_hosts) > 1:
        source_app = f"App_{available_hosts[0]}"
        for target_node in available_hosts[1:]:
            target_app = f"App_{target_node}"
            app.add_edge(source_app, target_app, packet_size_bytes=pkt_size, avg_packets_per_step=pkt_per_step)

    packet_evt = PacketGenerationEvent()
    routing_evt = RoutingEvent(step_duration_s=step_duration)
    metrics_evt = RoutingMetric()

    config = SimulationConfig(
        seed=GLOBAL_SEED,
        max_steps=steps,
        events=[packet_evt, routing_evt, metrics_evt],
        include_default_metrics=False,
        log_level="ERROR"
    )

    sim = Simulation(infra, simulation_config=config)
    sim.register(app, placement_strategy=StaticStrategy(mapping))

    start_time = time.perf_counter()
    sim.start()
    sim.wait()
    exec_time = time.perf_counter() - start_time

    # Retrieve the number of dropped packets from the infrastructure metrics
    dropped = getattr(infra, 'dropped_packets', 0)

    return {
        'exec_time': exec_time,
        'dropped': dropped
    }


# Benchmark configuration
# k=4 (36 nodes), k=6 (99 nodes), k=8 (208 nodes), k=10 (375 nodes), k=12 (612 nodes)
k_values = [6, 8, 10, 12]
traffic_rates = [10, 100, 300, 600, 900, 1200, 1500]
packet_size_bytes = 1500
num_runs = 1
sim_steps = 50

results = []

print("Starting Stress Test Benchmark: Scalability & Throughput...")
for k in k_values:
    total_nodes = int((5 * k**2) / 4 + (k**3) / 4) # Formula nodi Fat-Tree
    print(f"\n--- Testing Fat-Tree with k={k} ({total_nodes} nodes) ---")

    base_times = []

    # Execution of the base Eclypse simulation to get a reference time (no traffic)
    for run_idx in range(num_runs):
        shared_topology = get_fat_tree(k=k, seed=GLOBAL_SEED + run_idx)
        base_times.append(run_eclypse_base_shared(shared_topology.copy(), steps=sim_steps, run_idx=run_idx))

    avg_base_time = np.mean(base_times)

    # Execution of the Eclypse+NET simulation with different traffic rates
    for rate in traffic_rates:
        net_times = []
        net_drops = []

        for run_idx in range(num_runs):
            shared_topology = get_fat_tree(k=k, seed=GLOBAL_SEED + run_idx)

            net_metrics = run_eclypse_net_shared(
                shared_topology.copy(),
                rate_pps=rate,
                pkt_size=packet_size_bytes,
                steps=sim_steps,
                run_idx=run_idx
            )

            net_times.append(net_metrics['exec_time'])
            net_drops.append(net_metrics['dropped'])

        results.append({
            'k_param': k,
            'Total_Nodes': total_nodes,
            'Packet_Rate (pkt/s)': rate,
            'Packet_Size (Bytes)': packet_size_bytes,
            'Eclypse Base Time (s)': round(avg_base_time, 4),
            'Eclypse+NET Time (s)': round(np.mean(net_times), 4),
            'Avg_Dropped_Packets': round(np.mean(net_drops), 2)
        })

        print(f"  Rate: {rate} pkt/s -> Exec: {np.mean(net_times):.3f}s | Drops: {np.mean(net_drops):.1f}")

df_results = pd.DataFrame(results)

print(df_results)