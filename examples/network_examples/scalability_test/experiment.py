"""One trial of the Network benchmark: build a network, simulate traffic, measure.

The module does not depend on Ray: :func:`run_trial` receives the configuration of the
trial and the folder where ECLYPSE writes its outputs, and returns the metrics of the
trial as a dictionary.
"""

from __future__ import annotations

import math
import random
import time
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from eclypse.builders.network import (
    get_network_internet_as,
    get_network_random,
    get_network_scale_free,
)
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

if TYPE_CHECKING:
    from pathlib import Path

    from eclypse.network import Network


TOPOLOGIES = ("internet_as", "scale_free", "random")
"""Topologies supported by :func:`build_network`."""


def build_network(config: dict) -> Network:
    """Generate the network of a trial.

    Args:
        config (dict): The trial configuration. It uses ``topology``, ``routers``,
            ``seed``, ``max_queue_size``, ``ba_m`` and ``er_degree_factor``.

    Returns:
        Network: The generated network. Internet AS graphs have hosts on their
            customer and content-provider ASes; the other topologies have one host
            per router.

    Raises:
        ValueError: If the topology is not one of :data:`TOPOLOGIES`.
    """
    topology, n, seed = config["topology"], config["routers"], config["seed"]
    queue = config["max_queue_size"]
    if topology == "internet_as":
        return get_network_internet_as(n, max_queue_size=queue, seed=seed)
    if topology == "scale_free":
        return get_network_scale_free(
            n, config["ba_m"], max_queue_size=queue, seed=seed
        )
    if topology == "random":
        p = min(1.0, config["er_degree_factor"] * math.log(n) / n)
        return get_network_random(n, p, max_queue_size=queue, seed=seed)
    raise ValueError(f"Unknown topology {topology!r}; expected one of {TOPOLOGIES}.")


def reachable_hosts(net: Network) -> list[str]:
    """Return the hosts of the largest connected component of the network."""
    graph = nx.Graph()
    graph.add_nodes_from(net.nodes)
    graph.add_edges_from(net.edges)
    largest = max(nx.connected_components(graph), key=len)
    return [h for h in net.hosts if h in largest]


def build_application(net: Network, hosts: list[str], config: dict):
    """Create one service per host and ``flows`` random flows between them.

    The ``packets_per_step`` packets of a step are split evenly among the flows.

    Args:
        net (Network): The network.
        hosts (list[str]): The hosts that can exchange traffic.
        config (dict): The trial configuration. It uses ``flows``,
            ``packets_per_step``, ``packet_size_bytes`` and ``seed``.

    Returns:
        tuple[NetworkApplication, dict[str, str]]: The application and the
            placement of its services.
    """
    app = NetworkApplication(f"App_{net.id}")
    mapping = {}
    for host in hosts:
        app.add_node(f"svc_{host}", cpu=1, ram=1)
        mapping[f"svc_{host}"] = host

    n_flows = min(config["flows"], len(hosts) * (len(hosts) - 1))
    rate = config["packets_per_step"] / n_flows
    rng = random.Random(config["seed"])
    while app.number_of_edges() < n_flows:
        src, dst = rng.sample(hosts, 2)
        if not app.has_edge(f"svc_{src}", f"svc_{dst}"):
            app.add_edge(
                f"svc_{src}",
                f"svc_{dst}",
                packet_size_bytes=config["packet_size_bytes"],
                avg_packets_per_step=rate,
            )
    return app, mapping


class CountingGeneration(PacketGenerationEvent):
    """Packet generation event that counts the generated packets."""

    def __init__(self):
        """Initialize the event with a zero count."""
        super().__init__()
        self.generated = 0

    def __call__(self, app, placement, infra, **kwargs):
        """Generate the packets of the step and count them."""
        out = super().__call__(app, placement, infra, **kwargs)
        self.generated += app.num_generated
        return out


class TimedRouting(RoutingEvent):
    """Routing event that measures its duration and the hops it simulates."""

    def __init__(self, step_duration_s: float):
        """Initialize the event with empty measurements."""
        super().__init__(step_duration_s=step_duration_s)
        self.durations: list[float] = []
        self.hops: list[int] = []

    def __call__(self, app, placement, infra, **kwargs):
        """Forward the packets of the step and record time and hops."""
        start = time.perf_counter()
        out = super().__call__(app, placement, infra, **kwargs)
        self.durations.append(time.perf_counter() - start)
        self.hops.append(len(infra.step_columns["hop"]))
        return out


def record_deliveries(net: Network, step_duration_s: float) -> dict[str, list]:
    """Record the packets delivered by ``net.forward_batch``.

    The method is wrapped on the instance, so the simulation is unchanged.

    Args:
        net (Network): The network.
        step_duration_s (float): The duration of a step, used to compute the
            generation time of the packets.

    Returns:
        dict[str, list]: Lists filled during the simulation with the end-to-end
            latency, in ms, and the hops of each delivered packet.
    """
    record: dict[str, list] = {"latency_ms": [], "hops": []}
    forward = net.forward_batch

    def forward_and_record(batch, current_time):
        delivered = forward(batch, current_time)
        if len(delivered):
            generated_at = delivered.step * step_duration_s
            record["latency_ms"].append((delivered.arrival - generated_at) * 1000.0)
            record["hops"].append(delivered.hop)
        return delivered

    net.forward_batch = forward_and_record  # type: ignore[method-assign]
    return record


def run_trial(config: dict, output_dir: Path) -> dict:
    """Run one simulation and return its metrics.

    Args:
        config (dict): The trial configuration (see ``search_space.py``).
        output_dir (Path): The folder where ECLYPSE writes its outputs.

    Returns:
        dict: The metrics of the trial: size of the network, execution times,
            simulated hops, packet counts and end-to-end latency.
    """
    random.seed(config["seed"])
    np.random.seed(config["seed"])

    start = time.perf_counter()
    net = build_network(config)
    hosts = reachable_hosts(net)
    app, mapping = build_application(net, hosts, config)
    build_time = time.perf_counter() - start

    generation = CountingGeneration()
    routing = TimedRouting(config["step_duration_s"])
    events = [generation, routing]
    if config.get("telemetry", False):
        events.append(RoutingMetric())
    deliveries = record_deliveries(net, config["step_duration_s"])

    sim_config = SimulationConfig(
        seed=config["seed"],
        max_steps=config["steps"],
        path=output_dir,
        events=events,
        include_default_metrics=False,
        log_level="ERROR",
        remote=False,
    )
    sim = Simulation(net, simulation_config=sim_config)
    sim.register(app, placement_strategy=StaticStrategy(mapping))
    start = time.perf_counter()
    sim.start()
    sim.wait()
    sim_time = time.perf_counter() - start

    latency = (
        np.concatenate(deliveries["latency_ms"])
        if deliveries["latency_ms"]
        else np.empty(0)
    )
    path_hops = (
        np.concatenate(deliveries["hops"]) if deliveries["hops"] else np.empty(0)
    )
    routers = set(net.routers)
    router_degrees = [sum(v in routers for v in net.successors(r)) for r in routers]
    # The first step also builds the forwarding tables: throughput excludes it
    routing_rest = sum(routing.durations[1:])
    hops_rest = sum(routing.hops[1:])

    def stat(values: np.ndarray, fn) -> float:
        return float(fn(values)) if len(values) else float("nan")

    return {
        # Network
        "n_routers": len(net.routers),
        "n_hosts": len(net.hosts),
        "n_links": net.number_of_edges() // 2,
        "mean_router_degree": float(np.mean(router_degrees)),
        "max_router_degree": int(np.max(router_degrees)),
        "reachable_host_share": len(hosts) / len(net.hosts),
        "n_flows": app.number_of_edges(),
        # Execution time
        "build_time_s": build_time,
        "sim_time_s": sim_time,
        "routing_time_s": sum(routing.durations),
        "routing_first_step_s": routing.durations[0] if routing.durations else 0.0,
        "hops": sum(routing.hops),
        "hops_per_s": hops_rest / routing_rest if routing_rest > 0 else float("nan"),
        # Packets
        "generated": generation.generated,
        "delivered": len(latency),
        "dropped": net.dropped_packets,
        "in_transit_at_end": len(net.in_transit),
        "delivery_ratio": len(latency) / generation.generated
        if generation.generated
        else float("nan"),
        # Delivered packets
        "latency_mean_ms": stat(latency, np.mean),
        "latency_p50_ms": stat(latency, np.median),
        "latency_p95_ms": stat(latency, lambda v: np.percentile(v, 95)),
        "path_hops_mean": stat(path_hops, np.mean),
    }
