"""Scenario of the comparison: a network and its traffic, and their export for ns-3.

The network is generated with the Network builders, with zero processing time on
every node, since ns-3 has no per-node processing delay. The application has one
service per reachable host and ``flows`` random flows between them, which together
generate ``packets_per_s`` packets per second for ``traffic_time_s`` seconds.

ECLYPSE simulates the scenario first. The packets it generates, the topology and the
forwarding tables it uses are then written to CSV files, which the ns-3 program
``eclypse_scenario.cc`` reads to simulate exactly the same packets on the same paths.
"""

from __future__ import annotations

import csv
import math
import random
from typing import TYPE_CHECKING

import networkx as nx

from eclypse.builders.network import (
    get_network_internet_as,
    get_network_random,
    get_network_scale_free,
)
from eclypse.network import NetworkApplication

if TYPE_CHECKING:
    from pathlib import Path

    from eclypse.network import Network


TOPOLOGIES = ("internet_as", "scale_free", "random")
"""Topologies supported by :func:`build_network`."""

WIRE_OVERHEAD_BYTES = 30
"""Bytes that ns-3 adds to the UDP payload of a packet: PPP (2), IPv4 (20) and UDP (8).
The payload is ``size - WIRE_OVERHEAD_BYTES`` bytes, so that the packet occupies
``size`` bytes on the links, as in ECLYPSE."""

MIN_PACKET_BYTES = WIRE_OVERHEAD_BYTES + 8
"""Smallest packet size: the payload carries the 8-byte packet ID."""


def build_network(config: dict) -> Network:
    """Generate the network of a trial, with zero processing time on every node.

    Args:
        config (dict): The trial configuration. It uses ``topology``, ``routers``,
            ``seed``, ``max_queue_size``, ``ba_m`` and ``er_degree_factor``.

    Returns:
        Network: The generated network.

    Raises:
        ValueError: If the topology is not one of :data:`TOPOLOGIES`.
    """
    topology, n, seed = config["topology"], config["routers"], config["seed"]
    common = {
        "max_queue_size": config["max_queue_size"],
        "router_processing_time": 0.0,
        "host_processing_time": 0.0,
        "seed": seed,
    }
    if topology == "internet_as":
        return get_network_internet_as(n, **common)
    if topology == "scale_free":
        return get_network_scale_free(n, config["ba_m"], **common)
    if topology == "random":
        p = min(1.0, config["er_degree_factor"] * math.log(n) / n)
        return get_network_random(n, p, **common)
    raise ValueError(f"Unknown topology {topology!r}; expected one of {TOPOLOGIES}.")


def reachable_hosts(net: Network) -> list[str]:
    """Return the hosts of the largest connected component of the network."""
    graph = nx.Graph()
    graph.add_nodes_from(net.nodes)
    graph.add_edges_from(net.edges)
    largest = max(nx.connected_components(graph), key=len)
    return [h for h in net.hosts if h in largest]


def build_application(
    net: Network, config: dict
) -> tuple[NetworkApplication, dict[str, str]]:
    """Create one service per reachable host and ``flows`` random flows.

    The flows share ``packets_per_s`` packets per second evenly: each generates
    ``packets_per_s * step_duration_s / flows`` packets per step on average.

    Args:
        net (Network): The network.
        config (dict): The trial configuration. It uses ``flows``,
            ``packets_per_s``, ``packet_size_bytes``, ``step_duration_s`` and
            ``seed``.

    Returns:
        tuple[NetworkApplication, dict[str, str]]: The application and the
            placement of its services.

    Raises:
        ValueError: If the packets are smaller than :data:`MIN_PACKET_BYTES`.
    """
    if config["packet_size_bytes"] < MIN_PACKET_BYTES:
        raise ValueError(
            f"packet_size_bytes must be at least {MIN_PACKET_BYTES} "
            f"(got {config['packet_size_bytes']})."
        )
    hosts = reachable_hosts(net)
    app = NetworkApplication(f"App_{net.id}", seed=config["seed"])
    mapping = {}
    for host in hosts:
        app.add_node(f"svc_{host}", cpu=1, ram=1)
        mapping[f"svc_{host}"] = host

    n_flows = min(config["flows"], len(hosts) * (len(hosts) - 1))
    rate = config["packets_per_s"] * config["step_duration_s"] / n_flows
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


def write_csv(path: Path, header: list[str], rows):
    """Write a CSV file with a header row."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def export_for_ns3(net: Network, packets: list[tuple], folder: Path):
    """Write the scenario simulated by ECLYPSE in the files read by ns-3.

    The files are:

    - ``nodes.csv``: ``name``, ``role`` (``host`` or ``router``);
    - ``links.csv``: one row per bidirectional link, with the bandwidth in Mbps and
      the queue capacity in packets of each direction, and the propagation delay in
      seconds;
    - ``routes.csv``: ``node``, ``destination``, ``next_hop``, the forwarding table
      of every node towards every destination host of the traffic;
    - ``packets.csv``: ``id``, ``src``, ``dst``, ``size_bytes``, ``send_time_s``,
      in the order in which ECLYPSE injected them.

    Args:
        net (Network): The network, after the simulation.
        packets (list[tuple]): The injected packets, as ``(id, src, dst, size,
            send_time)`` tuples.
        folder (Path): The folder of the files; it must exist.

    Raises:
        ValueError: If a link is not bidirectional, or if a host does not have
            exactly one link.
    """
    write_csv(
        folder / "nodes.csv",
        ["name", "role"],
        (
            (n, "router" if d.get("role") == "router" else "host")
            for n, d in net.nodes(data=True)
        ),
    )

    links = []
    for u, v, d in net.edges(data=True):
        if not net.has_edge(v, u):
            raise ValueError(f"Link {u} -> {v} is not bidirectional.")
        if str(u) > str(v):  # each link once
            continue
        back = net.edges[v, u]
        links.append(
            (
                u,
                v,
                d["bandwidth_mbps"],
                back["bandwidth_mbps"],
                d["length_km"] / d["propagation_speed_km_s"],
                d["max_queue_size"],
                back["max_queue_size"],
            )
        )
    write_csv(
        folder / "links.csv",
        ["u", "v", "bw_uv_mbps", "bw_vu_mbps", "delay_s", "queue_uv", "queue_vu"],
        links,
    )
    for host in net.hosts:
        if net.out_degree(host) != 1:
            raise ValueError(f"Host {host} must have exactly one link.")

    # Forwarding tables, as built by Network for the simulation
    net._ensure_tables()  # pylint: disable=protected-access
    fwd = net._fwd  # pylint: disable=protected-access
    link_list = net._link_list  # pylint: disable=protected-access
    destinations = sorted({p[2] for p in packets})
    routes = []
    for node in net.nodes:
        u = net.node_index(node)
        for dst in destinations:
            if dst == node:
                continue
            link = fwd[u, net.node_index(dst)]
            if link >= 0:
                routes.append((node, dst, link_list[link].v))
    write_csv(folder / "routes.csv", ["node", "destination", "next_hop"], routes)

    write_csv(
        folder / "packets.csv",
        ["id", "src", "dst", "size_bytes", "send_time_s"],
        ((i, s, d, b, repr(t)) for i, s, d, b, t in packets),
    )
