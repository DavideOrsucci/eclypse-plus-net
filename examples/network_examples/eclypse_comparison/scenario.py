"""Scenario of the comparison: the same infrastructure, application and policies.

The scenario is built once with the classes of the Network extension
(:class:`~eclypse.network.Network` and :class:`~eclypse.network.NetworkApplication`)
and then copied into plain :class:`~eclypse.graph.Infrastructure` and
:class:`~eclypse.graph.Application` objects for base ECLYPSE. The copies have the
same nodes, links and attributes, in the same order, and the same seed, so the
update policies draw the same random numbers and change both versions in the same
way.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import networkx as nx

from eclypse import policies
from eclypse.builders.network import (
    get_network_internet_as,
    get_network_random,
    get_network_scale_free,
)
from eclypse.graph import (
    Application,
    Infrastructure,
)
from eclypse.network import NetworkApplication

if TYPE_CHECKING:
    from eclypse.graph.asset_graph import AssetGraph
    from eclypse.network import Network
    from eclypse.utils.types import UpdatePolicy


TOPOLOGIES = ("internet_as", "scale_free", "random")
"""Topologies supported by :func:`build_network`."""


# ---------------------------------------------------------------------- policies


class LinkFailures:
    """Update policy that takes router links down for a while and then restores them.

    Every ``period`` steps, starting from step ``period``, the policy removes a
    random ``fraction`` of the links between routers (both directions) and restores
    them, with the same attributes, ``duration`` steps later. Links are chosen with
    the random generator of the graph, in the order in which the graph lists them,
    so two graphs with the same links and seed fail the same links.

    The policy changes the topology: :class:`~eclypse.network.Network` rebuilds its
    forwarding tables at the next step, and base ECLYPSE recomputes the paths used to
    check the placement.

    Attributes:
        step (int): Number of calls so far.
        down (dict[tuple[str, str], dict]): Attributes of the links that are
            currently down, keyed by ``(u, v)``.
        failures (int): Number of link failures (one per bidirectional link).
    """

    def __init__(self, period: int, duration: int, fraction: float):
        """Initialize the policy.

        Args:
            period (int): Steps between two failures.
            duration (int): Steps a failed link stays down; less than ``period``.
            fraction (float): Share of the router links that fail each time.
        """
        if not 0 < duration < period:
            raise ValueError("duration must be positive and less than period.")
        self.period = period
        self.duration = duration
        self.fraction = fraction
        self.step = 0
        self.down: dict[tuple[str, str], dict] = {}
        self.failures = 0

    def __call__(self, graph: AssetGraph):
        """Fail or restore links, depending on the step."""
        self.step += 1
        if self.down and self.step % self.period == self.duration:
            for (u, v), data in self.down.items():
                graph.add_edge(u, v, strict=False, **data)
            self.down = {}
        if self.step % self.period == 0:
            links = _router_links(graph)
            k = max(1, round(self.fraction * len(links)))
            for u, v in graph.rnd.sample(links, min(k, len(links))):
                for a, b in ((u, v), (v, u)):
                    self.down[(a, b)] = dict(graph.edges[a, b])
                    graph.remove_edge(a, b)
                self.failures += 1

    @property
    def links_down(self) -> int:
        """Number of bidirectional links currently down."""
        return len(self.down) // 2


def _router_links(graph: AssetGraph) -> list[tuple[str, str]]:
    """Return the bidirectional links between routers, once each, in graph order."""
    nodes = graph.nodes
    links, seen = [], set()
    for u, v in graph.edges():
        if (
            nodes[u].get("role") == "router"
            and nodes[v].get("role") == "router"
            and graph.has_edge(v, u)
            and (v, u) not in seen
        ):
            links.append((u, v))
            seen.add((u, v))
    return links


def infrastructure_policies(config: dict) -> list[UpdatePolicy]:
    """Return new update policies for the infrastructure.

    - The bandwidth of every link fluctuates, at each step, between
      ``bandwidth_range`` times its initial value.
    - Router links fail and recover (see :class:`LinkFailures`).

    Policies keep a state, so each graph needs its own list.

    Args:
        config (dict): The trial configuration.

    Returns:
        list[UpdatePolicy]: The policies; the last one is the :class:`LinkFailures`.
    """
    return [
        policies.distribution.uniform(
            edge_assets="bandwidth_mbps",
            edge_distribution=tuple(config["bandwidth_range"]),
            basis="initial",
        ),
        LinkFailures(
            period=config["failure_period"],
            duration=config["failure_duration"],
            fraction=config["failure_fraction"],
        ),
    ]


def application_policies(config: dict) -> list[UpdatePolicy]:
    """Return new update policies for the application.

    - The packet rate of every flow follows a daily cycle: it oscillates around its
      initial value with relative amplitude ``load_amplitude`` and period
      ``load_period`` steps.
    - Every ``size_period`` steps, the packet size of every flow is redrawn between
      ``size_range`` times its initial value.

    Args:
        config (dict): The trial configuration.

    Returns:
        list[UpdatePolicy]: The policies.
    """
    return [
        policies.workload.diurnal_load(
            amplitude=config["load_amplitude"],
            period=config["load_period"],
            edge_assets="avg_packets_per_step",
            basis="initial",
        ),
        policies.every(
            config["size_period"],
            policies.distribution.uniform(
                edge_assets="packet_size_bytes",
                edge_distribution=tuple(config["size_range"]),
                basis="initial",
            ),
        ),
    ]


# ---------------------------------------------------------------------- network


def build_network(config: dict) -> Network:
    """Generate the network of a trial, with its update policies.

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
        "update_policies": infrastructure_policies(config),
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


def idle_latency_ms(net: Network, u: str, data: dict) -> float:
    """Return the delay of a 1500-byte packet on an idle link, in milliseconds."""
    processing = net.nodes[u].get("processing_time", 0.0)
    propagation = data["length_km"] / data["propagation_speed_km_s"]
    transmission = 1500 * 8 / (data["bandwidth_mbps"] * 1e6)
    return (processing + propagation + transmission) * 1000.0


def to_base_infrastructure(net: Network, config: dict) -> Infrastructure:
    """Copy a network into a base ECLYPSE infrastructure.

    Nodes and links keep their attributes and their order. Links also get a fixed
    ``latency``, the delay of a packet on the idle link, which base ECLYPSE uses as
    the cost of its paths. The infrastructure gets new update policies and the same
    seed as the network.

    Args:
        net (Network): The network to copy.
        config (dict): The trial configuration.

    Returns:
        Infrastructure: The copy.
    """
    infra = Infrastructure(
        infrastructure_id=f"{net.id}_base",
        update_policies=infrastructure_policies(config),
        seed=config["seed"],
    )
    for node, data in net.nodes(data=True):
        infra.add_node(node, **dict(data))
    for u, v, data in net.edges(data=True):
        infra.add_edge(
            u, v, strict=False, **dict(data), latency=idle_latency_ms(net, u, data)
        )
    return infra


# ---------------------------------------------------------------------- application


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

    The ``packets_per_step`` packets of a step are split evenly among the flows.

    Args:
        net (Network): The network.
        config (dict): The trial configuration. It uses ``flows``,
            ``packets_per_step``, ``packet_size_bytes`` and ``seed``.

    Returns:
        tuple[NetworkApplication, dict[str, str]]: The application, with its update
            policies, and the placement of its services.
    """
    hosts = reachable_hosts(net)
    app = NetworkApplication(
        f"App_{net.id}",
        update_policies=application_policies(config),
        seed=config["seed"],
    )
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


def to_base_application(app: NetworkApplication, config: dict) -> Application:
    """Copy a network application into a base ECLYPSE application.

    Services and flows keep their attributes and their order; base ECLYPSE stores
    the packet size and rate of the flows but does not use them. The application
    gets new update policies and the same seed.

    Args:
        app (NetworkApplication): The application to copy.
        config (dict): The trial configuration.

    Returns:
        Application: The copy.
    """
    base = Application(
        f"{app.id}_base",
        update_policies=application_policies(config),
        seed=config["seed"],
    )
    for node, data in app.nodes(data=True):
        base.add_node(node, **dict(data))
    for u, v, data in app.edges(data=True):
        base.add_edge(u, v, **dict(data))
    return base
