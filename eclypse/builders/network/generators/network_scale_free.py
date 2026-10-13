"""Scale-free network generator.

This module provides a factory function that generates a scale-free topology with the
Barabasi-Albert model of :func:`networkx.barabasi_albert_graph` and converts it into a
:class:`~eclypse.network.Network`.

In the Barabasi-Albert model, nodes are added one at a time and each new node connects
to ``m`` existing nodes, chosen with probability proportional to their degree
(preferential attachment). The result is a network with a few highly connected hubs and
many low-degree peripheral nodes, whose degree distribution follows a power law.

Such a topology is suited to studying congestion and bottlenecks: most flows traverse a
limited set of hubs, so their links carry most of the traffic.

The model is described in:
Albert-Laszlo Barabasi, Reka Albert. "Emergence of Scaling in Random Networks."
Science 286(5439), pp. 509-512, 1999.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

from eclypse.network import Network

if TYPE_CHECKING:
    from collections.abc import Callable

    from eclypse.graph.assets import Asset
    from eclypse.utils.types import (
        InitPolicy,
        UpdatePolicies,
    )


def get_network_scale_free(  # noqa: PLR0913 (configurable builder)
    n: int,
    m: int,
    *,
    hosts_per_router: int = 1,
    bandwidth_mbps: float = 1000.0,
    length_km: float = 10.0,
    host_bandwidth_mbps: float = 100.0,
    host_length_km: float = 1.0,
    max_queue_size: int = 100,
    router_processing_time: float = 0.0001,
    host_processing_time: float = 0.0001,
    host_cpu: int = 4,
    host_ram: int = 8,
    infrastructure_id: str = "scale_free_net",
    update_policies: UpdatePolicies = None,
    node_assets: dict[str, Asset] | None = None,
    link_assets: dict[str, Asset] | None = None,
    include_default_assets: bool = False,
    strict: bool = False,
    resource_init: InitPolicy = "min",
    path_algorithm: Callable[[nx.Graph, str, str], list[str]] | None = None,
    seed: int | None = None,
) -> Network:
    """Generate a scale-free network with the Barabasi-Albert model.

    The graph is generated with :func:`networkx.barabasi_albert_graph` and converted
    as follows:

    - node ``i`` becomes the router ``router_<i>``, with the attribute ``degree`` (its
      number of links to other routers);
    - every edge of the graph becomes a bidirectional link of class ``"router"``;
    - every router gets ``hosts_per_router`` hosts, named ``host_<i>_<h>``, with the
      attribute ``router`` (the router they are attached to) and attached to it
      through bidirectional links of class ``"host"``.

    Routers forward traffic and hosts run services, since hosts do not forward transit
    traffic in a :class:`~eclypse.network.Network`. Every link has the attribute
    ``link_class``.

    Args:
        n (int): Number of routers.
        m (int): Number of links from each new router to the existing ones; at least
            1 and less than ``n``.
        hosts_per_router (int): Hosts attached to each router. Defaults to 1.
        bandwidth_mbps (float): Bandwidth of the links between routers, in Mbps.
            Defaults to 1000.
        length_km (float): Length of the links between routers, in km. Defaults to
            10.
        host_bandwidth_mbps (float): Bandwidth of host links, in Mbps. Defaults to
            100.
        host_length_km (float): Length of host links, in km. Defaults to 1.
        max_queue_size (int): Queue capacity of every link, in packets. Defaults to
            100.
        router_processing_time (float): Processing time of routers, in seconds per
            packet. Defaults to 0.0001.
        host_processing_time (float): Processing time of hosts, in seconds per
            packet. Defaults to 0.0001.
        host_cpu (int): CPU capacity of each host. Defaults to 4.
        host_ram (int): RAM capacity of each host. Defaults to 8.
        infrastructure_id (str): Unique ID of the network. Defaults to
            "scale_free_net".
        update_policies (Callable | list[Callable] | None): Graph update policies.
            Defaults to None.
        node_assets (dict[str, Asset] | None): Asset definitions of the nodes.
            Defaults to None.
        link_assets (dict[str, Asset] | None): Asset definitions of the links.
            Defaults to None.
        include_default_assets (bool): Whether to include the default assets.
            Defaults to False.
        strict (bool): If True, raise an error if the asset values are not
            consistent with their spaces. Defaults to False.
        resource_init (InitPolicy): Initialization policy of the resources.
            Defaults to "min".
        path_algorithm (Callable[[nx.Graph, str, str], list[str]] | None): Ignored:
            :class:`~eclypse.network.Network` always uses its own minimum-cost
            path algorithm. Kept for signature compatibility with the other
            builders. Defaults to None.
        seed (int | None): Seed of the graph generation and of the network random
            generator. Defaults to None.

    Returns:
        Network: The generated network.

    Raises:
        ValueError: If ``m`` is not between 1 and ``n - 1``, or if
            ``hosts_per_router`` is negative.
    """
    if not 1 <= m < n:
        raise ValueError(f"m must be at least 1 and less than n (got m={m}, n={n}).")
    if hosts_per_router < 0:
        raise ValueError(
            f"hosts_per_router must be non-negative (got {hosts_per_router})."
        )

    graph = nx.barabasi_albert_graph(n=n, m=m, seed=seed)

    net = Network(
        infrastructure_id=infrastructure_id,
        update_policies=update_policies,
        node_assets=node_assets,
        edge_assets=link_assets,
        include_default_assets=include_default_assets,
        resource_init=resource_init,
        path_algorithm=path_algorithm,
        seed=seed,
    )

    for i in graph.nodes:
        net.add_router(
            _router(i),
            degree=graph.degree[i],
            processing_time=router_processing_time,
            strict=strict,
        )

    for u, v in graph.edges:
        net.add_edge(
            _router(u),
            _router(v),
            symmetric=True,
            strict=strict,
            bandwidth_mbps=bandwidth_mbps,
            length_km=length_km,
            max_queue_size=max_queue_size,
            link_class="router",
        )

    for i in graph.nodes:
        for h in range(hosts_per_router):
            host = f"host_{i}_{h}"
            net.add_host(
                host,
                router=_router(i),
                processing_time=host_processing_time,
                cpu=host_cpu,
                ram=host_ram,
                strict=strict,
            )
            net.add_edge(
                host,
                _router(i),
                symmetric=True,
                strict=strict,
                bandwidth_mbps=host_bandwidth_mbps,
                length_km=host_length_km,
                max_queue_size=max_queue_size,
                link_class="host",
            )

    return net


def _router(i: int) -> str:
    """Return the name of the router of graph node ``i``."""
    return f"router_{i}"
