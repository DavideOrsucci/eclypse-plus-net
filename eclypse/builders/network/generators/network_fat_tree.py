"""Fat-Tree network generator.

This module provides a factory function that builds a Fat-Tree data-center topology
as a :class:`~eclypse.network.Network`, with routers (switches), hosts and links
annotated with the parameters of the packet-level model.

A Fat-Tree with parameter ``k`` (an even number) has ``k`` pods. Each pod has
``k/2`` edge switches and ``k/2`` aggregation switches, fully connected to each other;
each edge switch serves ``k/2`` hosts, and each aggregation switch connects to ``k/2``
of the ``(k/2)^2`` core switches. The topology has ``k^3/4`` hosts and provides
``(k/2)^2`` equal-cost paths between hosts of different pods.

The implementation follows the definition from:
Mohammad Al-Fares, Alexander Loukissas, Amin Vahdat. "A Scalable, Commodity Data
Center Network Architecture." ACM SIGCOMM CCR 38(4), 2008,
https://dl.acm.org/doi/10.1145/1402958.1402967
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eclypse.network import Network

if TYPE_CHECKING:
    from collections.abc import Callable

    import networkx as nx

    from eclypse.graph.assets import Asset
    from eclypse.utils.types import (
        InitPolicy,
        UpdatePolicies,
    )

LINK_CLASSES = ("access", "aggregation", "core")
"""Link classes of the Fat-Tree: host-edge, edge-aggregation, aggregation-core."""

MIN_K = 2
"""Smallest valid Fat-Tree parameter."""


def get_network_fat_tree(
    k: int,
    infrastructure_id: str = "fat_tree_net",
    bandwidth_mbps: float | dict[str, float] = 1000,
    length_km: float | dict[str, float] = 2,
    host_cpu: int = 4,
    host_ram: int = 8,
    max_queue_size: int = 100,
    router_processing_time: float = 0.0001,
    host_processing_time: float = 0.0001,
    update_policies: UpdatePolicies = None,
    node_assets: dict[str, Asset] | None = None,
    link_assets: dict[str, Asset] | None = None,
    include_default_assets: bool = False,
    strict: bool = False,
    resource_init: InitPolicy = "min",
    path_algorithm: Callable[[nx.Graph, str, str], list[str]] | None = None,
    seed: int | None = None,
) -> Network:
    """Generate a Fat-Tree network.

    Nodes are named ``core_<i>``, ``agg_<pod>_<a>``, ``edge_<pod>_<e>`` and
    ``host_<pod>_<e>_<h>``, as in the ECLYPSE Fat-Tree infrastructure builder. Every
    node has the attribute ``tier`` (``"core"``, ``"aggregation"``, ``"edge"`` or
    ``"host"``), and every node except the core switches has the attribute ``pod``.
    Every link is bidirectional and has the attribute ``link_class``:
    ``"access"`` (host to edge), ``"aggregation"`` (edge to aggregation) or
    ``"core"`` (aggregation to core).

    Args:
        k (int): The Fat-Tree parameter: an even number, at least 2.
        infrastructure_id (str): Unique ID of the network. Defaults to
            "fat_tree_net".
        bandwidth_mbps (float | dict[str, float]): Bandwidth of the links, in Mbps:
            either one value for every link, or one value per link class
            (``"access"``, ``"aggregation"``, ``"core"``); classes missing from the
            dictionary use 1000. Defaults to 1000.
        length_km (float | dict[str, float]): Length of the links, in km: either one
            value for every link, or one value per link class; classes missing from
            the dictionary use 2. Defaults to 2.
        host_cpu (int): CPU capacity of each host. Defaults to 4.
        host_ram (int): RAM capacity of each host. Defaults to 8.
        max_queue_size (int): Queue capacity of every link, in packets. Defaults to
            100.
        router_processing_time (float): Processing time of the switches, in seconds
            per packet. Defaults to 0.0001.
        host_processing_time (float): Processing time of the hosts, in seconds per
            packet. Defaults to 0.0001.
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
        seed (int | None): Seed of the network random generator. Defaults to None.

    Returns:
        Network: The Fat-Tree network.

    Raises:
        ValueError: If ``k`` is not an even number of at least 2, or if
            ``bandwidth_mbps`` or ``length_km`` contain unknown link classes.
    """
    if k < MIN_K or k % 2 != 0:
        raise ValueError(f"k must be an even number of at least 2 (got {k}).")
    bandwidths = _per_class("bandwidth_mbps", bandwidth_mbps, default=1000.0)
    lengths = _per_class("length_km", length_km, default=2.0)

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

    half = k // 2
    for i in range(half**2):
        net.add_router(
            f"core_{i}",
            tier="core",
            processing_time=router_processing_time,
            strict=strict,
        )
    for pod in range(k):
        _add_pod(
            net,
            pod,
            half,
            bandwidths=bandwidths,
            lengths=lengths,
            max_queue_size=max_queue_size,
            router_processing_time=router_processing_time,
            host_attrs={
                "processing_time": host_processing_time,
                "cpu": host_cpu,
                "ram": host_ram,
            },
            strict=strict,
        )
    return net


def _add_pod(
    net: Network,
    pod: int,
    half: int,
    bandwidths: dict[str, float],
    lengths: dict[str, float],
    max_queue_size: int,
    router_processing_time: float,
    host_attrs: dict,
    strict: bool,
):
    """Add the switches and hosts of a pod and connect them to the core."""

    def link(u: str, v: str, link_class: str):
        net.add_edge(
            u,
            v,
            symmetric=True,
            strict=strict,
            bandwidth_mbps=bandwidths[link_class],
            length_km=lengths[link_class],
            max_queue_size=max_queue_size,
            link_class=link_class,
        )

    # Aggregation switches
    aggregation = [f"agg_{pod}_{a}" for a in range(half)]
    for agg in aggregation:
        net.add_router(
            agg,
            tier="aggregation",
            pod=pod,
            processing_time=router_processing_time,
            strict=strict,
        )

    # Edge switches, fully connected to the aggregation switches of the pod
    for e in range(half):
        edge = f"edge_{pod}_{e}"
        net.add_router(
            edge,
            tier="edge",
            pod=pod,
            processing_time=router_processing_time,
            strict=strict,
        )
        for agg in aggregation:
            link(edge, agg, "aggregation")

        # Hosts of the edge switch
        for h in range(half):
            host = f"host_{pod}_{e}_{h}"
            net.add_host(host, tier="host", pod=pod, strict=strict, **host_attrs)
            link(host, edge, "access")

    # Aggregation switch a connects to the core switches a*k/2 ... a*k/2 + k/2 - 1
    for a, agg in enumerate(aggregation):
        for j in range(half):
            link(agg, f"core_{a * half + j}", "core")


def _per_class(name: str, value: float | dict[str, float], default: float) -> dict:
    """Expand a scalar or a per-class dictionary into one value per link class."""
    if not isinstance(value, dict):
        return dict.fromkeys(LINK_CLASSES, float(value))
    unknown = set(value) - set(LINK_CLASSES)
    if unknown:
        raise ValueError(
            f"{name} has unknown link classes {sorted(unknown)}; "
            f"expected a subset of {list(LINK_CLASSES)}."
        )
    return {cls: float(value.get(cls, default)) for cls in LINK_CLASSES}
