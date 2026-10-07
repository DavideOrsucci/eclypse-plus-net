"""Internet AS-level network generator.

This module provides a factory function that generates a random graph resembling the
Internet Autonomous System (AS) network with :func:`networkx.random_internet_as_graph`
and converts it into a :class:`~eclypse.network.Network`.

The networkx generator implements the model of:
Ahmed Elmokashfi, Amund Kvalbein, Constantine Dovrolis. "On the Scalability of BGP:
The Role of Topology Growth." IEEE Journal on Selected Areas in Communications
28(8), pp. 1250-1261, 2010.

Each AS has a type: Tier-1 (``T``), mid-level (``M``), content provider (``CP``) or
customer (``C``). Each link is either a ``transit`` link between a provider and its
customer or a ``peer`` link between two ASes that exchange traffic as equals.

The conversion turns every AS into a router and attaches hosts to the customer and
content-provider ASes, which are the ones where services run. The networkx model has
no geography and no capacities, so every link gets the bandwidth and the length
associated with its class: the lowest tier among its two endpoints, in the order
``T``, ``M``, ``CP``, ``C``.
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

AS_TYPES = ("T", "M", "CP", "C")
"""AS types of the model, from the core to the edge."""

DEFAULT_BANDWIDTH_MBPS = {"T": 100_000.0, "M": 10_000.0, "CP": 10_000.0, "C": 1_000.0}
"""Default bandwidth of the AS links, in Mbps, by link class."""

DEFAULT_LENGTH_KM = {"T": 2_000.0, "M": 500.0, "CP": 200.0, "C": 50.0}
"""Default length of the AS links, in km, by link class."""


def get_network_internet_as(  # noqa: PLR0913 (configurable builder)
    n: int = 1000,
    *,
    hosts_per_customer: int = 1,
    hosts_per_content_provider: int = 1,
    bandwidth_mbps: dict[str, float] | None = None,
    length_km: dict[str, float] | None = None,
    host_bandwidth_mbps: float = 100.0,
    host_length_km: float = 1.0,
    max_queue_size: int = 100,
    router_processing_time: float = 0.0001,
    host_processing_time: float = 0.0001,
    host_cpu: int = 4,
    host_ram: int = 8,
    infrastructure_id: str = "internet_as_net",
    update_policies: UpdatePolicies = None,
    node_assets: dict[str, Asset] | None = None,
    link_assets: dict[str, Asset] | None = None,
    include_default_assets: bool = False,
    strict: bool = False,
    resource_init: InitPolicy = "min",
    path_algorithm: Callable[[nx.Graph, str, str], list[str]] | None = None,
    seed: int | None = None,
) -> Network:
    """Generate a network resembling the Internet AS graph.

    The AS graph is generated with :func:`networkx.random_internet_as_graph` and
    converted as follows:

    - AS ``i`` becomes the router ``as_<i>``, with the attribute ``as_type`` (``T``,
      ``M``, ``CP`` or ``C``);
    - every AS link becomes a bidirectional link with the attributes
      ``relationship`` (``"transit"`` or ``"peer"``), ``customer`` (the router of the
      customer AS of a transit link, None for peer links) and ``link_class`` (the
      lowest tier among its endpoints);
    - customer and content-provider ASes get ``hosts_per_customer`` and
      ``hosts_per_content_provider`` hosts, named ``host_<i>_<h>``, with the
      attribute ``as_router`` (the router of their AS) and attached to it through
      links of class ``"host"``.

    Args:
        n (int): Number of ASes. The networkx documentation validates the model for
            1000 to 10000 ASes; smaller values are accepted. Defaults to 1000.
        hosts_per_customer (int): Hosts attached to each customer (``C``) AS.
            Defaults to 1.
        hosts_per_content_provider (int): Hosts attached to each content-provider
            (``CP``) AS. Defaults to 1.
        bandwidth_mbps (dict[str, float] | None): Bandwidth of the AS links, in
            Mbps, by link class (``T``, ``M``, ``CP``, ``C``). Missing classes take
            the values of :data:`DEFAULT_BANDWIDTH_MBPS`. Defaults to None.
        length_km (dict[str, float] | None): Length of the AS links, in km, by link
            class. Missing classes take the values of :data:`DEFAULT_LENGTH_KM`.
            Defaults to None.
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
            "internet_as_net".
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
        seed (int | None): Seed of the AS graph generation and of the network
            random generator. Defaults to None.

    Returns:
        Network: The generated network.

    Raises:
        ValueError: If a number of hosts is negative, or if ``bandwidth_mbps`` or
            ``length_km`` contain unknown link classes.
    """
    for name, value in (
        ("hosts_per_customer", hosts_per_customer),
        ("hosts_per_content_provider", hosts_per_content_provider),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative (got {value}).")
    bandwidths = _per_class("bandwidth_mbps", bandwidth_mbps, DEFAULT_BANDWIDTH_MBPS)
    lengths = _per_class("length_km", length_km, DEFAULT_LENGTH_KM)

    as_graph = nx.random_internet_as_graph(n, seed=seed)

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

    for asn, data in as_graph.nodes(data=True):
        net.add_router(
            _router(asn),
            as_type=data["type"],
            processing_time=router_processing_time,
            strict=strict,
        )

    for u, v, data in as_graph.edges(data=True):
        link_class = max(
            as_graph.nodes[u]["type"], as_graph.nodes[v]["type"], key=AS_TYPES.index
        )
        customer = data.get("customer", "none")
        net.add_edge(
            _router(u),
            _router(v),
            symmetric=True,
            strict=strict,
            bandwidth_mbps=bandwidths[link_class],
            length_km=lengths[link_class],
            max_queue_size=max_queue_size,
            relationship=data["type"],
            customer=None if customer == "none" else _router(customer),
            link_class=link_class,
        )

    hosts_per_type = {"C": hosts_per_customer, "CP": hosts_per_content_provider}
    for asn, data in as_graph.nodes(data=True):
        for h in range(hosts_per_type.get(data["type"], 0)):
            host = f"host_{asn}_{h}"
            net.add_host(
                host,
                as_router=_router(asn),
                processing_time=host_processing_time,
                cpu=host_cpu,
                ram=host_ram,
                strict=strict,
            )
            net.add_edge(
                host,
                _router(asn),
                symmetric=True,
                strict=strict,
                bandwidth_mbps=host_bandwidth_mbps,
                length_km=host_length_km,
                max_queue_size=max_queue_size,
                link_class="host",
            )

    return net


def _router(asn) -> str:
    """Return the name of the router of an AS."""
    return f"as_{asn}"


def _per_class(
    name: str, values: dict[str, float] | None, defaults: dict[str, float]
) -> dict[str, float]:
    """Merge per-class values with their defaults, rejecting unknown classes."""
    values = values or {}
    unknown = set(values) - set(AS_TYPES)
    if unknown:
        raise ValueError(
            f"{name} has unknown link classes {sorted(unknown)}; "
            f"expected a subset of {list(AS_TYPES)}."
        )
    return {**defaults, **values}
