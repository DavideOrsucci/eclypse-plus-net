"""Minimum-cost paths of the network, computed with rustworkx.

:func:`path_algorithm` is the path algorithm that :class:`~eclypse.network.Network`
registers with ECLYPSE; :func:`to_rustworkx` converts the network into the rustworkx
graph used by the Dijkstra searches, also when the forwarding tables are built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import rustworkx as rx  # type: ignore

if TYPE_CHECKING:
    from eclypse.graph import Infrastructure


def path_algorithm(g: Infrastructure, source: str, target: str) -> list[str]:
    """Compute the minimum-cost path between two nodes.

    This is the path algorithm that :class:`Network` registers with ECLYPSE. Links
    are weighted by their ``cost`` attribute (inverse of the bandwidth, as in
    OSPF). Paths are computed with one single-source Dijkstra per requested source
    and cached on the graph instance, so that only the sources actually queried by
    ECLYPSE are computed. The rustworkx copy of the graph used by the Dijkstra is
    built once and shared by all the sources; it is cached with the paths, so both
    are discarded together when the cache is invalidated.

    Args:
        g (Infrastructure): The graph on which the path is computed.
        source (str): The source node.
        target (str): The target node.

    Returns:
        list[str]: The nodes of the path, from ``source`` to ``target``, or an empty
            list if ``target`` is unreachable.
    """
    cache = g.__dict__.get("_path_cache")
    if cache is None:
        cache = _PathCache()
        g.__dict__["_path_cache"] = cache

    paths = cache.paths.get(source)
    if paths is None:
        if cache.rx is None:
            cache.rx = to_rustworkx(g)
        rx_graph, node_to_rx, rx_to_node = cache.rx
        if source not in node_to_rx:
            return []
        rx_paths = rx.dijkstra_shortest_paths(
            rx_graph, node_to_rx[source], weight_fn=float
        )
        paths = {
            rx_to_node[t]: [rx_to_node[i] for i in p]
            for t, p in rx_paths.items()
            if t != node_to_rx[source]
        }
        cache.paths[source] = paths

    return paths.get(target, [])


class _PathCache:
    """Paths computed by :func:`path_algorithm` on one graph.

    Attributes:
        paths (dict[str, dict[str, list[str]]]): The paths from each queried
            source, keyed by source and then by target.
        rx (tuple | None): The rustworkx copy of the graph returned by
            :func:`to_rustworkx`, or None until the first Dijkstra.
    """

    __slots__ = ("paths", "rx")

    def __init__(self):
        """Initialize an empty cache."""
        self.paths: dict[str, dict[str, list[str]]] = {}
        self.rx: tuple | None = None


def to_rustworkx(g, nodes: list[str] | None = None) -> tuple:
    """Convert a graph into a rustworkx directed graph weighted by link cost.

    Args:
        g (nx.DiGraph): The graph to convert.
        nodes (list[str] | None): The nodes to keep, with the links between them.
            If None, all the nodes. Defaults to None.

    Returns:
        tuple: The rustworkx ``PyDiGraph``, the mapping from node names to
            rustworkx indices, and the list mapping rustworkx indices to node names.
    """
    rx_graph = rx.PyDiGraph()
    rx_to_node = list(g.nodes()) if nodes is None else nodes
    node_to_rx = {node: rx_graph.add_node(node) for node in rx_to_node}
    rx_graph.add_edges_from(
        [
            (node_to_rx[u], node_to_rx[v], data.get("cost", 1.0))
            for u, v, data in g.edges(data=True)
            if nodes is None or (u in node_to_rx and v in node_to_rx)
        ]
    )
    return rx_graph, node_to_rx, rx_to_node
