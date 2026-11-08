"""Network infrastructure with packet-level queuing for the ECLYPSE framework.

This module defines :class:`Network`, an ECLYPSE :class:`~eclypse.graph.Infrastructure`
whose links model store-and-forward transmission with finite FIFO queues. The data
structures it uses live in their own modules:

- :mod:`~eclypse.network.packets`: :class:`PacketBatch`, a set of packets stored as
  one numpy array per field;
- :mod:`~eclypse.network.links`: :class:`LinkState`, the runtime state of a directed
  link, and the vectorized per-link delay model;
- :mod:`~eclypse.network.paths`: :func:`path_algorithm`, the minimum-cost paths.

Every simulation step, the packets that have reached their current node are moved
by one hop with a single call to :meth:`Network.forward_batch`. Per-packet delays
are computed with vectorized numpy operations, while Python objects are touched
once per link.

The names moved to the other modules are re-exported here, for compatibility.
"""

from __future__ import annotations

import numpy as np
import rustworkx as rx  # type: ignore

from eclypse.graph import Infrastructure

from .constants import (
    BYTES_TO_BITS,
    DEFAULT_BANDWIDTH_MBPS,
    DEFAULT_LENGTH_KM,
    DEFAULT_PROPAGATION_SPEED_KM_S,
    MBPS_TO_BPS,
    SEC_TO_MS,
)
from .links import (
    QUEUE_COMPACT_MIN,
    TELEMETRY_COLUMNS,
    LinkState,
    Segments,
    propagation_delay_s,
)
from .packets import (
    PACKET_DTYPES,
    PACKET_FIELDS,
    PacketBatch,
)
from .paths import (
    path_algorithm,
    to_rustworkx,
)

__all__ = [
    "ARRIVAL_TOLERANCE_S",
    "PACKET_DTYPES",
    "PACKET_FIELDS",
    "QUEUE_COMPACT_MIN",
    "TELEMETRY_COLUMNS",
    "LinkState",
    "Network",
    "PacketBatch",
    "path_algorithm",
]

ARRIVAL_TOLERANCE_S = 1e-9
"""Tolerance, in seconds, of the comparison between arrival and current times.

A packet whose arrival time exceeds the current time by less than this amount is
considered arrived, so that floating point rounding in the sum of the hop delays
does not delay it by a whole step.
"""


def _packets(count: int) -> str:
    """Return ``"1 packet"`` or ``"<count> packets"``, for log messages."""
    return f"{count} packet" if count == 1 else f"{count} packets"


class Network(Infrastructure):
    """ECLYPSE infrastructure that simulates packet forwarding with queuing.

    Nodes are either hosts, which run services and generate traffic, or routers,
    which only forward it. Each directed link is a store-and-forward channel with a
    finite FIFO queue managed with the DropTail policy. The delay of a hop is the
    sum of the processing time of the sending node and the queuing, transmission
    and propagation delays of the link.

    Packets follow minimum-cost routes, with link costs inversely proportional to
    the bandwidth (as in OSPF). The forwarding tables are computed when first
    needed and rebuilt after every topology change.

    Node failures follow the availability model of ECLYPSE: a node whose
    ``availability`` is not positive is down, so routes avoid it and the packets
    on it or travelling towards it are lost, as if it had been removed. The
    forwarding tables are rebuilt when a node goes down or comes back up, so the
    node failure policies of ECLYPSE (e.g. ``availability_flap``, ``kill_nodes``)
    affect the traffic. Like ECLYPSE, the network ignores the ``availability`` of
    the links: a link failure must remove the edge (:meth:`remove_edge`) or set its
    bandwidth to zero.

    At every step, :meth:`forward_batch` moves by one hop the packets injected by
    the hosts and the packets that have reached a router, i.e. whose arrival time
    is not later than the current time. The other packets stay in transit until
    they reach their router. Per-hop telemetry is collected in
    :attr:`step_columns`, and the ``latency`` attribute of each edge is updated with
    the average delay observed on the link.

    Attributes:
        in_transit (PacketBatch): Packets travelling towards a router or waiting in
            its buffer, in forwarding order. Each packet is forwarded in the first
            step whose time is not earlier than its arrival time.
        step_columns (dict[str, list]): Telemetry of the current step, with one list
            per name in :data:`TELEMETRY_COLUMNS` and one row per hop.
        step_apps (list[int]): Index of the application of the packet of each
            telemetry row of the current step (see :meth:`application_index`).
        dropped_packets (int): Total number of packets dropped because a link queue
            was full.
        stranded_packets (int): Total number of packets discarded because they
            reached a host that is not their destination (hosts do not forward
            transit traffic).
        unroutable_packets (int): Total number of packets discarded because their
            destination was unreachable, e.g. removed, down or disconnected.
        lost_packets (int): Total number of packets discarded because they were
            on a node that was removed or is down, or travelling towards it.
        local_packets (int): Total number of packets delivered without crossing any
            link, because their source and destination are the same node (e.g. two
            services placed on the same host).
        router_source_packets (int): Total number of packets discarded when
            generated, because their source service is placed on a router (only
            hosts run services: see :class:`~eclypse.network.HostOnlyStrategy`).
        unplaced_packets (int): Total number of packets discarded when generated,
            because a service of their flow is not placed on a node of the network:
            the application is not placed yet, ECLYPSE discarded its placement
            (e.g. after a requirement stopped being met) and has not placed it
            again, or the service is mapped to a node that is not in the network or
            is neither a host nor a router.
        _app_ids (dict[str, int]): Index of every application whose packets have
            entered the network, in order of first appearance.
        _packet_counter (int): Number of packet identifiers allocated so far.
        _node_ids (dict[str, int]): Persistent index of every node ever added.
            Indices are never reused, so packets in transit keep valid references
            across topology changes.
        _node_names (list[str]): Node name of every index.
        _fwd (np.ndarray | None): Forwarding table: ``_fwd[u, d]`` is the position
            in ``_link_list`` of the next-hop link from node ``u`` towards node
            ``d``, or ``-1`` if ``d`` is unreachable. ``None`` when it has to be
            rebuilt.
        _link_list (list[LinkState]): The links, in the order used by ``_fwd``.
        _link_v (np.ndarray): Index of the destination node of each link.
        _link_hops (np.ndarray): Telemetry label (``"u->v"``) of each link.
        _alive (np.ndarray): Whether each node index belongs to the current
            topology and is available (``availability > 0``, see
            :meth:`~eclypse.graph.Infrastructure.is_available`).
        _availability_changed (bool): Whether the availability of a node has
            changed since the forwarding tables were built.
        _host_rank (np.ndarray): Position of each node in :attr:`hosts`, or ``-1``.
        _router_rank (np.ndarray): Position of each node in :attr:`routers`, or
            ``-1``.
        _links (dict[tuple[str, str], LinkState]): Runtime state of every link,
            keyed by ``(u, v)``.
        _touched (list[LinkState]): Links that received traffic in the current step.
        _prev_touched (list[LinkState]): Links that received traffic in the previous
            step.
        _latencies_initialized (bool): Whether every link has received a
            ``latency`` value since the last topology change.
        _hosts (list[str] | None): Cached result of :attr:`hosts`.
        _routers (list[str] | None): Cached result of :attr:`routers`.
    """

    def __init__(self, *args, **kwargs):
        """Initialize an empty network.

        Args:
            *args: Positional arguments passed to
                :class:`~eclypse.graph.Infrastructure`.
            **kwargs: Keyword arguments passed to
                :class:`~eclypse.graph.Infrastructure`. The ``path_algorithm``
                argument is always set to :func:`path_algorithm`.
        """
        # Tell ECLYPSE to use our custom path algorithm for routing
        kwargs["path_algorithm"] = path_algorithm

        super().__init__(*args, **kwargs)

        self._node_ids: dict[str, int] = {}
        self._node_names: list[str] = []
        self._fwd: np.ndarray | None = None
        self._link_list: list[LinkState] = []
        self._link_v = np.empty(0, dtype=np.int64)
        self._link_hops = np.empty(0, dtype=object)
        self._alive = np.empty(0, dtype=bool)
        self._availability_changed = False
        self._host_rank = np.empty(0, dtype=np.int64)
        self._router_rank = np.empty(0, dtype=np.int64)
        self._links: dict[tuple[str, str], LinkState] = {}
        self._touched: list[LinkState] = []
        self._prev_touched: list[LinkState] = []
        self._latencies_initialized = False
        self._hosts: list[str] | None = None
        self._routers: list[str] | None = None
        self.in_transit = PacketBatch.empty()
        self.step_columns: dict[str, list] = {c: [] for c in TELEMETRY_COLUMNS}
        self.step_apps: list[int] = []
        self._app_ids: dict[str, int] = {}
        self._packet_counter = 0
        self.dropped_packets = 0
        self.stranded_packets = 0
        self.unroutable_packets = 0
        self.lost_packets = 0
        self.local_packets = 0
        self.router_source_packets = 0
        self.unplaced_packets = 0

    # ------------------------------------------------------------------ topology

    def _invalidate_routing(self):
        """Discard every structure derived from the topology.

        The forwarding tables, the role caches and the cached paths are rebuilt
        when next needed. Link queues and packets in transit are preserved.
        """
        # The instance might not be fully initialized yet (base __init__)
        if "_fwd" not in self.__dict__:
            return
        self._fwd = None
        self._hosts = None
        self._routers = None
        self._latencies_initialized = False
        self.__dict__.pop("_path_cache", None)

    def _invalidate_node_attrs(self, keys):
        """Invalidate the caches affected by a change of node attributes.

        In addition to the base behavior, when the availability of a node changes,
        discards the cached paths of the ``available`` view and marks the
        forwarding tables for a check: they are rebuilt at the next forwarding if
        the set of available nodes has changed (see :meth:`ensure_routing_tables`).

        Args:
            keys (tuple): The names of the changed node attributes.
        """
        super()._invalidate_node_attrs(keys)
        if "availability" not in keys:
            return
        if self.__dict__.get("_available") is not None:
            self._available.__dict__.pop("_path_cache", None)
        # The instance might not be fully initialized yet (base __init__)
        if "_fwd" in self.__dict__:
            self._availability_changed = True

    def add_node(self, node_for_adding: str, strict: bool = False, **assets):
        """Add a node to the network.

        Prefer :meth:`add_host` and :meth:`add_router`, which also set the role of
        the node.

        Args:
            node_for_adding (str): The name of the node.
            strict (bool): If True, raise an error if the node assets are
                inconsistent. Defaults to False.
            **assets: The assets and attributes of the node.
        """
        super().add_node(node_for_adding, strict=strict, **assets)
        self._invalidate_routing()

    def add_edge(
        self,
        u_of_edge: str,
        v_of_edge: str,
        symmetric: bool = False,
        strict: bool = True,
        bandwidth_mbps: float = DEFAULT_BANDWIDTH_MBPS,
        length_km: float = DEFAULT_LENGTH_KM,
        propagation_speed_km_s: float = DEFAULT_PROPAGATION_SPEED_KM_S,
        max_queue_size: int = 100,
        **attr,
    ):
        """Add a link with its physical and queuing parameters.

        The routing cost of the link is set to the inverse of its bandwidth in bits
        per second.

        Args:
            u_of_edge (str): The source node of the link.
            v_of_edge (str): The destination node of the link.
            symmetric (bool): If True, also add the link in the opposite direction.
                Defaults to False.
            strict (bool): If True, raise an error if the edge assets are
                inconsistent; otherwise log a warning. Defaults to True.
            bandwidth_mbps (float): The bandwidth of the link, in Mbps. Defaults to
                100.0.
            length_km (float): The physical length of the link, in km. Defaults to
                1.0.
            propagation_speed_km_s (float): The signal propagation speed, in km/s.
                Defaults to 200000.0.
            max_queue_size (int): The maximum number of packets that can wait in the
                queue of the link. Defaults to 100.
            **attr: Additional attributes of the link.

        Raises:
            ValueError: If ``bandwidth_mbps`` is not positive.
        """
        if bandwidth_mbps <= 0:
            raise ValueError(
                f"bandwidth_mbps must be positive (got {bandwidth_mbps} for "
                f"{u_of_edge} -> {v_of_edge})."
            )
        attr["bandwidth_mbps"] = bandwidth_mbps
        attr["length_km"] = length_km
        attr["propagation_speed_km_s"] = propagation_speed_km_s
        attr["max_queue_size"] = max_queue_size
        attr["cost"] = 1 / (bandwidth_mbps * MBPS_TO_BPS)  # Cost for OSPF routing
        super().add_edge(
            u_of_edge, v_of_edge, symmetric=symmetric, strict=strict, **attr
        )
        self._invalidate_routing()

    def remove_node(self, n: str):
        """Remove a node, simulating a node failure.

        Packets waiting in the buffer of the node are lost, and routes are
        recomputed at the next forwarding.

        Args:
            n (str): The name of the node to remove.
        """
        super().remove_node(n)
        self._invalidate_routing()
        self.logger.warning(f"[FAILURE] Node {n} removed.")

    def remove_edge(self, u: str, v: str):
        """Remove a link, simulating a link failure.

        Routes are recomputed at the next forwarding.

        Args:
            u (str): The source node of the link.
            v (str): The destination node of the link.
        """
        super().remove_edge(u, v)
        self._invalidate_routing()
        self.logger.warning(f"[FAILURE] Link {u} -> {v} removed.")

    def node_index(self, name: str, register: bool = True) -> int:
        """Return the persistent index of a node, registering the node if needed.

        Args:
            name (str): The name of the node.
            register (bool): Whether to assign a new index to a node that has none
                yet. If False, such a node gets ``-1``. Defaults to True.

        Returns:
            int: The index of the node, or ``-1`` if the node has no index and
                ``register`` is False.
        """
        idx = self._node_ids.get(name)
        if idx is None:
            if not register:
                return -1
            idx = len(self._node_names)
            self._node_ids[name] = idx
            self._node_names.append(name)
        return idx

    def node_name(self, index: int) -> str | None:
        """Return the name of the node with the given index.

        Args:
            index (int): The index of the node.

        Returns:
            str | None: The name of the node, or None if ``index`` is ``-1``.
        """
        return None if index < 0 else self._node_names[index]

    def _sync_links(self):
        """Align the link states with the current edges of the graph.

        Creates a :class:`LinkState` for every new edge and discards those of the
        removed edges. Existing links keep their queues and timers.
        """
        old = self._links
        links: dict[tuple[str, str], LinkState] = {}
        node_attrs = self._node
        for u, v, data in self.edges(data=True):
            link = old.get((u, v))
            if link is None or link.attrs is not data:
                link = LinkState(u, v, data, node_attrs[u])
            else:
                link.u_attrs = node_attrs[u]
            links[(u, v)] = link
        self._links = links

    def build_routing_tables(self):
        """Compute the forwarding tables of all the nodes.

        Runs an all-pairs Dijkstra on the link costs and stores, for every pair of
        nodes, the first link of the minimum-cost path in the dense matrix
        ``_fwd``. This lets the next hop of a whole batch of packets be resolved
        with a single array lookup. Also refreshes the per-node role and
        availability arrays.

        As in ECLYPSE, a node is up if its ``availability`` is positive (see
        :meth:`~eclypse.graph.Infrastructure.is_available`). Paths only cross
        nodes that are up, so the forwarding tables match the paths that ECLYPSE
        computes on the ``available`` view of the network. A node that is down
        neither forwards nor receives packets, as if it had been removed.
        """
        self._sync_links()
        for node in self.nodes():
            self.node_index(node)
        link_pos = self._index_links()
        up = self._available_nodes()
        self._index_nodes(up)
        self._fwd = self._forwarding_matrix(up, link_pos)

    def _index_links(self) -> dict[tuple[str, str], int]:
        """Fix the order of the links and build the per-link arrays.

        Sets ``_link_list``, ``_link_v`` and ``_link_hops``.

        Returns:
            dict[tuple[str, str], int]: The position of every link ``(u, v)`` in
                ``_link_list``.
        """
        ids = self._node_ids
        link_list = list(self._links.values())
        self._link_list = link_list
        self._link_v = np.array([ids[link.v] for link in link_list], dtype=np.int64)
        self._link_hops = np.array([link.hop for link in link_list], dtype=object)
        return {(link.u, link.v): i for i, link in enumerate(link_list)}

    def _index_nodes(self, up: list[str]):
        """Build the per-node availability and role arrays.

        Sets ``_alive``, ``_host_rank`` and ``_router_rank``; the two rank arrays
        are read-only (see :attr:`host_ranks` and :attr:`router_ranks`).

        Args:
            up (list[str]): The nodes that are up.
        """
        n = len(self._node_names)
        ids = self._node_ids
        alive = np.zeros(n, dtype=bool)
        alive[[ids[node] for node in up]] = True
        self._alive = alive
        self._availability_changed = False
        self._host_rank = np.full(n, -1, dtype=np.int64)
        self._host_rank[[ids[h] for h in self.hosts]] = np.arange(len(self.hosts))
        self._host_rank.flags.writeable = False
        self._router_rank = np.full(n, -1, dtype=np.int64)
        self._router_rank[[ids[r] for r in self.routers]] = np.arange(len(self.routers))
        self._router_rank.flags.writeable = False

    def _forwarding_matrix(
        self, up: list[str], link_pos: dict[tuple[str, str], int]
    ) -> np.ndarray:
        """Compute the forwarding matrix with an all-pairs Dijkstra.

        Args:
            up (list[str]): The nodes that are up: paths only cross these nodes.
            link_pos (dict[tuple[str, str], int]): The position of every link in
                ``_link_list``.

        Returns:
            np.ndarray: The matrix ``fwd`` where ``fwd[u, d]`` is the position of
                the first link of the minimum-cost path from node ``u`` to node
                ``d``, or ``-1`` if there is none.
        """
        n = len(self._node_names)
        ids = self._node_ids
        all_up = len(up) == self.number_of_nodes()
        rx_graph, _, rx_to_node = to_rustworkx(self, None if all_up else up)
        rx_paths = rx.all_pairs_dijkstra_shortest_paths(rx_graph, float)

        fwd = np.full((n, n), -1, dtype=np.int32)
        for source_rx, targets in rx_paths.items():
            source = rx_to_node[source_rx]
            cols, vals = [], []
            for target_rx, p in targets.items():
                if len(p) > 1 and target_rx != source_rx:
                    cols.append(ids[rx_to_node[target_rx]])
                    vals.append(link_pos[(source, rx_to_node[p[1]])])
            if cols:
                fwd[ids[source], cols] = vals
        return fwd

    def _available_nodes(self) -> list[str]:
        """Return the nodes that are up, i.e. whose availability is positive."""
        return [n for n in self.nodes() if self.is_available(n)]

    def ensure_routing_tables(self):
        """Build the forwarding tables if they are missing or outdated.

        The tables are outdated after a change of the topology, or after a change
        of availability that turns a node up or down. Availability values that
        change without crossing zero (e.g. from 0.99 to 0.98) leave the tables
        valid, so they are not rebuilt.
        """
        if self._fwd is not None and self._availability_changed:
            self._availability_changed = False
            up = np.zeros(len(self._alive), dtype=bool)
            ids = self._node_ids
            up[[ids[node] for node in self._available_nodes()]] = True
            if not np.array_equal(up, self._alive):
                self._fwd = None
        if self._fwd is None:
            self.build_routing_tables()

    def next_hop(self, node: str, destination: str) -> str | None:
        """Return the next hop of the minimum-cost path from a node to another.

        The answer comes from the forwarding tables used by
        :meth:`forward_batch`, which are rebuilt first if they are outdated.

        Args:
            node (str): The node where the packet is.
            destination (str): The destination of the packet.

        Returns:
            str | None: The neighbour of ``node`` towards ``destination``, or None
                if ``destination`` is ``node`` itself, is unreachable, or one of the
                two nodes is not in the network or is down.
        """
        self.ensure_routing_tables()
        fwd = self._fwd
        u = self.node_index(node, register=False)
        d = self.node_index(destination, register=False)
        if fwd is None or not 0 <= u < len(fwd) or not 0 <= d < len(fwd) or u == d:
            return None
        link = int(fwd[u, d])
        return None if link < 0 else self._link_list[link].v

    # ------------------------------------------------------------------ traffic

    def allocate_packet_ids(self, n: int) -> np.ndarray:
        """Return ``n`` new packet identifiers, unique in the network.

        Identifiers are consecutive integers starting from 1, shared by all the
        applications whose traffic crosses the network.

        Args:
            n (int): The number of identifiers.

        Returns:
            np.ndarray: The identifiers, in increasing order.
        """
        first = self._packet_counter + 1
        self._packet_counter += n
        return np.arange(first, first + n, dtype=np.int64)

    def application_index(self, app_id: str) -> int:
        """Return the index of an application, registering it if needed.

        The index identifies the application of each packet (:attr:`PacketBatch.app`)
        and of each telemetry row (:attr:`step_apps`). Indices are assigned in order
        of registration, starting from 0.

        Args:
            app_id (str): The identifier of the application.

        Returns:
            int: The index of the application.
        """
        idx = self._app_ids.get(app_id)
        if idx is None:
            idx = len(self._app_ids)
            self._app_ids[app_id] = idx
        return idx

    @property
    def application_ids(self) -> list[str]:
        """Identifiers of the registered applications, in index order."""
        return list(self._app_ids)

    # ------------------------------------------------------------------ roles

    @property
    def hosts(self) -> list[str]:
        """Names of the host nodes (nodes without a role are hosts)."""
        if self._hosts is None:
            self._hosts = [
                n for n, d in self.nodes(data=True) if d.get("role", "host") == "host"
            ]
        return self._hosts

    @property
    def host_ranks(self) -> np.ndarray:
        """Position in :attr:`hosts` of each node index, or ``-1`` for non-hosts.

        The array is read-only and is replaced when the forwarding tables are
        rebuilt (see :meth:`ensure_routing_tables`).
        """
        return self._host_rank

    @property
    def router_ranks(self) -> np.ndarray:
        """Position in :attr:`routers` of each node index, or ``-1`` for non-routers.

        The array is read-only and is replaced when the forwarding tables are
        rebuilt (see :meth:`ensure_routing_tables`).
        """
        return self._router_rank

    @property
    def routers(self) -> list[str]:
        """Names of the router nodes."""
        if self._routers is None:
            self._routers = [
                n for n, d in self.nodes(data=True) if d.get("role") == "router"
            ]
        return self._routers

    def add_router(self, node_id: str, **attr):
        """Add a router to the network.

        Routers forward traffic but cannot host services or generate packets: their
        computational resources are set to zero, so that placement strategies never
        select them.

        Args:
            node_id (str): The name of the router.
            **attr: Additional attributes of the router, such as
                ``processing_time`` (seconds per packet).
        """
        attr["role"] = "router"
        # Routers have no computational resources, so placement always fails
        attr["cpu"] = 0
        attr["ram"] = 0
        attr["disk"] = 0
        self.add_node(node_id, **attr)
        self.logger.debug(f"Added Router node: {node_id}")

    def add_host(self, node_id: str, **attr):
        """Add a host to the network.

        Hosts can host services and generate traffic, but do not forward transit
        packets.

        Args:
            node_id (str): The name of the host.
            **attr: The computational resources and additional attributes of the
                host, such as ``processing_time`` (seconds per packet).
        """
        attr["role"] = "host"
        self.add_node(node_id, **attr)
        self.logger.debug(f"Added Host node: {node_id}")

    # ------------------------------------------------------------------ telemetry

    def clear_step_telemetry(self):
        """Start the telemetry of a new step.

        New column lists are allocated rather than cleared, so the lists returned
        by the metric of the previous step remain valid.
        """
        self.step_columns = {c: [] for c in TELEMETRY_COLUMNS}
        self.step_apps = []
        # Discard accumulators left over if update_link_latencies was not called
        for link in self._touched:
            link.delay_sum = 0.0
            link.count = 0
            link.dropped = 0
        self._prev_touched = self._touched
        self._touched = []

    def _default_latency(self, link: LinkState, backlog_s: float = 0.0) -> float:
        """Estimate the latency of a link from its parameters and backlog.

        The estimate is the delay of a 1500-byte packet that waits ``backlog_s``
        before being transmitted.

        Args:
            link (LinkState): The link.
            backlog_s (float): The queuing delay, in seconds. Defaults to 0.0, which
                gives the latency of an idle link.

        Returns:
            float: The estimated latency, in milliseconds.
        """
        data = link.attrs
        d_proc = link.u_attrs.get("processing_time", 0.0) * SEC_TO_MS
        d_prop = propagation_delay_s(data) * SEC_TO_MS
        R = data.get("bandwidth_mbps", DEFAULT_BANDWIDTH_MBPS) * MBPS_TO_BPS
        d_transm = ((1500 * BYTES_TO_BITS) / R) * SEC_TO_MS if R > 0 else 0.0
        return d_proc + backlog_s * SEC_TO_MS + d_prop + d_transm

    def update_link_latencies(self):
        """Update the ``latency`` attribute of the links.

        Links that accepted packets in the current step get the average delay of
        those packets; dropped packets are not latency samples. Links that dropped
        every packet offered to them get the delay that a new packet would
        experience behind their full queue. Idle links get the estimate of
        :meth:`_default_latency`. Since the estimate only changes when a link
        becomes idle, only the links used in the current or in the previous step
        are updated, except after a topology change, when all the links are.
        """
        self.ensure_routing_tables()

        if not self._latencies_initialized:
            idle = self._links.values()
            self._latencies_initialized = True
        else:
            idle = self._prev_touched

        for link in idle:
            if link.count == 0 and link.dropped == 0:
                link.attrs["latency"] = self._default_latency(link)

        for link in self._touched:
            if link.count:
                link.attrs["latency"] = link.delay_sum / link.count
            else:
                link.attrs["latency"] = self._default_latency(
                    link, link.backlog(link.step_time)
                )
            # Reset accumulators; the link stays in _touched until next clear
            link.delay_sum = 0.0
            link.count = 0
            link.dropped = 0

    # ------------------------------------------------------------------ forwarding

    def forward_batch(self, batch: PacketBatch, current_time: float) -> PacketBatch:
        """Forward every packet of a batch by one hop.

        Packets are processed in the batch order. Each packet is routed to the
        next link of its path and either enters the queue of the link or, if the
        queue is full or the bandwidth of the link is not positive (e.g. set to
        zero by an update policy), is dropped. Its hop delay is the sum of:

        - the processing time of the sending node;
        - the queuing delay, i.e. the time needed to transmit the bytes already in
          the queue;
        - the transmission delay of the packet itself;
        - the propagation delay of the link.

        One telemetry row per packet is appended to :attr:`step_columns`. Accepted
        packets advance to the next node and take its arrival time: those that
        reach their destination are returned, the others are appended to
        :attr:`in_transit`.

        Packets already on their destination node (source and destination services
        placed on the same node) do not cross any link: they are returned at once,
        unchanged and without telemetry rows, and counted in :attr:`local_packets`.

        Args:
            batch (PacketBatch): The packets to forward, in processing order.
            current_time (float): Current simulation time, in seconds.

        Returns:
            PacketBatch: The packets delivered to their destination by this hop.
        """
        batch, link_of, local = self._route(batch)
        if not len(batch):
            return local
        hops = self._transmit(batch, link_of, current_time)
        self._record_telemetry(batch, link_of, hops)
        return PacketBatch.concat([local, self._advance(batch, link_of, hops)])

    def _transmit(
        self, batch: PacketBatch, link_of: np.ndarray, current_time: float
    ) -> dict[str, np.ndarray]:
        """Enqueue every packet on its next-hop link and compute its hop delays.

        The packets are grouped by link, keeping the processing order inside a
        link, and each link is served with :class:`Segments`. The packets that are
        not accepted are counted in :attr:`dropped_packets`.

        Args:
            batch (PacketBatch): The packets, in processing order.
            link_of (np.ndarray): The position in ``_link_list`` of the next-hop
                link of every packet.
            current_time (float): Current simulation time, in seconds.

        Returns:
            dict[str, np.ndarray]: The per-packet results of
                :meth:`Segments.hop_delays`, in the order of ``batch``.
        """
        m = len(batch)
        perm = np.argsort(link_of, kind="stable")
        links_sorted = link_of[perm]
        starts = np.flatnonzero(
            np.concatenate(([True], links_sorted[1:] != links_sorted[:-1]))
        )
        seg = Segments(
            links=[self._link_list[i] for i in links_sorted[starts].tolist()],
            starts=starts,
            counts=np.diff(starts, append=m),
            sizes=batch.size[perm],
        )
        seg.serve(current_time)
        hops = seg.hop_delays(current_time)
        self.dropped_packets += seg.commit(hops, self._touched, self)

        # Back to the processing order
        inv = np.empty(m, dtype=np.int64)
        inv[perm] = np.arange(m)
        return {name: values[inv] for name, values in hops.items()}

    def _record_telemetry(
        self, batch: PacketBatch, link_of: np.ndarray, hops: dict[str, np.ndarray]
    ):
        """Append one row per packet to :attr:`step_columns` and :attr:`step_apps`.

        Args:
            batch (PacketBatch): The packets, in processing order.
            link_of (np.ndarray): The position in ``_link_list`` of the link that
                every packet has tried to cross.
            hops (dict[str, np.ndarray]): The per-packet results of
                :meth:`_transmit`.
        """
        acc = hops["accepted"]
        cols = self.step_columns
        cols["packet_id"].extend(batch.id.tolist())
        cols["hop_count"].extend((batch.hop + acc).tolist())
        cols["hop"].extend(self._link_hops[link_of].tolist())
        for name in (
            "processing_ms",
            "queue_ms",
            "transmission_ms",
            "propagation_ms",
            "queue_length",
            "arrival_at_next",
        ):
            cols[name].extend(hops[name].tolist())
        cols["dropped"].extend((~acc).tolist())
        self.step_apps.extend(batch.app.tolist())

    def _advance(
        self, batch: PacketBatch, link_of: np.ndarray, hops: dict[str, np.ndarray]
    ) -> PacketBatch:
        """Move the accepted packets to the next node of their path.

        The packets that reach their destination are returned; the others are
        appended to :attr:`in_transit`.

        Args:
            batch (PacketBatch): The packets, in processing order.
            link_of (np.ndarray): The position in ``_link_list`` of the link that
                every packet has tried to cross.
            hops (dict[str, np.ndarray]): The per-packet results of
                :meth:`_transmit`.

        Returns:
            PacketBatch: The packets delivered to their destination by this hop.
        """
        acc = hops["accepted"]
        moved = batch.take(acc)
        moved.prev = moved.cur
        moved.cur = self._link_v[link_of[acc]]
        moved.hop = moved.hop + 1
        moved.arrival = hops["arrival"][acc]
        delivered = moved.cur == moved.dst
        self.in_transit = PacketBatch.concat([self.in_transit, moved.take(~delivered)])
        return moved.take(delivered)

    def _route(self, batch: PacketBatch) -> tuple[PacketBatch, np.ndarray, PacketBatch]:
        """Find the next-hop link of every packet.

        Packets located on a node that was removed or is down are discarded and
        counted in :attr:`lost_packets` (see :meth:`_discard_lost`). Packets already
        on their destination node need no link: they are set apart and counted in
        :attr:`local_packets`. Packets whose destination is unreachable are
        discarded and counted in :attr:`unroutable_packets`, with one warning per
        destination.

        Args:
            batch (PacketBatch): The packets to route.

        Returns:
            tuple[PacketBatch, np.ndarray, PacketBatch]: The routable packets, the
                position in ``_link_list`` of their next-hop link, and the packets
                already on their destination node.
        """
        self.ensure_routing_tables()
        if not len(batch):
            return batch, np.empty(0, dtype=np.int64), PacketBatch.empty()
        batch = self._discard_lost(batch)
        here = batch.cur == batch.dst
        local = batch.take(here)
        if len(local):
            self.local_packets += len(local)
            batch = batch.take(~here)
        link_of = self._fwd[batch.cur, batch.dst]  # type: ignore[index]
        no_route = link_of < 0
        if no_route.any():
            self.unroutable_packets += int(no_route.sum())
            dsts, counts = np.unique(batch.dst[no_route], return_counts=True)
            for dst, count in zip(dsts.tolist(), counts.tolist(), strict=True):
                self.logger.warning(
                    f"{_packets(count)} dropped: No routes for {self._node_names[dst]}"
                )
            batch = batch.take(~no_route)
            link_of = link_of[~no_route]
        return batch, link_of, local

    def _discard_lost(self, batch: PacketBatch) -> PacketBatch:
        """Discard the packets located on a node that was removed or is down.

        They are counted in :attr:`lost_packets`, with one warning per node.

        Args:
            batch (PacketBatch): The packets, on nodes indexed by the current
                forwarding tables.

        Returns:
            PacketBatch: The packets located on nodes that are up.
        """
        alive = self._alive[batch.cur]
        if alive.all():
            return batch
        self.lost_packets += int(len(alive) - alive.sum())
        nodes, counts = np.unique(batch.cur[~alive], return_counts=True)
        for node, count in zip(nodes.tolist(), counts.tolist(), strict=True):
            self.logger.warning(
                f"{_packets(count)} lost: node {self._node_names[node]} was removed "
                "or is down"
            )
        return batch.take(alive)

    def take_router_batches(
        self, current_time: float | None = None
    ) -> list[tuple[str, PacketBatch]]:
        """Remove the packets that have arrived and group them by router.

        A packet has arrived if its arrival time is not later than ``current_time``,
        up to :data:`ARRIVAL_TOLERANCE_S`; the other packets stay in
        :attr:`in_transit`. Arrived packets located on a host are discarded and
        counted in :attr:`stranded_packets`, since hosts do not forward transit
        traffic. Arrived packets located on a node that was removed or is down are
        discarded and counted in :attr:`lost_packets`.

        Args:
            current_time (float | None): Current simulation time, in seconds. If
                None, every packet in transit is considered arrived. Defaults to
                None.

        Returns:
            list[tuple[str, PacketBatch]]: One ``(router, packets)`` pair per router
                with arrived packets, in :attr:`routers` order. The packets of each
                router are in forwarding order.
        """
        self.ensure_routing_tables()
        transit = self.in_transit
        if current_time is None:
            self.in_transit = PacketBatch.empty()
        else:
            arrived = transit.arrival <= current_time + ARRIVAL_TOLERANCE_S
            self.in_transit = transit.take(~arrived)
            transit = transit.take(arrived)
        if not len(transit):
            return []
        transit = self._discard_lost(transit)
        rank = self._router_rank[transit.cur]
        on_router = rank >= 0
        self.stranded_packets += int(len(rank) - on_router.sum())
        if not on_router.any():
            return []
        transit = transit.take(on_router)
        rank = rank[on_router]
        order = np.argsort(rank, kind="stable")
        transit = transit.take(order)
        rank = rank[order]
        starts = np.flatnonzero(np.concatenate(([True], rank[1:] != rank[:-1])))
        ends = np.append(starts[1:], len(rank))
        routers = self.routers
        return [
            (routers[int(rank[s])], transit.take(slice(s, e)))
            for s, e in zip(starts.tolist(), ends.tolist(), strict=True)
        ]
