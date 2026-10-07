"""Network infrastructure with packet-level queuing for the ECLYPSE framework.

This module defines :class:`Network`, an ECLYPSE :class:`~eclypse.graph.Infrastructure`
whose links model store-and-forward transmission with finite FIFO queues, together
with the data structures used to forward packets in batches:

- :class:`PacketBatch` stores a set of packets as one numpy array per field.
- :class:`LinkState` holds the runtime state (queue, timers, latency samples) of a
  directed link.

Every simulation step, all the packets to be forwarded are moved by one hop with a
single call to :meth:`Network.forward_batch`. Per-packet delays are computed with
vectorized numpy operations, while Python objects are touched once per link.
"""

import math
from bisect import bisect_right
from functools import reduce
from operator import add

import numpy as np
import rustworkx as rx  # type: ignore

from eclypse.graph import Infrastructure

from .constants import (
    BYTES_TO_BITS,
    DEFAULT_BANDWIDTH_MBPS,
    DEFAULT_LENGTH_KM,
    DEFAULT_PROPAGATION_SPEED_KM_S,
    MBPS_TO_BPS,
    MIN_LENGTH_KM,
    MIN_PROPAGATION_SPEED,
    SEC_TO_MS,
)

TELEMETRY_COLUMNS = (
    "packet_id",
    "hop_count",
    "hop",
    "processing_ms",
    "queue_ms",
    "transmission_ms",
    "propagation_ms",
    "queue_length",
    "arrival_at_next",
    "dropped",
)
"""Names of the per-hop telemetry columns recorded at every step.

Each forwarded or dropped packet contributes one row:

- ``packet_id``: unique identifier of the packet.
- ``hop_count``: hops traversed by the packet after this hop.
- ``hop``: traversed link, formatted as ``"u->v"``.
- ``processing_ms``: processing delay at the source node of the link.
- ``queue_ms``: queuing delay on the link.
- ``transmission_ms``: transmission (serialization) delay on the link.
- ``propagation_ms``: propagation delay along the link.
- ``queue_length``: packets already queued on the link when the packet arrived.
- ``arrival_at_next``: absolute arrival time at the next node, in milliseconds.
- ``dropped``: whether the packet was discarded because the queue was full.
"""

QUEUE_COMPACT_MIN = 1024
"""Minimum number of transmitted entries before a link queue list is compacted."""

PACKET_FIELDS = ("id", "src", "dst", "size", "step", "cur", "prev", "hop")
"""Names of the fields of a :class:`PacketBatch`."""


class PacketBatch:
    """A set of packets stored as a Structure of Arrays.

    Each field is a one-dimensional ``int64`` numpy array, and position ``i`` of
    every array describes the same packet. Node references are integer indices
    assigned by :meth:`Network.node_index`. The order of the packets is meaningful:
    it is the order in which they are processed.

    Attributes:
        id (np.ndarray): Unique identifier of each packet.
        src (np.ndarray): Index of the node that generated the packet.
        dst (np.ndarray): Index of the destination node.
        size (np.ndarray): Packet size, in bytes.
        step (np.ndarray): Simulation step in which the packet was created.
        cur (np.ndarray): Index of the node where the packet currently is.
        prev (np.ndarray): Index of the node the packet came from, or ``-1`` if it
            has just been injected by its source.
        hop (np.ndarray): Number of hops traversed so far.
    """

    __slots__ = PACKET_FIELDS

    id: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    size: np.ndarray
    step: np.ndarray
    cur: np.ndarray
    prev: np.ndarray
    hop: np.ndarray

    def __init__(self, **fields: np.ndarray):
        """Create a batch from its field arrays.

        Args:
            **fields (np.ndarray): One array per name in :data:`PACKET_FIELDS`, all
                with the same length.
        """
        for f in PACKET_FIELDS:
            setattr(self, f, fields[f])

    @classmethod
    def empty(cls) -> "PacketBatch":
        """Create a batch that contains no packets.

        Returns:
            PacketBatch: An empty batch.
        """
        return cls(**{f: np.empty(0, dtype=np.int64) for f in PACKET_FIELDS})

    def __len__(self) -> int:
        """Return the number of packets in the batch."""
        return len(self.id)

    def take(self, index: np.ndarray) -> "PacketBatch":
        """Select a subset of the packets.

        Args:
            index (np.ndarray): A boolean mask, an array of positions or a slice.

        Returns:
            PacketBatch: The selected packets, in the order given by ``index``.
        """
        return PacketBatch(**{f: getattr(self, f)[index] for f in PACKET_FIELDS})

    @staticmethod
    def concat(batches: list["PacketBatch"]) -> "PacketBatch":
        """Concatenate several batches.

        Args:
            batches (list[PacketBatch]): The batches to join.

        Returns:
            PacketBatch: The packets of all the batches, in the given order.
        """
        batches = [b for b in batches if len(b)]
        if not batches:
            return PacketBatch.empty()
        if len(batches) == 1:
            return batches[0]
        return PacketBatch(
            **{
                f: np.concatenate([getattr(b, f) for b in batches])
                for f in PACKET_FIELDS
            }
        )


class LinkState:
    """Runtime state of a directed link ``u -> v``.

    The state is kept outside the ECLYPSE edge attribute dictionary, whose writes
    invalidate the path caches of the infrastructure. Static parameters (bandwidth,
    length, propagation speed, queue capacity) are instead read from the edge
    attributes at every use, so that update policies acting on them take effect
    immediately.

    The link is a FIFO server that transmits one packet at a time at its full
    rate. Its queue is described by the time at which each packet finishes its
    transmission (``departures``): a packet arriving at time ``a`` waits
    ``max(0, busy_until - a)`` and then occupies the link for its transmission
    time (Lindley recursion). Partially transmitted packets are therefore
    accounted for exactly, whatever the times at which the queue is observed.

    Attributes:
        u (str): Source node of the link.
        v (str): Destination node of the link.
        hop (str): Link label used in the telemetry, formatted as ``"u->v"``.
        attrs (dict): Edge attribute dictionary of the link in the infrastructure.
        u_attrs (dict): Node attribute dictionary of the source node, used to read
            its processing time.
        departures (list[float]): Time, in seconds, at which each accepted packet
            finishes its transmission, in FIFO order. Entries before ``head`` belong
            to packets already transmitted.
        head (int): Index in ``departures`` of the first packet not yet fully
            transmitted at ``step_time``.
        busy_until (float): Time, in seconds, at which the link finishes
            transmitting all the accepted packets.
        step_time (float): Simulation time, in seconds, of the last service of the
            queue.
        delay_sum (float): Sum of the hop delays, in milliseconds, of the packets
            accepted on the link in the current step.
        count (int): Number of packets accepted on the link in the current step.
        dropped (int): Number of packets dropped by the link in the current step.
    """

    __slots__ = (
        "attrs",
        "busy_until",
        "count",
        "delay_sum",
        "departures",
        "dropped",
        "head",
        "hop",
        "step_time",
        "u",
        "u_attrs",
        "v",
    )

    def __init__(self, u: str, v: str, attrs: dict, u_attrs: dict):
        """Initialize the state of an idle link with an empty queue.

        Args:
            u (str): Source node of the link.
            v (str): Destination node of the link.
            attrs (dict): Edge attribute dictionary of the link.
            u_attrs (dict): Node attribute dictionary of the source node.
        """
        self.u = u
        self.v = v
        self.hop = f"{u}->{v}"
        self.attrs = attrs
        self.u_attrs = u_attrs
        self.departures: list[float] = []
        self.head = 0
        self.busy_until = 0.0
        self.step_time = 0.0
        self.delay_sum = 0.0
        self.count = 0
        self.dropped = 0

    @property
    def queue_length(self) -> int:
        """Number of packets not yet fully transmitted at the last service."""
        return len(self.departures) - self.head

    def backlog(self, current_time: float) -> float:
        """Return the time needed to transmit the packets already accepted.

        Args:
            current_time (float): Current simulation time, in seconds.

        Returns:
            float: The waiting time, in seconds, of a packet arriving at
                ``current_time``.
        """
        return max(0.0, self.busy_until - current_time)

    def serve(self, current_time: float):
        """Remove from the queue the packets transmitted by ``current_time``.

        A packet leaves the queue when its transmission ends, i.e. when its
        departure time is not later than ``current_time``.

        Args:
            current_time (float): Current simulation time, in seconds.
        """
        departures = self.departures
        if len(departures) > self.head and departures[self.head] <= current_time:
            self.head = bisect_right(departures, current_time, self.head)
            # Compact the list once the transmitted prefix dominates
            if self.head > QUEUE_COMPACT_MIN and 2 * self.head > len(departures):
                del departures[: self.head]
                self.head = 0
        self.step_time = current_time

    def link_params(self) -> tuple[float, float, float, float]:
        """Read the parameters of the delay model from the node and edge attributes.

        Returns:
            tuple[float, float, float, float]: The transmission rate in bits per
                second, the processing time of the source node in seconds, the
                propagation delay in seconds, and the queue capacity in packets
                (``inf`` if unbounded).
        """
        attrs = self.attrs
        rate = attrs.get("bandwidth_mbps", DEFAULT_BANDWIDTH_MBPS) * MBPS_TO_BPS
        d_proc = self.u_attrs.get("processing_time", 0.0)
        speed = attrs.get("propagation_speed_km_s", MIN_PROPAGATION_SPEED)
        d_prop = (attrs.get("length_km", MIN_LENGTH_KM) / speed) if speed > 0 else 0.0
        return rate, d_proc, d_prop, attrs.get("max_queue_size", float("inf"))


def path_algorithm(g: Infrastructure, source: str, target: str) -> list[str]:
    """Compute the minimum-cost path between two nodes.

    This is the path algorithm that :class:`Network` registers with ECLYPSE. Links
    are weighted by their ``cost`` attribute (inverse of the bandwidth, as in
    OSPF). Paths are computed with one single-source Dijkstra per requested source
    and cached on the graph instance, so that only the sources actually queried by
    ECLYPSE are computed.

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
        cache = {}
        g.__dict__["_path_cache"] = cache

    paths = cache.get(source)
    if paths is None:
        rx_graph, node_to_rx, rx_to_node = _to_rustworkx(g)
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
        cache[source] = paths

    return paths.get(target, [])


def _to_rustworkx(g) -> tuple:
    """Convert a graph into a rustworkx directed graph weighted by link cost.

    Args:
        g (nx.DiGraph): The graph to convert.

    Returns:
        tuple: The rustworkx ``PyDiGraph``, the mapping from node names to
            rustworkx indices, and the list mapping rustworkx indices to node names.
    """
    rx_graph = rx.PyDiGraph()
    rx_to_node = list(g.nodes())
    node_to_rx = {node: rx_graph.add_node(node) for node in rx_to_node}
    rx_graph.add_edges_from(
        [
            (node_to_rx[u], node_to_rx[v], data.get("cost", 1.0))
            for u, v, data in g.edges(data=True)
        ]
    )
    return rx_graph, node_to_rx, rx_to_node


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Divide element-wise, returning 0.0 where the denominator is not positive.

    Args:
        num (np.ndarray): The numerators.
        den (np.ndarray): The denominators.

    Returns:
        np.ndarray: ``num / den`` where ``den > 0``, and ``0.0`` elsewhere.
    """
    out = np.zeros(len(num), dtype=np.float64)
    np.divide(num, den, out=out, where=den > 0)
    return out


class _Segments:
    """Packets of a forwarding batch, grouped by the link they traverse.

    The packets are sorted by link, keeping their processing order within each
    link, so that the packets of a link form a contiguous segment. Per-link arrays
    have one entry per segment; per-packet arrays follow the link-sorted order.

    Within a step, all the packets of a link are offered to it at the same time.
    The ``i``-th packet of a segment therefore waits for the backlog of the link
    plus the transmission of the packets that precede it in the segment.

    Attributes:
        links (list[LinkState]): The link of each segment.
        starts (np.ndarray): Position of the first packet of each segment.
        counts (np.ndarray): Number of packets in each segment.
        sizes (np.ndarray): Size in bytes of each packet.
        rep (np.ndarray): Segment index of each packet.
        rate (np.ndarray): Transmission rate of each link, in bits per second.
        proc (np.ndarray): Processing time of the source node of each link, in
            seconds.
        prop (np.ndarray): Propagation delay of each link, in seconds.
        len0 (np.ndarray): Packets not yet fully transmitted on each link when the
            segment arrives.
        wait0 (np.ndarray): Backlog of each link when the segment arrives, in
            seconds.
        allowed (np.ndarray): Number of packets of each segment accepted by the
            DropTail policy; the remaining ones are dropped.
        csum (np.ndarray): Running total of ``sizes`` over the whole batch.
        before (np.ndarray): Value of ``csum`` just before the start of each
            segment.
    """

    def __init__(self, links, starts, counts, sizes):
        """Initialize the segments of a batch.

        Args:
            links (list[LinkState]): The link of each segment.
            starts (np.ndarray): Position of the first packet of each segment.
            counts (np.ndarray): Number of packets in each segment.
            sizes (np.ndarray): Size in bytes of each packet, in link-sorted order.
        """
        self.links: list[LinkState] = links
        self.starts: np.ndarray = starts
        self.counts: np.ndarray = counts
        self.sizes: np.ndarray = sizes
        self.rep = np.repeat(np.arange(len(links)), counts)
        n = len(links)
        self.rate = np.empty(n)
        self.proc = np.empty(n)
        self.prop = np.empty(n)
        self.len0 = np.empty(n, dtype=np.int64)
        self.wait0 = np.empty(n)
        self.allowed = np.empty(n, dtype=np.int64)
        self.csum = np.cumsum(sizes)
        self.before = self.csum[starts] - sizes[starts]

    def serve(self, current_time: float):
        """Serve the queue of every link and record its state and parameters.

        Args:
            current_time (float): Current simulation time, in seconds.
        """
        for j, (link, n) in enumerate(
            zip(self.links, self.counts.tolist(), strict=True)
        ):
            rate, d_proc, d_prop, max_q = link.link_params()
            link.serve(current_time)
            len0 = link.queue_length
            # DropTail: packet i is accepted while len0 + i < max_q
            free = max_q - len0
            self.allowed[j] = n if free >= n else max(0, math.ceil(free))
            self.rate[j] = rate
            self.proc[j] = d_proc
            self.prop[j] = d_prop
            self.len0[j] = len0
            self.wait0[j] = link.backlog(current_time)

    def hop_delays(self, current_time: float) -> dict[str, np.ndarray]:
        """Compute the delays and the outcome of every packet.

        Dropped packets get zero delays and an arrival time equal to the current
        time.

        Args:
            current_time (float): Current simulation time, in seconds.

        Returns:
            dict[str, np.ndarray]: Per-packet arrays, in link-sorted order:
                ``accepted`` (whether the packet entered the queue), the delay
                columns of :data:`TELEMETRY_COLUMNS` (``processing_ms``,
                ``queue_ms``, ``transmission_ms``, ``propagation_ms``),
                ``arrival_at_next``, ``queue_length``, ``hop_delay`` (the sum of
                the four delays, in milliseconds) and ``departure`` (the time at
                which the transmission of the packet ends, in seconds).
        """
        rep, sizes = self.rep, self.sizes
        pos = np.arange(len(sizes)) - self.starts[rep]
        allowed = self.allowed[rep]
        accepted = pos < allowed
        bytes_ahead = self.csum - sizes - self.before[rep]

        rate = self.rate[rep]
        d_proc = self.proc[rep]
        d_prop = self.prop[rep]
        d_queue = self.wait0[rep] + _safe_divide(bytes_ahead * BYTES_TO_BITS, rate)
        d_transm = _safe_divide(sizes * BYTES_TO_BITS, rate)

        out = {
            "accepted": accepted,
            "departure": current_time + d_queue + d_transm,
            "processing_ms": d_proc * SEC_TO_MS,
            "queue_ms": d_queue * SEC_TO_MS,
            "transmission_ms": d_transm * SEC_TO_MS,
            "propagation_ms": d_prop * SEC_TO_MS,
            "arrival_at_next": current_time * SEC_TO_MS
            + (d_proc + d_queue + d_transm + d_prop) * SEC_TO_MS,
            "queue_length": (self.len0[rep] + np.minimum(pos, allowed)).astype(
                np.float64
            ),
        }
        dropped = ~accepted
        if dropped.any():
            for name in ("processing_ms", "queue_ms", "transmission_ms"):
                out[name][dropped] = 0.0
            out["propagation_ms"][dropped] = 0.0
            out["arrival_at_next"][dropped] = current_time * SEC_TO_MS
        out["hop_delay"] = (
            out["processing_ms"]
            + out["queue_ms"]
            + out["transmission_ms"]
            + out["propagation_ms"]
        )
        return out

    def commit(self, hops: dict[str, np.ndarray], touched: list, network) -> int:
        """Enqueue the accepted packets and record the latency samples of each link.

        Args:
            hops (dict[str, np.ndarray]): The result of :meth:`hop_delays`.
            touched (list[LinkState]): Links that received traffic in the current
                step; links used for the first time in the step are appended.
            network (Network): The network, used for logging.

        Returns:
            int: The number of dropped packets.
        """
        departures = hops["departure"].tolist()
        delays = hops["hop_delay"].tolist()
        n_dropped = 0
        for link, start, n, a in zip(
            self.links,
            self.starts.tolist(),
            self.counts.tolist(),
            self.allowed.tolist(),
            strict=True,
        ):
            if link.count == 0 and link.dropped == 0:
                touched.append(link)
            if a:
                end = start + a
                link.departures.extend(departures[start:end])
                link.busy_until = departures[end - 1]
                # Left-to-right summation, for reproducible floating point results
                link.delay_sum = reduce(add, delays[start:end], link.delay_sum)
                link.count += a
            if a < n:
                link.dropped += n - a
                n_dropped += n - a
                network.logger.debug(
                    f"{n - a} packets DROPPED at {link.u}: Queue full on link "
                    f"{link.hop} (Limit: {link.attrs.get('max_queue_size')})"
                )
        return n_dropped


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

    At every step, :meth:`forward_batch` moves all the packets to be forwarded by
    one hop. Per-hop telemetry is collected in :attr:`step_columns`, and the
    ``latency`` attribute of each edge is updated with the average delay observed
    on the link.

    Attributes:
        in_transit (PacketBatch): Packets waiting in the router buffers, to be
            forwarded at the next step, in arrival order.
        step_columns (dict[str, list]): Telemetry of the current step, with one list
            per name in :data:`TELEMETRY_COLUMNS` and one row per hop.
        dropped_packets (int): Total number of packets dropped because a link queue
            was full.
        stranded_packets (int): Total number of packets discarded because they
            reached a host that is not their destination (hosts do not forward
            transit traffic).
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
            topology.
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
        self.dropped_packets = 0
        self.stranded_packets = 0

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

        In addition to the base behavior, discards the cached paths of the
        ``available`` view when the availability of a node changes.

        Args:
            keys (tuple): The names of the changed node attributes.
        """
        super()._invalidate_node_attrs(keys)
        if "availability" in keys and self.__dict__.get("_available") is not None:
            self._available.__dict__.pop("_path_cache", None)

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
        """
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

    def node_index(self, name: str) -> int:
        """Return the persistent index of a node, registering the node if needed.

        Args:
            name (str): The name of the node.

        Returns:
            int: The index of the node.
        """
        idx = self._node_ids.get(name)
        if idx is None:
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
        """
        self._sync_links()
        for node in self.nodes():
            self.node_index(node)
        n = len(self._node_names)
        ids = self._node_ids

        link_list = list(self._links.values())
        link_pos = {(link.u, link.v): i for i, link in enumerate(link_list)}
        self._link_list = link_list
        self._link_v = np.array([ids[link.v] for link in link_list], dtype=np.int64)
        self._link_hops = np.array([link.hop for link in link_list], dtype=object)

        alive = np.zeros(n, dtype=bool)
        alive[[ids[node] for node in self.nodes()]] = True
        self._alive = alive
        self._host_rank = np.full(n, -1, dtype=np.int64)
        self._host_rank[[ids[h] for h in self.hosts]] = np.arange(len(self.hosts))
        self._router_rank = np.full(n, -1, dtype=np.int64)
        self._router_rank[[ids[r] for r in self.routers]] = np.arange(len(self.routers))

        rx_graph, _, rx_to_node = _to_rustworkx(self)
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
        self._fwd = fwd

    def _ensure_tables(self):
        """Build the forwarding tables if they are missing or outdated."""
        if self._fwd is None:
            self.build_routing_tables()

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
        speed = data.get("propagation_speed_km_s", MIN_PROPAGATION_SPEED)
        length = data.get("length_km", MIN_LENGTH_KM)
        d_prop = (length / speed) * SEC_TO_MS if speed > 0 else 0.0
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
        self._ensure_tables()

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
        queue is full, is dropped. Its hop delay is the sum of:

        - the processing time of the sending node;
        - the queuing delay, i.e. the time needed to transmit the bytes already in
          the queue;
        - the transmission delay of the packet itself;
        - the propagation delay of the link.

        One telemetry row per packet is appended to :attr:`step_columns`. Accepted
        packets advance to the next node: those that reach their destination are
        returned, the others are appended to :attr:`in_transit`.

        Args:
            batch (PacketBatch): The packets to forward, in processing order.
            current_time (float): Current simulation time, in seconds.

        Returns:
            PacketBatch: The packets delivered to their destination by this hop.
        """
        batch, link_of = self._route(batch)
        m = len(batch)
        if m == 0:
            return PacketBatch.empty()

        # Group the packets by link, keeping the processing order inside a link
        perm = np.argsort(link_of, kind="stable")
        links_sorted = link_of[perm]
        starts = np.flatnonzero(
            np.concatenate(([True], links_sorted[1:] != links_sorted[:-1]))
        )
        seg = _Segments(
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
        acc = hops["accepted"][inv]
        new_hop = batch.hop + acc

        cols = self.step_columns
        cols["packet_id"].extend(batch.id.tolist())
        cols["hop_count"].extend(new_hop.tolist())
        cols["hop"].extend(self._link_hops[link_of].tolist())
        for name in (
            "processing_ms",
            "queue_ms",
            "transmission_ms",
            "propagation_ms",
            "queue_length",
            "arrival_at_next",
        ):
            cols[name].extend(hops[name][inv].tolist())
        cols["dropped"].extend((~acc).tolist())

        # Advance the accepted packets
        moved = batch.take(acc)
        moved.prev = moved.cur
        moved.cur = self._link_v[link_of[acc]]
        moved.hop = new_hop[acc]
        delivered = moved.cur == moved.dst
        self.in_transit = PacketBatch.concat([self.in_transit, moved.take(~delivered)])
        return moved.take(delivered)

    def _route(self, batch: PacketBatch) -> tuple[PacketBatch, np.ndarray]:
        """Find the next-hop link of every packet.

        Packets located on a node that is no longer in the network are discarded.
        Packets whose destination is unreachable are discarded with a warning.

        Args:
            batch (PacketBatch): The packets to route.

        Returns:
            tuple[PacketBatch, np.ndarray]: The routable packets and the position in
                ``_link_list`` of their next-hop link.
        """
        self._ensure_tables()
        if not len(batch):
            return batch, np.empty(0, dtype=np.int64)
        batch = batch.take(self._alive[batch.cur])
        link_of = self._fwd[batch.cur, batch.dst]  # type: ignore[index]
        no_route = link_of < 0
        if no_route.any():
            names = self._node_names
            for pid, dst in zip(
                batch.id[no_route].tolist(), batch.dst[no_route].tolist(), strict=True
            ):
                self.logger.warning(f"Packet {pid} dropped: No routes for {names[dst]}")
            batch = batch.take(~no_route)
            link_of = link_of[~no_route]
        return batch, link_of

    def take_router_batches(self) -> list[tuple[str, PacketBatch]]:
        """Remove the packets in transit and group them by router.

        Packets located on a host are discarded and counted in
        :attr:`stranded_packets`, since hosts do not forward transit traffic.

        Returns:
            list[tuple[str, PacketBatch]]: One ``(router, packets)`` pair per router
                with waiting packets, in :attr:`routers` order. The packets of each
                router are in arrival order.
        """
        self._ensure_tables()
        transit = self.in_transit
        self.in_transit = PacketBatch.empty()
        if not len(transit):
            return []
        transit = transit.take(self._alive[transit.cur])
        rank = self._router_rank[transit.cur]
        on_router = rank >= 0
        self.stranded_packets += int(len(rank) - on_router.sum())
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
