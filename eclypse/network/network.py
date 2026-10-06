"""Module containing the network infrastructure extension for the ECLYPSE framework."""

import math
from bisect import bisect_right
from functools import reduce
from operator import add

import msgspec  # type: ignore
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
"""Column names of the per-step hop telemetry (Structure of Arrays)."""


class NetNode:
    """Abstract base class representing a node in the network infrastructure."""

    def __init__(self, name: str, **assets):
        """Initialize a new network node.

        Args:
            name (str): The unique identifier of the node.
            **assets: Arbitrary keyword arguments representing the node's resources and\
            properties.
        """
        self.name = name
        self.assets = assets


class Router(NetNode):
    """Network routing node.

    Represents a node dedicated exclusively to routing network traffic.
    It does not possess computational capabilities and cannot host application services.
    """

    def __init__(self, name: str, **assets):
        """Initialize a new router node.

        Args:
            name (str): The unique identifier of the router.
            **assets: Arbitrary keyword arguments for additional properties.
        """
        assets["role"] = "router"
        # Reset computational resources to 0 to ensure standard placement always fails
        assets["cpu"] = 0
        assets["ram"] = 0
        assets["disk"] = 0
        super().__init__(name, **assets)


class Host(NetNode):
    """Computational network node.

    Represents an end-host capable of hosting application services, executing
    computational tasks, and generating network traffic.
    """

    def __init__(self, name: str, **assets):
        """Initialize a new host node.

        Args:
            name (str): The unique identifier of the host.
            **assets: Arbitrary keyword arguments representing computational assets\
            and properties.
        """
        assets["role"] = "host"
        super().__init__(name, **assets)


class HopInfo(msgspec.Struct, gc=False):  # type: ignore[call-arg]
    """Represent detailed telemetry information for a single network hop.

    Attributes:
        hop (str): The edge identifier representing the hop (e.g., 'A->B').
        processing_ms (float): The processing delay in milliseconds.
        queue_ms (float): The queuing delay in milliseconds.
        transmission_ms (float): The transmission delay in milliseconds.
        propagation_ms (float): The propagation delay in milliseconds.
        queue_length (int): The number of packets in the queue at arrival time.
        arrival_at_next (float): The absolute arrival time at the next node in ms.
        dropped (bool): Indicates if the packet was dropped at this hop. \
            Defaults to False.
    """

    hop: str
    processing_ms: float
    queue_ms: float
    transmission_ms: float
    propagation_ms: float
    queue_length: int
    arrival_at_next: float
    dropped: bool = False


class Packet(msgspec.Struct, gc=False):  # type: ignore[call-arg]
    """Represent a stateful network packet for hop-by-hop routing simulation.

    Attributes:
        id (int): The unique identifier of the packet.
        src (str): The source node identifier.
        dst (str): The destination node identifier.
        size (int): The size of the packet in bytes.
        step_created (int): The simulation step when the packet was created.
        current_node (str): The node where the packet is currently located.
        previous_node (str): The node the packet just left.
        hop_count (int): The number of hops the packet has traversed so far.
    """

    id: int
    src: str
    dst: str
    size: int
    step_created: int
    current_node: str = ""
    previous_node: str | None = None
    hop_count: int = 0


QUEUE_COMPACT_MIN = 1024
"""Minimum number of transmitted entries before a link queue list is compacted."""

PACKET_FIELDS = ("id", "src", "dst", "size", "step", "cur", "prev", "hop")
"""Fields of a PacketBatch. Node fields hold node indices (see Network.node_index)."""


class PacketBatch:
    """A set of packets stored as a Structure of Arrays (one numpy array per field).

    Node references (``src``, ``dst``, ``cur``, ``prev``) are integer node indices
    assigned by :meth:`Network.node_index`; ``prev == -1`` means "no previous node".
    The order of the packets is meaningful: it is the processing order.
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
        """Initialize the batch from one int64 array per field."""
        for f in PACKET_FIELDS:
            setattr(self, f, fields[f])

    @classmethod
    def empty(cls) -> "PacketBatch":
        """Return a batch without packets."""
        return cls(**{f: np.empty(0, dtype=np.int64) for f in PACKET_FIELDS})

    def __len__(self) -> int:
        """Return the number of packets in the batch."""
        return len(self.id)

    def take(self, index: np.ndarray) -> "PacketBatch":
        """Return the packets selected by a boolean mask or an index array."""
        return PacketBatch(**{f: getattr(self, f)[index] for f in PACKET_FIELDS})

    @staticmethod
    def concat(batches: list["PacketBatch"]) -> "PacketBatch":
        """Concatenate batches, preserving their order."""
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
    """Mutable runtime state of a directed link u -> v.

    The dynamic state lives here instead of in the edge attribute dictionary:
    ECLYPSE edge dictionaries are ``InvalidatingDict`` instances and every write to
    them invalidates the Infrastructure path caches.

    The FIFO queue is stored as the list of the *cumulative* bytes ever enqueued
    (``cum``), plus the index of the head (``head``) and the cumulative bytes
    already transmitted (``base``). This turns the service of the queue into a
    binary search and lets a whole batch of packets be enqueued with one
    ``extend``. Python integers keep every byte count exact.

    Static link parameters (bandwidth, length, speed) are read live from the edge
    attribute dictionary (``attrs``), so update policies that change them keep
    working.
    """

    __slots__ = (
        "attrs",
        "base",
        "count",
        "cum",
        "delay_sum",
        "head",
        "hop",
        "step_time",
        "u",
        "u_attrs",
        "v",
    )

    def __init__(self, u: str, v: str, attrs: dict, u_attrs: dict):
        """Initialize the runtime state of a link.

        Args:
            u (str): The source node of the link.
            v (str): The destination node of the link.
            attrs (dict): The edge attribute dictionary of the link.
            u_attrs (dict): The node attribute dictionary of the source node.
        """
        self.u = u
        self.v = v
        self.hop = f"{u}->{v}"
        self.attrs = attrs
        self.u_attrs = u_attrs
        self.cum: list[int] = []
        self.head = 0
        self.base = 0
        self.step_time = 0.0
        self.delay_sum = 0.0
        self.count = 0

    @property
    def queue_length(self) -> int:
        """Return the number of packets in the queue."""
        return len(self.cum) - self.head

    @property
    def queue_bytes(self) -> int:
        """Return the number of bytes in the queue."""
        return self.cum[-1] - self.base if len(self.cum) > self.head else 0

    def serve(self, current_time: float, rate_bps: float):
        """Transmit the packets that left the queue since the last service.

        Equivalent to popping head packets while the remaining capacity
        ``(current_time - step_time) * rate_bps`` covers their size in bits: since
        all byte counts are integers, ``8 * S <= capacity`` is evaluated exactly as
        ``S <= floor(capacity / 8)``.
        """
        if current_time > self.step_time:
            cum = self.cum
            capacity = (current_time - self.step_time) * rate_bps
            if len(cum) > self.head and capacity > 0:
                limit = capacity / BYTES_TO_BITS
                if limit >= cum[-1] - self.base:
                    self.head = len(cum)
                    self.base = cum[-1]
                else:
                    k = bisect_right(cum, self.base + math.floor(limit), self.head)
                    if k > self.head:
                        self.head = k
                        self.base = cum[k - 1]
                # Compact the list once the transmitted prefix dominates
                if self.head > QUEUE_COMPACT_MIN and 2 * self.head > len(cum):
                    del cum[: self.head]
                    self.head = 0
        self.step_time = current_time

    def link_params(self) -> tuple[float, float, float, float]:
        """Return (rate_bps, processing_s, propagation_s, max_queue_size)."""
        attrs = self.attrs
        rate = attrs.get("bandwidth_mbps", DEFAULT_BANDWIDTH_MBPS) * MBPS_TO_BPS
        d_proc = self.u_attrs.get("processing_time", 0.0)
        speed = attrs.get("propagation_speed_km_s", MIN_PROPAGATION_SPEED)
        d_prop = (attrs.get("length_km", MIN_LENGTH_KM) / speed) if speed > 0 else 0.0
        return rate, d_proc, d_prop, attrs.get("max_queue_size", float("inf"))


def path_algorithm(g: Infrastructure, source: str, target: str) -> list[str]:
    """Compute the shortest path (OSPF cost) from source to target.

    Paths are computed lazily, one single-source Dijkstra per requested source,
    and cached on the graph instance. ECLYPSE calls this on its ``available``
    view, which is a separate graph object: computing only what is requested
    avoids a second all-pairs Dijkstra on that view.
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
    """Convert a graph to a rustworkx PyDiGraph weighted by the OSPF cost."""
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
    """Element-wise num / den, with 0.0 where den <= 0 (as the scalar code did)."""
    out = np.zeros(len(num), dtype=np.float64)
    np.divide(num, den, out=out, where=den > 0)
    return out


class _Segments:
    """Packets of one forwarding batch grouped by link (one segment per link).

    Arrays indexed by segment hold per-link values; arrays indexed by packet are
    in link-sorted order (the processing order inside each link is preserved).
    """

    def __init__(self, links, starts, counts, sizes):
        """Initialize the segments of a batch."""
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
        self.bytes0 = np.empty(n, dtype=np.int64)
        self.allowed = np.empty(n, dtype=np.int64)
        self.last = np.empty(n, dtype=np.int64)
        self.csum = np.cumsum(sizes)
        self.before = self.csum[starts] - sizes[starts]

    def serve(self, current_time: float):
        """Serve the queue of every link and record its state and parameters."""
        for j, (link, n) in enumerate(
            zip(self.links, self.counts.tolist(), strict=True)
        ):
            rate, d_proc, d_prop, max_q = link.link_params()
            link.serve(current_time, rate)
            len0 = link.queue_length
            # DropTail: packet i is accepted while len0 + i < max_q
            free = max_q - len0
            self.allowed[j] = n if free >= n else max(0, math.ceil(free))
            self.rate[j] = rate
            self.proc[j] = d_proc
            self.prop[j] = d_prop
            self.len0[j] = len0
            self.bytes0[j] = link.queue_bytes
            self.last[j] = link.cum[-1] if len0 else link.base

    def hop_delays(self, current_time: float) -> dict[str, np.ndarray]:
        """Compute the delays of every packet (link-sorted order).

        Same operations, in the same order, as the scalar implementation, so the
        floating point results are identical.
        """
        rep, sizes = self.rep, self.sizes
        pos = np.arange(len(sizes)) - self.starts[rep]
        allowed = self.allowed[rep]
        accepted = pos < allowed
        queued_bytes = self.bytes0[rep] + (self.csum - sizes - self.before[rep])

        rate = self.rate[rep]
        d_proc = self.proc[rep]
        d_prop = self.prop[rep]
        d_queue = _safe_divide(queued_bytes * BYTES_TO_BITS, rate)
        d_transm = _safe_divide(sizes * BYTES_TO_BITS, rate)

        out = {
            "accepted": accepted,
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

    def commit(self, hop_delay: np.ndarray, touched: list, network) -> int:
        """Enqueue the accepted packets and record the latency samples.

        Returns:
            int: The number of dropped packets.
        """
        # Cumulative queue bytes of every packet (as if all were accepted)
        cum_bytes = (self.csum - self.before[self.rep] + self.last[self.rep]).tolist()
        delays = hop_delay.tolist()
        n_dropped = 0
        for link, start, n, a in zip(
            self.links,
            self.starts.tolist(),
            self.counts.tolist(),
            self.allowed.tolist(),
            strict=True,
        ):
            if a:
                end = start + a
                link.cum.extend(cum_bytes[start:end])
                # Sequential sum (C loop), as the scalar code accumulated it
                link.delay_sum = reduce(add, delays[start:end], link.delay_sum)
            if link.count == 0:
                touched.append(link)
            link.count += n  # A drop counts as a 0 ms sample in the link average
            if a < n:
                n_dropped += n - a
                network.logger.debug(
                    f"{n - a} packets DROPPED at {link.u}: Queue full on link "
                    f"{link.hop} (Limit: {link.attrs.get('max_queue_size')})"
                )
        return n_dropped


class Network(Infrastructure):
    """Extend the Infrastructure model of ECLYPSE to simulate physical queuing.

    This class simulates a network of routers using a queuing logic,
    managing physical packet delays and a precomputed OSPF forwarding table.

    Packets in transit are kept as a :class:`PacketBatch`. All the packets of a
    step are forwarded together by :meth:`forward_batch`, which computes the
    delays of every hop with numpy and touches Python objects only once per link.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the Network infrastructure.

        Args:
            *args: Variable length argument list passed to the base class.
            **kwargs: Arbitrary keyword arguments passed to the base class.
        """
        # Tell ECLYPSE to use our custom path algorithm for routing
        kwargs["path_algorithm"] = path_algorithm

        super().__init__(*args, **kwargs)

        # Persistent node registry: indices are never reused, so packets in
        # transit stay valid across topology changes
        self._node_ids: dict[str, int] = {}
        self._node_names: list[str] = []
        # Forwarding table: _fwd[u_idx, dst_idx] -> index in _link_list (-1: none)
        self._fwd: np.ndarray | None = None
        self._link_list: list[LinkState] = []
        self._link_v = np.empty(0, dtype=np.int64)
        self._link_hops = np.empty(0, dtype=object)
        # Per node index: is in the graph / position in hosts / routers (-1: no)
        self._alive = np.empty(0, dtype=bool)
        self._host_rank = np.empty(0, dtype=np.int64)
        self._router_rank = np.empty(0, dtype=np.int64)
        # Runtime state of every directed link, keyed by (u, v)
        self._links: dict[tuple[str, str], LinkState] = {}
        # Links that received telemetry in the current / previous step
        self._touched: list[LinkState] = []
        self._prev_touched: list[LinkState] = []
        self._latencies_initialized = False
        # Cached role lists
        self._hosts: list[str] | None = None
        self._routers: list[str] | None = None
        # Packets waiting in the router buffers, in arrival order
        self.in_transit = PacketBatch.empty()
        # Per-step hop telemetry as a Structure of Arrays
        self.step_columns: dict[str, list] = {c: [] for c in TELEMETRY_COLUMNS}
        # Count of the dropped packets
        self.dropped_packets = 0
        # Packets that reached a Host which is not their destination: Hosts do
        # not forward transit traffic, so these packets are discarded
        self.stranded_packets = 0

    # ------------------------------------------------------------------ topology

    def _invalidate_routing(self):
        """Drop every structure derived from the topology."""
        # The instance might not be fully initialized yet (base __init__)
        if "_fwd" not in self.__dict__:
            return
        self._fwd = None
        self._hosts = None
        self._routers = None
        self._latencies_initialized = False
        self.__dict__.pop("_path_cache", None)

    def _invalidate_node_attrs(self, keys):
        """Also drop the lazily computed paths of the ``available`` view."""
        super()._invalidate_node_attrs(keys)
        if "availability" in keys and self.__dict__.get("_available") is not None:
            self._available.__dict__.pop("_path_cache", None)

    def add_node(self, node_for_adding: str, strict: bool = False, **assets):
        """Add a node and invalidate the routing structures."""
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
        """Add an edge to the network with queuing parameters.

        Args:
            u_of_edge (str): The source node ID of the edge.
            v_of_edge (str): The destination node ID of the edge.
            symmetric (bool): If True, adds the edge in both directions.\
                Defaults to False.
            strict (bool): If True, raises an error if the assets are inconsistent.\
                If False, logs a warning. Defaults to True.
            bandwidth_mbps (float): The link bandwidth in Mbps. Defaults to 100.0.
            length_km (float): The length of the physical link in km. Defaults to 1.0.
            propagation_speed_km_s (float): The signal propagation speed in km/s.
                Defaults to 200000.0.
            max_queue_size (int): The maximum number of packets that can be queued\
                on this link. Defaults to 100.
            **attr: Additional attributes to apply to the edge.
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
        """Remove a node from the network and trigger an OSPF update.

        Packets waiting in the buffer of the removed node are lost.

        Args:
            n (str): The ID of the node to remove.
        """
        super().remove_node(n)
        self._invalidate_routing()
        self.logger.warning(f"[FAILURE] Node {n} removed.")

    def remove_edge(self, u: str, v: str):
        """Remove an edge from the network and trigger an OSPF update.

        Args:
            u (str): The source node ID of the edge.
            v (str): The destination node ID of the edge.
        """
        super().remove_edge(u, v)
        self._invalidate_routing()
        self.logger.warning(f"[FAILURE] Link {u} -> {v} removed.")

    def node_index(self, name: str) -> int:
        """Return the persistent integer index of a node, registering it if new."""
        idx = self._node_ids.get(name)
        if idx is None:
            idx = len(self._node_names)
            self._node_ids[name] = idx
            self._node_names.append(name)
        return idx

    def node_name(self, index: int) -> str | None:
        """Return the name of the node with the given index (None for -1)."""
        return None if index < 0 else self._node_names[index]

    def _sync_links(self):
        """Create LinkState objects for new edges and drop the removed ones.

        The state of links that still exist (queues, timers) is preserved.
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
        """Pre-calculate the Forwarding Information Base (FIB) for all nodes.

        Runs an all-pairs Dijkstra in Rust and stores, for every (node, target)
        pair, the index of the first-hop link in a dense integer matrix, so that
        the next hop of a whole batch of packets is a single numpy lookup.
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
        """Build the routing tables if the topology changed."""
        if self._fwd is None:
            self.build_routing_tables()

    def get_next_hop(self, source: str, target: str) -> str | None:
        """Retrieve the next-hop from the FIB in O(1) time."""
        self._ensure_tables()
        s, t = self._node_ids.get(source), self._node_ids.get(target)
        if s is None or t is None or not self._alive[s]:
            return None
        link = self._fwd[s, t]  # type: ignore[index]
        return None if link < 0 else self._link_list[link].v

    # ------------------------------------------------------------------ roles

    @property
    def hosts(self) -> list[str]:
        """Return a list of all nodes configured as hosts (cached)."""
        if self._hosts is None:
            self._hosts = [
                n for n, d in self.nodes(data=True) if d.get("role", "host") == "host"
            ]
        return self._hosts

    @property
    def routers(self) -> list[str]:
        """Return a list of all nodes configured as routers (cached)."""
        if self._routers is None:
            self._routers = [
                n for n, d in self.nodes(data=True) if d.get("role") == "router"
            ]
        return self._routers

    def add_router(self, node_id: str, **attr):
        """Add a router to the network topology.

        Args:
            node_id: The identifier of the router.
            **attr: Additional attributes for the router configuration.
        """
        router = Router(name=node_id, **attr)
        self.add_node(router.name, **router.assets)
        self.logger.debug(f"Added Router node: {node_id}")

    def add_host(self, node_id: str, **attr):
        """Add a computational host to the network topology.

        Args:
            node_id: The identifier of the host.
            **attr: Additional attributes for the host configuration.
        """
        host = Host(name=node_id, **attr)
        self.add_node(host.name, **host.assets)
        self.logger.debug(f"Added Host node: {node_id}")

    # ------------------------------------------------------------------ telemetry

    def clear_step_telemetry(self):
        """Start a new telemetry step.

        The column lists are replaced (not cleared) so that the lists handed to the
        metric of the previous step stay valid.
        """
        self.step_columns = {c: [] for c in TELEMETRY_COLUMNS}
        # Discard accumulators left over if update_link_latencies was not called
        for link in self._touched:
            link.delay_sum = 0.0
            link.count = 0
        self._prev_touched = self._touched
        self._touched = []

    @property
    def step_telemetry(self) -> list[HopInfo]:
        """Return the telemetry of the current step as HopInfo objects.

        Convenience view built on demand; hot paths use ``step_columns``.
        """
        c = self.step_columns
        return [
            HopInfo(
                hop=c["hop"][i],
                processing_ms=c["processing_ms"][i],
                queue_ms=c["queue_ms"][i],
                transmission_ms=c["transmission_ms"][i],
                propagation_ms=c["propagation_ms"][i],
                queue_length=int(c["queue_length"][i]),
                arrival_at_next=c["arrival_at_next"][i],
                dropped=c["dropped"][i],
            )
            for i in range(len(c["hop"]))
        ]

    def _default_latency(self, link: LinkState) -> float:
        """Estimate the latency of an idle link (1500 B packet, empty queue)."""
        data = link.attrs
        d_proc = link.u_attrs.get("processing_time", 0.0) * SEC_TO_MS
        speed = data.get("propagation_speed_km_s", MIN_PROPAGATION_SPEED)
        length = data.get("length_km", MIN_LENGTH_KM)
        d_prop = (length / speed) * SEC_TO_MS if speed > 0 else 0.0
        R = data.get("bandwidth_mbps", DEFAULT_BANDWIDTH_MBPS) * MBPS_TO_BPS
        d_transm = ((1500 * BYTES_TO_BITS) / R) * SEC_TO_MS if R > 0 else 0.0
        return d_proc + d_prop + d_transm

    def update_link_latencies(self):
        """Update the latency attribute of the links.

        Links that carried traffic in this step get the average observed hop delay.
        Idle links get a static estimate. The estimate only changes when a link goes
        from busy to idle, so after the first step only the links touched in this
        step or in the previous one are visited (O(hops) instead of O(edges)).
        """
        self._ensure_tables()

        if not self._latencies_initialized:
            idle = self._links.values()
            self._latencies_initialized = True
        else:
            idle = [link for link in self._prev_touched if link.count == 0]

        for link in idle:
            if link.count == 0:
                link.attrs["latency"] = self._default_latency(link)

        for link in self._touched:
            link.attrs["latency"] = link.delay_sum / link.count
            # Reset accumulators; the link stays in _touched until next clear
            link.delay_sum = 0.0
            link.count = 0

    # ------------------------------------------------------------------ forwarding

    def forward_batch(self, batch: PacketBatch, current_time: float) -> PacketBatch:
        """Forward every packet of the batch by one hop, in the batch order.

        The result is the same as calling the per-packet store-and-forward logic
        on each packet in order. Within a step a link is used only by the packets
        of its source node, and the time is the same for all of them, so for each
        link the queue is served once and the i-th packet sees the queue of the
        previous ones: queue delays become cumulative sums and DropTail drops all
        the packets after the first one that finds the queue full.

        The telemetry rows are appended in the batch order. Packets that did not
        reach their destination are appended to ``in_transit``.

        Args:
            batch (PacketBatch): The packets to forward, in processing order.
            current_time (float): The absolute simulation time in seconds.

        Returns:
            PacketBatch: The packets delivered to their destination in this hop.
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
        self.dropped_packets += seg.commit(hops["hop_delay"], self._touched, self)

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
        """Look up the next-hop link of every packet.

        Packets on a node that no longer exists are discarded silently; packets
        without a route are discarded with a warning.

        Returns:
            tuple[PacketBatch, np.ndarray]: The routable packets and the index
                (in ``_link_list``) of their next-hop link.
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

        Returns:
            list[tuple[str, PacketBatch]]: (router, packets in arrival order), in
                the order of ``routers``. Packets that are on a Host (which does not
                forward transit traffic) are discarded and counted in
                ``stranded_packets``.
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

    def forward_one_hop(self, packet: Packet, current_time: float) -> str | None:
        """Forward a single packet by one hop (per-packet API).

        Same model as :meth:`forward_batch`, for code that handles Packet objects
        directly. The simulation events use the batched version.

        Args:
            packet (Packet): The packet object to be forwarded.
            current_time (float): The absolute simulation time in seconds.

        Returns:
            str | None: The identifier of the next node, or None if no route exists
                or if the packet is dropped due to queue congestion.
        """
        self._ensure_tables()
        u = packet.current_node
        s, t = self._node_ids.get(u), self._node_ids.get(packet.dst)
        idx = -1 if s is None or t is None else int(self._fwd[s, t])  # type: ignore[index]
        if idx < 0:
            self.logger.warning(
                f"Packet {packet.id} dropped: No routes for {packet.dst}"
            )
            return None
        link = self._link_list[idx]
        rate, d_proc, d_prop, max_q = link.link_params()
        link.serve(current_time, rate)

        cols = self.step_columns
        if link.count == 0:
            self._touched.append(link)
        link.count += 1
        queue_length = link.queue_length
        now_ms = current_time * SEC_TO_MS

        if queue_length >= max_q:
            self.dropped_packets += 1
            values = (0.0, 0.0, 0.0, 0.0, now_ms, packet.hop_count, True)
            self.logger.debug(
                f"Packet {packet.id} DROPPED at {u}: Queue full on link {link.hop} "
                f"(Limit: {max_q})"
            )
            v = None
        else:
            size = packet.size
            if rate > 0:
                d_queue = link.queue_bytes * BYTES_TO_BITS / rate
                d_transm = (size * BYTES_TO_BITS) / rate
            else:
                d_queue = d_transm = 0.0
            last = link.cum[-1] if len(link.cum) > link.head else link.base
            link.cum.append(last + size)
            p_ms, q_ms = d_proc * SEC_TO_MS, d_queue * SEC_TO_MS
            t_ms, pr_ms = d_transm * SEC_TO_MS, d_prop * SEC_TO_MS
            link.delay_sum += p_ms + q_ms + t_ms + pr_ms
            arrival = now_ms + (d_proc + d_queue + d_transm + d_prop) * SEC_TO_MS
            v = link.v
            packet.previous_node = u
            packet.current_node = v
            packet.hop_count += 1
            values = (p_ms, q_ms, t_ms, pr_ms, arrival, packet.hop_count, False)

        p_ms, q_ms, t_ms, pr_ms, arrival, hop_count, dropped = values
        cols["packet_id"].append(packet.id)
        cols["hop_count"].append(hop_count)
        cols["hop"].append(link.hop)
        cols["processing_ms"].append(p_ms)
        cols["queue_ms"].append(q_ms)
        cols["transmission_ms"].append(t_ms)
        cols["propagation_ms"].append(pr_ms)
        cols["queue_length"].append(float(queue_length))
        cols["arrival_at_next"].append(arrival)
        cols["dropped"].append(dropped)
        return v
