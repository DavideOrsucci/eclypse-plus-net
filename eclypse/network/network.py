"""Module containing the network infrastructure extension for the ECLYPSE framework."""

from collections import deque

import msgspec  # type: ignore
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


class LinkState:
    """Mutable runtime state of a directed link u -> v.

    The dynamic state (queue, queued bytes, last service time, per-step delay
    accumulators) lives here instead of in the edge attribute dictionary. ECLYPSE
    edge dictionaries are ``InvalidatingDict`` instances: every write to them
    triggers a cache invalidation in the Infrastructure, which made each hop pay
    for several invalidations. Mutating attributes of this object is a plain
    attribute store.

    Static link parameters (bandwidth, length, speed) are still read live from the
    edge attribute dictionary (``attrs``), so update policies that change them keep
    working.
    """

    __slots__ = (
        "attrs",
        "count",
        "delay_sum",
        "hop",
        "queue",
        "queue_bytes",
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
        # Sizes (in bytes) of the packets currently queued on the link
        self.queue: deque[int] = deque()
        self.queue_bytes = 0
        self.step_time = 0.0
        self.delay_sum = 0.0
        self.count = 0


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


class Network(Infrastructure):
    """Extend the Infrastructure model of ECLYPSE to simulate physical queuing.

    This class simulates a network of routers using a queuing logic,
    managing physical packet delays and a precomputed OSPF forwarding table.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the Network infrastructure.

        Sets up the underlying ECLYPSE infrastructure and initializes the
        data structures required for tracking link queues and free times.

        Args:
            *args: Variable length argument list passed to the base class.
            **kwargs: Arbitrary keyword arguments passed to the base class.
        """
        # Tell ECLYPSE to use our custom path algorithm for routing
        kwargs["path_algorithm"] = path_algorithm

        super().__init__(*args, **kwargs)

        # Forwarding table: fwd[u][dst] -> LinkState of the next hop link
        self._fwd: dict[str, dict[str, LinkState]] | None = None
        # Runtime state of every directed link, keyed by (u, v)
        self._links: dict[tuple[str, str], LinkState] = {}
        # Links that received telemetry in the current / previous step
        self._touched: list[LinkState] = []
        self._prev_touched: list[LinkState] = []
        self._latencies_initialized = False
        # Cached role lists
        self._hosts: list[str] | None = None
        self._routers: list[str] | None = None
        # Per-step hop telemetry as a Structure of Arrays
        self.step_columns: dict[str, list] = {c: [] for c in TELEMETRY_COLUMNS}
        # Count of the dropped packets
        self.dropped_packets = 0

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

    def _invalidate_node_attrs(self, keys):
        """Also drop the lazily computed paths of the ``available`` view."""
        super()._invalidate_node_attrs(keys)
        if "availability" in keys and self.__dict__.get("_available") is not None:
            self._available.__dict__.pop("_path_cache", None)

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

        Runs an all-pairs Dijkstra in Rust and keeps, for every (source, target)
        pair, only the LinkState of the first hop. Full paths are not translated
        back to Python lists, which was the dominant cost on large topologies.
        """
        self._sync_links()
        links = self._links

        rx_graph, _, rx_to_node = _to_rustworkx(self)
        rx_paths = rx.all_pairs_dijkstra_shortest_paths(rx_graph, float)

        fwd: dict[str, dict[str, LinkState]] = {}
        for source_rx, targets in rx_paths.items():
            source = rx_to_node[source_rx]
            out = {}
            for target_rx, p in targets.items():
                if len(p) > 1 and target_rx != source_rx:
                    out[rx_to_node[target_rx]] = links[(source, rx_to_node[p[1]])]
            fwd[source] = out

        self._fwd = fwd

    def get_next_hop(self, source: str, target: str) -> str | None:
        """Retrieve the next-hop from the FIB in O(1) time."""
        if self._fwd is None:
            self.build_routing_tables()
        link = self._fwd.get(source, {}).get(target)  # type: ignore[union-attr]
        return None if link is None else link.v

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
        attr["router_buffer"] = []
        attr["local_injections"] = []
        router = Router(name=node_id, **attr)
        self.add_node(router.name, **router.assets)
        self.logger.debug(f"Added Router node: {node_id}")

    def add_host(self, node_id: str, **attr):
        """Add a computational host to the network topology.

        Args:
            node_id: The identifier of the host.
            **attr: Additional attributes for the host configuration.
        """
        attr["router_buffer"] = []
        attr["local_injections"] = []
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

        Writes go through the ECLYPSE edge dictionary (``latency`` is the ECLYPSE
        cost attribute), which invalidates its path caches only when the value
        actually changes.
        """
        if self._fwd is None:
            self.build_routing_tables()

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

    def forward_one_hop(  # noqa: PLR0915
        self, packet: Packet, current_time: float
    ) -> str | None:
        """Calculate the next hop for a packet and update its telemetry state.

        Resolves the next hop using the precomputed FIB, evaluates queuing,
        transmission, processing, and propagation delays using store-and-forward
        logic in O(1), and updates the packet's internal state. Also applies
        DropTail queue management.

        Args:
            packet (Packet): The packet object to be forwarded.
            current_time (float): The absolute simulation time in seconds.

        Returns:
            str | None: The identifier of the next node, or None if no route exists
                or if the packet is dropped due to queue congestion.
        """
        fwd = self._fwd
        if fwd is None:
            self.build_routing_tables()
            fwd = self._fwd

        u = packet.current_node
        table = fwd.get(u)  # type: ignore[union-attr]
        link = table.get(packet.dst) if table is not None else None

        if link is None:
            self.logger.warning(
                f"Packet {packet.id} dropped: No routes for {packet.dst}"
            )
            return None

        attrs = link.attrs
        R = attrs.get("bandwidth_mbps", DEFAULT_BANDWIDTH_MBPS) * MBPS_TO_BPS
        queue = link.queue

        # Empty the queue based on the elapsed time
        if current_time > link.step_time:
            bits_service_capacity = (current_time - link.step_time) * R
            queue_bytes = link.queue_bytes
            while queue and bits_service_capacity > 0:
                front_bits = queue[0] * BYTES_TO_BITS
                if bits_service_capacity >= front_bits:
                    bits_service_capacity -= front_bits
                    queue_bytes -= queue.popleft()
                else:
                    break
            link.queue_bytes = queue_bytes
        link.step_time = current_time

        cols = self.step_columns
        if link.count == 0:
            self._touched.append(link)

        queue_length = len(queue)

        # DropTail queue management: if the queue is full, drop the incoming packet
        max_q_size = attrs.get("max_queue_size", float("inf"))
        if queue_length >= max_q_size:
            self.dropped_packets += 1
            link.count += 1  # A drop counts as a 0 ms sample in the link average
            cols["packet_id"].append(packet.id)
            cols["hop_count"].append(packet.hop_count)
            cols["hop"].append(link.hop)
            cols["processing_ms"].append(0.0)
            cols["queue_ms"].append(0.0)
            cols["transmission_ms"].append(0.0)
            cols["propagation_ms"].append(0.0)
            cols["queue_length"].append(float(queue_length))
            cols["arrival_at_next"].append(current_time * SEC_TO_MS)
            cols["dropped"].append(True)
            self.logger.debug(
                f"Packet {packet.id} DROPPED at {u}: Queue full on link {link.hop} "
                f"(Limit: {max_q_size})"
            )
            return None

        d_proc = link.u_attrs.get("processing_time", 0.0)
        size = packet.size
        if R > 0:
            d_queue = link.queue_bytes * BYTES_TO_BITS / R
            d_transm = (size * BYTES_TO_BITS) / R
        else:
            d_queue = d_transm = 0.0
        speed = attrs.get("propagation_speed_km_s", MIN_PROPAGATION_SPEED)
        d_prop = (attrs.get("length_km", MIN_LENGTH_KM) / speed) if speed > 0 else 0.0

        # Enqueue the packet
        queue.append(size)
        link.queue_bytes += size

        processing_ms = d_proc * SEC_TO_MS
        queue_ms = d_queue * SEC_TO_MS
        transmission_ms = d_transm * SEC_TO_MS
        propagation_ms = d_prop * SEC_TO_MS
        hop_delay_ms = (d_proc + d_queue + d_transm + d_prop) * SEC_TO_MS

        link.delay_sum += processing_ms + queue_ms + transmission_ms + propagation_ms
        link.count += 1

        # Advance the packet's state to the next hop
        v = link.v
        packet.previous_node = u
        packet.current_node = v
        packet.hop_count += 1

        cols["packet_id"].append(packet.id)
        cols["hop_count"].append(packet.hop_count)
        cols["hop"].append(link.hop)
        cols["processing_ms"].append(processing_ms)
        cols["queue_ms"].append(queue_ms)
        cols["transmission_ms"].append(transmission_ms)
        cols["propagation_ms"].append(propagation_ms)
        cols["queue_length"].append(float(queue_length))
        cols["arrival_at_next"].append(current_time * SEC_TO_MS + hop_delay_ms)
        cols["dropped"].append(False)
        return v
