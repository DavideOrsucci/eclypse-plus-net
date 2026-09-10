"""Module containing the network infrastructure extension for the ECLYPSE framework."""

import heapq
from collections import (
    defaultdict,
    deque,
)
from dataclasses import dataclass

import networkx as nx

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


@dataclass(slots=True)
class HopInfo:
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


@dataclass(slots=True)
class Packet:
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


def path_algorithm(g: nx.Graph, source: str, target: str) -> list[str]:
    """Compute the shortest path using the precomputed FIB.

    This acts as a bridge between Eclypse's internal engine and our fast O(1)
    routing table, preventing Eclypse from repeatedly executing Dijkstra.
    """
    if getattr(g, "full_paths", None) is None:
        g.build_routing_tables()

    # Retrieve the full path in O(1)
    if source not in g.full_paths or target not in g.full_paths[source]:
        return []

    return g.full_paths[source][target]


class Network(Infrastructure):
    """Extend the Infrastructure model of ECLYPSE to simulate physical queuing.

    This class simulates a network of routers using a queuing logic,
    managing physical packet delays and an on-demand LRU cache for OSPF routing.
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

        # Initialization of the variables for the FIB to None
        self.fib = None
        self.full_paths = None
        # Hop telemetry tracking
        self.step_telemetry = []
        # Count of the dropped packets
        self.dropped_packets = 0

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

        Automatically calculates and injects physical queuing parameters
        such as bandwidth in bps and latency before passing the data to
        the underlying graph infrastructure.

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

        # Initialize the link queue and tracking variables for the new edge
        attr["queue"] = deque()
        attr["queue_bytes"] = 0
        attr["step_time"] = 0.0
        super().add_edge(
            u_of_edge, v_of_edge, symmetric=symmetric, strict=strict, **attr
        )

    def update_link_latencies(self):
        """Calculate and update the latency attribute for each link.

        The calculation evaluates the telemetry collected during the current step
        and updates the dynamic latency metric. If no telemetry is present, it
        estimates the latency based on default packet sizes and link properties.
        """
        link_delays = defaultdict(float)
        link_counts = defaultdict(int)

        # Calculate the total delay for each link based on the telemetry of the
        # current step
        for _, hop_info in self.step_telemetry:
            u, v = hop_info.hop.split("->")

            # The total delay for this hop is the sum of processing, queuing,
            # transmission and propagation delays
            total_hop_delay = (
                hop_info.processing_ms
                + hop_info.queue_ms
                + hop_info.transmission_ms
                + hop_info.propagation_ms
            )

            link_delays[(u, v)] += total_hop_delay
            link_counts[(u, v)] += 1

        # Update the attribute 'latency' for each link based on the average delay
        # observed during this step
        for u, v, data in self.edges(data=True):
            # Check the direction u -> v for the link
            # if we have telemetry data for it calculate the average delay
            if (u, v) in link_counts:
                avg_delay = link_delays[(u, v)] / link_counts[(u, v)]
                self[u][v]["latency"] = avg_delay

            # If we don't have telemetry data for this link during this step,
            # we can optionally set a default latency
            else:
                d_proc = self.processing_time(u, v) * SEC_TO_MS
                speed = data.get("propagation_speed_km_s", MIN_PROPAGATION_SPEED)
                length = data.get("length_km", MIN_LENGTH_KM)

                d_prop = (length / speed) * SEC_TO_MS if speed > 0 else 0.0
                # Estimate the transmission delay based on the bandwidth and a
                # typical packet size (e.g., 1500 bytes)
                R = data.get("bandwidth_mbps", DEFAULT_BANDWIDTH_MBPS) * MBPS_TO_BPS
                d_transm = ((1500 * BYTES_TO_BITS) / R) * SEC_TO_MS if R > 0 else 0.0

                self[u][v]["latency"] = d_proc + d_prop + d_transm

    def _all_pairs_dijkstra(self) -> dict:
        """Calculate the shortest paths for all nodes in a optimized manner."""
        full_paths = {}

        # Pre-construction of the adjacency list for ultra-fast O(1) access
        # Bypass NetworkX overhead during graph traversal
        adj = {
            u: [(v, data.get("cost", 1.0)) for v, data in self[u].items()]
            for u in self.nodes()
        }

        # Execution of Dijkstra for each source node
        for source in self.nodes():
            distances = {source: 0.0}
            predecessors = {source: None}
            pq = [(0.0, source)]  # Priority queue (distance, node)

            while pq:
                current_dist, u = heapq.heappop(pq)

                # Ignore obsolete paths if we have already found a better route.
                if current_dist > distances.get(u, float("inf")):
                    continue

                for v, weight in adj[u]:
                    distance = current_dist + weight

                    # Relax the edge
                    if distance < distances.get(v, float("inf")):
                        distances[v] = distance
                        predecessors[v] = u
                        heapq.heappush(pq, (distance, v))

            # Reconstruction of paths in the format expected by Eclypse
            source_paths = {}
            for target in distances:
                path = []
                curr = target
                while curr is not None:
                    path.append(curr)
                    curr = predecessors.get(curr)
                path.reverse()
                source_paths[target] = path

            full_paths[source] = source_paths

        return full_paths

    def build_routing_tables(self):
        """Pre-calculate the Forwarding Information Base (FIB) for all nodes.

        This is executed only once to reduce the computational cost of routing.
        """
        self.fib = defaultdict(dict)
        # Save all complete paths in a class variable
        self.full_paths = self._all_pairs_dijkstra()

        for source_node, targets in self.full_paths.items():
            for target_node, p in targets.items():
                if source_node != target_node and len(p) > 1:
                    # Populate the FIB only with the next-hop for the forward_one_hop
                    self.fib[source_node][target_node] = p[1]

    def get_next_hop(self, source: str, target: str) -> str | None:
        """Retrieve the next-hop from the FIB in O(1) time.

        Avoids calling the native path methods of Eclypse during routing.
        """
        if self.fib is None:
            self.build_routing_tables()

        if source not in self.fib or target not in self.fib[source]:
            return None

        return self.fib[source][target]

    def forward_one_hop(self, packet: Packet, current_time: float) -> str | None:
        """Calculate the next hop for a packet and update its telemetry state.

        Resolves the next hop using the precomputed FIB, evaluates queuing,
        transmission,
        processing, and propagation delays using store-and-forward logic in O(1), and
        updates the packet's internal state. Also applies DropTail queue management.

        Args:
            packet (Packet): The packet object to be forwarded.
            current_time (float): The absolute simulation time in seconds.

        Returns:
            str | None: The identifier of the next node, or None if no route exists
                or if the packet is dropped due to queue congestion.
        """
        u = packet.current_node

        # Resolve the next hop O(1) via FIB
        v = self.get_next_hop(u, packet.dst)

        if v is None:
            self.logger.warning(
                f"Packet {packet.id} dropped: No routes for {packet.dst}"
            )
            return None

        # Obtain the edge data for the link u -> v, including bandwidth and queue size
        edge = self[u][v]

        R = edge.get("bandwidth_mbps", DEFAULT_BANDWIDTH_MBPS) * MBPS_TO_BPS

        # Calculate the queuing delay based on the current queue length
        # and the link bandwidth
        queue = edge["queue"]

        # Empty the queue based on the elapsed time (O(1) tracking)
        if current_time > edge["step_time"]:
            delta_t = current_time - edge["step_time"]
            bits_service_capacity = delta_t * R

            while queue and bits_service_capacity > 0:
                front_packet = queue[0]
                front_packet_bits = front_packet.size * BYTES_TO_BITS
                if bits_service_capacity >= front_packet_bits:
                    bits_service_capacity -= front_packet_bits
                    queue.popleft()
                    # Update of the total bytes in the link queue after dequeuing a packet
                    edge["queue_bytes"] -= front_packet.size
                else:
                    break

        edge["step_time"] = current_time

        # DropTail queue management: if the queue is full, drop the incoming packet
        max_q_size = edge.get("max_queue_size", float("inf"))

        if len(queue) >= max_q_size:
            # If the queue is full, we drop the packet and log the event
            self.dropped_packets += 1
            drop_info = HopInfo(
                hop=f"{u}->{v}",
                processing_ms=0.0,
                queue_ms=0.0,
                transmission_ms=0.0,
                propagation_ms=0.0,
                queue_length=len(queue),
                arrival_at_next=current_time * SEC_TO_MS,
                dropped=True,
            )
            self.step_telemetry.append((packet, drop_info))

            self.logger.debug(
                f"Packet {packet.id} DROPPED at {u}: Queue full on link {u}->{v} "
                f"(Limit: {max_q_size})"
            )
            return None

        d_proc = self.processing_time(u, v)

        # The calculation of the queue delay is done in O(1) using the tracking variable
        d_queue = edge["queue_bytes"] * BYTES_TO_BITS / R if R > 0 else 0.0

        d_transm = ((packet.size * BYTES_TO_BITS) / R) if R > 0 else 0.0

        length = edge.get("length_km", MIN_LENGTH_KM)
        speed = edge.get("propagation_speed_km_s", MIN_PROPAGATION_SPEED)
        d_prop = (length / speed) if speed > 0 else 0.0

        # Save the current queue length for telemetry before adding the new packet
        current_queue_length = len(queue)

        # Enqueue the packet and update the total bytes in the link queue
        queue.append(packet)
        edge["queue_bytes"] += packet.size

        # Calculate the times for the telemetry
        hop_delay_ms = (d_proc + d_queue + d_transm + d_prop) * SEC_TO_MS
        arrival_time_ms = (current_time * SEC_TO_MS) + hop_delay_ms

        # Register the hop information for telemetry and later reporting
        hop_info = HopInfo(
            hop=f"{u}->{v}",
            processing_ms=d_proc * SEC_TO_MS,
            queue_ms=d_queue * SEC_TO_MS,
            transmission_ms=d_transm * SEC_TO_MS,
            propagation_ms=d_prop * SEC_TO_MS,
            queue_length=current_queue_length,
            arrival_at_next=arrival_time_ms,
            dropped=False,
        )
        self.step_telemetry.append((packet, hop_info))

        # Advance the packet's state to the next hop
        packet.previous_node = u
        packet.current_node = v

        packet.hop_count += 1
        return v

    def remove_node(self, n: str):
        """Remove a node from the network and trigger an OSPF cache update.

        Args:
            n (str): The ID of the node to remove.
        """
        super().remove_node(n)

        self.fib = None
        self.full_paths = None

        self.logger.warning(f"[FAILURE] Node {n} removed.")

    def remove_edge(self, u: str, v: str):
        """Remove an edge from the network and trigger an OSPF cache update.

        Args:
            u (str): The source node ID of the edge.
            v (str): The destination node ID of the edge.
        """
        super().remove_edge(u, v)

        self.fib = None
        self.full_paths = None

        self.logger.warning(f"[FAILURE] Link {u} -> {v} removed.")

    def add_router(self, node_id: str, **attr):
        """Add a router to the network topology.

        Args:
            node_id: The identifier of the router.
            **attr: Additional attributes for the router configuration.
        """
        attr["router_buffer"] = []
        attr["local_injections"] = []
        router = Router(name=node_id, **attr)
        super().add_node(router.name, **router.assets)
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
        super().add_node(host.name, **host.assets)
        self.logger.debug(f"Added Host node: {node_id}")

    @property
    def hosts(self) -> list[str]:
        """Return a list of all nodes configured as hosts."""
        return [n for n, d in self.nodes(data=True) if d.get("role", "host") == "host"]

    @property
    def routers(self) -> list[str]:
        """Return a list of all nodes configured as routers."""
        return [n for n, d in self.nodes(data=True) if d.get("role") == "router"]
