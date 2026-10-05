"""Module containing the traffic routing execution event for the ECLYPSE framework."""

import random
from bisect import bisect_right
from collections import (
    defaultdict,
    deque,
)
from itertools import accumulate

from eclypse.workflow.event import EclypseEvent
from eclypse.workflow.trigger import CascadeTrigger

from .constants import DEFAULT_BANDWIDTH_MBPS
from .network import Network
from .network_application import NetworkApplication


class RoutingEvent(EclypseEvent):
    """Worker event that executes routing logic and updates application state.

    This event is called exactly once per step by the simulation engine. It is
    responsible for taking generated packets, resolving their source and
    destination placements, performing the heavy routing calculations.
    """

    def __init__(self, step_duration_s=0.001):
        """Initialize the traffic routing execution event.

        Args:
            step_duration_s (float): The duration of a single simulation step
                in seconds. Defaults to 0.001.
        """
        self.step_duration_s = step_duration_s
        super().__init__(
            name="traffic_routing_execution",
            event_type="application",
            triggers=[CascadeTrigger("step")],
        )

    def _inject_generated_packets(
        self, app: NetworkApplication, placement, infra: Network
    ) -> None:
        """Inject new packets into the network infrastructure.

        Reads the generated packets from the application layer, resolves their physical
        source and destination nodes using the placement strategy, and places them into
        the local injection buffer of their source node.

        Args:
            app (NetworkApplication): The application layer containing generated \
            packets.
            placement: The placement strategy to resolve service to node mapping.
            infra (Network): The network infrastructure containing router buffers.
        """
        node_attrs = infra._node  # pylint: disable=protected-access
        # service -> node, or None if unmapped
        placement_cache: dict[str, str | None] = {}
        # service -> injection buffer of its node, or None if it cannot send
        source_cache: dict[str, list | None] = {}

        def resolve(service: str) -> str | None:
            try:
                node = placement.service_placement(service_id=service)
            except KeyError as e:
                infra.logger.debug(f"Packet dropped locally: Unmapped service {e}")
                node = None
            placement_cache[service] = node
            return node

        for packet in app.generated_packets:
            src_service = packet.src
            if src_service in source_cache:
                buffer = source_cache[src_service]
                src_node = placement_cache[src_service]
            else:
                src_node = (
                    placement_cache[src_service]
                    if src_service in placement_cache
                    else resolve(src_service)
                )
                buffer = None
                if src_node is not None:
                    attrs = node_attrs[src_node]
                    if attrs.get("role", "host") == "router":
                        infra.logger.warning(
                            f"Packet dropped: the service '{src_service}' is located "
                            f"on node '{src_node}' which is configured as a Router. "
                            f"Only Host nodes can generate traffic!"
                        )
                    else:
                        buffer = attrs["local_injections"]
                source_cache[src_service] = buffer

            if buffer is None:
                continue

            dst_service = packet.dst
            dst_node = (
                placement_cache[dst_service]
                if dst_service in placement_cache
                else resolve(dst_service)
            )
            if dst_node is None:
                continue

            packet.src = src_node
            packet.dst = dst_node
            packet.current_node = src_node
            packet.previous_node = None
            buffer.append(packet)

        app.generated_packets.clear()

    def _prepare_incoming_queues(
        self, router: str, buffer: list, infra: Network
    ) -> tuple[dict, dict]:
        """Separate incoming packets by source and determine link bandwidths.

        Args:
            router (str): The identifier of the router being processed.
            buffer (list): The list of packets currently in the router's buffer.
            infra (Network): The network infrastructure to query for link data.

        Returns:
            tuple[dict, dict]: A tuple containing two dictionaries:
                - The first maps the previous node ID to a deque of its packets.
                - The second maps the previous node ID to its link bandwidth in Mbps.
        """
        incoming_queues: dict[str, deque] = defaultdict(deque)
        for pkt in buffer:
            incoming_queues[pkt.previous_node].append(pkt)

        adj = infra._adj  # pylint: disable=protected-access
        bws: dict[str, float] = {}
        for prev_node in incoming_queues:
            edge_data = adj.get(prev_node, {}).get(router, {})
            bws[prev_node] = edge_data.get("bandwidth_mbps", DEFAULT_BANDWIDTH_MBPS)

        return incoming_queues, bws

    def _build_probabilistic_queue(
        self, incoming_queues: dict[str, deque], bandwidths: dict[str, float]
    ) -> list:
        """Merge input queues probabilistically based on bandwidth.

        Implement a probabilistic multiplexer: it extracts packets from the active
        input queues by tossing a 'weighted coin' based on the bandwidth values,
        simulating hardware fair-queuing based on link capacity.

        The draw is exactly what ``random.choices(sources, weights, k=1)`` does
        (cumulative weights + bisect on ``random() * total``), so the random stream
        and the results are identical, but the cumulative weights are rebuilt only
        when a queue empties instead of at every packet.

        Args:
            incoming_queues (dict[str, deque]): Packets grouped by their source.
            bandwidths (dict[str, float]): Bandwidth of each source link.

        Returns:
            list: A single, ordered list of interleaved packets ready for processing.
        """
        rnd = random.random
        active_sources = [src for src, q in incoming_queues.items() if q]

        if len(active_sources) == 1:
            # Only one input: the order is fixed. One draw per packet is still
            # consumed to keep the random stream identical to random.choices.
            queue = incoming_queues[active_sources[0]]
            for _ in range(len(queue)):
                rnd()
            return list(queue)

        merged_queue = []
        active_weights = [bandwidths[src] for src in active_sources]
        queues = [incoming_queues[src] for src in active_sources]

        while queues:
            cum_weights = list(accumulate(active_weights))
            total = cum_weights[-1] + 0.0
            hi = len(cum_weights) - 1
            # Draw until one queue becomes empty, then rebuild the weights
            while True:
                idx = bisect_right(cum_weights, rnd() * total, 0, hi)
                queue = queues[idx]
                merged_queue.append(queue.popleft())
                if not queue:
                    queues.pop(idx)
                    active_weights.pop(idx)
                    break

        return merged_queue

    def _forward_shuffled_packets(
        self,
        shuffled_packets: list,
        current_time_s: float,
        infra: Network,
        next_step_buffers: dict,
    ) -> None:
        """Route packets sequentially and distribute them to their next destination.

        Args:
            shuffled_packets (list): The probabilistically ordered list of packets.
            current_time_s (float): The current simulation time in seconds.
            infra (Network): The network infrastructure performing the routing.
            next_step_buffers (dict): A dictionary to hold packets bound for other \
                routers.
        """
        forward = infra.forward_one_hop
        for pkt in shuffled_packets:
            next_node = forward(pkt, current_time_s)
            if next_node is not None and next_node != pkt.dst:
                next_step_buffers[next_node].append(pkt)

    def __call__(self, app: NetworkApplication, placement, infra: Network, **_kwargs):
        """Execute the routing logic for packets generated in the current step.

        Args:
            app (NetworkApplication): The application layer with generated \
            packets and step info.
            placement: The placement strategy manager.
            infra (Network): The network infrastructure.
            **kwargs: Additional keyword arguments.
        """
        infra.clear_step_telemetry()
        current_time_s = app.current_step * self.step_duration_s
        node_attrs = infra._node  # pylint: disable=protected-access

        next_step_buffers: dict[str, list] = defaultdict(list)

        # Put the generated packets into the injection buffers of the hosts
        self._inject_generated_packets(app, placement, infra)

        # Elaboration of the hosts
        for host in infra.hosts:
            local_buffer = node_attrs[host]["local_injections"]
            if not local_buffer:
                continue
            self._forward_shuffled_packets(
                local_buffer, current_time_s, infra, next_step_buffers
            )
            local_buffer.clear()

        # Elaboration of the routers
        logger = infra.logger.opt(lazy=True)
        for router in infra.routers:
            buffer = node_attrs[router]["router_buffer"]
            if not buffer:
                continue

            incoming_queues, bws = self._prepare_incoming_queues(router, buffer, infra)
            shuffled_packets = self._build_probabilistic_queue(incoming_queues, bws)

            if len(shuffled_packets) > 1:
                # Lazy: the string is built only if the INFO level is enabled
                logger.info(  # noqa: PLE1205 (loguru {} formatting)
                    "{} order: [{}]",
                    lambda router=router: router,
                    lambda pkts=shuffled_packets: ", ".join(
                        f"{p.id} ({p.previous_node})" for p in pkts
                    ),
                )

            self._forward_shuffled_packets(
                shuffled_packets, current_time_s, infra, next_step_buffers
            )
            buffer.clear()

        # Move the packets in transit to the next router buffers for the next step
        for node, pkts in next_step_buffers.items():
            node_attrs[node]["router_buffer"].extend(pkts)

        # Update the link latencies based on the telemetry of the current step
        infra.update_link_latencies()
