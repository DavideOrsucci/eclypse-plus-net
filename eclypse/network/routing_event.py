"""Simulation event that forwards the network traffic of each step."""

import random
from itertools import accumulate

import numpy as np

from eclypse.workflow.event import EclypseEvent
from eclypse.workflow.trigger import CascadeTrigger

from .constants import DEFAULT_BANDWIDTH_MBPS
from .network import (
    Network,
    PacketBatch,
)
from .network_application import NetworkApplication


class RoutingEvent(EclypseEvent):
    """Event that forwards the packets of the current step by one hop.

    The event, named ``traffic_routing_execution``, runs at every step after the
    :class:`~eclypse.network.PacketGenerationEvent`. It:

    1. injects the packets generated in the step on the hosts where their source
       services are placed;
    2. merges the packets waiting on each router, received on different input
       links, with a probabilistic multiplexer weighted by the link bandwidths;
    3. forwards all the packets by one hop with
       :meth:`~eclypse.network.Network.forward_batch`;
    4. updates the ``latency`` attribute of the links.

    Packets are processed host by host, in :attr:`Network.hosts` order, and then
    router by router, in :attr:`Network.routers` order. The processing order
    determines which packets find a queue full, so it affects delays and drops.

    Attributes:
        step_duration_s (float): The simulated duration of a step, in seconds. The
            current time is the step number multiplied by this duration.
    """

    def __init__(self, step_duration_s=0.001):
        """Initialize the event to run every step.

        Args:
            step_duration_s (float): The simulated duration of a step, in seconds.
                Defaults to 0.001.
        """
        self.step_duration_s = step_duration_s
        super().__init__(
            name="traffic_routing_execution",
            event_type="application",
            triggers=[CascadeTrigger("step")],
        )

    def _inject_generated_packets(
        self, app: NetworkApplication, placement, infra: Network
    ) -> PacketBatch:
        """Inject the packets generated in the step into the network.

        The source and destination nodes of every flow are resolved through the
        placement. Packets are discarded if either service is not placed, or if the
        source service is placed on a router. The remaining packets are sorted by
        source host, in :attr:`Network.hosts` order, keeping the generation order
        within each host. The generated traffic of the application is consumed.

        Args:
            app (NetworkApplication): The application that generated the packets.
            placement (Placement): The placement of the application services.
            infra (Network): The network infrastructure.

        Returns:
            PacketBatch: The injected packets, in processing order.
        """
        generated = app.generated
        app.generated = None
        if generated is None or not len(generated):
            return PacketBatch.empty()

        flows = generated.flows
        node_ids = infra._node_ids  # pylint: disable=protected-access
        host_rank = infra._host_rank  # pylint: disable=protected-access
        router_rank = infra._router_rank  # pylint: disable=protected-access

        placement_cache: dict[str, int] = {}

        def resolve(service: str) -> int:
            if service not in placement_cache:
                try:
                    node = placement.service_placement(service_id=service)
                    idx = node_ids.get(node, -1)
                except KeyError as e:
                    infra.logger.debug(f"Packet dropped locally: Unmapped service {e}")
                    idx = -1
                placement_cache[service] = idx
            return placement_cache[service]

        n_flows = len(flows)
        flow_src = np.full(n_flows, -1, dtype=np.int64)
        flow_dst = np.full(n_flows, -1, dtype=np.int64)
        per_flow = np.bincount(generated.flow, minlength=n_flows)
        for f in np.flatnonzero(per_flow).tolist():
            src_service, dst_service, _ = flows[f]
            src, dst = resolve(src_service), resolve(dst_service)
            if src < 0 or dst < 0:
                continue
            if router_rank[src] >= 0:
                infra.logger.warning(
                    f"{per_flow[f]} packets dropped: the service '{src_service}' is "
                    f"located on node '{infra.node_name(src)}' which is configured as "
                    f"a Router. Only Host nodes can generate traffic!"
                )
                continue
            if host_rank[src] < 0:
                continue
            flow_src[f] = src
            flow_dst[f] = dst

        flow = generated.flow
        src = flow_src[flow]
        keep = (src >= 0) & (flow_dst[flow] >= 0)
        flow, src, ids = flow[keep], src[keep], generated.id[keep]
        order = np.argsort(host_rank[src], kind="stable")
        flow, src, ids = flow[order], src[order], ids[order]

        sizes = np.array([f[2] for f in flows], dtype=np.int64)
        n = len(ids)
        return PacketBatch(
            id=ids,
            src=src,
            dst=flow_dst[flow],
            size=sizes[flow],
            step=np.full(n, generated.step, dtype=np.int64),
            cur=src,
            prev=np.full(n, -1, dtype=np.int64),
            hop=np.zeros(n, dtype=np.int64),
        )

    def _input_bandwidths(
        self, router: str, prev_nodes: list[int], infra: Network
    ) -> list[float]:
        """Read the bandwidth of the input links of a router.

        Args:
            router (str): The router.
            prev_nodes (list[int]): The indices of the nodes upstream of the router.
            infra (Network): The network infrastructure.

        Returns:
            list[float]: The bandwidth, in Mbps, of the link from each node of
                ``prev_nodes`` to ``router``.
        """
        adj = infra._adj  # pylint: disable=protected-access
        return [
            adj.get(infra.node_name(p), {})
            .get(router, {})
            .get("bandwidth_mbps", DEFAULT_BANDWIDTH_MBPS)
            for p in prev_nodes
        ]

    def _build_probabilistic_queue(
        self, router: str, packets: PacketBatch, infra: Network
    ) -> PacketBatch:
        """Merge the input queues of a router with a probabilistic multiplexer.

        The packets of the router form one FIFO input queue per upstream node. The
        multiplexer repeatedly selects a non-empty input queue at random, with
        probability proportional to the bandwidth of its input link, and takes its
        first packet. This models a fair-queuing scheduler that serves each input in
        proportion to its capacity.

        Each selection draws one number from Python's global ``random`` generator,
        with the same semantics as ``random.choices(queues, weights, k=1)``.
        Selections are resolved in rounds of ``d`` draws, where ``d`` is the
        smallest number of packets left in an input queue: no queue can empty before
        the last draw of a round, so the weights are constant within it and the
        round is resolved with a single vectorized search.

        Args:
            router (str): The router whose packets are merged.
            packets (PacketBatch): The packets waiting on the router, in arrival
                order.
            infra (Network): The network infrastructure.

        Returns:
            PacketBatch: The packets of the router, in processing order.
        """
        rnd = random.random
        n = len(packets)
        uniq, first, inverse = np.unique(
            packets.prev, return_index=True, return_inverse=True
        )
        k = len(uniq)
        if k == 1:
            # A single input queue: the order is fixed, one draw per packet
            for _ in range(n):
                rnd()
            return packets

        # Input queues, in order of first appearance in the buffer
        appearance = np.argsort(first)
        rank = np.empty(k, dtype=np.int64)
        rank[appearance] = np.arange(k)
        source_of = rank[inverse.reshape(-1)]
        weights = self._input_bandwidths(router, uniq[appearance].tolist(), infra)
        remaining = np.bincount(source_of, minlength=k).tolist()

        active = list(range(k))
        sequence = []
        while active:
            if len(active) == 1:
                left = remaining[active[0]]
                for _ in range(left):
                    rnd()
                sequence.append(np.full(left, active[0], dtype=np.int64))
                break
            cum_weights = list(accumulate([weights[j] for j in active]))
            total = cum_weights[-1] + 0.0
            bounds = np.array(cum_weights[:-1])
            active_arr = np.array(active, dtype=np.int64)
            while True:
                d = min(remaining[j] for j in active)
                draws = np.array([rnd() for _ in range(d)])
                picks = np.searchsorted(bounds, draws * total, side="right")
                sequence.append(active_arr[picks])
                got = np.bincount(picks, minlength=len(active)).tolist()
                emptied = -1
                for pos, j in enumerate(active):
                    remaining[j] -= got[pos]
                    if remaining[j] == 0:
                        emptied = pos
                if emptied >= 0:
                    active.pop(emptied)
                    break

        # The i-th selection of queue j takes the i-th packet of queue j
        seq = np.concatenate(sequence)
        order = np.empty(n, dtype=np.int64)
        order[np.argsort(seq, kind="stable")] = np.argsort(source_of, kind="stable")
        return packets.take(order)

    def __call__(self, app: NetworkApplication, placement, infra: Network, **_kwargs):
        """Forward the packets of the current step by one hop.

        Args:
            app (NetworkApplication): The application, with the packets generated in
                the step.
            placement (Placement): The placement of the application services.
            infra (Network): The network infrastructure.
            **_kwargs: Additional arguments provided by the framework (unused).
        """
        infra.clear_step_telemetry()
        infra._ensure_tables()  # pylint: disable=protected-access
        current_time_s = app.current_step * self.step_duration_s

        # Hosts: the packets generated in this step
        batches = [self._inject_generated_packets(app, placement, infra)]

        # Routers: the packets that arrived in the previous step
        logger = infra.logger.opt(lazy=True)
        for router, packets in infra.take_router_batches():
            ordered = self._build_probabilistic_queue(router, packets, infra)
            if len(ordered) > 1:
                # Lazy: the string is built only if the INFO level is enabled
                logger.info(  # noqa: PLE1205 (loguru {} formatting)
                    "{} order: [{}]",
                    lambda router=router: router,
                    lambda p=ordered: ", ".join(
                        f"{i} ({infra.node_name(prev)})"
                        for i, prev in zip(p.id.tolist(), p.prev.tolist(), strict=True)
                    ),
                )
            batches.append(ordered)

        # Forward everything by one hop, in processing order
        infra.forward_batch(PacketBatch.concat(batches), current_time_s)

        # Update the link latencies based on the telemetry of the current step
        infra.update_link_latencies()
