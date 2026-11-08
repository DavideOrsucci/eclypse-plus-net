"""Simulation event that forwards the network traffic of each step."""

from typing import Any

import numpy as np

from eclypse.workflow.event import EclypseEvent
from eclypse.workflow.trigger import CascadeTrigger

from .multiplexer import probabilistic_merge
from .network import (
    Network,
    PacketBatch,
)
from .network_application import (
    GeneratedTraffic,
    NetworkApplication,
)


class RoutingEvent(EclypseEvent):
    """Event that forwards by one hop the packets that can move in the current step.

    The event, named ``traffic_routing_execution``, runs once per step, after the
    :class:`~eclypse.network.PacketGenerationEvent` of every application. It
    handles the traffic of the whole network, i.e. of all the
    :class:`~eclypse.network.NetworkApplication` instances of the simulation:

    1. injects the packets generated in the step by every application on the hosts
       where their source services are placed;
    2. takes the packets that have reached a router by the current time, and
       merges the packets of each router, received on different input links, with
       a probabilistic multiplexer weighted by the link bandwidths;
    3. forwards these packets by one hop with
       :meth:`~eclypse.network.Network.forward_batch`;
    4. updates the ``latency`` attribute of the links.

    A packet moves by at most one hop per step, and only once the current time has
    reached its arrival time at its node. If the step is shorter than the delay of a
    hop, the packet stays in transit for several steps.

    Packets are processed host by host, in :attr:`Network.hosts` order, and then
    router by router, in :attr:`Network.routers` order. The packets injected by a
    host are in order of application registration and then of generation. The
    processing order determines which packets find a queue full, so it affects
    delays and drops.

    Attributes:
        step_duration_s (float): The simulated duration of a step, in seconds. The
            current time is the step number multiplied by this duration. Packets
            are forwarded at the start of the first step that follows their
            arrival, so a shorter step gives a finer time resolution.
        current_step (int): The number of the current step, advanced at every
            call. It is the clock of the network, shared by all the applications.
    """

    def __init__(self, step_duration_s=0.001):
        """Initialize the event to run every step.

        Args:
            step_duration_s (float): The simulated duration of a step, in seconds.
                Defaults to 0.001.
        """
        self.step_duration_s = step_duration_s
        self.current_step = 0
        super().__init__(
            name="traffic_routing_execution",
            triggers=[CascadeTrigger("step")],
        )

    def _inject_generated_packets(
        self, app: NetworkApplication, placement, infra: Network, current_time: float
    ) -> PacketBatch:
        """Inject the packets generated in the step into the network.

        The source and destination nodes of every flow are resolved through the
        placement. Packets are discarded if either service is not placed on a node
        of the network (counted in :attr:`Network.unplaced_packets`, with one
        warning per application and step), or if the source service is placed on a
        router (counted in :attr:`Network.router_source_packets`). The remaining
        packets are sorted by source host, in :attr:`Network.hosts` order, keeping
        the generation order within each host. The generated traffic of the
        application is consumed, and the application is registered in the network
        (:meth:`Network.application_index`) if needed.

        Args:
            app (NetworkApplication): The application that generated the packets.
            placement (Placement): The placement of the application services.
            infra (Network): The network infrastructure.
            current_time (float): Current simulation time, in seconds, used as the
                arrival time of the packets on their source host.

        Returns:
            PacketBatch: The injected packets, in processing order.
        """
        app_index = infra.application_index(app.id)
        generated = app.generated
        app.generated = None
        if generated is None or not len(generated):
            return PacketBatch.empty()

        flows = generated.flows
        host_rank = infra.host_ranks
        flow_src, flow_dst = self._resolve_flows(app, generated, placement, infra)

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
            app=np.full(n, app_index, dtype=np.int64),
            cur=src,
            prev=np.full(n, -1, dtype=np.int64),
            hop=np.zeros(n, dtype=np.int64),
            arrival=np.full(n, current_time, dtype=np.float64),
        )

    def _resolve_flows(
        self,
        app: NetworkApplication,
        generated: GeneratedTraffic,
        placement,
        infra: Network,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resolve the source and destination node of every flow of an application.

        Only the flows that generated packets in the step are resolved. A flow is
        discarded (source and destination left at -1) if either service is not
        placed on a node of the network, or if its source service is placed on a
        router; its packets are counted in :attr:`Network.unplaced_packets` or
        :attr:`Network.router_source_packets`.

        Args:
            app (NetworkApplication): The application.
            generated (GeneratedTraffic): The packets it generated in the step.
            placement (Placement): The placement of the application services.
            infra (Network): The network infrastructure.

        Returns:
            tuple[np.ndarray, np.ndarray]: The index of the source and destination
                node of every flow, or -1 for the discarded flows.
        """
        flows = generated.flows
        host_rank = infra.host_ranks
        router_rank = infra.router_ranks

        placement_cache: dict[str, int] = {}

        def resolve(service: str) -> int:
            if service not in placement_cache:
                try:
                    node = placement.service_placement(service_id=service)
                    idx = infra.node_index(node, register=False)
                except KeyError:
                    idx = -1
                placement_cache[service] = idx
            return placement_cache[service]

        unplaced = 0
        unplaced_services: set[str] = set()

        n_flows = len(flows)
        flow_src = np.full(n_flows, -1, dtype=np.int64)
        flow_dst = np.full(n_flows, -1, dtype=np.int64)
        per_flow = np.bincount(generated.flow, minlength=n_flows)
        for f in np.flatnonzero(per_flow).tolist():
            src_service, dst_service, _ = flows[f]
            src, dst = resolve(src_service), resolve(dst_service)
            if src < 0 or dst < 0:
                unplaced += int(per_flow[f])
                unplaced_services.update(
                    s for s, i in ((src_service, src), (dst_service, dst)) if i < 0
                )
                continue
            if router_rank[src] >= 0:
                infra.router_source_packets += int(per_flow[f])
                infra.logger.warning(
                    f"{per_flow[f]} packets dropped: the service '{src_service}' is "
                    f"located on node '{infra.node_name(src)}' which is configured as "
                    f"a Router. Only Host nodes can generate traffic! Wrap the "
                    f"placement strategy in a HostOnlyStrategy to place services "
                    f"only on hosts."
                )
                continue
            if host_rank[src] < 0:
                # Neither a host nor a router: no node of the network runs it
                unplaced += int(per_flow[f])
                unplaced_services.add(src_service)
                continue
            flow_src[f] = src
            flow_dst[f] = dst

        if unplaced:
            infra.unplaced_packets += unplaced
            infra.logger.warning(
                f"{unplaced} packets of application '{app.id}' dropped: services "
                f"not placed on a node of the network: "
                f"{', '.join(sorted(unplaced_services))}"
            )
        return flow_src, flow_dst

    def _build_probabilistic_queue(
        self, router: str, packets: PacketBatch, infra: Network
    ) -> PacketBatch:
        """Order the packets of a router with :func:`probabilistic_merge`.

        Subclasses can override this method to use a different scheduler.

        Args:
            router (str): The router whose packets are merged.
            packets (PacketBatch): The packets waiting on the router, in arrival
                order.
            infra (Network): The network infrastructure.

        Returns:
            PacketBatch: The packets of the router, in processing order.
        """
        return probabilistic_merge(router, packets, infra)

    def __call__(self, *_args, **_kwargs):
        """Forward by one hop the packets that can move in the current step.

        The traffic of every :class:`NetworkApplication` registered in the
        simulation is forwarded through the infrastructure of the simulation.

        Args:
            *_args: The triggering event, provided by the framework (unused).
            **_kwargs: Additional arguments provided by the framework (unused).
        """
        simulator = self.simulator
        applications = [
            (placement.application, placement)
            for placement in simulator.placements.values()
            if isinstance(placement.application, NetworkApplication)
        ]
        self.route(applications, simulator.infrastructure)

    def route(
        self,
        applications: list[tuple[NetworkApplication, Any]],
        infra: Network,
        step: int | None = None,
    ):
        """Run one step of the network: inject, forward and update the latencies.

        Args:
            applications (list[tuple[NetworkApplication, Placement]]): The
                applications whose packets are injected, with their placements, in
                order of registration.
            infra (Network): The network infrastructure.
            step (int | None): The number of the step. If None, the step after
                :attr:`current_step`. Defaults to None.
        """
        self.current_step = self.current_step + 1 if step is None else step
        infra.clear_step_telemetry()
        infra.ensure_routing_tables()
        current_time_s = self.current_step * self.step_duration_s

        # Hosts: the packets generated in this step by all the applications
        batches = [
            self._order_by_source_host(
                [
                    self._inject_generated_packets(
                        app, placement, infra, current_time_s
                    )
                    for app, placement in applications
                ],
                infra,
            )
        ]

        # Routers: the packets that have arrived by the current time
        logger = infra.logger.opt(lazy=True)
        for router, packets in infra.take_router_batches(current_time_s):
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

        # Forward these packets by one hop, in processing order
        infra.forward_batch(PacketBatch.concat(batches), current_time_s)

        # Update the link latencies based on the telemetry of the current step
        infra.update_link_latencies()

    @staticmethod
    def _order_by_source_host(
        injected: list[PacketBatch], infra: Network
    ) -> PacketBatch:
        """Merge the packets injected by the applications, host by host.

        Args:
            injected (list[PacketBatch]): The packets of each application, each
                sorted by source host.
            infra (Network): The network infrastructure.

        Returns:
            PacketBatch: The packets sorted by source host, in :attr:`Network.hosts`
                order, and then by application and generation order.
        """
        batch = PacketBatch.concat(injected)
        if sum(1 for b in injected if len(b)) > 1:
            batch = batch.take(np.argsort(infra.host_ranks[batch.src], kind="stable"))
        return batch
