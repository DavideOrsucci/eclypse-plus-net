"""Simulate a scenario with ECLYPSE+NET and record the fate of every packet.

The simulation lasts ``traffic_time_s + drain_time_s`` seconds of simulated time: the
application generates packets for the first ``traffic_time_s`` seconds, then stops,
so that the packets still travelling can reach their destination.
"""

from __future__ import annotations

import random
import shutil
import time
from typing import TYPE_CHECKING

import numpy as np
from scenario import (
    build_application,
    build_network,
)

from eclypse.network import (
    PacketGenerationEvent,
    RoutingEvent,
)
from eclypse.placement.strategies import StaticStrategy
from eclypse.simulation import (
    Simulation,
    SimulationConfig,
)

if TYPE_CHECKING:
    from pathlib import Path

    from eclypse.network import (
        Network,
        PacketBatch,
    )


class LimitedGeneration(PacketGenerationEvent):
    """Packet generation that stops after ``traffic_steps`` steps."""

    def __init__(self, traffic_steps: int):
        """Initialize the event.

        Args:
            traffic_steps (int): Number of steps with traffic.
        """
        super().__init__()
        self.traffic_steps = traffic_steps

    def __call__(self, app, placement, infra, **kwargs):
        """Generate the packets of the step, or none after the traffic period."""
        if app.current_step >= self.traffic_steps:
            app.current_step += 1
            app.generated = None
            return {"packets_generated": 0}
        return super().__call__(app, placement, infra, **kwargs)


class RecordingRouting(RoutingEvent):
    """Routing event that records the packets injected into the network.

    Attributes:
        injected (list[tuple]): ``(id, src, dst, size, send_time)`` of every
            injected packet, in injection order.
    """

    def __init__(self, step_duration_s: float):
        """Initialize the event."""
        super().__init__(step_duration_s=step_duration_s)
        self.injected: list[tuple] = []

    def _inject_generated_packets(self, app, placement, infra, current_time):
        """Inject the packets of the step and record them."""
        batch = super()._inject_generated_packets(app, placement, infra, current_time)
        name = infra.node_name
        self.injected.extend(
            (i, name(s), name(d), b, current_time)
            for i, s, d, b in zip(
                batch.id.tolist(),
                batch.src.tolist(),
                batch.dst.tolist(),
                batch.size.tolist(),
                strict=True,
            )
        )
        return batch


def record_forwarding(net: Network) -> dict[str, dict]:
    """Record the packets delivered and dropped by ``net.forward_batch``.

    The method is wrapped on the instance, so the simulation is unchanged.

    Args:
        net (Network): The network.

    Returns:
        dict[str, dict]: Filled during the simulation: ``delivered`` maps the ID of
            every delivered packet to ``(arrival_time_s, hops)``, ``dropped`` maps
            the ID of every dropped packet to the link where it was dropped.
    """
    record: dict[str, dict] = {"delivered": {}, "dropped": {}}
    forward = net.forward_batch

    def forward_and_record(batch: PacketBatch, current_time: float):
        cols = net.step_columns
        first = len(cols["packet_id"])
        delivered = forward(batch, current_time)
        record["delivered"].update(
            zip(
                delivered.id.tolist(),
                zip(delivered.arrival.tolist(), delivered.hop.tolist(), strict=True),
                strict=True,
            )
        )
        for pid, hop, dropped in zip(
            cols["packet_id"][first:],
            cols["hop"][first:],
            cols["dropped"][first:],
            strict=True,
        ):
            if dropped:
                record["dropped"][pid] = hop
        return delivered

    net.forward_batch = forward_and_record  # type: ignore[method-assign]
    return record


def run_eclypse(config: dict, output_dir: Path) -> dict:
    """Simulate a scenario with ECLYPSE+NET.

    Args:
        config (dict): The trial configuration (see ``main.py``).
        output_dir (Path): The folder where ECLYPSE writes its outputs. It is
            deleted first if it exists.

    Returns:
        dict: ``net`` (the network after the simulation), ``injected`` (the injected
            packets, see :class:`RecordingRouting`), ``delivered`` and ``dropped``
            (see :func:`record_forwarding`), ``n_flows``, ``build_time_s``,
            ``sim_time_s`` and ``steps``.
    """
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    dt = config["step_duration_s"]
    traffic_steps = round(config["traffic_time_s"] / dt)
    steps = traffic_steps + round(config["drain_time_s"] / dt)

    start = time.perf_counter()
    net = build_network(config)
    app, mapping = build_application(net, config)
    build_time = time.perf_counter() - start

    routing = RecordingRouting(dt)
    record = record_forwarding(net)
    shutil.rmtree(output_dir, ignore_errors=True)
    sim = Simulation(
        net,
        simulation_config=SimulationConfig(
            seed=config["seed"],
            max_steps=steps,
            path=output_dir,
            events=[LimitedGeneration(traffic_steps), routing],
            include_default_metrics=False,
            log_level="ERROR",
            remote=False,
        ),
    )
    sim.register(app, placement_strategy=StaticStrategy(mapping))
    start = time.perf_counter()
    sim.start()
    sim.wait()
    sim_time = time.perf_counter() - start

    return {
        "net": net,
        "injected": routing.injected,
        "delivered": record["delivered"],
        "dropped": record["dropped"],
        "n_flows": app.number_of_edges(),
        "build_time_s": build_time,
        "sim_time_s": sim_time,
        "steps": steps,
    }
