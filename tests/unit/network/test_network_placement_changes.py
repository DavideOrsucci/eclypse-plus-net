"""Unit tests for the packets generated while the placement changes.

ECLYPSE discards the placement of an application when its requirements stop being
met, and places it again at the next step. In between, the services are not placed:
their packets are discarded and counted in ``Network.unplaced_packets``, so that
every generated packet is accounted for.
"""

from __future__ import annotations

import random

import numpy as np

from eclypse.graph.assets.defaults import (
    cpu,
    ram,
)
from eclypse.network import (
    HostOnlyStrategy,
    Network,
    NetworkApplication,
    PacketGenerationEvent,
    RoutingEvent,
)
from eclypse.placement.strategies import (
    FirstFitStrategy,
    RandomStrategy,
)
from eclypse.simulation import (
    Simulation,
    SimulationConfig,
)

STEPS = 20


def _assets():
    return {"cpu": cpu(init_fn_or_value=0), "ram": ram(init_fn_or_value=0)}


def _network() -> Network:
    """Router R with three hosts of 2 CPUs, the first added before the others."""
    net = Network("star", node_assets=_assets())
    net.add_router("R", processing_time=0.0001)
    for host in ("H0", "H1", "H2"):
        net.add_host(host, cpu=2, ram=4, processing_time=0.0001)
        net.add_edge(host, "R", symmetric=True, bandwidth_mbps=100, length_km=1)
    return net


class _RaiseCpu:
    """Application policy: from step ``at``, service ``service`` needs ``cpu``."""

    def __init__(self, at: int, service: str, value: int):
        self.at, self.service, self.value, self.step = at, service, value, 0

    def __call__(self, graph):
        self.step += 1
        if self.step == self.at:
            graph.nodes[self.service]["cpu"] = self.value


def _application(policies=None) -> NetworkApplication:
    app = NetworkApplication("app", node_assets=_assets(), update_policies=policies)
    for service in ("s0", "s1", "s2"):
        app.add_node(service, cpu=1, ram=1)
    for src, dst in (("s0", "s1"), ("s1", "s2"), ("s2", "s0")):
        app.add_edge(src, dst, packet_size_bytes=1000, avg_packets_per_step=3.0)
    return app


class _Counting(PacketGenerationEvent):
    def __init__(self):
        super().__init__()
        self.generated = 0

    def __call__(self, app, placement, infra, **kwargs):
        out = super().__call__(app, placement, infra, **kwargs)
        self.generated += app.num_generated
        return out


class _Recording(RoutingEvent):
    """Count the delivered packets and record the mapping seen at every step."""

    def __init__(self):
        super().__init__(step_duration_s=0.001)
        self.delivered = 0
        self.mappings: list[dict[str, str]] = []

    def route(self, applications, infra, step=None):
        if "_counted" not in infra.__dict__:
            forward = infra.forward_batch

            def counted(batch, now):
                out = forward(batch, now)
                self.delivered += len(out)
                return out

            infra.forward_batch = counted
            infra._counted = True
        self.mappings.append(dict(applications[0][1].mapping))
        super().route(applications, infra, step)


def _simulate(tmp_path, app, strategy):
    random.seed(7)
    np.random.seed(7)
    net = _network()
    generation, routing = _Counting(), _Recording()
    config = SimulationConfig(
        seed=7,
        max_steps=STEPS,
        path=tmp_path / "sim",
        events=[generation, routing],
        include_default_metrics=False,
        log_level="ERROR",
    )
    sim = Simulation(net, simulation_config=config)
    sim.register(app, placement_strategy=strategy)
    sim.start()
    sim.wait()
    return net, generation, routing


def _accounted(net: Network, routing: _Recording) -> int:
    return (
        routing.delivered
        + net.dropped_packets
        + net.stranded_packets
        + net.unroutable_packets
        + net.lost_packets
        + net.router_source_packets
        + net.unplaced_packets
        + len(net.in_transit)
    )


def test_packets_generated_while_the_placement_is_reset_are_counted(tmp_path):
    # From step 5, s1 needs 2 CPUs: its host no longer fits, ECLYPSE discards
    # the placement and places the application again at the next step
    app = _application(policies=[_RaiseCpu(5, "s1", 2)])

    net, generation, routing = _simulate(tmp_path, app, FirstFitStrategy())

    empty_steps = [i for i, mapping in enumerate(routing.mappings, 1) if not mapping]
    assert empty_steps, "the placement was never reset"
    assert routing.mappings[-1], "the application was not placed again"
    assert net.unplaced_packets > 0
    assert generation.generated == _accounted(net, routing)


def test_packets_of_an_application_that_cannot_be_placed_are_all_counted(tmp_path):
    # From step 5, s1 needs more CPUs than any host has
    app = _application(policies=[_RaiseCpu(5, "s1", 3)])

    net, generation, routing = _simulate(tmp_path, app, FirstFitStrategy())

    assert not routing.mappings[-1]
    assert net.unplaced_packets > 0
    assert generation.generated == _accounted(net, routing)


def test_random_placement_on_hosts_accounts_for_every_packet(tmp_path):
    app = _application()

    net, generation, routing = _simulate(
        tmp_path, app, HostOnlyStrategy(RandomStrategy(seed=7))
    )

    assert net.router_source_packets == 0
    assert all(node != "R" for mapping in routing.mappings for node in mapping.values())
    assert generation.generated == _accounted(net, routing)
