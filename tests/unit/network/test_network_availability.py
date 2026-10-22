"""Unit tests for node availability in Network: nodes that are down carry no traffic."""

from __future__ import annotations

import ast
import csv
import random

from eclypse.network import (
    Network,
    NetworkApplication,
    PacketBatch,
    PacketGenerationEvent,
    RoutingEvent,
    RoutingMetric,
)
from eclypse.network.network import path_algorithm
from eclypse.placement.strategies import StaticStrategy
from eclypse.policies.failure import kill_nodes
from eclypse.simulation import (
    Simulation,
    SimulationConfig,
)
from tests.unit.network._helpers import make_batch


def _diamond() -> Network:
    """A -> R1 -> B (slow) and A -> R2 -> B (fast), in both directions."""
    net = Network("diamond")
    net.add_host("A")
    net.add_host("B")
    net.add_router("R1")
    net.add_router("R2")
    for u, v, bw in (
        ("A", "R1", 10.0),
        ("R1", "B", 10.0),
        ("A", "R2", 100.0),
        ("R2", "B", 100.0),
    ):
        net.add_edge(u, v, symmetric=True, bandwidth_mbps=bw)
    return net


def _first_hop(net: Network, src: str, dst: str) -> str | None:
    """Forward one packet and return the link it traversed, if any."""
    net.clear_step_telemetry()
    net.forward_batch(make_batch(net, src, dst, 1), 1.0)
    hops = net.step_columns["hop"]
    return hops[0] if hops else None


def _eclypse_path(net: Network, src: str, dst: str) -> list[str]:
    """Return the path computed by ECLYPSE, as a list of nodes."""
    return [u for u, _, _ in net.path(src, dst)] + [dst]


# ---------------------------------------------------------------- routing


def test_traffic_avoids_a_router_that_is_down():
    net = _diamond()
    assert _first_hop(net, "A", "B") == "A->R2"

    net.nodes["R2"]["availability"] = 0

    assert _first_hop(net, "A", "B") == "A->R1"
    # Same path as ECLYPSE, which computes it on the available nodes
    assert _eclypse_path(net, "A", "B") == ["A", "R1", "B"]
    assert path_algorithm(net.available, "A", "B") == ["A", "R1", "B"]


def test_traffic_goes_back_to_a_router_that_comes_back_up():
    net = _diamond()
    net.nodes["R2"]["availability"] = 0
    assert _first_hop(net, "A", "B") == "A->R1"

    net.nodes["R2"]["availability"] = 0.99

    assert _first_hop(net, "A", "B") == "A->R2"
    assert _eclypse_path(net, "A", "B") == ["A", "R2", "B"]


def test_packets_are_discarded_when_the_only_router_is_down(line_network, net_logger):
    line_network.nodes["R"]["availability"] = 0

    assert _first_hop(line_network, "A", "B") is None
    assert line_network.dropped_packets == 0
    assert any("No routes for B" in m for m in net_logger.messages("warning"))


def test_packets_towards_a_host_that_is_down_are_discarded(line_network):
    line_network.nodes["B"]["availability"] = 0

    assert _first_hop(line_network, "A", "B") is None


def test_a_host_that_is_down_does_not_send_packets(line_network):
    line_network.nodes["A"]["availability"] = 0

    assert _first_hop(line_network, "A", "B") is None
    assert len(line_network.in_transit) == 0


# ---------------------------------------------------------------- packets


def test_packets_on_a_router_that_goes_down_are_lost(line_network):
    line_network.build_routing_tables()
    line_network.in_transit = make_batch(
        line_network, "A", "B", 3, cur="R", prev="A", hop=1
    )

    line_network.nodes["R"]["availability"] = 0

    assert line_network.take_router_batches() == []
    assert len(line_network.in_transit) == 0


def test_packets_in_transit_survive_if_the_router_is_back_up_in_time(line_network):
    line_network.build_routing_tables()
    line_network.in_transit = make_batch(
        line_network, "A", "B", 3, cur="R", prev="A", hop=1
    )

    line_network.nodes["R"]["availability"] = 0
    line_network.nodes["R"]["availability"] = 1

    groups = line_network.take_router_batches()
    assert [(router, len(batch)) for router, batch in groups] == [("R", 3)]


# ---------------------------------------------------------------- tables


def test_tables_are_not_rebuilt_if_no_node_goes_up_or_down(line_network):
    line_network.build_routing_tables()
    tables = line_network._fwd

    line_network.nodes["R"]["availability"] = 0.5
    line_network.nodes["A"]["availability"] = 0.98
    line_network._ensure_tables()

    assert line_network._fwd is tables


def test_tables_are_rebuilt_once_after_several_changes(line_network, monkeypatch):
    line_network.build_routing_tables()
    builds = []
    build = line_network.build_routing_tables

    def counting_build():
        builds.append(1)
        build()

    monkeypatch.setattr(line_network, "build_routing_tables", counting_build)

    line_network.nodes["R"]["availability"] = 0
    line_network.nodes["A"]["availability"] = 0
    line_network.clear_step_telemetry()
    line_network.forward_batch(PacketBatch.empty(), 1.0)
    line_network.take_router_batches()
    line_network.update_link_latencies()

    assert len(builds) == 1


# ---------------------------------------------------------------- policies


def test_eclypse_node_failure_policies_affect_the_traffic():
    net = _diamond()
    net.rnd = random.Random(1)
    for node in net.nodes:
        net.nodes[node]["availability"] = 1.0  # the policy reads the current value
    assert _first_hop(net, "A", "B") == "A->R2"

    kill_nodes(1.0, node_ids=["R2"])(net)

    assert _first_hop(net, "A", "B") == "A->R1"


class _R2DownBetween:
    """Update policy that turns R2 down at the 4th call and up at the 8th."""

    def __init__(self):
        self.calls = 0

    def __call__(self, graph):
        self.calls += 1
        if self.calls == 4:
            graph.nodes["R2"]["availability"] = 0.0
        elif self.calls == 8:
            graph.nodes["R2"]["availability"] = 1.0


def test_simulation_routes_around_a_router_while_it_is_down(tmp_path):
    net = Network("diamond", update_policies=[_R2DownBetween()])
    for host in ("A", "B"):
        net.add_host(host, cpu=4, ram=4)
    for router in ("R1", "R2"):
        net.add_router(router)
    for u, v, bw in (
        ("A", "R1", 10.0),
        ("R1", "B", 10.0),
        ("A", "R2", 100.0),
        ("R2", "B", 100.0),
    ):
        net.add_edge(u, v, symmetric=True, bandwidth_mbps=bw)
    app = NetworkApplication("app")
    app.add_node("s", cpu=1, ram=1)
    app.add_node("d", cpu=1, ram=1)
    app.add_edge("s", "d", avg_packets_per_step=3.0)

    sim = Simulation(
        net,
        simulation_config=SimulationConfig(
            seed=1,
            max_steps=12,
            path=tmp_path / "sim",
            events=[
                PacketGenerationEvent(),
                RoutingEvent(step_duration_s=0.01),
                RoutingMetric(),
            ],
            include_default_metrics=False,
            log_level="ERROR",
            remote=False,
            report_format="csv",
        ),
    )
    sim.register(app, StaticStrategy({"s": "A", "d": "B"}))
    sim.start()
    sim.wait()

    # Links used at each step (see test_network_multiple_applications for the rows)
    hops: dict[int, set[str]] = {}
    with (tmp_path / "sim" / "csv" / "application.csv").open() as f:
        for row in csv.reader(f):
            if len(row) >= 7 and row[3] == "traffic_routing" and row[5] == "hop":
                values = ast.literal_eval(",".join(row[6:]))
                hops.setdefault(int(row[2]), set()).update(values)

    first_hops = {
        step: {h for h in used if h.startswith("A->")} for step, used in hops.items()
    }
    down = range(5, 9)  # the policies of step k run before its routing
    assert all(first_hops[k] == {"A->R1"} for k in down)
    assert all(first_hops[k] == {"A->R2"} for k in first_hops if k not in down)
    assert not any("R2->B" in hops[k] for k in down)
