"""Unit tests for several NetworkApplication instances sharing one Network."""

from __future__ import annotations

import ast
import csv

import pytest

from eclypse.network import (
    Network,
    NetworkApplication,
    PacketGenerationEvent,
    RoutingEvent,
    RoutingMetric,
)
from eclypse.placement.strategies import StaticStrategy
from eclypse.simulation import (
    Simulation,
    SimulationConfig,
)
from tests.unit.network._helpers import StubPlacement


def _star() -> Network:
    """Hosts H1, H2, H3, H4 attached to the router R."""
    net = Network("star")
    for host in ("H1", "H2", "H3", "H4"):
        net.add_host(host, cpu=4, ram=4)
    net.add_router("R")
    for host in ("H1", "H2", "H3", "H4"):
        net.add_edge(host, "R", symmetric=True, bandwidth_mbps=10.0, length_km=10.0)
    return net


def _app(name: str, src: str, dst: str, rate: float = 5.0):
    """Application with one flow from a service on ``src`` to one on ``dst``."""
    app = NetworkApplication(name)
    app.add_node(f"{name}_s", cpu=1, ram=1)
    app.add_node(f"{name}_d", cpu=1, ram=1)
    app.add_edge(f"{name}_s", f"{name}_d", avg_packets_per_step=rate)
    return app, StubPlacement({f"{name}_s": src, f"{name}_d": dst})


def _step(net, routing, apps):
    for app, placement in apps:
        PacketGenerationEvent()(app, placement, net)
    routing.route(apps, net)


# ---------------------------------------------------------------- network


def test_packet_ids_are_allocated_by_the_network():
    net = _star()

    assert net.allocate_packet_ids(3).tolist() == [1, 2, 3]
    assert net.allocate_packet_ids(0).tolist() == []
    assert net.allocate_packet_ids(2).tolist() == [4, 5]


def test_applications_are_indexed_in_order_of_registration():
    net = _star()

    assert net.application_index("a") == 0
    assert net.application_index("b") == 1
    assert net.application_index("a") == 0
    assert net.application_ids == ["a", "b"]


# ---------------------------------------------------------------- routing


@pytest.mark.usefixtures("seeded")
def test_packet_ids_are_unique_across_applications():
    net = _star()
    apps = [_app("app1", "H1", "H2"), _app("app2", "H3", "H4")]

    _step(net, RoutingEvent(), apps)

    ids = net.step_columns["packet_id"]
    assert len(ids) > 0
    assert len(ids) == len(set(ids))


@pytest.mark.usefixtures("seeded")
def test_one_routing_step_forwards_the_packets_of_every_application():
    net = _star()
    apps = [_app("app1", "H1", "H2"), _app("app2", "H3", "H4")]
    generated = {}

    for app, placement in apps:
        PacketGenerationEvent()(app, placement, net)
        generated[app.id] = app.num_generated
    RoutingEvent().route(apps, net)

    hops = net.step_columns["hop"]
    assert hops.count("H1->R") == generated["app1"]
    assert hops.count("H3->R") == generated["app2"]
    assert net.step_apps == [0] * generated["app1"] + [1] * generated["app2"]


@pytest.mark.usefixtures("seeded")
def test_packets_are_injected_host_by_host_across_applications():
    net = _star()
    # app1 sends from H2, app2 from H1: H1 comes first in Network.hosts
    apps = [_app("app1", "H2", "H3"), _app("app2", "H1", "H3")]

    _step(net, RoutingEvent(), apps)

    hops = net.step_columns["hop"]
    first_h2 = hops.index("H2->R")
    assert set(hops[:first_h2]) == {"H1->R"}
    assert set(hops[first_h2:]) == {"H2->R"}


@pytest.mark.usefixtures("seeded")
def test_the_routing_clock_advances_once_per_step():
    net = _star()
    apps = [_app("app1", "H1", "H2"), _app("app2", "H3", "H4")]
    routing = RoutingEvent(step_duration_s=0.5)

    _step(net, routing, apps)
    _step(net, routing, apps)

    assert routing.current_step == 2
    injected = [
        t
        for t, hop in zip(
            net.step_columns["arrival_at_next"], net.step_columns["hop"], strict=True
        )
        if hop in ("H1->R", "H3->R")
    ]
    # Injected at t = 1 s: they reach R after transmission and propagation
    assert injected
    assert all(1000.0 < t < 1100.0 for t in injected)


# ---------------------------------------------------------------- metric


@pytest.mark.usefixtures("seeded")
def test_metric_reports_only_the_packets_of_its_application():
    net = _star()
    apps = [_app("app1", "H1", "H2"), _app("app2", "H3", "H4")]
    _step(net, RoutingEvent(), apps)

    report1 = RoutingMetric()(apps[0][0], apps[0][1], net)
    report2 = RoutingMetric()(apps[1][0], apps[1][1], net)

    assert set(report1["hop"]) == {"H1->R"}
    assert set(report2["hop"]) == {"H3->R"}
    assert len(report1["packet_id"]) + len(report2["packet_id"]) == len(
        net.step_columns["packet_id"]
    )
    assert len(report1["step"]) == len(report1["hop"])


@pytest.mark.usefixtures("seeded")
def test_metric_returns_none_for_an_application_without_packets_in_the_step():
    net = _star()
    busy = _app("busy", "H1", "H2")
    idle = _app("idle", "H3", "H4", rate=0.0)
    _step(net, RoutingEvent(), [busy, idle])

    assert RoutingMetric()(idle[0], idle[1], net) is None


# ---------------------------------------------------------------- simulation


def test_simulation_with_two_applications_keeps_their_telemetry_apart(tmp_path):
    net = _star()
    routing = RoutingEvent(step_duration_s=0.01)
    sim = Simulation(
        net,
        simulation_config=SimulationConfig(
            seed=1,
            max_steps=5,
            path=tmp_path / "sim",
            events=[PacketGenerationEvent(), routing, RoutingMetric()],
            include_default_metrics=False,
            log_level="ERROR",
            remote=False,
            report_format="csv",
        ),
    )
    for name, (src, dst) in {"app1": ("H1", "H2"), "app2": ("H3", "H4")}.items():
        app, placement = _app(name, src, dst, rate=3.0)
        sim.register(app, StaticStrategy(placement.mapping))
    sim.start()
    sim.wait()

    # Rows: timestamp, event, n_event, callback, application, column, values (the
    # list of values is written unquoted, so it spans the remaining fields)
    hops: dict[str, set[str]] = {"app1": set(), "app2": set()}
    ids: dict[str, list[int]] = {"app1": [], "app2": []}
    with (tmp_path / "sim" / "csv" / "application.csv").open() as f:
        for row in csv.reader(f):
            if len(row) < 7 or row[3] != "traffic_routing":
                continue
            values = ast.literal_eval(",".join(row[6:]))
            if row[5] == "hop":
                hops[row[4]].update(values)
            elif row[5] == "packet_id":
                ids[row[4]].extend(values)

    assert routing.current_step == 5
    assert hops["app1"] and hops["app1"] <= {"H1->R", "R->H2"}
    assert hops["app2"] and hops["app2"] <= {"H3->R", "R->H4"}
    assert not set(ids["app1"]) & set(ids["app2"])
