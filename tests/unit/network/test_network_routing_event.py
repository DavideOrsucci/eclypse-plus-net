"""Unit tests for RoutingEvent and RoutingMetric."""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import pytest

from eclypse.network import (
    Network,
    NetworkApplication,
    PacketBatch,
    PacketGenerationEvent,
    RoutingEvent,
    RoutingMetric,
)
from eclypse.network.network import TELEMETRY_COLUMNS
from tests.unit.network._helpers import (
    PROC_MS,
    PROP_MS,
    TX_MS,
    StubPlacement,
)


def _run_step(app, placement, infra, routing=None):
    PacketGenerationEvent()(app, placement, infra)
    (routing or RoutingEvent(step_duration_s=0.001))(app, placement, infra)
    return infra.step_columns


# ---------------------------------------------------------------- end to end


@pytest.mark.usefixtures("seeded")
def test_first_step_injects_the_packets_on_the_source_host(line_network, line_app):
    app, placement = line_app

    cols = _run_step(app, placement, line_network)

    n = len(cols["packet_id"])
    assert n > 0
    assert cols["hop"] == ["A->R"] * n
    assert len(line_network.in_transit) == n
    assert app.generated is None


@pytest.mark.usefixtures("seeded")
def test_packets_advance_one_hop_per_step_and_are_delivered(line_network, line_app):
    app, placement = line_app
    routing = RoutingEvent(step_duration_s=0.001)

    first = list(_run_step(app, placement, line_network, routing)["packet_id"])
    cols = _run_step(app, placement, line_network, routing)

    forwarded_by_router = [
        pid
        for pid, hop in zip(cols["packet_id"], cols["hop"], strict=True)
        if hop == "R->B"
    ]
    assert sorted(forwarded_by_router) == sorted(first)
    assert not set(first) & set(line_network.in_transit.id.tolist())


@pytest.mark.usefixtures("seeded")
def test_forwarding_time_is_the_step_number_times_the_step_duration(
    line_network, line_app
):
    app, placement = line_app
    routing = RoutingEvent(step_duration_s=0.25)

    cols = _run_step(app, placement, line_network, routing)

    first_arrival = cols["arrival_at_next"][0]
    assert first_arrival == pytest.approx(250.0 + PROC_MS + TX_MS + PROP_MS)


@pytest.mark.usefixtures("seeded")
def test_routing_event_updates_the_link_latencies(line_network, line_app):
    app, placement = line_app

    _run_step(app, placement, line_network)

    assert all("latency" in data for _, _, data in line_network.edges(data=True))


@pytest.mark.usefixtures("seeded")
def test_packets_of_unplaced_services_are_discarded(line_network):
    app = NetworkApplication("app")
    for service in ("sa", "ghost"):
        app.add_node(service, cpu=1, ram=1)
    app.add_edge("sa", "ghost", avg_packets_per_step=20.0)

    cols = _run_step(app, StubPlacement({"sa": "A"}), line_network)

    assert cols["packet_id"] == []
    assert len(line_network.in_transit) == 0


@pytest.mark.usefixtures("seeded")
def test_packets_generated_on_a_router_are_discarded_with_a_warning(
    line_network, line_app, net_logger
):
    app, _ = line_app
    placement = StubPlacement({"sa": "R", "sb": "B"})

    cols = _run_step(app, placement, line_network)

    assert cols["packet_id"] == []
    assert any("Router" in message for message in net_logger.messages("warning"))


@pytest.mark.usefixtures("seeded")
def test_injected_packets_are_ordered_by_source_host():
    infra = Network("n")
    for host in ("H1", "H2"):
        infra.add_host(host)
    infra.add_router("R")
    infra.add_host("H3")
    for host in ("H1", "H2", "H3"):
        infra.add_edge(host, "R", symmetric=True)
    app = NetworkApplication("app")
    for service in ("s1", "s2", "s3"):
        app.add_node(service, cpu=1, ram=1)
    app.add_edge("s2", "s3", avg_packets_per_step=10.0)  # flow from H2 first
    app.add_edge("s1", "s3", avg_packets_per_step=10.0)
    placement = StubPlacement({"s1": "H1", "s2": "H2", "s3": "H3"})

    cols = _run_step(app, placement, infra)

    hops = cols["hop"]
    first_h2 = hops.index("H2->R")
    assert set(hops[:first_h2]) == {"H1->R"}
    assert set(hops[first_h2:]) == {"H2->R"}


# ---------------------------------------------------------------- multiplexer


def _mux_network(bandwidths: dict[str, float]) -> Network:
    infra = Network("mux")
    infra.add_router("R")
    infra.add_host("OUT")
    infra.add_edge("R", "OUT")
    for host, bw in bandwidths.items():
        infra.add_host(host)
        infra.add_edge(host, "R", bandwidth_mbps=bw)
    infra.build_routing_tables()
    return infra


def _buffer(infra: Network, prev_hosts: list[str]) -> PacketBatch:
    n = len(prev_hosts)
    out = infra.node_index("OUT")
    return PacketBatch(
        id=np.arange(1, n + 1, dtype=np.int64),
        src=np.array([infra.node_index(h) for h in prev_hosts], dtype=np.int64),
        dst=np.full(n, out, dtype=np.int64),
        size=np.full(n, 100, dtype=np.int64),
        step=np.zeros(n, dtype=np.int64),
        cur=np.full(n, infra.node_index("R"), dtype=np.int64),
        prev=np.array([infra.node_index(h) for h in prev_hosts], dtype=np.int64),
        hop=np.ones(n, dtype=np.int64),
    )


def _reference_order(prev_hosts: list[str], bandwidths: dict[str, float]) -> list[int]:
    """Plain implementation: one weighted random.choices draw per packet."""
    queues: dict[str, deque[int]] = {}
    for packet_id, host in enumerate(prev_hosts, start=1):
        queues.setdefault(host, deque()).append(packet_id)
    order = []
    while queues:
        hosts = list(queues)
        chosen = random.choices(hosts, weights=[bandwidths[h] for h in hosts], k=1)[0]
        order.append(queues[chosen].popleft())
        if not queues[chosen]:
            del queues[chosen]
    return order


@pytest.mark.parametrize("seed", range(8))
def test_multiplexer_matches_one_weighted_draw_per_packet(seed):
    rng = random.Random(seed)
    bandwidths = {"H1": 10.0, "H2": 40.0, "H3": 100.0}
    prev_hosts = [rng.choice(list(bandwidths)) for _ in range(rng.randint(2, 60))]
    infra = _mux_network(bandwidths)

    random.seed(seed)
    expected = _reference_order(prev_hosts, bandwidths)
    after_reference = random.random()

    random.seed(seed)
    ordered = RoutingEvent()._build_probabilistic_queue(
        "R", _buffer(infra, prev_hosts), infra
    )

    assert ordered.id.tolist() == expected
    assert random.random() == after_reference  # same number of draws consumed


def test_multiplexer_keeps_the_fifo_order_of_each_input():
    bandwidths = {"H1": 1.0, "H2": 1.0}
    prev_hosts = ["H1", "H2", "H1", "H1", "H2", "H2", "H1"]
    infra = _mux_network(bandwidths)
    random.seed(3)

    ordered = RoutingEvent()._build_probabilistic_queue(
        "R", _buffer(infra, prev_hosts), infra
    )

    ids = ordered.id.tolist()
    assert sorted(ids) == list(range(1, len(prev_hosts) + 1))
    for host in bandwidths:
        own = [i + 1 for i, h in enumerate(prev_hosts) if h == host]
        assert [i for i in ids if i in own] == own


def test_multiplexer_with_a_single_input_keeps_the_arrival_order():
    infra = _mux_network({"H1": 10.0})
    random.seed(5)

    ordered = RoutingEvent()._build_probabilistic_queue(
        "R", _buffer(infra, ["H1"] * 4), infra
    )
    after = random.random()

    assert ordered.id.tolist() == [1, 2, 3, 4]
    random.seed(5)
    for _ in range(4):
        random.random()
    assert random.random() == after


def test_multiplexer_favours_the_faster_input():
    bandwidths = {"H1": 1000.0, "H2": 1.0}
    prev_hosts = ["H2"] * 50 + ["H1"] * 50
    infra = _mux_network(bandwidths)
    random.seed(11)

    ordered = RoutingEvent()._build_probabilistic_queue(
        "R", _buffer(infra, prev_hosts), infra
    )

    first_half = ordered.id.tolist()[:50]
    from_fast_input = sum(packet_id > 50 for packet_id in first_half)
    assert from_fast_input >= 45


# ---------------------------------------------------------------- metric


def test_metric_returns_none_without_telemetry(line_network, line_app):
    app, placement = line_app
    line_network.clear_step_telemetry()

    assert RoutingMetric()(app, placement, line_network) is None


@pytest.mark.usefixtures("seeded")
def test_metric_reports_the_columns_of_the_step(line_network, line_app):
    app, placement = line_app
    _run_step(app, placement, line_network)

    report = RoutingMetric()(app, placement, line_network)

    n = len(line_network.step_columns["packet_id"])
    assert set(report) == {"step", *TELEMETRY_COLUMNS}
    assert report["step"] == [app.current_step] * n
    for name in TELEMETRY_COLUMNS:
        assert report[name] is line_network.step_columns[name]


@pytest.mark.usefixtures("seeded")
def test_metric_reports_of_previous_steps_stay_valid(line_network, line_app):
    app, placement = line_app
    _run_step(app, placement, line_network)
    report = RoutingMetric()(app, placement, line_network)
    snapshot = {name: list(values) for name, values in report.items()}

    _run_step(app, placement, line_network)

    assert report == snapshot
