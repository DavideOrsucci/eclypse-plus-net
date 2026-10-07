"""Unit tests for packet forwarding in Network."""

from __future__ import annotations

import numpy as np
import pytest

from eclypse.network import PacketBatch
from eclypse.network.network import TELEMETRY_COLUMNS
from tests.unit.network._helpers import (
    PROC_MS,
    PROP_MS,
    TX_MS,
    build_line,
    make_batch,
)

T0 = 1.0  # forwarding time, in seconds
T0_MS = 1000.0


def _forward(net, batch, t=T0):
    net.clear_step_telemetry()
    delivered = net.forward_batch(batch, t)
    return delivered, net.step_columns


# ---------------------------------------------------------------- delays


def test_single_packet_on_an_empty_link_has_no_queuing_delay(line_network):
    _, cols = _forward(line_network, make_batch(line_network, "A", "B", 1))

    assert cols["hop"] == ["A->R"]
    assert cols["processing_ms"] == [pytest.approx(PROC_MS)]
    assert cols["queue_ms"] == [0.0]
    assert cols["transmission_ms"] == [pytest.approx(TX_MS)]
    assert cols["propagation_ms"] == [pytest.approx(PROP_MS)]
    assert cols["queue_length"] == [0.0]
    assert cols["arrival_at_next"] == [pytest.approx(T0_MS + PROC_MS + TX_MS + PROP_MS)]
    assert cols["dropped"] == [False]


def test_each_packet_waits_for_the_bytes_queued_before_it(line_network):
    _, cols = _forward(line_network, make_batch(line_network, "A", "B", 4))

    assert cols["queue_length"] == [0.0, 1.0, 2.0, 3.0]
    assert cols["queue_ms"] == pytest.approx([0.0, TX_MS, 2 * TX_MS, 3 * TX_MS])
    base = T0_MS + PROC_MS + TX_MS + PROP_MS
    assert cols["arrival_at_next"] == pytest.approx(
        [base, base + TX_MS, base + 2 * TX_MS, base + 3 * TX_MS]
    )


def test_transmission_delay_scales_with_packet_size(line_network):
    batch = PacketBatch.concat(
        [
            make_batch(line_network, "A", "B", 1, size=500, first_id=1),
            make_batch(line_network, "A", "B", 1, size=2000, first_id=2),
        ]
    )

    _, cols = _forward(line_network, batch)

    assert cols["transmission_ms"] == pytest.approx([0.5 * TX_MS, 2 * TX_MS])
    assert cols["queue_ms"] == pytest.approx([0.0, 0.5 * TX_MS])


def test_backlog_from_a_previous_step_delays_new_packets(line_network):
    _forward(line_network, make_batch(line_network, "A", "B", 3), t=T0)

    # 1.5 ms later one packet has been transmitted, two are still queued
    _, cols = _forward(
        line_network, make_batch(line_network, "A", "B", 1, first_id=9), t=T0 + 0.0015
    )

    assert cols["queue_length"] == [2.0]
    assert cols["queue_ms"] == pytest.approx([2 * TX_MS])


def test_queue_is_empty_after_enough_time(line_network):
    _forward(line_network, make_batch(line_network, "A", "B", 3), t=T0)

    _, cols = _forward(
        line_network, make_batch(line_network, "A", "B", 1, first_id=9), t=T0 + 1.0
    )

    assert cols["queue_length"] == [0.0]
    assert cols["queue_ms"] == [0.0]


def test_links_of_different_nodes_have_independent_queues(line_network):
    batch = PacketBatch.concat(
        [
            make_batch(line_network, "A", "B", 2, first_id=1),
            make_batch(line_network, "B", "A", 2, first_id=3),
        ]
    )

    _, cols = _forward(line_network, batch)

    assert cols["hop"] == ["A->R", "A->R", "B->R", "B->R"]
    assert cols["queue_length"] == [0.0, 1.0, 0.0, 1.0]


# ---------------------------------------------------------------- DropTail


def test_droptail_drops_the_packets_that_find_the_queue_full():
    net = build_line(max_queue_size=2)

    _, cols = _forward(net, make_batch(net, "A", "B", 4))

    assert cols["dropped"] == [False, False, True, True]
    assert cols["queue_length"] == [0.0, 1.0, 2.0, 2.0]
    assert net.dropped_packets == 2
    assert net._links["A", "R"].queue_length == 2


def test_dropped_packets_have_zero_delays_and_do_not_advance():
    net = build_line(max_queue_size=1)

    _, cols = _forward(net, make_batch(net, "A", "B", 2))

    assert cols["dropped"] == [False, True]
    for name in ("processing_ms", "queue_ms", "transmission_ms", "propagation_ms"):
        assert cols[name][1] == 0.0
    assert cols["arrival_at_next"][1] == pytest.approx(T0_MS)
    assert cols["hop_count"] == [1, 0]
    assert len(net.in_transit) == 1


def test_dropped_packets_accumulate_across_steps():
    net = build_line(max_queue_size=1)

    _forward(net, make_batch(net, "A", "B", 3), t=T0)
    _forward(net, make_batch(net, "A", "B", 3, first_id=10), t=T0 + 1e-6)

    assert net.dropped_packets == 2 + 3


# ---------------------------------------------------------------- packet state


def test_accepted_packets_advance_and_wait_in_transit(line_network):
    delivered, cols = _forward(line_network, make_batch(line_network, "A", "B", 2))

    assert len(delivered) == 0
    transit = line_network.in_transit
    assert transit.id.tolist() == [1, 2]
    assert transit.cur.tolist() == [line_network.node_index("R")] * 2
    assert transit.prev.tolist() == [line_network.node_index("A")] * 2
    assert transit.hop.tolist() == [1, 1]
    assert cols["hop_count"] == [1, 1]


def test_packets_reaching_their_destination_are_returned(line_network):
    batch = make_batch(line_network, "A", "B", 2, cur="R", prev="A", hop=1)

    delivered, cols = _forward(line_network, batch)

    assert delivered.id.tolist() == [1, 2]
    assert delivered.cur.tolist() == [line_network.node_index("B")] * 2
    assert delivered.hop.tolist() == [2, 2]
    assert cols["hop"] == ["R->B", "R->B"]
    assert len(line_network.in_transit) == 0


def test_telemetry_has_one_row_per_packet_in_processing_order(line_network):
    batch = PacketBatch.concat(
        [
            make_batch(line_network, "B", "A", 1, first_id=7),
            make_batch(line_network, "A", "B", 1, first_id=3),
            make_batch(line_network, "B", "A", 1, first_id=5),
        ]
    )

    _, cols = _forward(line_network, batch)

    assert set(cols) == set(TELEMETRY_COLUMNS)
    assert cols["packet_id"] == [7, 3, 5]
    assert cols["hop"] == ["B->R", "A->R", "B->R"]
    assert all(len(values) == 3 for values in cols.values())


def test_telemetry_values_are_python_scalars(line_network):
    _, cols = _forward(line_network, make_batch(line_network, "A", "B", 1))

    assert type(cols["packet_id"][0]) is int
    assert type(cols["queue_ms"][0]) is float
    assert type(cols["dropped"][0]) is bool


def test_forwarding_an_empty_batch_does_nothing(line_network):
    delivered, cols = _forward(line_network, PacketBatch.empty())

    assert len(delivered) == 0
    assert all(values == [] for values in cols.values())


# ---------------------------------------------------------------- failures


def test_packets_without_a_route_are_discarded_with_a_warning(line_network, net_logger):
    line_network.add_host("C")

    delivered, cols = _forward(line_network, make_batch(line_network, "A", "C", 2))

    assert len(delivered) == 0
    assert cols["packet_id"] == []
    assert len(line_network.in_transit) == 0
    assert line_network.dropped_packets == 0
    warnings = net_logger.messages("warning")
    assert len(warnings) == 2
    assert all("No routes for C" in message for message in warnings)


def test_packets_on_a_removed_node_are_discarded(line_network):
    batch = make_batch(line_network, "A", "B", 2, cur="R", prev="A", hop=1)

    line_network.remove_node("R")
    delivered, cols = _forward(line_network, batch)

    assert len(delivered) == 0
    assert cols["packet_id"] == []


# ---------------------------------------------------------------- router batches


def test_take_router_batches_groups_packets_by_router_in_router_order():
    net = build_line()
    net.add_router("R0")  # added after R: comes second in net.routers
    net.add_edge("A", "R0", symmetric=True)
    net.build_routing_tables()
    net.in_transit = PacketBatch.concat(
        [
            make_batch(net, "A", "B", 1, cur="R0", prev="A", first_id=1),
            make_batch(net, "A", "B", 1, cur="R", prev="A", first_id=2),
            make_batch(net, "A", "B", 1, cur="R0", prev="A", first_id=3),
            make_batch(net, "A", "B", 1, cur="R", prev="A", first_id=4),
        ]
    )

    groups = net.take_router_batches()

    assert [router for router, _ in groups] == ["R", "R0"]
    assert [batch.id.tolist() for _, batch in groups] == [[2, 4], [1, 3]]
    assert len(net.in_transit) == 0


def test_take_router_batches_discards_packets_stranded_on_hosts(line_network):
    line_network.in_transit = PacketBatch.concat(
        [
            make_batch(line_network, "A", "B", 1, cur="R", prev="A", first_id=1),
            make_batch(line_network, "B", "A", 2, cur="B", prev="R", first_id=2),
        ]
    )

    groups = line_network.take_router_batches()

    assert [(router, batch.id.tolist()) for router, batch in groups] == [("R", [1])]
    assert line_network.stranded_packets == 2


def test_take_router_batches_without_packets(line_network):
    assert line_network.take_router_batches() == []


def test_two_hops_deliver_the_packets(line_network):
    _forward(line_network, make_batch(line_network, "A", "B", 3), t=T0)

    [(router, batch)] = line_network.take_router_batches()
    delivered, cols = _forward(line_network, batch, t=T0 + 0.01)

    assert router == "R"
    assert delivered.id.tolist() == [1, 2, 3]
    assert np.all(delivered.hop == 2)
    assert cols["hop"] == ["R->B"] * 3
    assert len(line_network.in_transit) == 0
