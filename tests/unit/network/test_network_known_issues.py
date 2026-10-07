"""Expected physical behavior that the network extension does not model yet.

Each test describes the correct behavior and is marked ``xfail(strict=True)``:
it is expected to fail with the current model. When the model is fixed the test
passes, pytest reports it as an error (XPASS strict), and the marker must be
removed so that the test guards the fix.
"""

from __future__ import annotations

import pytest

from eclypse.network import RoutingEvent
from tests.unit.network._helpers import (
    PROC_MS,
    PROP_MS,
    TX_MS,
    build_line,
    make_batch,
)


@pytest.mark.xfail(
    strict=True,
    reason="LinkState.serve discards the capacity left after the last whole packet",
)
def test_partial_transmissions_carry_over_between_services():
    net = build_line()
    net.clear_step_telemetry()
    net.forward_batch(make_batch(net, "A", "B", 2), 1.0)
    link = net._links["A", "R"]

    # Two 1 ms services: together they transmit exactly the two packets,
    # but each one alone covers 1.5 packets at most
    link.serve(1.0015, 8e6)
    link.serve(1.002, 8e6)

    assert link.queue_length == 0


@pytest.mark.xfail(
    strict=True,
    reason="Dropped packets are averaged into the link latency as 0 ms samples",
)
def test_dropped_packets_do_not_lower_the_link_latency():
    net = build_line(max_queue_size=2)
    net.clear_step_telemetry()

    net.forward_batch(make_batch(net, "A", "B", 4), 1.0)
    net.update_link_latencies()

    # Average of the two accepted packets only: 2.5 ms and 3.5 ms
    accepted = PROC_MS + TX_MS + PROP_MS
    assert net.edges["A", "R"]["latency"] == pytest.approx(accepted + TX_MS / 2)


@pytest.mark.xfail(
    strict=True,
    reason="Packets advance one hop per step regardless of their arrival time",
)
def test_a_packet_is_not_forwarded_before_it_arrives(line_network):
    routing = RoutingEvent(step_duration_s=0.001)
    line_network.clear_step_telemetry()
    # 50 packets on a link that needs 1 ms per packet: the last one reaches R
    # after about 52 ms
    line_network.forward_batch(make_batch(line_network, "A", "B", 50), 0.001)
    last_arrival_ms = line_network.step_columns["arrival_at_next"][-1]
    assert last_arrival_ms > 50.0

    class _App:
        current_step = 2
        generated = None

    routing(_App(), None, line_network)  # step 2: t = 2 ms

    assert 50 not in line_network.step_columns["packet_id"]
