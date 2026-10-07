"""Unit tests for the link latencies and the per-step telemetry of Network."""

from __future__ import annotations

import pytest

from eclypse.network.network import TELEMETRY_COLUMNS
from tests.unit.network._helpers import (
    PROC_MS,
    PROP_MS,
    TX_MS,
    make_batch,
)

# Idle estimate: processing + propagation + transmission of a 1500-byte packet
IDLE_MS = PROC_MS + PROP_MS + 1.5 * TX_MS


def _step(net, batch=None, t=1.0):
    net.clear_step_telemetry()
    if batch is not None:
        net.forward_batch(batch, t)
    net.update_link_latencies()


def _latency(net, u, v):
    return net.edges[u, v]["latency"]


def test_every_link_gets_the_idle_estimate_at_the_first_update(line_network):
    _step(line_network)

    for u, v in line_network.edges():
        assert _latency(line_network, u, v) == pytest.approx(IDLE_MS)


def test_used_link_gets_the_average_hop_delay(line_network):
    _step(line_network, make_batch(line_network, "A", "B", 3))

    # Hop delays: 2.5, 3.5 and 4.5 ms (0, 1 and 2 packets ahead in the queue)
    assert _latency(line_network, "A", "R") == pytest.approx(3.5)
    assert _latency(line_network, "R", "B") == pytest.approx(IDLE_MS)


def test_link_returns_to_the_idle_estimate_when_unused(line_network):
    _step(line_network, make_batch(line_network, "A", "B", 3), t=1.0)
    assert _latency(line_network, "A", "R") != pytest.approx(IDLE_MS)

    _step(line_network, t=2.0)

    assert _latency(line_network, "A", "R") == pytest.approx(IDLE_MS)


def test_latency_reflects_only_the_current_step(line_network):
    _step(line_network, make_batch(line_network, "A", "B", 3), t=1.0)

    _step(line_network, make_batch(line_network, "A", "B", 1, first_id=10), t=2.0)

    assert _latency(line_network, "A", "R") == pytest.approx(PROC_MS + TX_MS + PROP_MS)


def test_idle_estimate_uses_the_current_link_parameters(line_network):
    _step(line_network)

    line_network.edges["A", "R"]["length_km"] = 400.0
    line_network.add_host("C")  # topology change: every link is refreshed
    _step(line_network, t=2.0)

    assert _latency(line_network, "A", "R") == pytest.approx(IDLE_MS + PROP_MS)


def test_clear_step_telemetry_allocates_new_columns(line_network):
    line_network.clear_step_telemetry()
    line_network.forward_batch(make_batch(line_network, "A", "B", 2), 1.0)
    previous = line_network.step_columns

    line_network.clear_step_telemetry()

    assert line_network.step_columns is not previous
    assert set(line_network.step_columns) == set(TELEMETRY_COLUMNS)
    assert all(values == [] for values in line_network.step_columns.values())
    assert previous["packet_id"] == [1, 2]


def test_clear_step_telemetry_resets_latency_accumulators(line_network):
    line_network.clear_step_telemetry()
    line_network.forward_batch(make_batch(line_network, "A", "B", 2), 1.0)
    link = line_network._links["A", "R"]
    assert link.count == 2

    line_network.clear_step_telemetry()

    assert (link.delay_sum, link.count) == (0.0, 0)
