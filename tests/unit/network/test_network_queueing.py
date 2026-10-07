"""Validation of the link queue against an exact FIFO model and queueing theory."""

from __future__ import annotations

import numpy as np
import pytest

from eclypse.network import PacketBatch
from tests.unit.network._helpers import (
    BW_MBPS,
    build_line,
    make_batch,
)

RATE_BPS = BW_MBPS * 1e6


def _offer(net, sizes: list[int], t: float, first_id: int) -> dict[str, list]:
    """Offer packets of the given sizes to link A->R at time ``t``."""
    batch = PacketBatch.concat(
        [
            make_batch(net, "A", "B", 1, size=size, first_id=first_id + i)
            for i, size in enumerate(sizes)
        ]
    )
    net.clear_step_telemetry()
    net.forward_batch(batch, t)
    net.in_transit = PacketBatch.empty()  # observe link A->R only
    return net.step_columns


def _lindley(arrivals: list[tuple[float, int]]) -> list[float]:
    """Waiting times (ms) of a FIFO link serving the packets one at a time."""
    busy_until, waits = 0.0, []
    for t, size in arrivals:
        wait = max(0.0, busy_until - t)
        busy_until = t + wait + size * 8 / RATE_BPS
        waits.append(wait * 1e3)
    return waits


def test_queuing_delays_match_an_exact_fifo_link():
    rng = np.random.default_rng(42)
    net = build_line(max_queue_size=10**9)
    t, pid = 1.0, 1
    arrivals, observed = [], []
    for _ in range(300):
        t += float(rng.uniform(0.0, 0.004))  # irregular observation times
        sizes = rng.integers(64, 1500, size=rng.poisson(2.0)).tolist()
        if not sizes:
            continue
        observed += _offer(net, sizes, t, pid)["queue_ms"]
        arrivals += [(t, size) for size in sizes]
        pid += len(sizes)

    assert observed == pytest.approx(_lindley(arrivals), abs=1e-9)
    assert max(observed) > 0.0


def test_queue_length_counts_the_packets_not_yet_transmitted():
    net = build_line(max_queue_size=10**9)
    _offer(net, [1000] * 4, 1.0, 1)  # departures at 1, 2, 3 and 4 ms

    cols = _offer(net, [1000], 1.0025, 10)

    assert cols["queue_length"] == [2.0]  # packets 3 and 4
    assert cols["queue_ms"] == pytest.approx([1.5])


def test_link_is_stable_below_its_capacity():
    """4 packets every 3 steps of 1.5 ms: 89% of the capacity."""
    net = build_line(max_queue_size=10**9)
    lengths = []
    for k in range(600):
        n = 2 if k % 3 == 0 else 1
        cols = _offer(net, [1000] * n, 1.0 + 0.0015 * k, 10 * k)
        lengths.append(cols["queue_length"][0])

    assert max(lengths) <= 3


def test_mean_waiting_time_matches_md1_theory():
    """Poisson arrivals, constant service time: Wq = rho / (2 mu (1 - rho))."""
    rho, tx_s, substeps = 0.7, 1000 * 8 / RATE_BPS, 5
    dt = tx_s / substeps
    rng = np.random.default_rng(0)
    net = build_line(max_queue_size=10**9)
    waits, pid, k = [], 1, 0
    while pid < 10_000:
        k += 1
        n = int(rng.poisson(rho / substeps))
        if n:
            waits += _offer(net, [1000] * n, 1.0 + k * dt, pid)["queue_ms"]
            pid += n

    expected_ms = rho / (2 * (1 - rho)) * tx_s * 1e3
    assert np.mean(waits[1000:]) == pytest.approx(expected_ms, rel=0.15)
