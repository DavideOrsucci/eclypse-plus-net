"""Unit tests for PacketBatch and LinkState."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eclypse.network import PacketBatch
from eclypse.network.constants import (
    DEFAULT_BANDWIDTH_MBPS,
    MBPS_TO_BPS,
)
from eclypse.network.network import (
    PACKET_FIELDS,
    QUEUE_COMPACT_MIN,
    LinkState,
)


def _batch(ids: list[int]) -> PacketBatch:
    arr = np.array(ids, dtype=np.int64)
    return PacketBatch(**{f: arr * (i + 1) for i, f in enumerate(PACKET_FIELDS)})


# ---------------------------------------------------------------- PacketBatch


def test_empty_batch_has_every_field_as_empty_int64_array():
    batch = PacketBatch.empty()

    assert len(batch) == 0
    for field in PACKET_FIELDS:
        values = getattr(batch, field)
        assert values.dtype == np.int64
        assert values.shape == (0,)


def test_len_counts_packets():
    assert len(_batch([1, 2, 3])) == 3


@pytest.mark.parametrize(
    "index",
    [
        np.array([True, False, True, False]),
        np.array([0, 2]),
        slice(0, 3, 2),
    ],
    ids=["mask", "positions", "slice"],
)
def test_take_selects_the_same_packets_in_every_field(index):
    batch = _batch([10, 20, 30, 40])

    taken = batch.take(index)

    assert taken.id.tolist() == [10, 30]
    for i, field in enumerate(PACKET_FIELDS):
        assert getattr(taken, field).tolist() == [10 * (i + 1), 30 * (i + 1)]


def test_take_follows_the_order_of_the_index():
    taken = _batch([10, 20, 30]).take(np.array([2, 0, 1]))

    assert taken.id.tolist() == [30, 10, 20]


def test_concat_preserves_the_order_of_batches_and_packets():
    joined = PacketBatch.concat([_batch([1, 2]), _batch([3]), _batch([4, 5])])

    assert joined.id.tolist() == [1, 2, 3, 4, 5]
    assert joined.hop.tolist() == [8, 16, 24, 32, 40]


def test_concat_skips_empty_batches_and_returns_a_single_batch_unchanged():
    only = _batch([7, 8])

    joined = PacketBatch.concat([PacketBatch.empty(), only, PacketBatch.empty()])

    assert joined is only


@pytest.mark.parametrize("batches", [[], [PacketBatch.empty(), PacketBatch.empty()]])
def test_concat_of_no_packets_is_empty(batches):
    assert len(PacketBatch.concat(batches)) == 0


# ---------------------------------------------------------------- LinkState


def _link(**attrs) -> LinkState:
    return LinkState("u", "v", attrs, {"processing_time": 0.002})


def _enqueue(link: LinkState, sizes: list[int]):
    last = link.cum[-1] if len(link.cum) > link.head else link.base
    for size in sizes:
        last += size
        link.cum.append(last)


def test_new_link_is_idle_with_an_empty_queue():
    link = _link()

    assert link.hop == "u->v"
    assert link.queue_length == 0
    assert link.queue_bytes == 0
    assert link.step_time == 0.0
    assert (link.delay_sum, link.count) == (0.0, 0)


def test_queue_length_and_bytes_follow_the_cumulative_list():
    link = _link()

    _enqueue(link, [100, 200, 300])

    assert link.queue_length == 3
    assert link.queue_bytes == 600


def test_serve_transmits_only_whole_packets_in_fifo_order():
    link = _link()
    _enqueue(link, [100, 200, 300])

    # 1 s at 2400 bit/s = 300 bytes: the first two packets fit exactly
    link.serve(1.0, 2400.0)

    assert link.queue_length == 1
    assert link.queue_bytes == 300
    assert link.step_time == 1.0


def test_serve_with_enough_capacity_empties_the_queue():
    link = _link()
    _enqueue(link, [100, 200, 300])

    link.serve(10.0, 1e6)

    assert link.queue_length == 0
    assert link.queue_bytes == 0


def test_serve_keeps_a_packet_that_does_not_fit_entirely():
    link = _link()
    _enqueue(link, [1000])

    link.serve(1.0, 7999.0)  # one bit short of a 1000-byte packet

    assert link.queue_length == 1


@pytest.mark.parametrize("time", [0.0, -1.0])
def test_serve_does_nothing_if_time_does_not_advance(time):
    link = _link()
    _enqueue(link, [100])
    link.step_time = 0.0

    link.serve(time, 1e9)

    assert link.queue_length == 1


def test_serve_compacts_a_long_transmitted_prefix_preserving_the_queue():
    link = _link()
    n = 3 * QUEUE_COMPACT_MIN
    _enqueue(link, [100] * n)
    drained = 2 * QUEUE_COMPACT_MIN

    # 2 s at 400 * drained bit/s = 100 * drained bytes
    link.serve(2.0, 400.0 * drained)

    assert link.head == 0
    assert len(link.cum) == n - drained
    assert link.queue_length == n - drained
    assert link.queue_bytes == 100 * (n - drained)


def test_enqueue_after_compaction_continues_the_cumulative_count():
    link = _link()
    _enqueue(link, [100] * (3 * QUEUE_COMPACT_MIN))
    link.serve(2.0, 1e12)  # drain everything

    _enqueue(link, [50])

    assert link.queue_length == 1
    assert link.queue_bytes == 50


def test_link_params_read_the_edge_and_node_attributes():
    link = _link(
        bandwidth_mbps=10.0,
        length_km=400.0,
        propagation_speed_km_s=200000.0,
        max_queue_size=7,
    )

    rate, d_proc, d_prop, max_q = link.link_params()

    assert rate == pytest.approx(10.0 * MBPS_TO_BPS)
    assert d_proc == pytest.approx(0.002)
    assert d_prop == pytest.approx(0.002)
    assert max_q == 7


def test_link_params_reflect_attribute_updates():
    attrs = {"bandwidth_mbps": 10.0}
    link = LinkState("u", "v", attrs, {})

    attrs["bandwidth_mbps"] = 20.0

    assert link.link_params()[0] == pytest.approx(20.0 * MBPS_TO_BPS)


def test_link_params_fallbacks_for_missing_attributes():
    rate, d_proc, d_prop, max_q = LinkState("u", "v", {}, {}).link_params()

    assert rate == pytest.approx(DEFAULT_BANDWIDTH_MBPS * MBPS_TO_BPS)
    assert d_proc == 0.0
    assert d_prop == 0.0
    assert math.isinf(max_q)
