"""Unit tests for the packets whose destination service leaves its node.

A packet is addressed to the node of its destination service when it is generated.
If the service moves while the packet travels, the packet still reaches the old
node: it is not delivered to the service, and is counted in
``Network.misdelivered_packets``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tests.unit.network._helpers import make_batch

if TYPE_CHECKING:
    from eclypse.network import Network


def _addressed(net: Network, batch, service: str):
    batch.dst_service = np.full(len(batch), net.service_index("app", service))
    return batch


def _forward_until_delivered(net: Network, batch, steps: int = 10) -> list[int]:
    delivered = net.forward_batch(batch, 0.0).id.tolist()
    for step in range(1, steps):
        for _, packets in net.take_router_batches(step * 1.0):
            delivered += net.forward_batch(packets, step * 1.0).id.tolist()
    return delivered


def test_packets_are_delivered_while_the_service_stays(line_network):
    line_network.update_service_hosts("app", {"s": "B"})
    batch = _addressed(line_network, make_batch(line_network, "A", "B", 3), "s")

    delivered = _forward_until_delivered(line_network, batch)

    assert len(delivered) == 3
    assert line_network.misdelivered_packets == 0


def test_packets_reaching_a_node_the_service_left_are_misdelivered(
    line_network, net_logger
):
    line_network.update_service_hosts("app", {"s": "B"})
    batch = _addressed(line_network, make_batch(line_network, "A", "B", 3), "s")
    line_network.forward_batch(batch, 0.0)

    # The service moves to A while the packets are on R
    line_network.update_service_hosts("app", {"s": "A"})
    (_, packets), *_ = line_network.take_router_batches(1.0)
    delivered = line_network.forward_batch(packets, 1.0)

    assert len(delivered) == 0
    assert line_network.misdelivered_packets == 3
    assert any(
        "misdelivered" in m and "'s'" in m and "node A" in m
        for m in net_logger.messages("warning")
    )


def test_packets_of_a_service_that_is_no_longer_placed_are_misdelivered(
    line_network, net_logger
):
    line_network.update_service_hosts("app", {"s": "B", "t": "A"})
    batch = _addressed(line_network, make_batch(line_network, "A", "B", 2), "s")
    line_network.forward_batch(batch, 0.0)

    # The placement of the application is discarded: no service is placed
    line_network.update_service_hosts("app", {})
    (_, packets), *_ = line_network.take_router_batches(1.0)
    delivered = line_network.forward_batch(packets, 1.0)

    assert len(delivered) == 0
    assert line_network.misdelivered_packets == 2
    assert any("not placed" in m for m in net_logger.messages("warning"))


def test_only_the_packets_of_the_moved_service_are_misdelivered(line_network):
    line_network.update_service_hosts("app", {"s": "B", "t": "B"})
    to_s = _addressed(line_network, make_batch(line_network, "A", "B", 2), "s")
    to_t = _addressed(
        line_network, make_batch(line_network, "A", "B", 3, first_id=10), "t"
    )
    line_network.forward_batch(to_s, 0.0)
    line_network.forward_batch(to_t, 0.0)

    line_network.update_service_hosts("app", {"s": "A", "t": "B"})
    (_, packets), *_ = line_network.take_router_batches(1.0)
    delivered = line_network.forward_batch(packets, 1.0)

    assert sorted(delivered.id.tolist()) == [10, 11, 12]
    assert line_network.misdelivered_packets == 2


def test_packets_not_addressed_to_a_service_are_always_delivered(line_network):
    line_network.update_service_hosts("app", {"s": "A"})
    batch = make_batch(line_network, "A", "B", 3)

    delivered = _forward_until_delivered(line_network, batch)

    assert (batch.dst_service == -1).all()
    assert len(delivered) == 3
    assert line_network.misdelivered_packets == 0


def test_a_service_mapped_to_a_node_outside_the_network_is_not_placed(line_network):
    line_network.update_service_hosts("app", {"s": "ghost"})
    batch = _addressed(line_network, make_batch(line_network, "A", "B", 2), "s")

    assert _forward_until_delivered(line_network, batch) == []
    assert line_network.misdelivered_packets == 2
    assert "ghost" not in line_network.nodes


def test_service_indices_are_persistent_and_grow(line_network):
    first = line_network.service_index("app", "s")
    others = [line_network.service_index("app", f"x{i}") for i in range(40)]

    assert line_network.service_index("app", "s") == first
    assert line_network.service_index("other", "s") != first
    assert others == list(range(first + 1, first + 41))

    line_network.update_service_hosts("app", {"x39": "B"})
    line_network.update_service_hosts("other", {"s": "A"})
    batch = _addressed(line_network, make_batch(line_network, "A", "B", 1), "x39")
    assert len(_forward_until_delivered(line_network, batch)) == 1
