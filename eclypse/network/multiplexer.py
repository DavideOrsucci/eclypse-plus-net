"""Probabilistic multiplexer that merges the input queues of a router.

The packets waiting on a router form one FIFO input queue per upstream node.
:func:`probabilistic_merge` decides the order in which the router processes them,
selecting the input queues at random with probability proportional to the bandwidth
of their input links.
"""

from __future__ import annotations

import random
from itertools import accumulate
from typing import TYPE_CHECKING

import numpy as np

from .constants import DEFAULT_BANDWIDTH_MBPS

if TYPE_CHECKING:
    from .network import Network
    from .packets import PacketBatch


def input_bandwidths(router: str, prev_nodes: list[int], infra: Network) -> list[float]:
    """Read the bandwidth of the input links of a router.

    Args:
        router (str): The router.
        prev_nodes (list[int]): The indices of the nodes upstream of the router.
        infra (Network): The network infrastructure.

    Returns:
        list[float]: The bandwidth, in Mbps, of the link from each node of
            ``prev_nodes`` to ``router``.
    """
    adj = infra.adj
    return [
        adj.get(infra.node_name(p), {})
        .get(router, {})
        .get("bandwidth_mbps", DEFAULT_BANDWIDTH_MBPS)
        for p in prev_nodes
    ]


def probabilistic_merge(
    router: str, packets: PacketBatch, infra: Network
) -> PacketBatch:
    """Merge the input queues of a router with a probabilistic multiplexer.

    The packets of the router form one FIFO input queue per upstream node. The
    multiplexer repeatedly selects a non-empty input queue at random, with
    probability proportional to the bandwidth of its input link, and takes its
    first packet. This models a fair-queuing scheduler that serves each input in
    proportion to its capacity.

    Each selection draws one number from Python's global ``random`` generator,
    with the same semantics as ``random.choices(queues, weights, k=1)``.
    Selections are resolved in rounds of ``d`` draws, where ``d`` is the
    smallest number of packets left in an input queue: no queue can empty before
    the last draw of a round, so the weights are constant within it and the
    round is resolved with a single vectorized search.

    Args:
        router (str): The router whose packets are merged.
        packets (PacketBatch): The packets waiting on the router, in arrival
            order.
        infra (Network): The network infrastructure.

    Returns:
        PacketBatch: The packets of the router, in processing order.
    """
    rnd = random.random
    n = len(packets)
    uniq, first, inverse = np.unique(
        packets.prev, return_index=True, return_inverse=True
    )
    k = len(uniq)
    if k == 1:
        # A single input queue: the order is fixed, one draw per packet
        for _ in range(n):
            rnd()
        return packets

    # Input queues, in order of first appearance in the buffer
    appearance = np.argsort(first)
    rank = np.empty(k, dtype=np.int64)
    rank[appearance] = np.arange(k)
    source_of = rank[inverse.reshape(-1)]
    weights = input_bandwidths(router, uniq[appearance].tolist(), infra)
    remaining = np.bincount(source_of, minlength=k).tolist()

    active = list(range(k))
    sequence = []
    while active:
        if len(active) == 1:
            left = remaining[active[0]]
            for _ in range(left):
                rnd()
            sequence.append(np.full(left, active[0], dtype=np.int64))
            break
        cum_weights = list(accumulate([weights[j] for j in active]))
        total = cum_weights[-1] + 0.0
        bounds = np.array(cum_weights[:-1])
        active_arr = np.array(active, dtype=np.int64)
        while True:
            d = min(remaining[j] for j in active)
            draws = np.array([rnd() for _ in range(d)])
            picks = np.searchsorted(bounds, draws * total, side="right")
            sequence.append(active_arr[picks])
            got = np.bincount(picks, minlength=len(active)).tolist()
            emptied = -1
            for pos, j in enumerate(active):
                remaining[j] -= got[pos]
                if remaining[j] == 0:
                    emptied = pos
            if emptied >= 0:
                active.pop(emptied)
                break

    # The i-th selection of queue j takes the i-th packet of queue j
    seq = np.concatenate(sequence)
    order = np.empty(n, dtype=np.int64)
    order[np.argsort(seq, kind="stable")] = np.argsort(source_of, kind="stable")
    return packets.take(order)
