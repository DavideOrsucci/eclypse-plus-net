"""Runtime state and delay model of the links of the network.

- :class:`LinkState` holds the runtime state (FIFO queue, timers, latency samples)
  of a directed link, served with the Lindley recursion.
- :class:`Segments` groups the packets of a forwarding batch by link and computes,
  with vectorized numpy operations, the delays and the DropTail outcome of every
  packet, touching Python objects once per link.
- :data:`TELEMETRY_COLUMNS` names the per-hop telemetry recorded for each packet.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from functools import reduce
from operator import add

import numpy as np

from .constants import (
    BYTES_TO_BITS,
    DEFAULT_BANDWIDTH_MBPS,
    DEFAULT_PROPAGATION_SPEED_KM_S,
    MBPS_TO_BPS,
    MIN_LENGTH_KM,
    SEC_TO_MS,
)

TELEMETRY_COLUMNS = (
    "packet_id",
    "hop_count",
    "hop",
    "processing_ms",
    "queue_ms",
    "transmission_ms",
    "propagation_ms",
    "queue_length",
    "arrival_at_next",
    "dropped",
)
"""Names of the per-hop telemetry columns recorded at every step.

Each forwarded or dropped packet contributes one row:

- ``packet_id``: unique identifier of the packet.
- ``hop_count``: hops traversed by the packet after this hop.
- ``hop``: traversed link, formatted as ``"u->v"``.
- ``processing_ms``: processing delay at the source node of the link.
- ``queue_ms``: queuing delay on the link.
- ``transmission_ms``: transmission (serialization) delay on the link.
- ``propagation_ms``: propagation delay along the link.
- ``queue_length``: packets already queued on the link when the packet arrived.
- ``arrival_at_next``: absolute arrival time at the next node, in milliseconds.
- ``dropped``: whether the packet was discarded because the queue was full.
"""

QUEUE_COMPACT_MIN = 1024
"""Minimum number of transmitted entries before a link queue list is compacted."""


class LinkState:
    """Runtime state of a directed link ``u -> v``.

    The state is kept outside the ECLYPSE edge attribute dictionary, whose writes
    invalidate the path caches of the infrastructure. Static parameters (bandwidth,
    length, propagation speed, queue capacity) are instead read from the edge
    attributes at every use, so that update policies acting on them take effect
    immediately.

    The link is a FIFO server that transmits one packet at a time at its full
    rate. Its queue is described by the time at which each packet finishes its
    transmission (``departures``): a packet arriving at time ``a`` waits
    ``max(0, busy_until - a)`` and then occupies the link for its transmission
    time (Lindley recursion). Partially transmitted packets are therefore
    accounted for exactly, whatever the times at which the queue is observed. A
    link whose bandwidth is not positive cannot transmit and drops the packets
    offered to it; the packets it had already accepted are still delivered.

    Attributes:
        u (str): Source node of the link.
        v (str): Destination node of the link.
        hop (str): Link label used in the telemetry, formatted as ``"u->v"``.
        attrs (dict): Edge attribute dictionary of the link in the infrastructure.
        u_attrs (dict): Node attribute dictionary of the source node, used to read
            its processing time.
        departures (list[float]): Time, in seconds, at which each accepted packet
            finishes its transmission, in FIFO order. Entries before ``head`` belong
            to packets already transmitted.
        head (int): Index in ``departures`` of the first packet not yet fully
            transmitted at ``step_time``.
        busy_until (float): Time, in seconds, at which the link finishes
            transmitting all the accepted packets.
        step_time (float): Simulation time, in seconds, of the last service of the
            queue.
        delay_sum (float): Sum of the hop delays, in milliseconds, of the packets
            accepted on the link in the current step.
        count (int): Number of packets accepted on the link in the current step.
        dropped (int): Number of packets dropped by the link in the current step.
    """

    __slots__ = (
        "attrs",
        "busy_until",
        "count",
        "delay_sum",
        "departures",
        "dropped",
        "head",
        "hop",
        "step_time",
        "u",
        "u_attrs",
        "v",
    )

    def __init__(self, u: str, v: str, attrs: dict, u_attrs: dict):
        """Initialize the state of an idle link with an empty queue.

        Args:
            u (str): Source node of the link.
            v (str): Destination node of the link.
            attrs (dict): Edge attribute dictionary of the link.
            u_attrs (dict): Node attribute dictionary of the source node.
        """
        self.u = u
        self.v = v
        self.hop = f"{u}->{v}"
        self.attrs = attrs
        self.u_attrs = u_attrs
        self.departures: list[float] = []
        self.head = 0
        self.busy_until = 0.0
        self.step_time = 0.0
        self.delay_sum = 0.0
        self.count = 0
        self.dropped = 0

    @property
    def queue_length(self) -> int:
        """Number of packets not yet fully transmitted at the last service."""
        return len(self.departures) - self.head

    def backlog(self, current_time: float) -> float:
        """Return the time needed to transmit the packets already accepted.

        Args:
            current_time (float): Current simulation time, in seconds.

        Returns:
            float: The waiting time, in seconds, of a packet arriving at
                ``current_time``.
        """
        return max(0.0, self.busy_until - current_time)

    def serve(self, current_time: float):
        """Remove from the queue the packets transmitted by ``current_time``.

        A packet leaves the queue when its transmission ends, i.e. when its
        departure time is not later than ``current_time``.

        Args:
            current_time (float): Current simulation time, in seconds.
        """
        departures = self.departures
        if len(departures) > self.head and departures[self.head] <= current_time:
            self.head = bisect_right(departures, current_time, self.head)
            # Compact the list once the transmitted prefix dominates
            if self.head > QUEUE_COMPACT_MIN and 2 * self.head > len(departures):
                del departures[: self.head]
                self.head = 0
        self.step_time = current_time

    def link_params(self) -> tuple[float, float, float, float]:
        """Read the parameters of the delay model from the node and edge attributes.

        Returns:
            tuple[float, float, float, float]: The transmission rate in bits per
                second, the processing time of the source node in seconds, the
                propagation delay in seconds, and the queue capacity in packets
                (``inf`` if unbounded).
        """
        attrs = self.attrs
        rate = attrs.get("bandwidth_mbps", DEFAULT_BANDWIDTH_MBPS) * MBPS_TO_BPS
        d_proc = self.u_attrs.get("processing_time", 0.0)
        d_prop = propagation_delay_s(attrs)
        return rate, d_proc, d_prop, attrs.get("max_queue_size", float("inf"))


def propagation_delay_s(attrs: dict) -> float:
    """Return the propagation delay of a link from its attributes.

    A link without ``length_km`` is assumed to have length :data:`MIN_LENGTH_KM`
    (no propagation delay), and a link without ``propagation_speed_km_s`` to
    propagate at :data:`DEFAULT_PROPAGATION_SPEED_KM_S`. A speed that is not
    positive gives no propagation delay, to avoid divisions by zero.

    Args:
        attrs (dict): The attributes of the link.

    Returns:
        float: The propagation delay, in seconds.
    """
    speed = attrs.get("propagation_speed_km_s", DEFAULT_PROPAGATION_SPEED_KM_S)
    if speed <= 0:
        return 0.0
    return attrs.get("length_km", MIN_LENGTH_KM) / speed


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Divide element-wise, returning 0.0 where the denominator is not positive.

    Args:
        num (np.ndarray): The numerators.
        den (np.ndarray): The denominators.

    Returns:
        np.ndarray: ``num / den`` where ``den > 0``, and ``0.0`` elsewhere.
    """
    out = np.zeros(len(num), dtype=np.float64)
    np.divide(num, den, out=out, where=den > 0)
    return out


class Segments:
    """Packets of a forwarding batch, grouped by the link they traverse.

    The packets are sorted by link, keeping their processing order within each
    link, so that the packets of a link form a contiguous segment. Per-link arrays
    have one entry per segment; per-packet arrays follow the link-sorted order.

    Within a step, all the packets of a link are offered to it at the same time.
    The ``i``-th packet of a segment therefore waits for the backlog of the link
    plus the transmission of the packets that precede it in the segment.

    Attributes:
        links (list[LinkState]): The link of each segment.
        starts (np.ndarray): Position of the first packet of each segment.
        counts (np.ndarray): Number of packets in each segment.
        sizes (np.ndarray): Size in bytes of each packet.
        rep (np.ndarray): Segment index of each packet.
        rate (np.ndarray): Transmission rate of each link, in bits per second.
        proc (np.ndarray): Processing time of the source node of each link, in
            seconds.
        prop (np.ndarray): Propagation delay of each link, in seconds.
        len0 (np.ndarray): Packets not yet fully transmitted on each link when the
            segment arrives.
        wait0 (np.ndarray): Backlog of each link when the segment arrives, in
            seconds.
        allowed (np.ndarray): Number of packets of each segment accepted by the
            DropTail policy; the remaining ones are dropped. It is zero on links
            whose bandwidth is not positive, which cannot transmit.
        csum (np.ndarray): Running total of ``sizes`` over the whole batch.
        before (np.ndarray): Value of ``csum`` just before the start of each
            segment.
    """

    def __init__(self, links, starts, counts, sizes):
        """Initialize the segments of a batch.

        Args:
            links (list[LinkState]): The link of each segment.
            starts (np.ndarray): Position of the first packet of each segment.
            counts (np.ndarray): Number of packets in each segment.
            sizes (np.ndarray): Size in bytes of each packet, in link-sorted order.
        """
        self.links: list[LinkState] = links
        self.starts: np.ndarray = starts
        self.counts: np.ndarray = counts
        self.sizes: np.ndarray = sizes
        self.rep = np.repeat(np.arange(len(links)), counts)
        n = len(links)
        self.rate = np.empty(n)
        self.proc = np.empty(n)
        self.prop = np.empty(n)
        self.len0 = np.empty(n, dtype=np.int64)
        self.wait0 = np.empty(n)
        self.allowed = np.empty(n, dtype=np.int64)
        self.csum = np.cumsum(sizes)
        self.before = self.csum[starts] - sizes[starts]

    def serve(self, current_time: float):
        """Serve the queue of every link and record its state and parameters.

        Args:
            current_time (float): Current simulation time, in seconds.
        """
        for j, (link, n) in enumerate(
            zip(self.links, self.counts.tolist(), strict=True)
        ):
            rate, d_proc, d_prop, max_q = link.link_params()
            link.serve(current_time)
            len0 = link.queue_length
            if rate <= 0:
                # A link without bandwidth cannot transmit: it drops every packet
                self.allowed[j] = 0
            else:
                # DropTail: packet i is accepted while len0 + i < max_q
                free = max_q - len0
                self.allowed[j] = n if free >= n else max(0, math.ceil(free))
            self.rate[j] = rate
            self.proc[j] = d_proc
            self.prop[j] = d_prop
            self.len0[j] = len0
            self.wait0[j] = link.backlog(current_time)

    def hop_delays(self, current_time: float) -> dict[str, np.ndarray]:
        """Compute the delays and the outcome of every packet.

        Dropped packets get zero delays and an arrival time equal to the current
        time.

        Args:
            current_time (float): Current simulation time, in seconds.

        Returns:
            dict[str, np.ndarray]: Per-packet arrays, in link-sorted order:
                ``accepted`` (whether the packet entered the queue), the delay
                columns of :data:`TELEMETRY_COLUMNS` (``processing_ms``,
                ``queue_ms``, ``transmission_ms``, ``propagation_ms``),
                ``arrival_at_next``, ``queue_length``, ``hop_delay`` (the sum of
                the four delays, in milliseconds), ``departure`` (the time at
                which the transmission of the packet ends, in seconds) and
                ``arrival`` (the arrival time at the next node, in seconds).
        """
        rep, sizes = self.rep, self.sizes
        pos = np.arange(len(sizes)) - self.starts[rep]
        allowed = self.allowed[rep]
        accepted = pos < allowed
        bytes_ahead = self.csum - sizes - self.before[rep]

        rate = self.rate[rep]
        d_proc = self.proc[rep]
        d_prop = self.prop[rep]
        d_queue = self.wait0[rep] + _safe_divide(bytes_ahead * BYTES_TO_BITS, rate)
        d_transm = _safe_divide(sizes * BYTES_TO_BITS, rate)

        out = {
            "accepted": accepted,
            "departure": current_time + d_queue + d_transm,
            "arrival": current_time + d_proc + d_queue + d_transm + d_prop,
            "processing_ms": d_proc * SEC_TO_MS,
            "queue_ms": d_queue * SEC_TO_MS,
            "transmission_ms": d_transm * SEC_TO_MS,
            "propagation_ms": d_prop * SEC_TO_MS,
            "arrival_at_next": current_time * SEC_TO_MS
            + (d_proc + d_queue + d_transm + d_prop) * SEC_TO_MS,
            "queue_length": (self.len0[rep] + np.minimum(pos, allowed)).astype(
                np.float64
            ),
        }
        dropped = ~accepted
        if dropped.any():
            for name in ("processing_ms", "queue_ms", "transmission_ms"):
                out[name][dropped] = 0.0
            out["propagation_ms"][dropped] = 0.0
            out["arrival_at_next"][dropped] = current_time * SEC_TO_MS
            out["arrival"][dropped] = current_time
        out["hop_delay"] = (
            out["processing_ms"]
            + out["queue_ms"]
            + out["transmission_ms"]
            + out["propagation_ms"]
        )
        return out

    def commit(self, hops: dict[str, np.ndarray], touched: list, network) -> int:
        """Enqueue the accepted packets and record the latency samples of each link.

        Args:
            hops (dict[str, np.ndarray]): The result of :meth:`hop_delays`.
            touched (list[LinkState]): Links that received traffic in the current
                step; links used for the first time in the step are appended.
            network (Network): The network, used for logging.

        Returns:
            int: The number of dropped packets.
        """
        departures = hops["departure"].tolist()
        delays = hops["hop_delay"].tolist()
        n_dropped = 0
        for link, start, n, a, rate in zip(
            self.links,
            self.starts.tolist(),
            self.counts.tolist(),
            self.allowed.tolist(),
            self.rate.tolist(),
            strict=True,
        ):
            if link.count == 0 and link.dropped == 0:
                touched.append(link)
            if a:
                end = start + a
                link.departures.extend(departures[start:end])
                link.busy_until = departures[end - 1]
                # Left-to-right summation, for reproducible floating point results
                link.delay_sum = reduce(add, delays[start:end], link.delay_sum)
                link.count += a
            if a < n:
                link.dropped += n - a
                n_dropped += n - a
                reason = (
                    "No bandwidth on link "
                    f"{link.hop} (bandwidth_mbps: {link.attrs.get('bandwidth_mbps')})"
                    if rate <= 0
                    else f"Queue full on link {link.hop} "
                    f"(Limit: {link.attrs.get('max_queue_size')})"
                )
                network.logger.debug(f"{n - a} packets DROPPED at {link.u}: {reason}")
        return n_dropped
