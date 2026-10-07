"""Helpers shared by the tests of the network extension."""

from __future__ import annotations

from typing import Any

import numpy as np

from eclypse.network import (
    Network,
    PacketBatch,
)

# Link parameters chosen so that delays are round numbers:
# 8 Mbps -> a 1000-byte packet takes 1 ms to transmit;
# 200 km at 200000 km/s -> 1 ms of propagation.
BW_MBPS = 8.0
LENGTH_KM = 200.0
PKT_BYTES = 1000
TX_MS = 1.0
PROP_MS = 1.0
PROC_S = 0.0005  # 0.5 ms of processing at the sending node
PROC_MS = 0.5


class RecordingLogger:
    """Logger stand-in that records the calls made by the network extension."""

    def __init__(self):
        self.records: list[tuple[str, tuple[Any, ...]]] = []

    def bind(self, **_: Any) -> RecordingLogger:
        return self

    def opt(self, **_: Any) -> RecordingLogger:
        return self

    def _record(self, level: str, args: tuple[Any, ...]):
        self.records.append((level, args))

    def debug(self, *args: Any, **_: Any):
        self._record("debug", args)

    def info(self, *args: Any, **_: Any):
        self._record("info", args)

    def warning(self, *args: Any, **_: Any):
        self._record("warning", args)

    def error(self, *args: Any, **_: Any):
        self._record("error", args)

    def messages(self, level: str) -> list[str]:
        """Return the first argument of the calls made at the given level."""
        return [str(args[0]) for lvl, args in self.records if lvl == level and args]


class StubPlacement:
    """Placement stand-in mapping services to nodes; unknown services raise."""

    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def service_placement(self, service_id: str) -> str:
        return self.mapping[service_id]


def make_batch(
    net: Network,
    src: str,
    dst: str,
    n: int,
    size: int = PKT_BYTES,
    cur: str | None = None,
    prev: str | None = None,
    hop: int = 0,
    first_id: int = 1,
) -> PacketBatch:
    """Build a batch of ``n`` identical packets from ``src`` to ``dst``."""
    cur_idx = net.node_index(cur or src)
    prev_idx = -1 if prev is None else net.node_index(prev)
    return PacketBatch(
        id=np.arange(first_id, first_id + n, dtype=np.int64),
        src=np.full(n, net.node_index(src), dtype=np.int64),
        dst=np.full(n, net.node_index(dst), dtype=np.int64),
        size=np.full(n, size, dtype=np.int64),
        step=np.zeros(n, dtype=np.int64),
        cur=np.full(n, cur_idx, dtype=np.int64),
        prev=np.full(n, prev_idx, dtype=np.int64),
        hop=np.full(n, hop, dtype=np.int64),
    )


def build_line(max_queue_size: int = 100) -> Network:
    """Build the line topology A (host) - R (router) - B (host).

    Links are bidirectional, with the parameters defined at module level.
    """
    net = Network("line")
    net.add_host("A", processing_time=PROC_S, cpu=4, ram=4)
    net.add_router("R", processing_time=PROC_S)
    net.add_host("B", processing_time=PROC_S, cpu=4, ram=4)
    for u, v in (("A", "R"), ("R", "B")):
        net.add_edge(
            u,
            v,
            symmetric=True,
            bandwidth_mbps=BW_MBPS,
            length_km=LENGTH_KM,
            max_queue_size=max_queue_size,
        )
    return net
