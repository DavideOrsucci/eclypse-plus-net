"""Fixtures for the tests of the network extension."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import pytest

from eclypse.network import NetworkApplication
from tests.unit.network._helpers import (
    PKT_BYTES,
    RecordingLogger,
    StubPlacement,
    build_line,
)

if TYPE_CHECKING:
    from eclypse.network import Network


@pytest.fixture
def line_network() -> Network:
    """Line topology A - R - B with large queues."""
    return build_line()


@pytest.fixture
def net_logger(monkeypatch) -> RecordingLogger:
    """Record the log calls made by Network instances."""
    recorder = RecordingLogger()
    monkeypatch.setattr("eclypse.graph.asset_graph.logger", recorder)
    return recorder


@pytest.fixture
def seeded():
    """Seed the random generators used by the extension."""
    random.seed(1234)
    np.random.seed(1234)


@pytest.fixture
def line_app() -> tuple[NetworkApplication, StubPlacement]:
    """Application with one flow from service ``sa`` (on A) to ``sb`` (on B)."""
    app = NetworkApplication("app")
    app.add_node("sa", cpu=1, ram=1)
    app.add_node("sb", cpu=1, ram=1)
    app.add_edge("sa", "sb", packet_size_bytes=PKT_BYTES, avg_packets_per_step=5.0)
    return app, StubPlacement({"sa": "A", "sb": "B"})
