"""Unit tests for HostOnlyStrategy: services are placed only on hosts."""

from __future__ import annotations

import pytest

from eclypse.network import (
    HostOnlyStrategy,
    Network,
    NetworkApplication,
)
from eclypse.placement import PlacementView
from eclypse.placement.strategies import (
    BestFitStrategy,
    FirstFitStrategy,
    RandomStrategy,
    RoundRobinStrategy,
    StaticStrategy,
)
from tests.unit.network._helpers import build_line


def _star(n_hosts: int = 3) -> Network:
    """Router R0 first (as the builders do), then hosts H0..Hn around it."""
    net = Network("star")
    net.add_router("R0")
    for i in range(n_hosts):
        net.add_host(f"H{i}", cpu=4, ram=4)
        net.add_edge("R0", f"H{i}", symmetric=True)
    return net


def _app(n_services: int) -> NetworkApplication:
    app = NetworkApplication("app")
    for i in range(n_services):
        app.add_node(f"s{i}", cpu=1, ram=1)
    return app


def _place(strategy, net, app):
    return strategy.place(net, app, {}, PlacementView(net))


@pytest.mark.parametrize("seed", range(20))
def test_random_strategy_never_chooses_a_router(seed):
    net, app = _star(), _app(6)

    mapping = _place(HostOnlyStrategy(RandomStrategy(seed=seed)), net, app)

    assert set(mapping) == set(app.nodes)
    assert set(mapping.values()) <= {"H0", "H1", "H2"}


def test_round_robin_spreads_the_services_over_the_hosts():
    net, app = _star(), _app(3)

    # Without the wrapper, round robin starts from the router added first
    assert _place(RoundRobinStrategy(), net, app)["s0"] == "R0"

    mapping = _place(HostOnlyStrategy(RoundRobinStrategy()), net, app)

    assert mapping == {"s0": "H0", "s1": "H1", "s2": "H2"}


@pytest.mark.parametrize("strategy", [FirstFitStrategy, BestFitStrategy])
def test_fit_strategies_keep_working(strategy):
    net, app = _star(), _app(2)

    mapping = _place(HostOnlyStrategy(strategy()), net, app)

    assert set(mapping) == {"s0", "s1"}
    assert set(mapping.values()) <= {"H0", "H1", "H2"}


def test_hosts_that_are_down_are_not_chosen():
    net, app = _star(), _app(6)
    net.nodes["H1"]["availability"] = 0

    mapping = _place(HostOnlyStrategy(RandomStrategy(seed=1)), net, app)

    assert set(mapping.values()) <= {"H0", "H2"}


def test_services_mapped_to_a_router_are_left_unplaced(monkeypatch):
    net, app = build_line(), _app(2)
    warnings = []
    monkeypatch.setattr(
        type(net), "logger", property(lambda self: _Recorder(warnings)), raising=False
    )

    mapping = _place(HostOnlyStrategy(StaticStrategy({"s0": "A", "s1": "R"})), net, app)

    assert mapping == {"s0": "A"}
    assert len(warnings) == 1
    assert "'s1': 'R'" in warnings[0]


def test_the_wrapped_strategy_sees_the_same_view_without_routers():
    net, app = _star(), _app(1)
    seen = {}

    class Spy(RandomStrategy):
        def place(self, infrastructure, application, placements, placement_view):
            seen["nodes"] = set(placement_view.residual.nodes)
            seen["infrastructure"] = placement_view.infrastructure
            return super().place(
                infrastructure, application, placements, placement_view
            )

    view = PlacementView(net)
    HostOnlyStrategy(Spy(seed=1)).place(net, app, {}, view)

    assert seen["nodes"] == {"H0", "H1", "H2"}
    assert seen["infrastructure"] is view.infrastructure


class _Recorder:
    def __init__(self, sink):
        self.sink = sink

    def warning(self, message, *_args, **_kwargs):
        self.sink.append(message)
