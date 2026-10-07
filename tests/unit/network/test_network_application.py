"""Unit tests for NetworkApplication and PacketGenerationEvent."""

from __future__ import annotations

import numpy as np
import pytest

from eclypse.network import (
    NetworkApplication,
    PacketGenerationEvent,
)
from eclypse.network.constants import (
    DEFAULT_AVG_PACKETS_PER_STEP,
    DEFAULT_PACKET_SIZE_BYTES,
)


def _app(**flows: tuple[float, int]) -> NetworkApplication:
    """Build an application; each keyword ``u_v=(rate, size)`` is a flow u -> v."""
    app = NetworkApplication("app")
    for name, (rate, size) in flows.items():
        u, v = name.split("_")
        for service in (u, v):
            if service not in app.nodes:
                app.add_node(service, cpu=1, ram=1)
        app.add_edge(u, v, packet_size_bytes=size, avg_packets_per_step=rate)
    return app


# ---------------------------------------------------------------- flows


def test_add_edge_stores_the_traffic_parameters():
    app = _app(a_b=(2.5, 64))

    assert app.edges["a", "b"]["avg_packets_per_step"] == 2.5
    assert app.edges["a", "b"]["packet_size_bytes"] == 64


def test_add_edge_defaults():
    app = NetworkApplication("app")
    app.add_node("a", cpu=1, ram=1)
    app.add_node("b", cpu=1, ram=1)

    app.add_edge("a", "b")

    assert app.edges["a", "b"]["avg_packets_per_step"] == DEFAULT_AVG_PACKETS_PER_STEP
    assert app.edges["a", "b"]["packet_size_bytes"] == DEFAULT_PACKET_SIZE_BYTES


@pytest.mark.parametrize(
    ("size", "rate"),
    [(0, 1.0), (-10, 1.0), (100, -0.5)],
    ids=["zero-size", "negative-size", "negative-rate"],
)
def test_add_edge_rejects_invalid_traffic_parameters(size, rate):
    app = NetworkApplication("app")
    app.add_node("a", cpu=1, ram=1)
    app.add_node("b", cpu=1, ram=1)

    with pytest.raises(ValueError):
        app.add_edge("a", "b", packet_size_bytes=size, avg_packets_per_step=rate)


# ---------------------------------------------------------------- generation


def test_new_application_has_no_traffic():
    app = _app(a_b=(1.0, 100))

    assert app.generated is None
    assert app.num_generated == 0
    assert app.current_step == 0


def test_application_without_flows_generates_nothing():
    app = NetworkApplication("app")
    app.add_node("a", cpu=1, ram=1)

    app.generate_traffic_for_step(1)

    assert app.generated is None
    assert app.num_generated == 0


def test_packet_counts_are_poisson_draws_from_the_global_numpy_state():
    app = _app(a_b=(5.0, 100), b_a=(2.0, 200))

    np.random.seed(7)
    app.generate_traffic_for_step(3)
    np.random.seed(7)
    expected = np.random.poisson([5.0, 2.0])

    generated = app.generated
    assert generated.flows == [("a", "b", 100), ("b", "a", 200)]
    assert np.bincount(generated.flow, minlength=2).tolist() == expected.tolist()
    assert generated.step == 3
    assert app.num_generated == int(expected.sum())


def test_packets_are_grouped_by_flow_and_numbered_consecutively():
    app = _app(a_b=(20.0, 100), b_a=(20.0, 100))
    np.random.seed(0)

    app.generate_traffic_for_step(1)

    flow = app.generated.flow
    assert np.all(np.diff(flow) >= 0)
    assert app.generated.id.tolist() == list(range(1, len(flow) + 1))


def test_packet_ids_are_unique_across_steps():
    app = _app(a_b=(10.0, 100))
    np.random.seed(0)

    app.generate_traffic_for_step(1)
    first = app.generated.id.tolist()
    app.generate_traffic_for_step(2)
    second = app.generated.id.tolist()

    assert first and second
    assert second[0] == first[-1] + 1


def test_zero_rate_flows_generate_no_packets():
    app = _app(a_b=(0.0, 100), b_a=(50.0, 100))
    np.random.seed(0)

    for step in range(1, 6):
        app.generate_traffic_for_step(step)
        assert not np.any(app.generated.flow == 0)


def test_flow_parameters_are_read_at_every_generation():
    app = _app(a_b=(0.0, 100))
    np.random.seed(0)
    app.generate_traffic_for_step(1)
    assert app.num_generated == 0

    app.edges["a", "b"]["avg_packets_per_step"] = 50.0
    app.edges["a", "b"]["packet_size_bytes"] = 300
    app.generate_traffic_for_step(2)

    assert app.num_generated > 0
    assert app.generated.flows == [("a", "b", 300)]


def test_generation_replaces_the_traffic_of_the_previous_step():
    app = _app(a_b=(10.0, 100))
    np.random.seed(0)
    app.generate_traffic_for_step(1)
    first = app.generated

    app.generate_traffic_for_step(2)

    assert app.generated is not first
    assert app.generated.step == 2


# ---------------------------------------------------------------- event


def test_packet_generation_event_advances_the_step_and_reports_the_count():
    app = _app(a_b=(10.0, 100))
    event = PacketGenerationEvent()
    np.random.seed(0)

    result = event(app, None, None)

    assert app.current_step == 1
    assert app.generated.step == 1
    assert result == {"packets_generated": app.num_generated}

    event(app, None, None)
    assert app.current_step == 2


def test_packet_generation_event_metadata():
    event = PacketGenerationEvent()

    assert event.name == "packet_generation"
