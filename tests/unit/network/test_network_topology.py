"""Unit tests for the topology, roles and routing of Network."""

from __future__ import annotations

import pytest

from eclypse.network import Network
from eclypse.network.constants import MBPS_TO_BPS
from eclypse.network.network import path_algorithm
from tests.unit.network._helpers import make_batch


def _diamond(fast_mbps: float = 100.0, slow_mbps: float = 10.0) -> Network:
    """A -> R1 -> B (slow) and A -> R2 -> B (fast)."""
    net = Network("diamond")
    net.add_host("A")
    net.add_host("B")
    net.add_router("R1")
    net.add_router("R2")
    for u, v, bw in (
        ("A", "R1", slow_mbps),
        ("R1", "B", slow_mbps),
        ("A", "R2", fast_mbps),
        ("R2", "B", fast_mbps),
    ):
        net.add_edge(u, v, bandwidth_mbps=bw)
    return net


def _first_hop(net: Network, src: str, dst: str) -> str:
    """Forward one packet and return the link it traversed."""
    net.clear_step_telemetry()
    net.forward_batch(make_batch(net, src, dst, 1), 1.0)
    return net.step_columns["hop"][0]


# ---------------------------------------------------------------- roles


def test_add_router_sets_role_and_removes_computational_resources():
    net = Network("n")

    net.add_router("R", cpu=8, ram=8, processing_time=0.001)

    attrs = net.nodes["R"]
    assert attrs["role"] == "router"
    assert (attrs["cpu"], attrs["ram"], attrs["disk"]) == (0, 0, 0)
    assert attrs["processing_time"] == 0.001


def test_add_host_sets_role_and_keeps_resources():
    net = Network("n")

    net.add_host("H", cpu=8, ram=16)

    assert net.nodes["H"]["role"] == "host"
    assert net.nodes["H"]["cpu"] == 8


def test_hosts_and_routers_lists(line_network):
    assert line_network.hosts == ["A", "B"]
    assert line_network.routers == ["R"]


def test_nodes_without_role_are_hosts():
    net = Network("n")

    net.add_node("X")

    assert net.hosts == ["X"]
    assert net.routers == []


def test_role_lists_are_updated_when_nodes_are_added(line_network):
    assert line_network.hosts == ["A", "B"]

    line_network.add_host("C")
    line_network.add_router("R2")

    assert line_network.hosts == ["A", "B", "C"]
    assert line_network.routers == ["R", "R2"]


# ---------------------------------------------------------------- links


def test_add_edge_stores_link_parameters_and_routing_cost():
    net = Network("n")
    net.add_host("A")
    net.add_host("B")

    net.add_edge(
        "A",
        "B",
        bandwidth_mbps=50.0,
        length_km=3.0,
        propagation_speed_km_s=100000.0,
        max_queue_size=10,
    )

    attrs = net.edges["A", "B"]
    assert attrs["bandwidth_mbps"] == 50.0
    assert attrs["length_km"] == 3.0
    assert attrs["propagation_speed_km_s"] == 100000.0
    assert attrs["max_queue_size"] == 10
    assert attrs["cost"] == pytest.approx(1 / (50.0 * MBPS_TO_BPS))
    assert not net.has_edge("B", "A")


def test_symmetric_edges_are_two_independent_links(line_network):
    line_network.build_routing_tables()

    forward = line_network._links["A", "R"]
    backward = line_network._links["R", "A"]

    assert forward is not backward
    assert forward.attrs is not backward.attrs


# ---------------------------------------------------------------- routing


def test_routing_follows_the_minimum_cost_path():
    net = _diamond()

    assert _first_hop(net, "A", "B") == "A->R2"
    assert path_algorithm(net, "A", "B") == ["A", "R2", "B"]


def test_infrastructure_path_uses_the_network_path_algorithm():
    net = _diamond()

    hops = net.path("A", "B")

    assert [(u, v) for u, v, _ in hops] == [("A", "R2"), ("R2", "B")]


def test_path_algorithm_returns_empty_list_for_unreachable_nodes():
    net = _diamond()
    net.add_host("C")

    assert path_algorithm(net, "A", "C") == []
    assert path_algorithm(net, "missing", "B") == []


def test_removing_a_link_reroutes_traffic():
    net = _diamond()
    assert _first_hop(net, "A", "B") == "A->R2"

    net.remove_edge("A", "R2")

    assert _first_hop(net, "A", "B") == "A->R1"
    assert path_algorithm(net, "A", "B") == ["A", "R1", "B"]


def test_removing_a_node_reroutes_traffic():
    net = _diamond()

    net.remove_node("R2")

    assert _first_hop(net, "A", "B") == "A->R1"
    assert "R2" not in net.routers


def test_adding_a_better_link_reroutes_traffic():
    net = _diamond(fast_mbps=10.0, slow_mbps=10.0)
    net.add_router("R3")
    net.add_edge("A", "R3", bandwidth_mbps=1000.0)
    net.add_edge("R3", "B", bandwidth_mbps=1000.0)

    assert _first_hop(net, "A", "B") == "A->R3"


def test_link_queues_survive_a_routing_rebuild(line_network):
    line_network.clear_step_telemetry()
    line_network.forward_batch(make_batch(line_network, "A", "B", 3), 1.0)
    link = line_network._links["A", "R"]
    assert link.queue_length == 3

    line_network.add_host("C")
    line_network.build_routing_tables()

    assert line_network._links["A", "R"] is link
    assert link.queue_length == 3


# ---------------------------------------------------------------- node indices


def test_node_index_is_stable_and_registers_new_names(line_network):
    a = line_network.node_index("A")

    assert line_network.node_index("A") == a
    new = line_network.node_index("Z")
    assert new not in (a, line_network.node_index("B"))
    assert line_network.node_name(new) == "Z"
    assert line_network.node_name(-1) is None


def test_node_indices_are_not_reused_after_removal(line_network):
    line_network.build_routing_tables()
    b = line_network.node_index("B")
    r = line_network.node_index("R")

    line_network.remove_node("R")
    line_network.add_router("R_new")
    line_network.build_routing_tables()

    assert line_network.node_index("B") == b
    assert line_network.node_index("R_new") not in (b, r)
    assert line_network.node_name(r) == "R"
