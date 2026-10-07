"""Expected physical behavior that the network extension does not model yet.

Each test describes the correct behavior and is marked ``xfail(strict=True)``:
it is expected to fail with the current model. When the model is fixed the test
passes, pytest reports it as an error (XPASS strict), and the marker must be
removed so that the test guards the fix.
"""

from __future__ import annotations

import pytest

from eclypse.network import RoutingEvent
from tests.unit.network._helpers import make_batch


@pytest.mark.xfail(
    strict=True,
    reason="Packets advance one hop per step regardless of their arrival time",
)
def test_a_packet_is_not_forwarded_before_it_arrives(line_network):
    routing = RoutingEvent(step_duration_s=0.001)
    line_network.clear_step_telemetry()
    # 50 packets on a link that needs 1 ms per packet: the last one reaches R
    # after about 52 ms
    line_network.forward_batch(make_batch(line_network, "A", "B", 50), 0.001)
    last_arrival_ms = line_network.step_columns["arrival_at_next"][-1]
    assert last_arrival_ms > 50.0

    class _App:
        current_step = 2
        generated = None

    routing(_App(), None, line_network)  # step 2: t = 2 ms

    assert 50 not in line_network.step_columns["packet_id"]
