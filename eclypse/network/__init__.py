"""Packet-level network simulation for the ECLYPSE framework.

The extension models the infrastructure as a network of hosts and routers
connected by links with finite FIFO queues, and the application as a set of
stochastic traffic flows between services. At every step, packets are generated,
the packets that have reached their current node are forwarded by one hop along
minimum-cost routes, and their per-hop delays are recorded.

A simulation typically registers, in this order:

- :class:`PacketGenerationEvent`, which generates the packets of the step;
- :class:`RoutingEvent`, which forwards the packets of all the applications
  through the :class:`Network`, once per step;
- :class:`RoutingMetric`, which reports the per-hop telemetry of each application.

Several :class:`NetworkApplication` instances can share the same network: packet
identifiers are allocated by the network, so they are unique across applications.
"""

from .network import Network, PacketBatch
from .network_application import NetworkApplication
from .packet_generation import PacketGenerationEvent
from .routing_event import RoutingEvent
from .routing_metric import RoutingMetric

__all__ = [
    "Network",
    "NetworkApplication",
    "PacketBatch",
    "PacketGenerationEvent",
    "RoutingEvent",
    "RoutingMetric",
]
