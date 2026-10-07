"""Simulation event that generates the network traffic of each step."""

from eclypse.workflow.event import EclypseEvent
from eclypse.workflow.trigger import CascadeTrigger

from .network import Network
from .network_application import NetworkApplication


class PacketGenerationEvent(EclypseEvent):
    """Event that generates the packets of a :class:`NetworkApplication`.

    The event runs at every simulation step. It advances the step counter of the
    application and generates the packets of all its flows, which are then injected
    into the network by the :class:`~eclypse.network.RoutingEvent`. It must
    therefore be registered before the routing event.
    """

    def __init__(self):
        """Initialize the event, named ``packet_generation``, to run every step."""
        super().__init__(
            name="packet_generation",
            event_type="application",
            triggers=[CascadeTrigger("step")],
        )

    def __call__(self, app: NetworkApplication, _placement, _infra: Network, **_kwargs):
        """Advance the application to the next step and generate its packets.

        Args:
            app (NetworkApplication): The application that generates the traffic.
            _placement (Placement): The placement of the application (unused).
            _infra (Network): The network infrastructure (unused).
            **_kwargs: Additional arguments provided by the framework (unused).

        Returns:
            dict[str, int]: The number of generated packets, under the key
                ``packets_generated``.
        """
        app.current_step += 1
        app.generate_traffic_for_step(app.current_step)
        self.logger.debug(
            f"step {app.current_step}: "
            f"App '{app.id}' has generated "
            f"{app.num_generated} packets."
        )

        return {"packets_generated": app.num_generated}
