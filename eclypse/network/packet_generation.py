"""Module containing the packet generation event for the ECLYPSE framework."""

from eclypse.workflow.event import EclypseEvent
from eclypse.workflow.trigger import CascadeTrigger

from .network import Network
from .network_application import NetworkApplication


class PacketGenerationEvent(EclypseEvent):
    """Worker event that generates network packets to be routed.

    This event triggers on every simulation step to instruct the application
    to generate new traffic flows based on its internal graph and patterns.
    """

    def __init__(self):
        """Initialize the packet generation event.

        Sets up the event name, type, and triggers it to run on every
        simulation step using a CascadeTrigger.
        """
        super().__init__(
            name="packet_generation",
            event_type="application",
            triggers=[CascadeTrigger("step")],
        )

    def __call__(self, app: NetworkApplication, _placement, _infra: Network, **_kwargs):
        """Generate packets for the current simulation step.

        Advances the internal application clock and triggers the creation of
        new traffic flows based on the application's configured patterns.

        Args:
            app (NetworkApplication): The application instance responsible for
                generating the packets.
            placement: The placement service used in the simulation.
            infra (Network): The network infrastructure model.
            **kwargs: Additional keyword arguments provided by the framework.

        Returns:
            dict: A dictionary containing the count of generated packets under
                the key 'packets_generated'.
        """
        app.current_step += 1
        app.generate_traffic_for_step(app.current_step)
        self.logger.debug(
            f"step {app.current_step}: "
            f"App '{app.id}' has generated "
            f"{app.num_generated} packets."
        )

        return {"packets_generated": app.num_generated}
