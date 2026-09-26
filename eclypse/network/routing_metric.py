"""Module containing the traffic routing metric for the ECLYPSE framework."""

from eclypse.report.metrics.metric import application

from .network import Network
from .network_application import NetworkApplication


@application(name="traffic_routing", activates_on="step")
class RoutingMetric:
    """Observer metric that extracts routing results using Structure of Arrays (SoA).

    This approach avoids the massive overhead of dynamic string formatting
    by appending object attributes to pre-allocated list columns.
    """

    def __call__(
        self, app: NetworkApplication, _placement, infra: Network, **_kwargs
    ) -> dict[str, list[int | float | str | bool]] | None:
        """Extract and format metrics from the application's completed packets.

        Args:
            app (NetworkApplication): The application instance being observed.
            placement: The placement service used in the simulation.
            infra (Network): The network infrastructure model.
            **kwargs: Additional keyword arguments provided by the framework.

        Returns:
            dict | None: A dictionary containing the SoA metrics for the current
                step, or None if no packets were completed.
        """
        if not infra.step_telemetry:
            return None

        # Initialize the Structure of Arrays for the current step's metrics.
        step_results: dict[str, list[int | float | str | bool]] = {
            "step": [],
            "packet_id": [],
            "hop_count": [],
            "hop": [],
            "processing_ms": [],
            "queue_ms": [],
            "transmission_ms": [],
            "propagation_ms": [],
            "queue_length": [],
            "arrival_at_next": [],
            "dropped": [],
        }

        current_step = app.current_step

        # Iterate and append the observed packets and their hop information.
        for packet, hop_info in infra.step_telemetry:
            step_results["step"].append(current_step)
            step_results["packet_id"].append(packet.id)
            step_results["hop_count"].append(packet.hop_count)
            step_results["hop"].append(hop_info.hop)
            step_results["processing_ms"].append(float(hop_info.processing_ms))
            step_results["queue_ms"].append(float(hop_info.queue_ms))
            step_results["transmission_ms"].append(float(hop_info.transmission_ms))
            step_results["propagation_ms"].append(float(hop_info.propagation_ms))
            step_results["queue_length"].append(float(hop_info.queue_length))
            step_results["arrival_at_next"].append(float(hop_info.arrival_at_next))
            step_results["dropped"].append(bool(hop_info.dropped))

        return step_results
