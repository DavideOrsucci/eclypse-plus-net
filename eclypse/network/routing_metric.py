"""Module containing the traffic routing metric for the ECLYPSE framework."""

from eclypse.report.metrics.metric import application

from .network import Network
from .network_application import NetworkApplication


@application(name="traffic_routing", activates_on="step")
class RoutingMetric:
    """Observer metric that extracts routing results from completed packets.

    This metric is called by the Reporter. It performs no complex calculations;
    it simply reads the completed packets processed during the current step and
    formats their delay and hop information for reporting.
    """

    def __call__(self, app: NetworkApplication, _placement, infra: Network, **_kwargs):
        """Extract and format metrics from the application's completed packets.

        Args:
            app (NetworkApplication): The application instance being observed.
            placement: The placement service used in the simulation.
            infra (Network): The network infrastructure model.
            **kwargs: Additional keyword arguments provided by the framework.

        Returns:
            dict | None: A dictionary containing the flattened metrics for the current
                step, or None if no packets were completed.
        """
        step_results = {}

        if infra.step_telemetry:
            for _, (packet, hop_data) in enumerate(infra.step_telemetry):
                prefix = (
                    f"step_{app.current_step}_pkt_{packet.id}_hop_{packet.hop_count}"
                )

                step_results[f"{prefix}_hop"] = hop_data.hop
                step_results[f"{prefix}_processing_ms"] = float(hop_data.processing_ms)
                step_results[f"{prefix}_queue_ms"] = float(hop_data.queue_ms)
                step_results[f"{prefix}_transmission_ms"] = float(
                    hop_data.transmission_ms
                )
                step_results[f"{prefix}_propagation_ms"] = float(
                    hop_data.propagation_ms
                )
                step_results[f"{prefix}_queue_length"] = float(hop_data.queue_length)
                step_results[f"{prefix}_arrival_at_next"] = float(
                    hop_data.arrival_at_next
                )
                step_results[f"{prefix}_dropped"] = bool(hop_data.dropped)

        return step_results if step_results else None
