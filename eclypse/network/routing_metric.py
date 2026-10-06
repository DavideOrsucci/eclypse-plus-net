"""Module containing the traffic routing metric for the ECLYPSE framework."""

from eclypse.report.metrics.metric import application

from .network import Network
from .network_application import NetworkApplication


@application(name="traffic_routing", activates_on="step")
class RoutingMetric:
    """Observer metric that extracts routing results using Structure of Arrays (SoA).

    The columns are filled directly during forwarding, so the metric only has to
    add the ``step`` column.
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
        columns = infra.step_columns
        n = len(columns["hop"])
        if n == 0:
            return None

        # The telemetry is already stored as a Structure of Arrays by
        # Network.forward_batch: hand the column lists over without copying
        # them (Network allocates fresh lists at the start of every step).
        step_results: dict[str, list[int | float | str | bool]] = {
            "step": [app.current_step] * n
        }
        step_results.update(columns)
        return step_results
