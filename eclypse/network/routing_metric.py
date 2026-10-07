"""Metric that reports the per-hop telemetry of the network traffic."""

from eclypse.report.metrics.metric import application

from .network import Network
from .network_application import NetworkApplication


@application(name="traffic_routing", activates_on="step")
class RoutingMetric:
    """Application metric that reports the telemetry of every hop of the step.

    The metric, named ``traffic_routing``, is collected at every step. It reports
    one row per forwarded or dropped packet, with the columns listed in
    :data:`~eclypse.network.network.TELEMETRY_COLUMNS` plus the ``step`` column.
    """

    def __call__(
        self, app: NetworkApplication, _placement, infra: Network, **_kwargs
    ) -> dict[str, list[int | float | str | bool]] | None:
        """Collect the telemetry of the current step.

        Args:
            app (NetworkApplication): The observed application.
            _placement (Placement): The placement of the application (unused).
            infra (Network): The network infrastructure.
            **_kwargs: Additional arguments provided by the framework (unused).

        Returns:
            dict[str, list[int | float | str | bool]] | None: One list per column,
                with one entry per hop, or None if no packet was forwarded in the
                step.
        """
        columns = infra.step_columns
        n = len(columns["hop"])
        if n == 0:
            return None

        # The columns are returned without copying: the network allocates new
        # lists at the start of every step.
        step_results: dict[str, list[int | float | str | bool]] = {
            "step": [app.current_step] * n
        }
        step_results.update(columns)
        return step_results
