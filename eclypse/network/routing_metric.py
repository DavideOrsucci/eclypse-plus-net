"""Metric that reports the per-hop telemetry of the network traffic."""

from eclypse.report.metrics.metric import application

from .network import Network
from .network_application import NetworkApplication


@application(name="traffic_routing", activates_on="step")
class RoutingMetric:
    """Application metric that reports the telemetry of every hop of the step.

    The metric, named ``traffic_routing``, is collected at every step for every
    application. It reports one row per packet of the application forwarded or
    dropped in the step, with the columns listed in
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
                with one entry per hop of a packet of the application, or None if
                no such packet was forwarded in the step.
        """
        columns = infra.step_columns
        if not columns["hop"]:
            return None

        if len(infra.application_ids) > 1:
            # Several applications share the network: keep the rows of this one
            ids = infra.application_ids
            index = ids.index(app.id) if app.id in ids else -1
            mask = [a == index for a in infra.step_apps]
            if not any(mask):
                return None
            columns = {
                name: [v for v, keep in zip(values, mask, strict=True) if keep]
                for name, values in columns.items()
            }

        # With a single application the columns are returned without copying: the
        # network allocates new lists at the start of every step.
        step_results: dict[str, list[int | float | str | bool]] = {
            "step": [app.current_step] * len(columns["hop"])
        }
        step_results.update(columns)
        return step_results
