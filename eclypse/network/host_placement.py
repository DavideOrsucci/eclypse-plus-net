"""A placement strategy wrapper that places services only on hosts.

In a :class:`~eclypse.network.Network` only hosts run services: routers forward
packets, and the traffic of a service placed on a router is discarded. Some
strategies of ECLYPSE choose among all the nodes without checking any requirement
(:class:`~eclypse.placement.strategies.RandomStrategy`,
:class:`~eclypse.placement.strategies.RoundRobinStrategy`), so they can pick a
router. :class:`HostOnlyStrategy` wraps any strategy and lets it choose only among
the hosts.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import networkx as nx

from eclypse.placement.strategies import PlacementStrategy

if TYPE_CHECKING:
    from eclypse.graph import (
        Application,
        Infrastructure,
    )
    from eclypse.placement import (
        Placement,
        PlacementView,
    )


class _HostView:
    """A placement view whose residual graph contains only the eligible hosts.

    The residual graph keeps the residual resources of the hosts (a copy of their
    attributes) and no links: placement strategies choose among nodes. Every other
    attribute is read from the wrapped view.
    """

    def __init__(self, view: PlacementView, residual: Any):
        self._view = view
        self.residual = residual

    def __getattr__(self, name: str) -> Any:
        return getattr(self._view, name)


class HostOnlyStrategy(PlacementStrategy):
    """Wrap a placement strategy so that it places services only on hosts.

    The wrapped strategy receives a placement view restricted to the hosts that are
    available (``availability > 0``), so it cannot choose a router. Strategies that
    do not use the view, such as a
    :class:`~eclypse.placement.strategies.StaticStrategy`, are checked afterwards:
    services mapped to a router are left unplaced, with a warning: ECLYPSE then
    treats the placement as partial, discards it and tries again at the next step.

    Example::

        sim.register(app, placement_strategy=HostOnlyStrategy(RandomStrategy(seed=42)))
    """

    def __init__(self, strategy: PlacementStrategy):
        """Initialize the wrapper.

        Args:
            strategy (PlacementStrategy): The strategy that chooses the hosts.
        """
        self.strategy = strategy

    def place(
        self,
        infrastructure: Infrastructure,
        application: Application,
        placements: dict[str, Placement],
        placement_view: PlacementView,
    ) -> dict[str, str]:
        """Place the services of an application with the wrapped strategy, on hosts.

        Args:
            infrastructure (Infrastructure): The infrastructure to place onto.
            application (Application): The application to place.
            placements (dict[str, Placement]): The placements of all the applications.
            placement_view (PlacementView): The current state of the infrastructure.

        Returns:
            dict[str, str]: The mapping of the services onto hosts. Services the
                wrapped strategy mapped to a router are missing.
        """
        residual = placement_view.residual
        hosts = [
            node
            for node, attrs in residual.nodes(data=True)
            if attrs.get("role") != "router" and attrs.get("availability", 1) > 0
        ]
        host_residual = nx.DiGraph()
        host_residual.add_nodes_from((node, residual.nodes[node]) for node in hosts)
        view = _HostView(placement_view, host_residual)
        mapping = self.strategy.place(
            infrastructure,
            application,
            placements,
            view,  # type: ignore[arg-type]
        )

        on_routers = {
            service: node
            for service, node in mapping.items()
            if node is not None
            and node in residual
            and residual.nodes[node].get("role") == "router"
        }
        if on_routers:
            infrastructure.logger.warning(
                f"Services not placed, mapped to a router: {on_routers}. "
                "Only hosts can run services."
            )
        return {s: n for s, n in mapping.items() if s not in on_routers}

    def is_feasible(self, infrastructure: Infrastructure, application: Application):
        """Return whether the wrapped strategy can place the application."""
        return self.strategy.is_feasible(infrastructure, application)
