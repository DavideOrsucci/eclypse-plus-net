"""A fault injection event that simulates a link failure in the network."""

from eclypse.workflow.event import once_at

from eclypse.network import Network


@once_at(step=50, event_type="infrastructure", name="fault_injection")
def FaultInjectionEvent(infr: Network, _placement_view, **_kwargs):
    """Event that fires EXACTLY at step 50.

    Removes the link and forces an OSPF recalculation.
    """
    infr.logger.warning("step 50: Gateway->R1 link broken! Routes recalculated.\n")
    infr.remove_edge("Gateway", "R1")
