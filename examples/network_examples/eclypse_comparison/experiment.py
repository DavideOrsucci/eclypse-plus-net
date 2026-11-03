"""One trial of the comparison: simulate a scenario with base ECLYPSE or ECLYPSE+NET.

Both modes simulate the same infrastructure and application, changed at every step
by the same update policies (see ``scenario.py``), with the same static placement.
The events of a step are:

==============  ======================================  ======  ===
Phase           Event                                   base    net
==============  ======================================  ======  ===
``enact``       placement check and enactment            yes    yes
``policies``    update policies (``step`` event)         yes    yes
``generation``  packet generation                        no     yes
``routing``     packet forwarding by one hop             no     yes
==============  ======================================  ======  ===

Base ECLYPSE neither generates packets nor routes them, so its simulation runs only
the first two phases. Every phase is timed at every step, so the cost of each phase
can be compared between the two modes, and the cost of the packet-level model can
be told apart from the cost of the rest of the simulation.

Inside the ``enact`` phase, two methods of ECLYPSE are also measured: the placement
check (``PlacementManager.audit``) and the computation of a path
(``Infrastructure._compute_path``), which ECLYPSE repeats whenever its path cache has
been invalidated. Without update policies (``policies`` set to False), base ECLYPSE
keeps its cached paths, while ECLYPSE+NET writes the ``latency`` of the links at
every step, which invalidates them: the count of path computations shows the cost of
this integration.
"""

from __future__ import annotations

import random
import shutil
import time
from typing import TYPE_CHECKING

import numpy as np
from scenario import (
    build_application,
    build_network,
    to_base_application,
    to_base_infrastructure,
)

from eclypse.graph import Infrastructure
from eclypse.network import (
    PacketGenerationEvent,
    RoutingEvent,
)
from eclypse.placement._manager import PlacementManager
from eclypse.placement.strategies import StaticStrategy
from eclypse.simulation import (
    Simulation,
    SimulationConfig,
)
from eclypse.workflow.event.defaults import (
    EnactEvent,
    StepEvent,
)

if TYPE_CHECKING:
    from pathlib import Path

    from scenario import LinkFailures


MODES = ("base", "net")
"""Simulation modes: base ECLYPSE and ECLYPSE+NET."""

PHASES = ("enact", "policies", "generation", "routing")
"""Timed phases of a step, in execution order."""

FRAMEWORK = {"audit_s": 0.0, "audit_calls": 0, "path_s": 0.0, "path_calls": 0}
"""Time spent in, and calls of, two methods of ECLYPSE (see :func:`instrument`)."""


def instrument():
    """Measure ``PlacementManager.audit`` and ``Infrastructure._compute_path``.

    The two methods are wrapped on their class, so every call adds its duration and
    one call to :data:`FRAMEWORK`; the behaviour is unchanged. Each trial runs in a
    process of its own, so the wrappers do not affect other trials. Calling this
    function again resets the counters without wrapping the methods twice.
    """
    FRAMEWORK.update(audit_s=0.0, audit_calls=0, path_s=0.0, path_calls=0)
    for cls, method, key in (
        (PlacementManager, "audit", "audit"),
        (Infrastructure, "_compute_path", "path"),
    ):
        original = getattr(cls, method)
        if getattr(original, "_measured", False):
            continue

        def wrapper(self, *args, _original=original, _key=key, **kwargs):
            start = time.perf_counter()
            try:
                return _original(self, *args, **kwargs)
            finally:
                FRAMEWORK[f"{_key}_s"] += time.perf_counter() - start
                FRAMEWORK[f"{_key}_calls"] += 1

        wrapper._measured = True  # type: ignore[attr-defined]
        setattr(cls, method, wrapper)


def timed(event_class: type) -> type:
    """Return a subclass of an event class that times every call.

    The durations, in seconds, are appended to the ``durations`` attribute of the
    event. The event keeps the name of its base class, so it replaces the default
    event of ECLYPSE with the same name.

    Args:
        event_class (type): The event class to time.

    Returns:
        type: The timed subclass.
    """

    class Timed(event_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.durations: list[float] = []

        def __call__(self, *args, **kwargs):
            start = time.perf_counter()
            out = super().__call__(*args, **kwargs)
            self.durations.append(time.perf_counter() - start)
            return out

    Timed.__name__ = Timed.__qualname__ = f"Timed{event_class.__name__}"
    return Timed


TimedGeneration = timed(PacketGenerationEvent)


class TimedEnact(timed(EnactEvent)):
    """Timed ``enact`` event that also records the path computations of each step."""

    def __init__(self):
        """Initialize the event."""
        super().__init__()
        self.path_calls: list[int] = []

    def __call__(self, *args, **kwargs):
        """Check and enact the placement, and record the paths computed meanwhile."""
        before = FRAMEWORK["path_calls"]
        out = super().__call__(*args, **kwargs)
        self.path_calls.append(FRAMEWORK["path_calls"] - before)
        return out


class TimedPolicies(timed(StepEvent)):
    """Timed ``step`` event that also records the links down after each step."""

    def __init__(self, failures: LinkFailures | None):
        """Initialize the event.

        Args:
            failures (LinkFailures | None): The link-failure policy of the
                infrastructure, or None if the trial has no policies.
        """
        super().__init__()
        self.failures = failures
        self.links_down: list[int] = []

    def __call__(self, *args, **kwargs):
        """Apply the update policies and record the links down."""
        out = super().__call__(*args, **kwargs)
        self.links_down.append(self.failures.links_down if self.failures else 0)
        return out


class TimedRouting(timed(RoutingEvent)):
    """Timed routing event that also records the hops simulated at each step."""

    def __init__(self, step_duration_s: float):
        """Initialize the event."""
        super().__init__(step_duration_s=step_duration_s)
        self.hops: list[int] = []
        self.packets: list[int] = []

    def route(self, applications, infra, step=None):
        """Forward the packets of the step and record the packets and hops."""
        self.packets.append(sum(app.num_generated for app, _ in applications))
        super().route(applications, infra, step)
        self.hops.append(len(infra.step_columns["hop"]))


def fingerprint(infra: Infrastructure, app) -> dict:
    """Summarize the state left by the update policies.

    The two modes apply the same policies to the same graphs, so they must end
    with the same values.
    """
    return {
        "end_links": infra.number_of_edges(),
        "end_bandwidth_sum": float(
            sum(d["bandwidth_mbps"] for _, _, d in infra.edges(data=True))
        ),
        "end_rate_sum": float(
            sum(d["avg_packets_per_step"] for _, _, d in app.edges(data=True))
        ),
        "end_size_sum": int(
            sum(d["packet_size_bytes"] for _, _, d in app.edges(data=True))
        ),
    }


def run_trial(config: dict, output_dir: Path) -> tuple[dict, list[dict]]:
    """Run one simulation and return its metrics.

    Args:
        config (dict): The trial configuration (see ``main.py``); ``mode`` is
            ``"base"`` or ``"net"``.
        output_dir (Path): The folder where ECLYPSE writes its outputs. It is
            deleted first if it exists.

    Returns:
        tuple[dict, list[dict]]: The metrics of the trial, and one row per step
            with the duration of each phase.

    Raises:
        ValueError: If the mode is not one of :data:`MODES`.
    """
    mode = config["mode"]
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}.")
    random.seed(config["seed"])
    np.random.seed(config["seed"])

    start = time.perf_counter()
    net = build_network(config)
    app, mapping = build_application(net, config)
    infra = net
    if mode == "base":
        infra = to_base_infrastructure(net, config)
        app = to_base_application(app, config)
    build_time = time.perf_counter() - start
    scenario = {
        "n_routers": len(net.routers),
        "n_hosts": len(net.hosts),
        "n_links": net.number_of_edges() // 2,
        "n_services": app.number_of_nodes(),
        "n_flows": app.number_of_edges(),
    }

    # The LinkFailures policy is the last one of the infrastructure, if any
    failures = infra.update_policies[-1] if infra.update_policies else None
    instrument()
    enact, policies = TimedEnact(), TimedPolicies(failures)
    events = [enact, policies]
    generation = routing = None
    if mode == "net":
        generation = TimedGeneration()
        routing = TimedRouting(config["step_duration_s"])
        events += [generation, routing]

    shutil.rmtree(output_dir, ignore_errors=True)
    sim_config = SimulationConfig(
        seed=config["seed"],
        max_steps=config["steps"],
        path=output_dir,
        events=events,
        include_default_metrics=False,
        log_level="ERROR",
        remote=False,
    )
    sim = Simulation(infra, simulation_config=sim_config)
    sim.register(app, placement_strategy=StaticStrategy(mapping))
    start = time.perf_counter()
    sim.start()
    sim.wait()
    sim_time = time.perf_counter() - start

    durations = {
        "enact": enact.durations,
        "policies": policies.durations,
        "generation": generation.durations if generation else [],
        "routing": routing.durations if routing else [],
    }
    totals = {f"{p}_time_s": float(sum(durations[p])) for p in PHASES}

    n_steps = len(policies.durations)
    steps = []
    for i in range(n_steps):
        row = {"step": i + 1, "links_down": policies.links_down[i]}
        for p in PHASES:
            row[f"{p}_s"] = durations[p][i] if i < len(durations[p]) else 0.0
        row["path_calls"] = enact.path_calls[i] if i < len(enact.path_calls) else 0
        row["packets"] = routing.packets[i] if routing else 0
        row["hops"] = routing.hops[i] if routing else 0
        steps.append(row)

    metrics = {
        **scenario,
        # Execution time
        "build_time_s": build_time,
        "sim_time_s": sim_time,
        **totals,
        "other_time_s": sim_time - sum(totals.values()),
        "steps_run": n_steps,
        # Inside the enact phase: placement checks and path computations
        "audit_time_s": FRAMEWORK["audit_s"],
        "path_time_s": FRAMEWORK["path_s"],
        "path_calls": FRAMEWORK["path_calls"],
        # Policies
        "link_failures": failures.failures if failures else 0,
        **fingerprint(infra, app),
        # Traffic (ECLYPSE+NET only)
        "generated": int(sum(routing.packets)) if routing else 0,
        "hops": int(sum(routing.hops)) if routing else 0,
        "dropped": net.dropped_packets if mode == "net" else 0,
        "in_transit_at_end": len(net.in_transit) if mode == "net" else 0,
    }
    return metrics, steps
