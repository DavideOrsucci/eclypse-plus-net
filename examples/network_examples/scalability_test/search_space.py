"""Search space of the Network benchmark run with Ray Tune.

Every combination of the values given with ``tune.grid_search`` is one trial: one
simulation on a freshly generated network. The other entries are the same for all the
trials.
"""

from ray import tune

search_space = {
    # Varied parameters
    "topology": tune.grid_search(["internet_as", "scale_free", "random"]),
    "routers": tune.grid_search([128, 256, 512]),
    "packets_per_step": tune.grid_search([100, 500, 1000, 2000]),
    "seed": tune.grid_search([3997, 151195, 300425]),
    # Traffic: `flows` random host pairs share `packets_per_step` packets per step
    "flows": 100,
    "packet_size_bytes": 1500,
    # Simulation: `steps` steps of `step_duration_s` seconds
    "steps": 100,
    "step_duration_s": 0.01,
    # Network
    "max_queue_size": 100,
    "ba_m": 2,  # links of each new router in the Barabasi-Albert model
    "er_degree_factor": 1.5,  # Erdos-Renyi p = er_degree_factor * ln(n) / n
    # Whether to also write the per-hop telemetry of RoutingMetric (large files)
    "telemetry": False,
}
