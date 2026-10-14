# Network benchmark with Ray Tune

A grid search that simulates traffic on networks of different topology, size and load,
and measures the execution time and the behaviour of the packet-level model. Each
combination of parameters is one Ray Tune trial, so the trials run in parallel on all
the cores of a machine or on a Ray cluster.

| File | Content |
|---|---|
| `search_space.py` | The grid: topologies, numbers of routers, packets per step, seeds, and the fixed parameters |
| `experiment.py` | One trial, without Ray: builds the network and the application, runs the simulation, returns the metrics |
| `main.py` | Runs the grid with Ray Tune and saves `results.csv` |
| `merge_results.py` | Rebuilds `results.csv` from the trial folders, e.g. after an interrupted run |
| `analyse.ipynb` | Charts of the results |

## Running

Ray Tune is needed only here; install it in the environment of the project:

```bash
uv add --group bench "ray[tune]>=2.59.0"
```

Then, from this folder:

```bash
uv run python main.py --smoke          # quick check, without Ray
uv run python main.py                  # whole grid on a local Ray instance
uv run python main.py --address auto   # whole grid on an existing Ray cluster
```

Options: `--cpus-per-trial` (CPUs reserved for each trial, default 1), `--storage`
(where Ray Tune stores the experiment, default `~/eclypse-sim/ray`) and `--name`
(experiment name, default `network_grid`).

## Outputs

Ray Tune creates one folder per trial in `<storage>/<name>/`, with the configuration of
the trial (`params.json`), its metrics (`result.json`) and the ECLYPSE outputs
(`output/`). At the end, `main.py` saves the metrics of all the trials in
`<storage>/<name>/results.csv`, one row per trial, with the parameters in the columns
`config/<parameter>`.

The metrics of a trial are:

- **network**: `n_routers`, `n_hosts`, `n_links` (all bidirectional links),
  `mean_router_degree`, `max_router_degree`, `reachable_host_share` (hosts in the
  largest connected component, the only ones that exchange traffic), `n_flows`;
- **execution time**: `build_time_s` (network and application), `sim_time_s`
  (simulation), `routing_time_s` and `routing_first_step_s` (time spent in the routing
  event; the first step also builds the forwarding tables), `hops` (simulated hops),
  `hops_per_s` (hops per second of routing time, first step excluded);
- **packets**: `generated`, `delivered`, `dropped` (queue full), `in_transit_at_end`
  (still travelling when the simulation ends), `delivery_ratio`;
- **delivered packets**: `latency_mean_ms`, `latency_p50_ms`, `latency_p95_ms`
  (end-to-end, from generation to arrival) and `path_hops_mean`.

The per-hop telemetry of `RoutingMetric` is not written by default, since it produces
one row per hop; set `"telemetry": True` in `search_space.py` to write it.
