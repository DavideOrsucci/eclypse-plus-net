# Base ECLYPSE vs ECLYPSE+NET

A comparison of the execution time of base ECLYPSE and of ECLYPSE with the Network
extension, on the same scenarios, changed at every step by the same update policies.

| File | Content |
|---|---|
| `scenario.py` | Network, application and update policies of a scenario, and their copies for base ECLYPSE |
| `experiment.py` | One simulation in one mode, with every phase of every step timed |
| `main.py` | Runs every scenario in both modes and saves `results.csv` and `steps.csv` |
| `analyse.ipynb` | Charts of the results |

## The scenario

The network is generated with one of the Network builders (`get_network_internet_as`,
`get_network_scale_free`, `get_network_random`). The application has one service per
host of the largest connected component, placed on its host, and `flows` random flows
between them that share `packets_per_step` packets per step.

For base ECLYPSE, network and application are copied into a plain `Infrastructure` and a
plain `Application`, with the same nodes, links, attributes, order and seed. Links also
get a fixed `latency` (the delay of a packet on the idle link), which base ECLYPSE uses
as the cost of its paths. Base ECLYPSE keeps the packet size and rate of the flows, but
does not use them.

Both versions get the same update policies:

| Graph | Policy | Effect |
|---|---|---|
| Infrastructure | `policies.distribution.uniform` on `bandwidth_mbps` | At every step, the bandwidth of every link is redrawn between 0.8 and 1.2 times its initial value |
| Infrastructure | `LinkFailures` (in `scenario.py`) | Every 20 steps, 2% of the links between routers go down for 5 steps (topology change) |
| Application | `policies.workload.diurnal_load` on `avg_packets_per_step` | The packet rate of every flow follows a daily cycle of 50 steps, between 0.5 and 1.5 times its initial value |
| Application | `policies.every(10, policies.distribution.uniform)` on `packet_size_bytes` | Every 10 steps, the packet size of every flow is redrawn between 0.5 and 1 times its initial value |

The policies draw from the random generator of their graph, and the copies have the same
seed and order, so they change both versions in the same way: `analyse.ipynb` checks it
on the final state of the graphs (columns `end_*` of `results.csv`). Base ECLYPSE
ignores the bandwidth, rate and size changes, while ECLYPSE+NET uses them to forward the
packets.

## What is timed

Each step runs these events, and each is timed:

| Phase | Event | Base | NET |
|---|---|---|---|
| `enact` | placement check (`audit`) and enactment | yes | yes |
| `policies` | `step`: update policies of application and infrastructure | yes | yes |
| `generation` | `PacketGenerationEvent` | no | yes |
| `routing` | `RoutingEvent`: forwarding of the packets by one hop | no | yes |

`generation` and `routing` are the cost of the packet-level model, which base ECLYPSE
does not have. `enact` and `policies` exist in both modes, but they can cost more with
ECLYPSE+NET: the routing event updates the `latency` of the links at every step, and the
placement check computes its paths with the `Network` path algorithm. `other` is the
rest of the simulation time (scheduling of the events).

## Running

From this folder:

```bash
uv run python main.py --smoke                 # quick check: 32 routers, 20 steps
uv run python main.py                         # whole grid, one simulation at a time
uv run python main.py --routers 64 128 --seed 1
```

Options: `--workers` (simulations in parallel, default 1: more workers finish sooner
but make the times noisier), `--output` (folder of the results, default
`~/eclypse-sim/eclypse_comparison`), and `--topology`, `--routers`, `--seed` to replace
the values of the grid in `main.py`. The parameters shared by all the simulations are
in `FIXED`, in `main.py`.

Every simulation runs in a new process. Its ECLYPSE outputs are written in
`<output>/trials/<topology>_<routers>_<seed>_<mode>/`.

## Outputs

`results.csv` has one row per simulation, with its parameters (`mode`, `topology`,
`routers`, `seed`) and:

- **scenario**: `n_routers`, `n_hosts`, `n_links`, `n_services`, `n_flows`;
- **execution time**: `build_time_s` (scenario, including the copies for base ECLYPSE),
  `sim_time_s` (simulation), `enact_time_s`, `policies_time_s`, `generation_time_s`,
  `routing_time_s`, `other_time_s`, `steps_run`;
- **policies**: `link_failures`, and the final state `end_links`, `end_bandwidth_sum`,
  `end_rate_sum`, `end_size_sum`, equal in the two modes;
- **traffic**, ECLYPSE+NET only: `generated`, `hops`, `dropped`, `in_transit_at_end`.

`steps.csv` has one row per step of every simulation, with the duration of each phase
(`enact_s`, `policies_s`, `generation_s`, `routing_s`), the links down after the
policies (`links_down`), and, for ECLYPSE+NET, the packets generated and the hops
simulated in the step.

## Step duration

The simulations use steps of 10 ms (`step_duration_s` in `main.py`). A packet moves by
at most one hop per step, and only at the start of a step, so a packet that reaches a
router waits for the next step: when the delay of a hop is much shorter than a step, a
packet takes about `hops - 1` steps to cross the network, and its end-to-end latency is
about `(hops - 1) * 10 ms`, mostly the time resolution of the simulation (see the
comparison with ns-3 in `../ns3_comparison`).

This comparison measures execution times, not latencies, so its results stay valid.
The step duration still shapes the work of the routing event: it decides how many
packets are in transit at the same time and how the queues fill. Results obtained with
different step durations should not be compared directly.
