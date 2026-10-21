# ECLYPSE+NET vs ns-3

A comparison of ECLYPSE+NET with [ns-3](https://www.nsnam.org/), a packet-level
network simulator: both simulate the same networks, with the same packets on the same
paths, and every packet is matched by ID. The comparison measures how close the
end-to-end latencies and the losses of ECLYPSE+NET are to those of ns-3, how this
depends on the step duration, and how long each simulator takes.

| File | Content |
|---|---|
| `scenario.py` | Network and traffic of a scenario, and their export for ns-3 |
| `eclypse_run.py` | Simulation with ECLYPSE+NET, recording every injected, delivered and dropped packet |
| `eclypse_scenario.cc` | ns-3 program that simulates an exported scenario |
| `ns3_run.py` | Runs the ns-3 program and reads its results |
| `main.py` | Runs the grid and saves `results.csv` and the packet-by-packet comparison of every trial |
| `analyse.ipynb` | Charts of the results |

## How the two simulations are aligned

ECLYPSE+NET simulates the scenario first. The packets it generates, the topology and
its forwarding tables are then exported (`export_for_ns3`) and read by ns-3:

| ECLYPSE+NET | ns-3 |
|---|---|
| link with `bandwidth_mbps`, `length_km`, `propagation_speed_km_s` | point-to-point channel: data rate of each direction, delay = length / speed |
| FIFO DropTail queue of `max_queue_size` packets, counting the packet in transmission | DropTail queue of `max_queue_size - 1` packets on the device, which transmits one packet outside its queue; traffic control layer removed |
| minimum-cost paths of `Network` | static host routes towards the same next hops |
| hosts do not forward transit traffic | `IpForward` disabled on hosts |
| packet of `size` bytes | UDP payload of `size - 30` bytes, carrying the packet ID: with the UDP, IPv4 and PPP headers, `size` bytes on the links |
| packet injected at step `k` | packet sent at `k * step_duration_s` |
| `processing_time` | set to zero on every node, since ns-3 has no per-node processing delay |

What remains different is the model of time. ns-3 is event-driven: a packet is
forwarded as soon as it reaches a router. ECLYPSE+NET moves packets at the start of
every step: a packet that reaches a router waits for the next step, so each hop can
add up to one step duration to its latency, and the queues see the packets of a step
at the same time. The grid repeats every scenario with step durations of 10, 1 and
0.1 ms, to show how the difference shrinks with the step.

## Setting up ns-3

ns-3 is needed only here. It runs on Linux and macOS; on Windows use WSL. Download
and build it (the `internet` and `point-to-point` modules are enough), then build the
program of this folder in its `scratch` folder:

```bash
git clone --depth 1 --branch ns-3.49 https://gitlab.com/nsnam/ns-3-dev.git ~/ns-3-dev
cd ~/ns-3-dev
./ns3 configure -d release --enable-modules=point-to-point,internet
cp /path/to/eclypse-plus-net/examples/network_examples/ns3_comparison/eclypse_scenario.cc scratch/
./ns3 build eclypse_scenario
```

The `release` profile is needed for meaningful execution times. After changing
`eclypse_scenario.cc`, copy it again and rebuild.

## Running

From this folder:

```bash
uv run python main.py --ns3 ~/ns-3-dev --smoke   # quick check: 16 routers
uv run python main.py --ns3 ~/ns-3-dev           # whole grid
uv run python main.py --ns3 ~/ns-3-dev --routers 16 32 --step-duration-s 0.001
```

Options: `--output` (folder of the results, default `~/eclypse-sim/ns3_comparison`),
and `--topology`, `--routers`, `--step-duration-s`, `--packets-per-s`, `--seed` to
replace the values of the grid in `main.py`. The parameters shared by all the trials
are in `FIXED`, in `main.py`. Trials run one at a time, each in a new process.

## Outputs

`results.csv` has one row per trial, with its parameters and:

- **scenario**: `n_routers`, `n_hosts`, `n_links`, `n_flows`, `steps`;
- **packets**: `sent`, `eclypse_delivered`, `eclypse_dropped`,
  `eclypse_in_transit_at_end`, `ns3_delivered`, `ns3_dropped`, `both_delivered`,
  `hops_mean` (links traversed by the delivered packets);
- **latency** (end-to-end, from sending to arrival, in ms): mean and percentiles in
  each simulator (`eclypse_latency_ms_*`, `ns3_latency_ms_*`), and the difference
  between the two on the packets delivered by both (`latency_diff_mean_ms`, signed,
  and `latency_absdiff_mean_ms`, `latency_absdiff_p95_ms`, `latency_absdiff_max_ms`);
- **execution time**: `eclypse_build_time_s` (network and application),
  `eclypse_sim_time_s` (simulation), `export_time_s`, `ns3_setup_time_s` (building
  the ns-3 scenario), `ns3_run_time_s` (`Simulator::Run`), `ns3_wall_time_s` (the
  whole ns-3 process).

Every trial folder, `trials/<topology>_<routers>_<step>_<load>_<seed>/`, has the
exported scenario (`nodes.csv`, `links.csv`, `routes.csv`, `packets.csv`), the
outputs of ns-3 (`ns3/received.csv`, `ns3/stats.csv`) and `comparison.csv`, one row
per packet: `id`, `src`, `dst`, `send_time_s`, `eclypse_latency_ms`, `ns3_latency_ms`
(empty if not delivered), `hops` and `eclypse_dropped_on` (the link where ECLYPSE
dropped it).
