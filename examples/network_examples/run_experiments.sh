#!/usr/bin/env bash
# Run the experiments of examples/network_examples.
#
# Usage: ./run_experiments.sh [options] <command>... [-- <options of main.py>]
#
# Experiments:
#   scalability  scalability_test: Ray Tune grid over topology, size and load
#   vs-eclypse   eclypse_comparison: base ECLYPSE vs ECLYPSE+NET
#   vs-ns3       ns3_comparison: ECLYPSE+NET vs ns-3 (needs ns3-setup)
#   stress       stress_test.py on Fat-Trees, pinned to one CPU core
#   all          stress, scalability, vs-eclypse and vs-ns3
#
# Preparation:
#   setup        install uv if missing and the dependencies of the project (with Ray Tune)
#   ns3-setup    download ns-3 and build the program of ns3_comparison
#   smoke        quick check of scalability, vs-eclypse and vs-ns3 (same as --smoke)
#
# Options:
#   --smoke                run the quick version of scalability, vs-eclypse and vs-ns3
#   -w, --workers N        simulations in parallel in vs-eclypse (default 1: cleanest times)
#   -c, --cpus-per-trial N CPUs reserved for each Ray trial in scalability (default 2)
#   --core N               CPU core used by stress (default: 2, or the first one available)
#   --ns3-dir DIR          ns-3 folder (default ~/ns-3-dev)
#   --performance          set the CPU governor to `performance` during the run (sudo)
#   -n, --dry-run          print the commands without running them
#   -h, --help             show this help
#   -- ARGS                pass ARGS to main.py of vs-eclypse or vs-ns3 (one experiment
#                          only), e.g. -- --routers 64 128 --seed 1
#
# Every experiment writes its log to ~/eclypse-sim/logs/<date>_<experiment>.log; the
# results stay where the experiments write them (~/eclypse-sim/...). If an experiment
# fails, the following ones still run, and the summary at the end lists the failures.
# For long runs over SSH, start the script with nohup, so that it survives a disconnect.
#
# Examples:
#   ./run_experiments.sh setup ns3-setup smoke
#   ./run_experiments.sh scalability
#   ./run_experiments.sh --smoke vs-ns3
#   ./run_experiments.sh -w 4 vs-eclypse
#   ./run_experiments.sh vs-ns3 -- --routers 16 32 --step-duration-s 0.001
#   nohup ./run_experiments.sh all > run.out 2>&1 &

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SIM_DIR="$HOME/eclypse-sim"
LOG_DIR="$SIM_DIR/logs"
NS3_VERSION="ns-3.49"

SMOKE=0
WORKERS=1
CPUS_PER_TRIAL=2
CORE=""
NS3_DIR="$HOME/ns-3-dev"
PERFORMANCE=0
DRY_RUN=0
COMMANDS=()
EXTRA=()

usage() {
    sed -n '2,/^$/{s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"
}

die() {
    echo "Error: $*" >&2
    exit 2
}

# ---------------------------------------------------------------------- arguments

while (($#)); do
    case "$1" in
        --smoke) SMOKE=1 ;;
        -w | --workers) WORKERS="${2:?missing value of $1}"; shift ;;
        -c | --cpus-per-trial) CPUS_PER_TRIAL="${2:?missing value of $1}"; shift ;;
        --core) CORE="${2:?missing value of $1}"; shift ;;
        --ns3-dir) NS3_DIR="${2:?missing value of $1}"; shift ;;
        --performance) PERFORMANCE=1 ;;
        -n | --dry-run) DRY_RUN=1 ;;
        -h | --help) usage; exit 0 ;;
        --) shift; EXTRA=("$@"); break ;;
        setup | ns3-setup | stress | scalability | vs-eclypse | vs-ns3) COMMANDS+=("$1") ;;
        all) COMMANDS+=(stress scalability vs-eclypse vs-ns3) ;;
        smoke) SMOKE=1; COMMANDS+=(scalability vs-eclypse vs-ns3) ;;
        *) usage >&2; die "unknown argument '$1'" ;;
    esac
    shift
done
((${#COMMANDS[@]})) || { usage; exit 2; }
NS3_DIR="${NS3_DIR/#\~/$HOME}"

# Options after --: only for one experiment that has a command line
if ((${#EXTRA[@]})); then
    [[ ${#COMMANDS[@]} -eq 1 && ("${COMMANDS[0]}" == vs-eclypse || "${COMMANDS[0]}" == vs-ns3) ]] ||
        die "options after -- need exactly one command, vs-eclypse or vs-ns3" \
            "(scalability reads its grid from scalability_test/search_space.py)"
fi

# --performance: set the governor, and restore the previous one at the end
GOVERNOR_FILE=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
PREVIOUS_GOVERNOR=""
restore_governor() {
    [[ -n "$PREVIOUS_GOVERNOR" ]] || return 0
    if sudo -n cpupower frequency-set -g "$PREVIOUS_GOVERNOR" >/dev/null 2>&1; then
        echo "CPU governor restored to '$PREVIOUS_GOVERNOR'."
    else
        echo "Restore the CPU governor with: sudo cpupower frequency-set -g $PREVIOUS_GOVERNOR"
    fi
}
if ((PERFORMANCE)) && ((!DRY_RUN)); then
    command -v cpupower >/dev/null ||
        die "cpupower is missing (sudo apt install linux-tools-common linux-tools-\$(uname -r))"
    PREVIOUS_GOVERNOR="$(cat "$GOVERNOR_FILE" 2>/dev/null)"
    sudo cpupower frequency-set -g performance >/dev/null || die "cannot set the CPU governor"
    echo "CPU governor: performance (was '$PREVIOUS_GOVERNOR')."
fi
trap restore_governor EXIT

# ---------------------------------------------------------------------- helpers

RESULTS=()
FAILED=0

# Run a command in a folder, print it and save its output to a log file
run() {
    local name="$1" dir="$2"
    shift 2
    local log
    log="$LOG_DIR/$(date +%Y%m%d-%H%M%S)_${name}.log"
    echo
    echo "=== $name ($(date '+%H:%M:%S')) ==="
    echo "    cd $dir"
    printf '    '
    printf '%q ' "$@"
    echo
    if ((DRY_RUN)); then
        RESULTS+=("dry-run  $name")
        return 0
    fi
    mkdir -p "$LOG_DIR"
    echo "    log: $log"
    local start=$SECONDS status
    (cd "$dir" && "$@") 2>&1 | tee "$log"
    status=${PIPESTATUS[0]}
    local elapsed=$((SECONDS - start))
    local took
    took="$(printf '%dh%02dm%02ds' $((elapsed / 3600)) $((elapsed % 3600 / 60)) $((elapsed % 60)))"
    if ((status == 0)); then
        RESULTS+=("ok       $name  ($took)")
    else
        RESULTS+=("FAILED   $name  ($took, exit $status, log $log)")
        FAILED=1
    fi
    return "$status"
}

skip() {
    echo
    echo "=== $1: skipped: $2 ==="
    RESULTS+=("skipped  $1  ($2)")
    FAILED=1
}

require_uv() {
    ((DRY_RUN)) && return 0
    if ! command -v uv >/dev/null; then
        [[ -x "$HOME/.local/bin/uv" ]] && export PATH="$HOME/.local/bin:$PATH"
    fi
    command -v uv >/dev/null || die "uv is not installed: run '$0 setup' first"
}

ns3_program() {
    find "$NS3_DIR/build/scratch" -maxdepth 1 -type f -name '*eclypse_scenario*' -perm -u+x \
        2>/dev/null | head -n 1
}

# Name of a run: the quick versions end with -smoke
label() {
    if ((SMOKE)); then echo "$1-smoke"; else echo "$1"; fi
}

# ---------------------------------------------------------------------- commands

cmd_setup() {
    if ! command -v uv >/dev/null && [[ ! -x "$HOME/.local/bin/uv" ]]; then
        run install-uv "$REPO" sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
    fi
    require_uv
    run setup "$REPO" uv sync --group bench &&
        run check-imports "$REPO" uv run python -c \
            'import eclypse.network, ray; print("eclypse.network and ray: ok")'
}

cmd_ns3_setup() {
    local missing=()
    for tool in git g++ cmake python3; do
        command -v "$tool" >/dev/null || missing+=("$tool")
    done
    if ((${#missing[@]})) && ((!DRY_RUN)); then
        skip ns3-setup "missing ${missing[*]}: sudo apt install -y git g++ cmake ninja-build python3"
        return
    fi
    if [[ ! -d "$NS3_DIR" ]]; then
        run ns3-clone "$HOME" git clone --depth 1 --branch "$NS3_VERSION" \
            https://gitlab.com/nsnam/ns-3-dev.git "$NS3_DIR" || return
    fi
    run ns3-configure "$NS3_DIR" ./ns3 configure -d release \
        --enable-modules=point-to-point,internet &&
        run ns3-copy "$NS3_DIR" cp "$HERE/ns3_comparison/eclypse_scenario.cc" scratch/ &&
        run ns3-build "$NS3_DIR" ./ns3 build eclypse_scenario
}

cmd_stress() {
    require_uv
    local core="$CORE"
    if [[ -z "$core" ]] && command -v taskset >/dev/null; then
        # Core 2, or the first core this process may use
        core=2
        taskset -c "$core" true 2>/dev/null ||
            core="$(taskset -cp $$ | sed 's/.*: *//; s/[-,].*//')"
    fi
    if command -v taskset >/dev/null && taskset -c "$core" true 2>/dev/null; then
        run stress "$HERE" taskset -c "$core" uv run python stress_test.py
    else
        echo "Warning: cannot pin stress_test.py to core ${core:-?}; running it unpinned." >&2
        run stress "$HERE" uv run python stress_test.py
    fi
}

cmd_scalability() {
    require_uv
    if ((SMOKE)); then
        run "$(label scalability)" "$HERE/scalability_test" uv run python main.py --smoke
    else
        run scalability "$HERE/scalability_test" uv run python main.py \
            --cpus-per-trial "$CPUS_PER_TRIAL"
    fi
}

cmd_vs_eclypse() {
    require_uv
    local args=(--workers "$WORKERS")
    ((SMOKE)) && args+=(--smoke --output "$SIM_DIR/smoke/eclypse_comparison")
    run "$(label vs-eclypse)" "$HERE/eclypse_comparison" uv run python main.py \
        "${args[@]}" "${EXTRA[@]}"
}

cmd_vs_ns3() {
    require_uv
    if ((!DRY_RUN)) && [[ -z "$(ns3_program)" ]]; then
        skip "$(label vs-ns3)" "eclypse_scenario is not built in $NS3_DIR: run '$0 ns3-setup' first"
        return
    fi
    local args=(--ns3 "$NS3_DIR")
    ((SMOKE)) && args+=(--smoke --output "$SIM_DIR/smoke/ns3_comparison")
    run "$(label vs-ns3)" "$HERE/ns3_comparison" uv run python main.py \
        "${args[@]}" "${EXTRA[@]}"
}

# ---------------------------------------------------------------------- main

echo "Repository: $REPO"
if ((SMOKE)); then
    echo "Commands:   ${COMMANDS[*]} (quick versions)"
else
    echo "Commands:   ${COMMANDS[*]}"
fi
((DRY_RUN)) && echo "Dry run: nothing is executed."
start_all=$SECONDS

for command in "${COMMANDS[@]}"; do
    case "$command" in
        setup) cmd_setup ;;
        ns3-setup) cmd_ns3_setup ;;
        stress) cmd_stress ;;
        scalability) cmd_scalability ;;
        vs-eclypse) cmd_vs_eclypse ;;
        vs-ns3) cmd_vs_ns3 ;;
    esac
done

elapsed=$((SECONDS - start_all))
echo
echo "=== Summary ($((elapsed / 60)) min $((elapsed % 60)) s) ==="
printf '  %s\n' "${RESULTS[@]}"
echo "Results in $SIM_DIR, logs in $LOG_DIR"
exit "$FAILED"
