"""Run the ns-3 program ``eclypse_scenario`` on a scenario exported by ECLYPSE."""

from __future__ import annotations

import csv
import subprocess
import time
from pathlib import Path

PROGRAM = "eclypse_scenario"
"""Name of the ns-3 program, built from ``eclypse_scenario.cc`` in ``scratch/``."""


def find_program(ns3_dir: Path) -> Path:
    """Return the executable of :data:`PROGRAM` built in an ns-3 folder.

    Args:
        ns3_dir (Path): The ns-3 folder (the one with the ``ns3`` script).

    Returns:
        Path: The executable, in ``build/scratch/``.

    Raises:
        FileNotFoundError: If the program has not been built.
    """
    candidates = sorted((Path(ns3_dir) / "build" / "scratch").glob(f"*{PROGRAM}*"))
    executables = [p for p in candidates if p.is_file() and p.stat().st_mode & 0o111]
    if not executables:
        raise FileNotFoundError(
            f"{PROGRAM} is not built in {ns3_dir}: copy {PROGRAM}.cc into "
            f"{Path(ns3_dir) / 'scratch'} and run './ns3 build {PROGRAM}' there."
        )
    return executables[-1]


def run_ns3(executable: Path, scenario_dir: Path, output_dir: Path) -> dict:
    """Simulate an exported scenario with ns-3.

    Args:
        executable (Path): The executable returned by :func:`find_program`.
        scenario_dir (Path): The folder with the files of ``export_for_ns3``.
        output_dir (Path): The folder where ns-3 writes its results; it is created
            if needed.

    Returns:
        dict: ``received`` (the arrival time of every received packet, by ID),
            ``wall_time_s`` (duration of the whole process) and the values of
            ``stats.csv`` (``sent``, ``received_count``, ``queue_drops``,
            ``ip_drops``, ``setup_time_s``, ``run_time_s``, ...).

    Raises:
        subprocess.CalledProcessError: If the program fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    subprocess.run(
        [
            str(executable),
            f"--input={scenario_dir.resolve()}",
            f"--output={output_dir.resolve()}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wall_time = time.perf_counter() - start

    with (output_dir / "stats.csv").open() as f:
        stats = {row["key"]: float(row["value"]) for row in csv.DictReader(f)}
    stats["received_count"] = stats.pop("received")
    with (output_dir / "received.csv").open() as f:
        received = {int(row["id"]): float(row["time_s"]) for row in csv.DictReader(f)}
    return {**stats, "received": received, "wall_time_s": wall_time}
