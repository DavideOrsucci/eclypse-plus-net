"""Run the ns-3 program ``eclypse_scenario`` on a scenario exported by ECLYPSE."""

from __future__ import annotations

import csv
import subprocess
import time
from pathlib import Path

PROGRAM = "eclypse_scenario"
"""Name of the ns-3 program, built from ``eclypse_scenario.cc`` in ``scratch/``."""

STDERR_LINES = 30
"""Lines of the standard error of a failed ns-3 run reported in the error."""


class Ns3Error(RuntimeError):
    """Error raised when the ns-3 program fails, with the end of its standard error.

    Attributes:
        command (list[str]): The command that was run.
        returncode (int): The exit code of the program (negative if it was killed by
            a signal).
        stderr (str): The whole standard error of the program.
    """

    def __init__(self, command: list[str], returncode: int, stderr: str):
        """Initialize the error.

        Args:
            command (list[str]): The command that was run.
            returncode (int): The exit code of the program.
            stderr (str): The standard error of the program.
        """
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        status = (
            f"was killed by signal {-returncode}"
            if returncode < 0
            else f"exited with code {returncode}"
        )
        lines = stderr.strip().splitlines()
        tail = "\n".join(lines[-STDERR_LINES:]) if lines else "(empty)"
        skipped = (
            f" (last {STDERR_LINES} of {len(lines)} lines)"
            if len(lines) > STDERR_LINES
            else ""
        )
        super().__init__(
            f"ns-3 {status}: {' '.join(command)}\nstderr{skipped}:\n{tail}"
        )

    def __reduce__(self):
        """Rebuild the error from its arguments, e.g. when it crosses processes."""
        return (self.__class__, (self.command, self.returncode, self.stderr))


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
        Ns3Error: If the program fails; the error includes the end of its standard
            error.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(executable),
        f"--input={scenario_dir.resolve()}",
        f"--output={output_dir.resolve()}",
    ]
    start = time.perf_counter()
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    wall_time = time.perf_counter() - start
    if result.returncode != 0:
        raise Ns3Error(command, result.returncode, result.stderr)

    with (output_dir / "stats.csv").open() as f:
        stats = {row["key"]: float(row["value"]) for row in csv.DictReader(f)}
    stats["received_count"] = stats.pop("received")
    with (output_dir / "received.csv").open() as f:
        received = {int(row["id"]): float(row["time_s"]) for row in csv.DictReader(f)}
    return {**stats, "received": received, "wall_time_s": wall_time}
