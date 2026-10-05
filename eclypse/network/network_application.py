"""Module extending the ECLYPSE application model with network capabilities."""

import numpy as np

from eclypse.graph import Application

from .constants import (
    DEFAULT_AVG_PACKETS_PER_STEP,
    DEFAULT_PACKET_SIZE_BYTES,
)
from .network import Packet


class NetworkApplication(Application):
    """Extension of the standard ECLYPSE application class.

    Incorporates network traffic characteristics and stochastic
    packet generation logic.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the NetworkApplication.

        Sets up the internal state required for packet generation tracking,
        including sequential IDs and packet buffers.

        Args:
            *args: Variable length argument list passed to the parent class.
            **kwargs: Arbitrary keyword arguments passed to the parent class.
        """
        super().__init__(*args, **kwargs)
        # Internal counter to give unique IDs to generated packets
        self._packet_counter = 0
        self.current_step = 0
        self.generated_packets = []

    def add_edge(
        self,
        u_of_edge: str,
        v_of_edge: str,
        symmetric: bool = False,
        strict: bool = True,
        packet_size_bytes: int = DEFAULT_PACKET_SIZE_BYTES,
        avg_packets_per_step: float = DEFAULT_AVG_PACKETS_PER_STEP,
        **attr,
    ):
        """Add a logical flow edge to the application graph.

        Overrides the default add_edge method to automatically validate and inject
        traffic parameters (packet size and expected rate) required for the
        stochastic Poisson distribution generation.

        Args:
            u_of_edge (str): The source service ID of the logical flow.
            v_of_edge (str): The destination service ID of the logical flow.
            symmetric (bool): If True, adds the edge in both directions.\
                Defaults to False.
            strict (bool): If True, raises an error if the assets are inconsistent.\
                If False, logs a warning. Defaults to True.
            packet_size_bytes (int, optional): The fixed size in bytes for packets
                generated on this flow. Defaults to DEFAULT_PACKET_SIZE_BYTES.
            avg_packets_per_step (float, optional): The expected average number of
                packets generated per step (Lambda). Defaults to \
                DEFAULT_AVG_PACKETS_PER_STEP.
            **attr: Additional attributes to assign to the edge.

        Raises:
            ValueError: If packet_size_bytes is less than or equal to 0, or if
                avg_packets_per_step is less than 0.
        """
        if packet_size_bytes <= 0:
            raise ValueError(f"Packet size must be > 0. Found: {packet_size_bytes}")
        if avg_packets_per_step < 0:
            raise ValueError(f"Packet rate must be >= 0. Found: {avg_packets_per_step}")
        attr["packet_size_bytes"] = packet_size_bytes
        attr["avg_packets_per_step"] = avg_packets_per_step
        super().add_edge(
            u_of_edge, v_of_edge, symmetric=symmetric, strict=strict, **attr
        )

        self.logger.info(
            f"Added flow {u_of_edge}->{v_of_edge}, "
            f"{avg_packets_per_step} pkt/step (avg), "
            f"size {packet_size_bytes}B"
        )

    def generate_traffic_for_step(self, step: int) -> None:
        """Generate traffic based on the defined flows in the application graph.

        Reads the flow definitions stored in the edges and produces a list of packet
        objects for the current simulation step, drawing the quantity from a
        Poisson distribution.

        Args:
            step (int): The current simulation step number to stamp on packets.
        """
        self.generated_packets.clear()

        # Flow parameters are read live (update policies may change them)
        flows = [
            (
                u,
                v,
                data.get("avg_packets_per_step", DEFAULT_AVG_PACKETS_PER_STEP),
                data.get("packet_size_bytes", DEFAULT_PACKET_SIZE_BYTES),
            )
            for u, nbrs in self._adj.items()
            for v, data in nbrs.items()
        ]
        if not flows:
            return

        # One vectorized Poisson draw for all the flows. With the legacy numpy
        # generator this yields the same numbers as one scalar call per flow
        # (lam == 0 returns 0 without consuming randomness).
        counts = np.random.poisson([f[2] for f in flows]).tolist()

        packets = self.generated_packets
        counter = self._packet_counter
        for (u, v, _, size), n_packets in zip(flows, counts, strict=True):
            for _ in range(n_packets):
                counter += 1
                packets.append(
                    Packet(id=counter, src=u, dst=v, size=size, step_created=step)
                )
        self._packet_counter = counter
