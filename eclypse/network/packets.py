"""Packets of the network, stored in batches as a Structure of Arrays.

:class:`PacketBatch` stores a set of packets as one numpy array per field, named in
:data:`PACKET_FIELDS`. Batches are what :class:`~eclypse.network.Network` forwards
at every step and keeps in transit between steps.
"""

from __future__ import annotations

import numpy as np

PACKET_FIELDS = (
    "id",
    "src",
    "dst",
    "size",
    "step",
    "app",
    "cur",
    "prev",
    "hop",
    "arrival",
    "dst_service",
)
"""Names of the fields of a :class:`PacketBatch`."""

PACKET_DTYPES = {**dict.fromkeys(PACKET_FIELDS, np.int64), "arrival": np.float64}
"""Numpy data type of each field of a :class:`PacketBatch`."""


class PacketBatch:
    """A set of packets stored as a Structure of Arrays.

    Each field is a one-dimensional numpy array, of the type given in
    :data:`PACKET_DTYPES`, and position ``i`` of every array describes the same
    packet. Node references are integer indices assigned by
    :meth:`Network.node_index`. The order of the packets is meaningful: it is the
    order in which they are processed.

    Attributes:
        id (np.ndarray): Unique identifier of each packet.
        src (np.ndarray): Index of the node that generated the packet.
        dst (np.ndarray): Index of the destination node.
        size (np.ndarray): Packet size, in bytes.
        step (np.ndarray): Simulation step in which the packet was created.
        app (np.ndarray): Index of the application that generated the packet, as
            returned by :meth:`Network.application_index`.
        cur (np.ndarray): Index of the node where the packet currently is.
        prev (np.ndarray): Index of the node the packet came from, or ``-1`` if it
            has just been injected by its source.
        hop (np.ndarray): Number of hops traversed so far.
        arrival (np.ndarray): Time, in seconds, at which the packet reaches the node
            ``cur``: its generation time for a packet just injected by its source,
            and the arrival time at the end of the last hop otherwise. A packet is
            forwarded only once the current time has reached its arrival time.
        dst_service (np.ndarray): Index of the destination service, as returned by
            :meth:`Network.service_index`, or ``-1`` if the packet is not
            addressed to a service. When a packet reaches its destination node, the
            network checks that the service still runs there (see
            :attr:`Network.misdelivered_packets`).
    """

    __slots__ = PACKET_FIELDS

    id: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    size: np.ndarray
    step: np.ndarray
    app: np.ndarray
    cur: np.ndarray
    prev: np.ndarray
    hop: np.ndarray
    arrival: np.ndarray
    dst_service: np.ndarray

    def __init__(self, **fields: np.ndarray):
        """Create a batch from its field arrays.

        Args:
            **fields (np.ndarray): One array per name in :data:`PACKET_FIELDS`, all
                with the same length. ``dst_service`` can be omitted: the packets
                are then not addressed to a service (``-1``).
        """
        if "dst_service" not in fields:
            fields["dst_service"] = np.full(len(fields["id"]), -1, dtype=np.int64)
        for f in PACKET_FIELDS:
            setattr(self, f, fields[f])

    @classmethod
    def empty(cls) -> PacketBatch:
        """Create a batch that contains no packets.

        Returns:
            PacketBatch: An empty batch.
        """
        return cls(**{f: np.empty(0, dtype=PACKET_DTYPES[f]) for f in PACKET_FIELDS})

    def __len__(self) -> int:
        """Return the number of packets in the batch."""
        return len(self.id)

    def take(self, index: np.ndarray) -> PacketBatch:
        """Select a subset of the packets.

        Args:
            index (np.ndarray): A boolean mask, an array of positions or a slice.

        Returns:
            PacketBatch: The selected packets, in the order given by ``index``.
        """
        return PacketBatch(**{f: getattr(self, f)[index] for f in PACKET_FIELDS})

    @staticmethod
    def concat(batches: list[PacketBatch]) -> PacketBatch:
        """Concatenate several batches.

        Args:
            batches (list[PacketBatch]): The batches to join.

        Returns:
            PacketBatch: The packets of all the batches, in the given order.
        """
        batches = [b for b in batches if len(b)]
        if not batches:
            return PacketBatch.empty()
        if len(batches) == 1:
            return batches[0]
        return PacketBatch(
            **{
                f: np.concatenate([getattr(b, f) for b in batches])
                for f in PACKET_FIELDS
            }
        )
