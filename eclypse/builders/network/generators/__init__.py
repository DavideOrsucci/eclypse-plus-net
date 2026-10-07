"""Network generators (e.g. get_network_fat_tree, get_network_internet_as).

The package collects builders that return a :class:`~eclypse.network.Network`, with
hosts, routers and links annotated with the parameters of the packet-level model.
"""

from .network_fat_tree import get_network_fat_tree
from .network_internet_as import get_network_internet_as

__all__ = [
    "get_network_fat_tree",
    "get_network_internet_as",
]
