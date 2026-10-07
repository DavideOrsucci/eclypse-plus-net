"""Builders of networks for the packet-level network extension."""

from .generators import (
    get_network_fat_tree,
    get_network_internet_as,
)

__all__ = [
    "get_network_fat_tree",
    "get_network_internet_as",
]
