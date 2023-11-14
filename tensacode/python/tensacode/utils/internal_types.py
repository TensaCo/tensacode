from __future__ import annotations
from typing import TypeVar


K, V = TypeVar("K"), TypeVar("V")
nested_dict = dict[K, "nested_dict[K, V]"] | dict[K, V]
