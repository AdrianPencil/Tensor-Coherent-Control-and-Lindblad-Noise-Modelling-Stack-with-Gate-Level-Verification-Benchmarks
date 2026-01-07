"""
core.cache

A tiny LRU cache intended for expensive, pure computations.
We keep it generic: "key -> value" with bounded capacity.

Higher layers can build stronger keys (e.g., parameters + array digests).
"""

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import blake2b
from typing import Generic, Hashable, Iterable, TypeVar

import numpy as np
import numpy.typing as npt

__all__ = [
    "LRUCache",
    "ArrayDigest",
    "array_digest",
]

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """
    Minimal LRU cache with O(1) get/set based on OrderedDict.

    Intended use:
      cache = LRUCache(max_items=256)
      v = cache.get(key)
      if v is None: v = ...; cache.set(key, v)
    """

    __all__ = ["get", "set", "clear", "items", "max_items"]

    def __init__(self, max_items: int) -> None:
        self._max_items = int(max_items)
        self._store: OrderedDict[K, V] = OrderedDict()

    @property
    def max_items(self) -> int:
        return self._max_items

    def get(self, key: K) -> V | None:
        v = self._store.get(key)
        if v is None:
            return None
        self._store.move_to_end(key, last=True)
        return v

    def set(self, key: K, value: V) -> None:
        if key in self._store:
            self._store.move_to_end(key, last=True)
        self._store[key] = value
        while len(self._store) > self._max_items:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def items(self) -> Iterable[tuple[K, V]]:
        return self._store.items()


@dataclass(frozen=True, slots=True)
class ArrayDigest:
    """
    Small digest for numpy arrays suitable for cache keys.

    Note: digesting has cost. Use only when repeated computations dominate.
    """

    shape: tuple[int, ...]
    dtype: str
    digest: str


def array_digest(x: npt.ArrayLike) -> ArrayDigest:
    """
    Produce a stable digest for a numpy array based on C-ordered bytes.

    The digest is intended for cache keys in pure computations; it is not a
    cryptographic guarantee of uniqueness.
    """
    a = np.asarray(x)
    a_c = np.ascontiguousarray(a)
    h = blake2b(a_c.view(np.uint8), digest_size=16)
    return ArrayDigest(
        shape=tuple(int(s) for s in a_c.shape),
        dtype=str(a_c.dtype),
        digest=h.hexdigest(),
    )
