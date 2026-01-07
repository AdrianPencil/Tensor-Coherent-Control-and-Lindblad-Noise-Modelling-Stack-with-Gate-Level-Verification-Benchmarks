"""
circuits.compilation

Optional layer: scheduling / constraints / compilation passes.

This file provides a minimal "compilation pass" container to keep an explicit
extension point without committing to a large framework.
"""

from dataclasses import dataclass
from typing import Protocol

from qip.circuits.circuits import Circuit

__all__ = [
    "CompilationPass",
    "Compiler",
]


class CompilationPass(Protocol):
    def __call__(self, circuit: Circuit) -> Circuit:
        """
        Transform a circuit (e.g., optimize, schedule, rewrite).
        """
        ...


@dataclass(frozen=True, slots=True)
class Compiler:
    """
    Compiler that applies an ordered list of passes.

    If passes is empty, compilation is an identity transform.
    """

    passes: tuple[CompilationPass, ...] = ()

    def compile(self, circuit: Circuit) -> Circuit:
        """
        Apply passes in order and return the transformed circuit.
        """
        out = circuit
        for p in self.passes:
            out = p(out)
        return out
