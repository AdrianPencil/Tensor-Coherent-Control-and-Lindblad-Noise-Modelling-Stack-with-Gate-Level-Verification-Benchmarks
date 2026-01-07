"""
circuits.transpile

Optional layer: map circuits -> device-native representations (controls/pulses).

For now, this is a deliberately minimal identity-like interface so the rest of
the stack can compile cleanly. Concrete device mappers can be built later.

The main purpose is to standardize the type boundary:
  Circuit -> TranspiledProgram
"""

from dataclasses import dataclass
from typing import Any

from qip.circuits.circuits import Circuit

__all__ = [
    "TranspiledProgram",
    "Transpiler",
]


@dataclass(frozen=True, slots=True)
class TranspiledProgram:
    """
    Output of transpilation.

    payload is intentionally untyped (Any) because different devices will
    produce different representations (pulse lists, schedules, H(t) objects).
    """

    payload: Any
    description: str


@dataclass(frozen=True, slots=True)
class Transpiler:
    """
    Base transpiler.

    Current behavior is a pass-through that preserves the circuit.
    """

    target: str = "identity"

    def transpile(self, circuit: Circuit) -> TranspiledProgram:
        """
        Transpile a circuit into a target-specific payload.

        In the identity target, payload is the original circuit.
        """
        return TranspiledProgram(payload=circuit, description=f"transpiled(target={self.target})")
