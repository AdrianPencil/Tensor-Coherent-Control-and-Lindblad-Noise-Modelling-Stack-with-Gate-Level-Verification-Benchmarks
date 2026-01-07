"""
circuits.circuits

Circuit container and composition.

We keep one class:
- Circuit: ordered list of Gate objects with n_qubits metadata

This is intentionally light; simulation lives in circuits.simulation later.
"""

from dataclasses import dataclass

from qip.circuits.gates import Gate

__all__ = [
    "Circuit",
]


@dataclass(frozen=True, slots=True)
class Circuit:
    n_qubits: int
    gates: tuple[Gate, ...]

    @staticmethod
    def empty(n_qubits: int) -> "Circuit":
        n = int(n_qubits)
        if n < 0:
            raise ValueError("n_qubits must be >= 0")
        return Circuit(n_qubits=n, gates=())

    def append(self, gate: Gate) -> "Circuit":
        n = int(self.n_qubits)
        for q in gate.qubits:
            if int(q) < 0 or int(q) >= n:
                raise ValueError("gate qubit index out of range for this circuit")
        return Circuit(n_qubits=n, gates=self.gates + (gate,))
