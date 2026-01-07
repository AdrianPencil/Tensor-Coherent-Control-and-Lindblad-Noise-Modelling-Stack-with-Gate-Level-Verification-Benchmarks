"""
circuits.simulation

Circuit simulation backends:
- statevector simulation for pure states
- density-matrix simulation for mixed states (and simple noisy gate models)

Tensor convention (important):
- We represent an n-qubit statevector as a tensor of shape (2, 2, ..., 2)
  with axis order matching qubit index order: axis i corresponds to qubit i.
- Basis ordering in the flattened vector is consistent with that tensor order.

We avoid constructing full 2^n x 2^n matrices for each gate when possible.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from qip.core.config import DEFAULT, QIPConfig
from qip.core.linalg import as_c128
from qip.circuits.circuits import Circuit
from qip.circuits.gates import Gate
from qip.states.density import DensityMatrix
from qip.states.state import StateVector

__all__ = [
    "GateModel",
    "CircuitSimulator",
]


class GateModel(Protocol):
    def unitary(self, gate: Gate) -> npt.NDArray[np.complex128]:
        """
        Return the ideal unitary for the gate.

        A noisy model should instead be used with `simulate_density_with_model`.
        """
        ...


def _apply_gate_to_statevector(
    psi: npt.NDArray[np.complex128],
    U: npt.NDArray[np.complex128],
    qubits: tuple[int, ...],
    n_qubits: int,
) -> npt.NDArray[np.complex128]:
    """
    Apply a k-qubit gate U to an n-qubit statevector psi without building a full matrix.

    psi: shape (2^n,)
    U: shape (2^k, 2^k)
    qubits: which qubits the gate acts on (ordered)
    """
    n = int(n_qubits)
    qs = tuple(int(q) for q in qubits)
    k = len(qs)

    if psi.ndim != 1 or psi.size != (1 << n):
        raise ValueError("psi has wrong shape for n_qubits")
    if U.shape != (1 << k, 1 << k):
        raise ValueError("U has wrong shape for given qubits")

    psi_t = psi.reshape((2,) * n)
    other = tuple(i for i in range(n) if i not in qs)
    perm = qs + other
    inv_perm = np.argsort(np.array(perm, dtype=np.int64))

    psi_p = np.transpose(psi_t, axes=perm).reshape((1 << k, 1 << (n - k)))
    psi_p2 = U @ psi_p
    out = np.transpose(psi_p2.reshape((2,) * n), axes=inv_perm).reshape(-1)
    return np.ascontiguousarray(out)


def _apply_gate_to_density(
    rho: npt.NDArray[np.complex128],
    U: npt.NDArray[np.complex128],
    qubits: tuple[int, ...],
    n_qubits: int,
) -> npt.NDArray[np.complex128]:
    """
    Apply ρ -> U ρ U† on selected qubits using tensor reshaping.

    rho: shape (2^n, 2^n)
    U: shape (2^k, 2^k)
    """
    n = int(n_qubits)
    qs = tuple(int(q) for q in qubits)
    k = len(qs)

    if rho.ndim != 2 or rho.shape != (1 << n, 1 << n):
        raise ValueError("rho has wrong shape for n_qubits")
    if U.shape != (1 << k, 1 << k):
        raise ValueError("U has wrong shape for given qubits")

    rho_t = rho.reshape((2,) * n + (2,) * n)

    ket_axes = qs
    bra_axes = tuple(q + n for q in qs)
    other_ket = tuple(i for i in range(n) if i not in qs)
    other_bra = tuple(i + n for i in range(n) if i not in qs)

    perm = ket_axes + other_ket + bra_axes + other_bra
    inv_perm = np.argsort(np.array(perm, dtype=np.int64))

    rho_p = np.transpose(rho_t, axes=perm)
    rho_p = rho_p.reshape((1 << k, 1 << (n - k), 1 << k, 1 << (n - k)))

    tmp = np.tensordot(U, rho_p, axes=(1, 0))
    tmp = tmp.reshape((1 << k, 1 << (n - k), 1 << k, 1 << (n - k)))

    Ud = np.conjugate(U).T
    tmp2 = np.tensordot(tmp, Ud, axes=(2, 0))
    tmp2 = tmp2.transpose(0, 1, 3, 2)
    tmp2 = tmp2.reshape((2,) * n + (2,) * n)

    out = np.transpose(tmp2, axes=inv_perm).reshape((1 << n, 1 << n))
    return np.ascontiguousarray(out)


@dataclass(frozen=True, slots=True)
class CircuitSimulator:
    """
    Simulator for `Circuit` objects.

    This class provides:
    - statevector simulation assuming ideal unitaries
    - density simulation driven by a gate-noise model (optional)
    """

    cfg: QIPConfig = DEFAULT

    def simulate_statevector(
        self,
        circuit: Circuit,
        psi0: npt.ArrayLike | None = None,
        gate_model: GateModel | None = None,
    ) -> StateVector:
        """
        Simulate a circuit in the statevector backend.

        If psi0 is None, uses |0...0>.
        If gate_model is None, uses each gate's stored matrix directly.
        """
        n = int(circuit.n_qubits)
        if n < 0:
            raise ValueError("circuit.n_qubits must be >= 0")

        if psi0 is None:
            psi = np.zeros((1 << n,), dtype=self.cfg.dtype_complex)
            psi[0] = np.complex128(1.0 + 0.0j)
        else:
            psi = as_c128(psi0, cfg=self.cfg).reshape(-1)
            if psi.size != (1 << n):
                raise ValueError("psi0 has wrong dimension")

        psi = np.ascontiguousarray(psi)

        for g in circuit.gates:
            U = g.U if gate_model is None else gate_model.unitary(g)
            U = np.ascontiguousarray(as_c128(U, cfg=self.cfg))
            psi = _apply_gate_to_statevector(psi, U, g.qubits, n)

        return StateVector.from_array(psi, cfg=self.cfg)

    def simulate_density(
        self,
        circuit: Circuit,
        rho0: npt.ArrayLike | None = None,
    ) -> DensityMatrix:
        """
        Simulate a circuit in the density backend using ideal unitaries.

        If rho0 is None, uses |0...0><0...0|.
        """
        n = int(circuit.n_qubits)
        if n < 0:
            raise ValueError("circuit.n_qubits must be >= 0")

        if rho0 is None:
            rho = np.zeros((1 << n, 1 << n), dtype=self.cfg.dtype_complex)
            rho[0, 0] = np.complex128(1.0 + 0.0j)
        else:
            rho = as_c128(rho0, cfg=self.cfg)
            if rho.shape != (1 << n, 1 << n):
                raise ValueError("rho0 has wrong shape")

        rho = np.ascontiguousarray(rho)

        for g in circuit.gates:
            U = np.ascontiguousarray(as_c128(g.U, cfg=self.cfg))
            rho = _apply_gate_to_density(rho, U, g.qubits, n)

        return DensityMatrix.from_matrix(rho, cfg=self.cfg)

    def simulate_density_with_model(
        self,
        circuit: Circuit,
        model: object,
        rho0: npt.ArrayLike | None = None,
    ) -> DensityMatrix:
        """
        Simulate a circuit in the density backend with a gate model.

        The model must expose:
          apply_to_density(gate: Gate, rho: ArrayLike) -> NDArray[complex128]

        This supports simple per-gate noise without forcing a global Lindblad solver.
        """
        if not hasattr(model, "apply_to_density"):
            raise ValueError("model must define apply_to_density(gate, rho)")

        n = int(circuit.n_qubits)
        if n < 0:
            raise ValueError("circuit.n_qubits must be >= 0")

        if rho0 is None:
            rho = np.zeros((1 << n, 1 << n), dtype=self.cfg.dtype_complex)
            rho[0, 0] = np.complex128(1.0 + 0.0j)
        else:
            rho = as_c128(rho0, cfg=self.cfg)
            if rho.shape != (1 << n, 1 << n):
                raise ValueError("rho0 has wrong shape")

        rho = np.ascontiguousarray(rho)

        for g in circuit.gates:
            rho = model.apply_to_density(g, rho)
            rho = np.ascontiguousarray(as_c128(rho, cfg=self.cfg))

        return DensityMatrix.from_matrix(rho, cfg=self.cfg)
