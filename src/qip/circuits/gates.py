"""
circuits.gates

Gate library and parametrisations.

We store each gate as:
- name
- qubits (tuple of int)
- matrix (dense complex128)

This is intentionally strict -no implicit device mapping here.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

from qip.core.config import DEFAULT, QIPConfig
from qip.core.linalg import as_c128, kron

__all__ = [
    "Gate",
    "I",
    "X",
    "Y",
    "Z",
    "H",
    "S",
    "T",
    "Rx",
    "Ry",
    "Rz",
    "CNOT",
]


@dataclass(frozen=True, slots=True)
class Gate:
    name: str
    qubits: tuple[int, ...]
    U: npt.NDArray[np.complex128]
    cfg: QIPConfig = DEFAULT

    @property
    def arity(self) -> int:
        return int(len(self.qubits))

    @property
    def dim(self) -> int:
        return int(self.U.shape[0])


def _u2(a00: complex, a01: complex, a10: complex, a11: complex, *, cfg: QIPConfig) -> npt.NDArray[np.complex128]:
    m = np.array([[a00, a01], [a10, a11]], dtype=cfg.dtype_complex)
    return np.ascontiguousarray(m)


def I(q: int, *, cfg: QIPConfig = DEFAULT) -> Gate:
    U = np.eye(2, dtype=cfg.dtype_complex)
    return Gate(name="I", qubits=(int(q),), U=np.ascontiguousarray(U), cfg=cfg)


def X(q: int, *, cfg: QIPConfig = DEFAULT) -> Gate:
    U = _u2(0.0, 1.0, 1.0, 0.0, cfg=cfg)
    return Gate(name="X", qubits=(int(q),), U=U, cfg=cfg)


def Y(q: int, *, cfg: QIPConfig = DEFAULT) -> Gate:
    U = _u2(0.0, -1.0j, 1.0j, 0.0, cfg=cfg)
    return Gate(name="Y", qubits=(int(q),), U=U, cfg=cfg)


def Z(q: int, *, cfg: QIPConfig = DEFAULT) -> Gate:
    U = _u2(1.0, 0.0, 0.0, -1.0, cfg=cfg)
    return Gate(name="Z", qubits=(int(q),), U=U, cfg=cfg)


def H(q: int, *, cfg: QIPConfig = DEFAULT) -> Gate:
    s = 1.0 / np.sqrt(2.0)
    U = _u2(s, s, s, -s, cfg=cfg)
    return Gate(name="H", qubits=(int(q),), U=U, cfg=cfg)


def S(q: int, *, cfg: QIPConfig = DEFAULT) -> Gate:
    U = _u2(1.0, 0.0, 0.0, 1.0j, cfg=cfg)
    return Gate(name="S", qubits=(int(q),), U=U, cfg=cfg)


def T(q: int, *, cfg: QIPConfig = DEFAULT) -> Gate:
    U = _u2(1.0, 0.0, 0.0, np.exp(1.0j * np.pi / 4.0), cfg=cfg)
    return Gate(name="T", qubits=(int(q),), U=U, cfg=cfg)


def Rx(q: int, theta: float, *, cfg: QIPConfig = DEFAULT) -> Gate:
    th = float(theta)
    c = np.cos(0.5 * th)
    s = -1.0j * np.sin(0.5 * th)
    U = _u2(c, s, s, c, cfg=cfg)
    return Gate(name="Rx", qubits=(int(q),), U=U, cfg=cfg)


def Ry(q: int, theta: float, *, cfg: QIPConfig = DEFAULT) -> Gate:
    th = float(theta)
    c = np.cos(0.5 * th)
    s = np.sin(0.5 * th)
    U = _u2(c, -s, s, c, cfg=cfg)
    return Gate(name="Ry", qubits=(int(q),), U=U, cfg=cfg)


def Rz(q: int, theta: float, *, cfg: QIPConfig = DEFAULT) -> Gate:
    th = float(theta)
    a = np.exp(-0.5j * th)
    b = np.exp(0.5j * th)
    U = _u2(a, 0.0, 0.0, b, cfg=cfg)
    return Gate(name="Rz", qubits=(int(q),), U=U, cfg=cfg)


def CNOT(control: int, target: int, *, cfg: QIPConfig = DEFAULT) -> Gate:
    c = int(control)
    t = int(target)
    if c == t:
        raise ValueError("control and target must be different")
    U = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=cfg.dtype_complex,
    )
    return Gate(name="CNOT", qubits=(c, t), U=np.ascontiguousarray(U), cfg=cfg)
