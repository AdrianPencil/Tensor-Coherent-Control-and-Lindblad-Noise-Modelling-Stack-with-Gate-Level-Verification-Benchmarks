"""
circuits

Gate/circuit layer built on dense operators and state/density backends.

Public surface:
- Gate, Circuit
- common gate constructors (via gates.py)
"""

__all__ = [
    "Gate",
    "Circuit",
]

from qip.circuits.gates import Gate
from qip.circuits.circuits import Circuit
