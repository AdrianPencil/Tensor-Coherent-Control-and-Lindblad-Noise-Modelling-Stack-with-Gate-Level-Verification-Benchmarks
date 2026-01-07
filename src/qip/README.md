# qip-stack

A minimal, modular Quantum Information Processing stack that connects:

**states + operators + dynamics + noise + control + circuits + workflows + end-to-end case studies**

The emphasis is on **clean boundaries**, **typed APIs**, **contiguous NumPy kernels**, and **brutally concrete pipelines** that can serve as examples and integration tests.

---

## Why this exists (motivation)

Most QIP codebases are either:

- **too high-level** (excellent for circuits, but detached from device/noise/control realism), or
- **too device-specific** (excellent physics, but hard to reuse across platforms), or
- **too heavy** (large dependencies and frameworks before you even know what you need).

`qip-stack` is a scaffold you can grow in either direction:

- Start from **math-first primitives** (states/operators).
- Add **dynamics** (Hamiltonian + Lindblad).
- Add **noise** in multiple views (PSD models, Kraus channels, identification).
- Add **control scaffolding** (pulses, calibration, light optimization).
- Add a **gate/circuit layer** that can later transpile to device-native programs.
- Tie it together with **workflows** and **case studies** that behave like integration tests.

---

## What it does (high level)

### Core primitives
- **StateVector** for pure states.
- **DensityMatrix** for mixed states.
- Dense **Operator** wrappers + Pauli/basis helpers.
- Linear algebra utilities that enforce contiguous arrays and stable numeric behavior.

### Dynamics
- Static and time-dependent **Hamiltonian** objects.
- **LindbladModel** master equation RHS.
- A minimal **RK4** integrator for matrices/vectors.
- Convenience propagators for unitary evolution and Lindblad evolution.

### Noise
- **Kraus channels** for common noise models (phase flip, depolarizing, amplitude damping).
- **PSD models** (white, ohmic, 1/f^α).
- A small **identification** fitter for “1/f + white”.
- Pragmatic mapping from PSD → **effective rates** (scaffold, replace later).

### Control
- Pulse envelopes (Gaussian, square, cosine-rise/fall, DRAG).
- Control Hamiltonian composition: `H(t) = H0 + Σ u_k(t) H_k`.
- Calibration JSON load/save.
- A minimal random-search optimizer (prototype-friendly).

### Circuits
- Gate library (I,X,Y,Z,H,S,T,Rx,Ry,Rz,CNOT).
- Circuit container for composition.
- Tensor-based circuit simulation:
  - statevector backend (ideal)
  - density backend (ideal or with a simple gate noise model)

### Workflows + Case Studies
- Workflows are reusable “experiment bundles” returning structured results.
- Case studies are concrete pipelines that stitch:
  **Hamiltonian → noise → control → verification**
- Included: superconducting-qubit end-to-end case study using measured-ish PSD + calibration.

---

## Quickstart

### Install (editable dev install)
```bash
pip install -e .
