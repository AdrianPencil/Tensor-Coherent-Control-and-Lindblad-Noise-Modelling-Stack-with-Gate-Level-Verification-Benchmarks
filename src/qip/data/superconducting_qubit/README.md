# Superconducting qubit example data

This folder contains small, realistic-ish inputs used by the `case_studies` pipeline.

## Files

### `calibration.json`
A minimal calibration object. Expected keys:

- `omega_01` (rad/s) - qubit angular frequency (optional for the current rotating-frame case study)
- `drive_scale` (rad/s) - effective Rabi-rate scale used to define an `X` gate time via `T_gate = pi / drive_scale`
- `phase_offset` (rad) - optional global phase offset (default 0)

Example:
```json
{
  "omega_01": 3.141592653589793e10,
  "drive_scale": 6.283185307179586e8,
  "phase_offset": 0.0
}
