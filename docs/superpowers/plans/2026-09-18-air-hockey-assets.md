# Air hockey primitive assets implementation plan

**Goal:** Provide a configurable SAPIEN air hockey table, puck, and gripper-held striker prototype in RoboTwin.

**Architecture:** `env_cfg/air_hockey_assets.yml` holds dimensions and initial physical parameters in SI units. `air_hockey_assets.py` validates that configuration and creates static table geometry and dynamic puck/striker bodies from SAPIEN primitives. A physics-only smoke test uses the installed RoboTwin environment without requiring Vulkan.

**Tech stack:** Python 3.10, PyYAML, SAPIEN 3.0.0b1, `unittest`.

**Scope:** Asset generation only. No AUBO, gripper attachment, policy or data collection integration.

**Provisional values:** Table length 2.128 m and width 1.218 m, puck diameter 0.0633 m and mass 0.01 kg come from Air Hockey Challenge `table.xml`. RoboTwin's 0.74 m table height is used as a provisional scene height. Goal and striker dimensions are provisional until the user supplies their measurements.

## Files

- `env_cfg/air_hockey_assets.yml`: dimensions, masses, contact parameters, source/provisional markers.
- `air_hockey_assets.py`: config validation and SAPIEN builders.
- `tests/test_air_hockey_assets.py`: config and physics-only smoke checks.

## Task 1: Validate the physical contract

- [x] Write a failing `unittest` for the three builders' actor names, body types, and puck mass.
- [x] Run it with `python -m unittest tests.test_air_hockey_assets` and confirm the feature is missing.
- [x] Add the YAML and minimal builder module.
- [x] Re-run the same command and confirm pass.

## Task 2: Verify contact behavior

- [x] Add physics-only tests for level resting and rail rebound; manually verify sliding and open goals.
- [x] Verify failures before adjusting cylinder orientation and PhysX bounce threshold.
- [x] Make the minimum corrections and run the full test module.
- [x] Run final verification and review changed files, then report provisional parameters and Vulkan rendering limitation.
