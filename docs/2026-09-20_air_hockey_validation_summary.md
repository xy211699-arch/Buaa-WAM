# Air Hockey DEFEND + HIT Validation Summary

Date: 2026-09-20

## Goal

Validate the minimum RoboTwin/SAPIEN air-hockey algorithm and prove that
continuous RGB demonstration data can be collected correctly.

Validated pipeline:

    automatic random inbound puck
    -> DEFEND prediction
    -> Cartesian goalie motion
    -> physical contact
    -> actual post-contact puck state
    -> HIT preparation
    -> separated search seed
    -> candidate forward simulation
    -> best candidate selection
    -> restore seed
    -> execute best candidate
    -> puck flight
    -> episode termination
    -> RGB + synchronized state data

## DEFEND

Validated:

- Puck ground-truth position and velocity are read directly from SAPIEN/PhysX.
- A 2D predictor computes the defensive-line intercept.
- The left UR5 striker moves laterally along the defensive line.
- Real puck/striker contact is detected.
- The controller waits for the real post-contact puck state to settle.
- DEFEND transitions to HIT_READY.
- No analytical post-collision puck state is assumed.

## HIT forward search

Validated:

- Save robot and puck state.
- Move the striker laterally to create a separated HIT seed.
- Confirm the separated puck remains stable.
- Restore the same seed before each candidate.
- Evaluate 20 angle/speed candidates in forward simulation.
- Score the resulting puck trajectories.
- Select the best candidate.
- Restore the same seed and execute the selected candidate live.

Search and live replay reached zero measured replay error in the validated runs.

## Kinematic striker tool

The original dynamic striker could slip away from the gripper during candidate
simulation.

The left striker was changed to an end-effector-following kinematic tool:

    robot EE pose
    -> fixed EE-to-striker transform
    -> PhysX kinematic target
    -> dynamic puck collision

The UR5 itself still moves using IK and joint drive targets.

Candidate validation showed approximately 0.8-1.3 mm maximum tool-follow error.

## Snapshot and restore

The snapshot architecture was adapted to the kinematic tool.

The left striker is no longer independently restored as the source of truth.
The robot state is restored and the striker is synchronized from the robot EE.

Validated:

- puck pose restore
- puck velocity restore
- robot q restore
- right striker restore
- EE-to-tool synchronization
- stable separated search seed
- stable post-search restore

## RGB acquisition

A standalone SAPIEN RGB camera test validated:

- scene.add_camera
- camera.take_picture
- camera.get_picture("Color")
- 640x360 RGB capture
- float-to-uint8 RGB conversion
- PNG output

RGB collection therefore does not depend on viewer refresh smoothness.

## Planner timeline isolation

The real demonstration timeline contains:

    DEFEND
    CONTACT_HOLD
    HIT_PREP
    HIT_EXECUTE
    FLIGHT

Internal planner operations are excluded:

    snapshot validation
    candidate 01
    ...
    candidate 20

During candidate search:

- RGB recording is paused.
- demonstration episode time is paused.
- internal physics steps are counted separately.

After candidate selection:

- the search seed is restored
- recording resumes
- only the final selected HIT execution enters the demonstration

This prevents candidate restore operations from appearing as visual
teleportation in training RGB.

## Synchronized frame data

Each recorded RGB frame is synchronized with:

- episode time
- phase
- left robot q/dq
- right robot q/dq
- left TCP pose
- left joint drive target
- puck ground-truth pose
- puck ground-truth velocity
- left striker pose

A validation episode produced exactly:

    147 frames.jsonl records
    147 RGB PNG files
    episode time 0.000 s -> 4.857 s

More than 65,000 planner physics steps were excluded from that demonstration.

## Automatic random episodes

The final validation script supports deterministic random inbound puck states
through a command-line seed.

It automatically samples a legal no-side-wall-bounce inbound trajectory and
then runs the complete pipeline without a manual puck impulse.

Seeds 1, 2, 3, and 4 completed automatic episodes.

Observed terminal conditions included STOP, TIMEOUT, and GOAL.

One validated GOAL run produced:

    plane_x = +1.0190 m
    y_cross = +0.1061 m
    cross_time = 2.0861 s
    tool follow max error = 1.00 mm

The selected HIT search result and live replay were identical.

## Conclusion

The minimum DEFEND + HIT pipeline has been validated end-to-end:

    random puck spawn
    -> DEFEND
    -> physical contact
    -> actual post-contact state
    -> HIT planning
    -> candidate search
    -> best candidate replay
    -> continuous RGB recording
    -> synchronized robot/state recording
    -> episode termination
    -> dataset files written to disk

The RGB demonstration collection architecture is therefore validated.

## Deferred work

Not part of this milestone:

- side-wall bounce physics correction
- final physical goal/table geometry
- large-scale multi-episode collection
- final WAM dataset conversion
- WAM training
- AUBO sim-to-real deployment
