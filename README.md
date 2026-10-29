# Landing a quadcopter on a moving target — ROS 2 implementation and validation

Implementation of Wickramanayake, Rajasooriya, Ranawella & Karunarathne, [*"Landing a
Quadcopter on to a moving landing target using Computer Vision"*](https://www.researchgate.net/publication/330467111_Landing_a_Quadcopter_on_to_a_moving_landing_target_using_Computer_Vision)
(Univ. of Peradeniya, 2019), revisited with a headless closed-loop simulator. The rebuild
refines the paper's estimation for closed-loop flight (tilt compensation and a Kalman
filter) and measures what each step adds, with tests and experiments in this repo.

Write-up with an interactive replay of the recorded runs:
[billacode.com/blogs/landing-a-quadcopter-on-a-moving-target-revisiting-my-2019-paper](https://billacode.com/blogs/landing-a-quadcopter-on-a-moving-target-revisiting-my-2019-paper)

## Motivation of the paper

A quadcopter can take off and land vertically and reach places a piloted aircraft can't,
but landing on a **moving** platform is hard. The drone has to find the target, estimate
its relative velocity and heading, cope with acceleration and wind, and still touch down
precisely. Earlier work often relies on the drone and the vehicle **communicating**
(sharing GPS or odometry), and in the real world that link cannot be guaranteed. The paper
therefore uses **only an onboard camera**, low-cost hardware (Raspberry Pi + Arduino Nano)
and an **ArUco landing pad with three marker sizes**, so that some marker stays fully
visible from the highest altitude down to touchdown. The pipeline is:

1. pad design (large / medium / small marker)
2. detection and pose estimation (`estimatePoseSingleMarkers` → `rvec`, `tvec`)
3. following/tracking (relative distance, velocity, angle and altitude from `tvec`)
4. landing: near the pad, cut the rotors and fall onto it ballistically, provided the drop
   lands on the pad; otherwise abort and climb again

## From the paper to closed-loop flight

The 2019 work built detection and speed measurement on real hardware, with a smooth
landing left as future work. Flying the drone in a closed loop needs every quantity to be
exact and to hold up while the drone tilts, so this rebuild implements Section III-C as
follows.

| Quantity | In the paper (Sec. III-C) | In this rebuild | Why |
|---|---|---|---|
| Rotation | `rvecs` | `R = cv2.Rodrigues(rvec)` | `solvePnP` returns `rvec` directly; as a matrix it drives tilt compensation and pad heading |
| Distance moved | `√((tx₁²+ty₁²) − (tx₂²+ty₂²))` | `‖p₂ − p₁‖` on world-frame positions | Plain straight-line distance between two positions |
| Speed | `d / duration` | Constant-velocity Kalman filter on the pad's world position; relative velocity `v_pad − v_drone` | Smooths frame-to-frame noise at 30 Hz |
| Direction | `angle = tyᵢ / txᵢ` | `atan2(v_y, v_x)` of the filtered velocity; pad yaw from `rvec` | Heading of the motion in all four quadrants |
| Altitude | `tz` | `p_rel = R_world_body · (R_body_cam · tvec + p_cam)`, altitude = −p_rel,z | Stays correct while the drone tilts to follow the pad (at 4 m a 20° tilt shifts the apparent pad by ~1.5 m) |
| Speed reference | phone speedometer on the vehicle | target ground velocity and relative velocity reported separately | The two are equal only while the drone hovers |

The distance formula is a case where exact notation matters. Coded literally as printed, it
is undefined whenever the pad moves away from the image centre (E1 below). The 2019 Table I
speeds match the phone's readings, so the implementation evidently used a plain distance and
only the typeset formula differs. The `paper` variant in this repo codes the printed formulas
literally so the effect can be measured.

## Architecture (ROS 2 Jazzy, `ros2_ws/src/mtl_landing`)

```
 sim_world ──/camera/image_raw──▶ aruco_detector ──/pad/pose──▶ target_estimator
     ▲   └──/drone/odom (flight-controller state)──────────────▶   │ mode: paper|naive|
     │                                                             │ compensated|corrected
     └──/drone/cmd_vel, /drone/motors◀── landing_controller ◀──/target/estimate
 flight_logger records truth + all estimators + debug frames → results/ros_run_*.json
```

* `core/` is pure Python (numpy + OpenCV) and shared by the ROS nodes, the batch runner and
  the sim lab. `core/pipeline.py` is the one closed loop (camera → detector → estimators →
  controller → physics) that the batch runner, the sim lab and the speed sweep all drive.
  Estimators implement `core.estimation.Estimator` and return a typed `core.state.Estimate`;
  the estimator node sends that same `Estimate` to the controller node on `/target/estimate`.
* **sim_world** has quadrotor dynamics (velocity loop with integral action, tilt with
  first-order lag, drag, Ornstein–Uhlenbeck wind gusts), a ground vehicle with 7 motion
  scenarios, and a **rendered downward camera** (800×600, 70° HFOV, 30 Hz) that warps the
  pad, deck and a cluttered ground texture into real images, with noise and blur. The
  **real ArUco detector** runs on those images. The flight-controller state carries
  attitude bias, velocity noise and position drift. Gazebo and a GPU are not needed.
* **aruco_detector** runs single-marker IPPE pose like the paper, then refines with every
  visible corner of all pad markers (board pose).
* **target_estimator** runs four variants on every detection, and `mode` picks the one that
  flies: `paper` (printed formulas, read literally), `naive` (plain distance, camera frame),
  `compensated` (tilt compensation) and `corrected` (tilt compensation + Kalman filter).
* **landing_controller**: SEARCH → TRACK → DESCEND → FINAL → CUT → LANDED, with ABORT
  (climb and retry), following the paper's Sec. III-D. The rotors are cut only when the
  predicted ballistic impact point is within 12 cm of the pad centre.

## Running

```bash
docker build -f docker/Dockerfile -t mtl_landing:jazzy .        # once
docker/run_lab.sh                            # interactive sim lab on http://localhost:8765
docker/run_ros_sim.sh weave corrected        # full ROS 2 graph, one landing
docker/run_ros_sim.sh straight paper         # same, flown with the printed formulas read literally

# batch experiments (core runner, same physics/detector/estimators), inside the image
docker run --rm -v $PWD/ros2_ws/src:/ws/src -v $PWD/tools:/tools -v $PWD/results:/results \
  -e PYTHONPATH=/ws/src/mtl_landing --entrypoint bash mtl_landing:jazzy \
  -c "python3 /tools/experiments.py all"
python3 tools/export_blog_replay.py <out_dir> straight_corrected weave_corrected   # replay data for the blog player

# unit tests (equation properties, detector accuracy on rendered images)
docker run --rm -v $PWD/ros2_ws/src:/ws/src -e PYTHONPATH=/ws/src/mtl_landing \
  -w /ws/src/mtl_landing --entrypoint bash mtl_landing:jazzy -c "python3 -m pytest -q test"
```

`run_ros_sim.sh` runs each container with `--network none` and its own `ROS_DOMAIN_ID`.
Without that, parallel runs discover each other over DDS and mix their `/clock` topics.

On a machine with ROS 2 Jazzy installed you can skip Docker: `colcon build` the workspace
and run `ros2 launch mtl_landing landing_sim.launch.py scenario:=circle mode:=corrected`.
RViz can show the `/tf` frames (`world`, `base_link`, `pad`) and `/camera/debug`.

## Interactive sim lab

`docker/run_lab.sh` starts a local web UI on http://localhost:8765. It runs the same
simulator, ArUco detector, estimators and landing controller live, with every parameter
on a slider:

- **Vehicle:** pick a scenario or drive it yourself (speed and turn rate).
- **Environment:** gusts and steady wind.
- **Drone:** velocity response, tilt and acceleration limits, attitude and velocity noise.
- **Camera:** field of view, resolution, frame rate, latency, noise, blur, minimum marker size.
- **Landing pad:** the three marker sizes.
- **Estimation:** which variant flies the drone, plus the Kalman filter settings.
- **Landing controller:** gains, heights, the drop rules and timeouts.

Most knobs apply immediately; the ones marked "on reset" apply after Reset. The page shows
the onboard camera with detections, a top view with every estimator's guess of the pad
position, and rolling charts of height, offset and relative speed (truth against each
estimator). Pause, Step (0.1 s) and simulation speed control are in the header.

### Hosting it on a k3s server

`.github/workflows/deploy-sim-lab.yml` deploys the lab to a k3s cluster over SSH:

- **Pull requests:** unit tests and chart validation.
- **Push to `main`:** also builds `docker/lab.Dockerfile` (no ROS, just numpy and OpenCV)
  to `ghcr.io/<owner>/sim-lab:<sha>`, then runs `helm upgrade` with the chart in
  `deploy/helm/sim-lab`.

The chart adds its own Gateway and HTTPRoute on k3s's Traefik. With a Cloudflare tunnel
pointed at the host's port 80, adding a public hostname to the tunnel is the only
network change needed.

The server details (SSH target, public hostname, where to copy the image pull secret
from) come from secrets on a `deploy` environment; the header of the workflow lists them.
They are secrets rather than variables because Actions logs of a public repo are public,
and only secrets are masked.

The hosted lab runs with `LAB_PUBLIC=1`. Everyone shares one simulation, so the costly
knobs are capped, viewers are capped at 20, and nothing is simulated while no one has
the page open.

## Validation results

Numbers from `results/speed_sweep.json` and `results/monte_carlo.json` (re-run with
`tools/experiments.py all 4`).

**E1: speed sweep (paper Table I analogue).** The drone hovers 3.5 m above the deck and the
vehicle drives underneath at 1–9 m/s, 0.6 m off-axis, 3 seeds per speed. Every estimator
sees identical images.

| Estimator | Mean reported speed for a 1 → 9 m/s pad | RMSE | Undefined samples |
|---|---|---|---|
| Printed formulas, read literally | 9.3 → 20.6 m/s | 9.5–13.8 m/s | 41–52 % |
| Plain distance (camera frame) | 1.4 → 8.7 m/s | 0.56–0.98 m/s | 0 % |
| Tilt-compensated | 1.2 → 8.9 m/s | 0.48–0.95 m/s | 0 % |
| Tilt-compensated + Kalman filter | 1.06 → 9.0 m/s | 0.07–0.14 m/s | filter warm-up only |

**E2: estimation in flight.** These are all detections from the 28 corrected-mode closed-loop
flights (~22 k frames). Every estimator was evaluated on the same frames.

| Estimator | Position error p50 / p95 | Speed MAE | Altitude RMSE |
|---|---|---|---|
| Camera frame (printed / plain distance) | 21 / 96 cm | 3.79 / 0.66 m/s | 10.8 cm |
| Tilt-compensated | 1.4 / 5.5 cm | 0.15 m/s | 2.5 cm |
| Tilt-compensated + Kalman filter | 2.2 / 6.4 cm | 0.07 m/s | 1.9 cm |

The camera-frame position error grows from ~3 cm when level to over 1.5 m at 20–30° of tilt.

**E3: closed-loop landings.** The runs cover 7 scenarios × 4 seeds per estimator. A run counts
as landed only if it ends on the pad, ≤ 0.35 m from centre, with vertical impact
≤ 2.2 m/s and horizontal slip ≤ 0.6 m/s.

| Flown by | Landed | Notes |
|---|---|---|
| Printed formulas (camera frame) | 0 / 28 | loses the pad once the drone tilts |
| Plain distance (camera frame) | 0 / 28 | loses the pad once the drone tilts |
| Tilt-compensated | 6 / 28 | mostly abort/retry timeouts |
| Tilt-compensated + Kalman filter | 7 / 28 | static 3/4, straight 2/4, accelerating 1/4, circle 1/4; weave, stop-and-go and fast 0/4 |

Estimation is solid. The weak point is the **landing controller** on manoeuvring targets:
the pad shifts or brakes during the final hover, the camera loses it, and the drone aborts.
That is the open work item.

**Pad.** The paper gives no marker sizes. With a 10 cm centre marker the camera lost the pad
in the last 30 cm unless the drone was within ~3 cm of centre, so the pad uses
32 / 20 / 16 cm markers. Between 1.2 and 4 m the pad is detected in 86–90 % of frames.
Below 0.5 m the small marker carries almost all detections.

## Limitations

* The simulation is a planar-scene renderer, not a photorealistic one. Lighting changes,
  motion blur at high angular rates and lens distortion are simplified.
* Quadrotor dynamics are a velocity-controlled model (what a PX4/ArduPilot offboard
  interface exposes), not rotor-level physics. Ground effect is not modelled.
* The deck is flat; vehicle pitch/roll from road bumps is not modelled.
* Below ~0.3 m camera height even the small marker leaves the frame unless the drone is
  well centred, so the last moments before the drop rely on the Kalman prediction. That
  matches the paper's remark about losing the target very close to it.
* The landing controller does not yet handle weaving, stop-and-go or fast targets reliably
  (see E3).
