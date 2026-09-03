# UUV — localisation with AprilTags, an IMU and a Kalman filter

Locating an underwater vehicle with a RealSense D435i that looks at AprilTags
through the wall of a sealed tube, underwater.

---

## Start here — one command

```
python kalman/proof_imu_kalman.py
```

It re-runs the checks live and answers the two requests that were made: **IMU
read from the Intel SDK, maths applied to it**, and a Kalman filter **based on
a kinematic model**. Nothing is read from a saved results file — every number
is recomputed as it prints.

No camera needed for that one. To *see* the filter working rather than read
that it does (needs `matplotlib`):

```
python -m pip install matplotlib
python demos/demo_kalman.py
```

**One honest reservation, stated up front:** position cannot come from the IMU
alone. A MEMS bias double-integrates into 2.5 cm after 1 s but 1 m after 10 s.
The IMU is what carries the estimate through a **tag dropout of a few seconds**
(9 mm of error over 1.5 s, against 377 mm without it). The tags remain the only
drift-free source. That is why the two requests are one system, not two.

---

## What is done, and what is not

| | State | Evidence / what is left |
|---|---|---|
| Underwater calibration | done | `fx = 838.45`, `fy = 652.10` — verified in the field |
| IMU read from the Intel SDK | done | `python kalman/imu_realsense.py` |
| IMU maths → orientation | done | 0.03° on a known quarter turn |
| Kinematic Kalman filter | done | `python kalman/kalman_reference_check.py` — Becker's 9 published values, to the 4th decimal |
| Gyro / accelerometer noise | measured | vehicle at rest, 400 Hz |
| **The vehicle's real dynamics** | **to do** | needs the vehicle **moving in the water** — see below |
| `SIGMA_PIXEL` underwater | to do | measured in air (0.215 px); `calibration/measure_tag_noise.py` |

### The measurement that is left — protocol step 5

Two settings (`SIGMA_ACCELERATION`, `GYRO_DRIFT_DEG_S`) still hold their
**assumed** value. They describe how fast the vehicle accelerates and turns:
that depends on its mass, its thrusters and the water, so no calculation and no
datasheet can provide them.

**It is not blocking while the IMU is connected** — in that case the filter
never reads them. Demonstrated with numbers:

```
python kalman/settings_sensitivity.py
```

Varying them by a factor of 4572 does not change the result by a millimetre.
But the day the filter runs **without** the IMU, they govern everything.

The scripts print the procedure themselves, and the reminder switches itself
off once the measurement is made. There is nothing to disable by hand.

---

## Where to start, depending on what you want

| I want to… | Command |
|---|---|
| prove the IMU and filter work is done | `python kalman/proof_imu_kalman.py` |
| **see the filter working, as a figure** | `python demos/demo_kalman.py` |
| **the course figures LIVE, on real measurements** | `python localization/world_frame_check.py --plots` |
| show the calibration gives the right distance | `python calibration/demo_distance.py --mounting tube_water` |
| check a known distance, against a tape measure | `python calibration/check_distance.py --reel 1.5 --tag 0.22389` |
| **do step 5 (vehicle in the water)** | `python localization/world_frame_check.py` |
| watch the IMU run live | `python kalman/imu_realsense.py` |
| the same maths with no camera | `python kalman/imu_realsense.py --simulation` |
| run the filter's self-tests | `python kalman/kalman_filter.py` |
| set this machine's mounting | `python calibration/set_mounting.py` |

---

## Do this once per computer

The physical mounting (bare camera in air / in the tube / in the water) is a
property of **the machine**, not of the code — the office laptop and the pool PC
do not give the same answer. Set it once:

```
python calibration/set_mounting.py tube_water
```

It is written to `calibration/local_mounting.txt`, which is **not** versioned.
Getting it wrong crashes nothing: distances are simply wrong by tens of percent,
silently. The scripts warn when the image contradicts the declared mounting, but
they cannot catch everything.

The three mountings are `bare_air`, `tube_air` and `tube_water`. The French
names used before the handover (`nue_air`, `tube_eau`) are still accepted
everywhere, so a machine already set up keeps working untouched.

---

## The repository

| Folder | Contents |
|---|---|
| `kalman/` | The filter, the IMU, the live plots, the proofs. **Everything still to be picked up is here.** |
| `localization/` | Tags, world frame, maps. `world_frame_check.py` is the pool script. |
| `calibration/` | Optics, calibration, distance checks. `optics.py` is the single source of truth for every constant. |
| `ros/` | The two ROS 2 nodes. |
| `tools/` | List cameras, list RealSense devices, print tags. |
| `demos/` | Standalone demonstrations. Some guess a focal length — do not confuse them with the calibrated chain. |
| `docs/` | `kalman_protocol.md`, the full protocol from pool to filter. |

**The filter's reference document** is Alex Becker, *Kalman Filter Explained
Through Examples* (kalmanfilter.net), constant-velocity kinematic model. This is
not a distant inspiration: `kalman/kalman_reference_check.py` runs his worked
example through the class that actually runs on the vehicle and recovers his 9
published values.

Every script opens with a **HOW TO USE IT** block, then explains **why** it
exists and what it cannot prove. That is what to read before changing anything.
