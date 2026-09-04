# Handover — Kalman and IMU

Written for whoever picks the Kalman filter and the IMU back up. It assumes
nothing: everything below has been run and its output checked.

If you read one thing, read **The three commands**. The rest is there when
you hit something.

---

## The three commands

**1. Understand what exists, without any hardware.**

```
python kalman/proof_imu_kalman.py
```

Two minutes, no camera. It re-runs the checks as sub-processes and recomputes
every number in front of you — nothing is read from a saved results file. It
answers the two things that were asked for: the IMU read from Intel's SDK,
and a Kalman filter built on a kinematic model.

**2. Work: camera, tags, filter, live figures.**

```
python localization/world_frame_check.py
```

This is *the* script. It opens the camera, detects the tags, runs the filter
and opens the six course figures **by itself**, in their own window next to
the video. Nothing to type, no flag to remember. `--no-plots` if you ever
want it without them.

Keys once it runs: `o` sets the reference tag (the world origin), `m`
switches distance/rotation mode, `f` turns the filter on and off, `s` records
a measurement, `q` quits.

**3. Check the IMU on its own.**

```
python kalman/imu_realsense.py               camera plugged in
python kalman/imu_realsense.py --simulation  same maths, no camera
```

Three parts, about two minutes: 5 seconds at rest (do not touch anything),
a soundness check (the accelerometer must read 9.81 m/s² at rest), then the
orientation live — turn the camera a quarter turn, yaw must read 90°.

---

## Two things that will surprise you

**1. On your very first run, the program asks you a question.**

Something like:

```
WHAT IS THIS MACHINE'S MOUNTING?
  1) bare_air    bare camera, in air     (bench, desk, table)
  2) tube_air    camera in the tube, in air
  3) tube_water  camera in the tube, IN WATER
```

This is normal, and it is asked **once per computer**, not once per session.
Answer according to where the camera physically is right now — on a desk, it
is `bare_air`. Your answer is written to `calibration/local_mounting.txt`,
which is deliberately **not** versioned, so the poolside PC and the office
laptop each keep their own without fighting over git.

To change it later: `python calibration/set_mounting.py`.

Why it matters: each mounting has its own calibration. Underwater the tube's
wall refracts and the focal length goes from 604 to 838 px. Get the mounting
wrong and every distance is out by about a quarter, silently.

**2. The underwater calibration will NOT arrive with `git pull`.**

`calibration/mountings/*.npz` is gitignored on purpose (same reason as
above: one machine's calibration must never overwrite another's). So on a
fresh machine you have `bare_air` only.

The day you work in the pool you need `tube_water.npz` copied onto that
machine by hand. Its contents, for reference:

```
fx = 838.45   fy = 652.10   cx = 320.76   cy = 267.37
dist = [0.25503774, 0.43545221, 0.01411297, -0.01478373, -2.02963755]
```

`python calibration/set_mounting.py` shows you which calibrations are
actually present on the machine you are sitting at. If it says `tube_water
no`, the file is missing and the scripts will serve `bare_air`'s numbers
while telling you so.

---

## The numbers, and which ones are still assumed

They all live in **one block** of `kalman/kalman_filter.py`, around line 230,
under `THE NUMBERS TO MEASURE`. Nothing else in the repository needs editing:
every script reads them from there.

| Number | Value | State | How to measure it |
|---|---|---|---|
| underwater focal length | 838.45 / 652.10 px | verified in the field | `tube_water` calibration |
| `GYRO_NOISE_DEG_S` | 0.106 | measured | `kalman/imu_realsense.py`, at rest |
| `ACCEL_NOISE` | 0.015 | measured | `kalman/imu_realsense.py`, at rest |
| `SIGMA_PIXEL` | 0.215 px | measured **in air** | `calibration/measure_tag_noise.py` |
| `SIGMA_ACCELERATION` | 0.4 m/s² | **assumed** | `localization/world_frame_check.py`, step 5 |
| `GYRO_DRIFT_DEG_S` | 10 deg/s | **assumed** | `localization/world_frame_check.py`, step 5 |

The last two describe how hard the vehicle accelerates and turns. That
depends on its mass, its thrusters and the water, so no datasheet and no
calculation can supply them — only the vehicle moving in the water can.

**They are not blocking while the IMU is connected.** In that case the filter
never reads them. Demonstrated with numbers:

```
python kalman/settings_sensitivity.py
```

Varying them by a factor of 4572 does not change the result by a millimetre.
The day the filter runs **without** the IMU, they govern everything.

The scripts print the procedure themselves, and the reminder switches itself
off once the measurement is made. There is nothing to disable by hand.

---

## Which file is which

The trap: `kalman/kalman_filter.py` sounds like the program to run. It is
**not** — it is a library, imported by six other files. Run it directly and
it only executes its own self-tests (17 of them, no hardware; they print
`ALL TESTS PASS`).

| File | What it is | Do you run it? |
|---|---|---|
| `kalman/kalman_filter.py` | the filter itself, a library | no — the numbers to edit are in it |
| `kalman/kalman_live_plots.py` | the six figures, a library | no — it opens from the script below |
| `localization/world_frame_check.py` | camera + tags + filter + figures | **yes, this is the one** |
| `kalman/imu_realsense.py` | the IMU alone | yes |
| `kalman/proof_imu_kalman.py` | re-runs and proves everything | yes |
| `kalman/kalman_reference_check.py` | reproduces Becker's 9 published values | yes, to convince yourself |
| `demos/demo_kalman.py` | simulated pool, produces a figure | yes, no hardware |
| `calibration/optics.py` | focal lengths, tube, tag sizes, a library | no |
| `calibration/set_mounting.py` | the mounting setting | yes, once per machine |

Every file in the repository — all 41 — starts with a `HOW TO USE IT` block
giving the exact command and the keys. If you open one at random, the answer
is at the top of it.

---

## Where the rest is written

- `README.md` — the "where to start" table, by what you want to do.
- `docs/kalman_protocol.md` — the full protocol, step by step, and the
  reasoning behind each number.
- `kalman/kalman_filter.py`, first 200 lines — why the filter is built this
  way: the constant-velocity model, where `Q` comes from, and why a tag's
  error is not isotropic (it says very well where it is sideways, and very
  badly how far away it is).

## One honest reservation

Position cannot come from the IMU alone. A MEMS bias double-integrates into
2.5 cm after 1 s, but 1 m after 10 s. The IMU is what carries the estimate
through a **tag dropout of a few seconds** (9 mm of error over 1.5 s, against
377 mm without it) — it is not a way to navigate blind. The tags remain the
only drift-free source. That is why the two are one system, not two.
