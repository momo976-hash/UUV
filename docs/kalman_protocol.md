# Protocol — from the pool to the Kalman filter

What remains to be done AFTER the `tube_water` calibration, in order.

---

## Reference document

The filter follows **Alex Becker, "Kalman Filter Explained Through Examples"**,
kalmanfilter.net — a **constant-velocity kinematic model**.

This is not a distant inspiration: it is the same filter. To check that, run

```
python kalman/kalman_reference_check.py
```

That script runs the document's worked example (a 1D radar tracking an
aircraft) through the class that actually filters the vehicle's position, and
compares the 9 published values — `Q`, `x(1,0)`, `P(1,0)`, `K(1)`, `x(1,1)`,
`P(1,1)`, `x(2,1)`, `P(2,1)` — with the computed ones. They are recovered to
the fourth decimal. The same check is locked into `kalman_filter.py`'s
self-tests: touching the equations makes it fail immediately.

The filter is written and tested (`python kalman/kalman_filter.py` runs 17
self-tests). There is therefore **nothing to code**. What is missing is
**four measured numbers** the filter expects, currently either assumed or
measured in air.

| Number | Current value | How to measure it | Step |
|---|---|---|---|
| underwater focal length | **838.45 / 652.10 px**, verified in the field | `tube_water` calibration | 1 and 2 |
| `SIGMA_PIXEL` | 0.215 px, **measured IN AIR** | `calibration/measure_tag_noise.py` | 4 |
| `SIGMA_ACCELERATION` | 0.4 m/s2, **assumed** | `localization/world_frame_check.py` | 5 |
| `GYRO_DRIFT_DEG_S` | 10 deg/s, **assumed** | `localization/world_frame_check.py` | 5 |

Two more are already measured, and they are the ones that govern whenever the
IMU is connected: `GYRO_NOISE_DEG_S = 0.106` and `ACCEL_NOISE = 0.015`,
both from `kalman/imu_realsense.py` with the vehicle at rest.

## In total: a few lines to change, in ONE file

Nothing else. Every script that uses them reads them from there:
**`kalman/kalman_filter.py`**, block "THE NUMBERS TO MEASURE", near line 200.

```python
SIGMA_PIXEL        = 0.215
SIGMA_ACCELERATION = 0.4
GYRO_DRIFT_DEG_S   = 10.0
```

The scripts that measure these values **print the exact line to copy** at the
end of a session. You do not have to work out where it goes.

**The mounting (step 2) is no longer changed by editing code at all.** It is
set once per computer:

```
python calibration/set_mounting.py
```

---

## Step 1 — Calibrate underwater

```
python calibration/calibrate.py --mounting tube_water
```

Same procedure as `tube_air`: the 50 mm checkerboard, about thirty views, the
board well into the IMAGE CORNERS — that is where the distortion coefficients
are read from.

What it writes:
- `calibration/mountings/tube_water.npz` — the 9 parameters
- `calibration/mountings/tube_water_ros.yaml` — for the ROS node

Immediate check:

```
python calibration/optics.py
```

The RECORDED CALIBRATIONS section must now list `tube_water` with its measured
fx and fy. Compare them with the values **predicted** by the optical model. A
few percent of difference is normal; 30 % means the mounting has been
misunderstood, and that must be understood before going any further.

---

## Step 2 — Switch to `tube_water`

**No Python file to edit.** On the computer that will do the measuring:

```
python calibration/set_mounting.py tube_water
```

Do this **once per machine**, not once per session. The setting is written to
`calibration/local_mounting.txt`, which is **not** versioned: the poolside PC
stays on `tube_water` and the office laptop on `bare_air`, and a git pull
never changes either.

For a single command, without disturbing anything:

```
UUV_MOUNTING=bare_air python localization/world_frame_check.py
```

---

## Step 3 — Place the tags and build the map

The tag layout is in `localization/pool_layout_3d.py`. To see it in 3D:

```
python localization/pool_layout_3d.py
```

The map can also build itself, with no tape measure, by showing the camera
pairs of tags that overlap:

```
python localization/auto_mapping.py
```

**Mechanical stability matters more than anything here.** The tags sit on
ballasted acrylic boxes, not sealed into concrete. If a tag moves by delta,
the camera position deduced from it moves by delta too, exactly. That is a
BIAS, and a Kalman filter follows a bias instead of averaging it out. A box
displaced by 1 cm produces on its own five times the whole rest of the error
budget.

`TagWatchdog` in the filter detects it — `demo_kalman.py` shows it catching a
22 mm displacement to within 1 mm — but detecting is not correcting.

---

## Step 4 — Re-measure detection noise UNDERWATER

```
python calibration/measure_tag_noise.py
```

Camera still, tag still. The current value (0.215 px) was measured **in air**.
Murky water and poorer contrast will make it worse, and a filter that believes
the tags more than it should reports an over-confident uncertainty.

The script prints the line to copy:

```
    SIGMA_PIXEL = 0.312
```

---

## Step 5 — Measure the vehicle's real dynamics

```
python localization/world_frame_check.py
```

1. Aim at a tag and press **`o`**. It becomes the world origin.
2. Move towards the second tag — linking happens by itself when both are
   visible together for a moment.
3. **Keep going for about 30 seconds** after the "tag linked" message. That
   message is a confirmation, not a signal to stop.
4. Drive it **like a real mission**: usual speeds and accelerations, neither
   parked nor deliberately shaken.
5. Press **`q`**.

You do NOT need the `m` or `s` keys for this step. They belong to step 6.

The script prints the two exact lines to copy:

```
OBSERVED DYNAMICS
  rotation      median   12.4 deg/s   95th percentile   31.0 deg/s
  acceleration  median   0.18 m/s2    95th percentile    0.62 m/s2
------------------------------------------------------------------
      SIGMA_ACCELERATION = 0.6
      GYRO_DRIFT_DEG_S  = 31
```

These are the 95th percentiles — wide enough to cover what the vehicle really
does, without latching onto an isolated spike.

**Physical meaning**, to explain it to someone:
- `SIGMA_ACCELERATION` = how hard the vehicle can accelerate without the
  filter knowing. Too small → the filter lags in turns. Too large → it stops
  smoothing anything.
- `GYRO_DRIFT_DEG_S` = how fast the orientation can change between two frames
  with no measurement.

### Is this step blocking?

**Not while the IMU is connected.** Both settings sit on an `is None` branch in
the filter: as soon as the D435i's IMU feeds it, they are never read.
Demonstrated with numbers:

```
python kalman/settings_sensitivity.py
```

Varying `SIGMA_ACCELERATION` by a factor of 4572 changes the position RMS by
0.0000 mm with the IMU connected — and from 17.9 to 52.8 mm without it.

So this step is needed for the day the IMU fails, is unplugged, or is simply
not used for a given run. Until then it is not a prerequisite, and the scripts
say so themselves rather than blocking.

**A warning worth repeating:** this measurement is only meaningful with the
REAL vehicle in the water. A camera waved by hand on a desk gives about 3 g in
median — 500 times what a UUV does. Copying that in would tell the filter the
vehicle can accelerate at 23 g without its knowledge, and it would stop
filtering altogether. The assumed 0.4 is closer to the truth for a UUV than
anything measurable by hand.

---

## Step 6 — Check the filter really improves things

Still in `localization/world_frame_check.py`. The procedure:

1. Press `o` on the reference tag
2. Check the display reads `filter : ON` (key `f`)
3. Move the camera by a distance **measured with a tape**
4. Type the real value on the keyboard, press `s` to record it
5. Repeat about **fifteen** times, at varied distances

Each `s` records the raw AND the filtered value for the same instant, on the
same line. There is no need to redo the series with the filter off: comparing
two series would require repeating exactly the same gesture twice, and the
gesture would dominate the difference.

Press `q` and the script prints the verdict on its own.

### The two questions in the verdict

**Question 1 — does the filter reduce the error?**

```
  RMS error   raw      16.1 mm
               filter   8.6 mm     -> gain 1.88x
  [OK] the filter reduces the error.
```

Threshold: gain >= 1.2. Below that the script answers `[INCONCLUSIVE]` — on
fifteen measurements, a few percent cannot be told apart from chance. A gain
below 1.0 almost always points at `SIGMA_ACCELERATION` (step 5).

**Announce no expected gain in advance.** The self-tests show 33x, but that is
a simulation in which the noise is exactly what the filter assumes and the
trajectory has constant velocity — both of the filter's assumptions are true by
construction there. In the pool it will be far less. The only defensible number
is the one YOU measure here.

**Question 2 — does the filter tell the truth about its precision?**

This is the more important question, and it cannot be seen on screen.

```
  uncertainty reported by the filter:   7.0 mm (median)
  error actually observed            :   6.5 mm (median)
  actual / reported ratio: 0.9
  [OK] the filter tells the truth about its precision.
```

| Ratio | Verdict |
|---|---|
| < 0.5 | cautious — reports more error than it makes, harmless |
| 0.5 to 2 | honest |
| 2 to 4 | believes itself more precise than it is; do not trust the `+/-` |
| > 4 | **it lies** — check `SIGMA_PIXEL`, then the tag map |

### Watching it work while it runs

```
python localization/world_frame_check.py --plots
```

Six live figures, on the real measurements: the Bayesian update (prior,
likelihood, posterior), estimate vs raw measurement with the ±1σ band, the
uncertainty over time, the Kalman gain, each tag's quality, and the outlier
test. Requires `matplotlib`; without it the measurement runs anyway.

None of those figures says whether the position is CORRECT — there is no
ground truth in a real run. They say whether the filter behaves the way a
Kalman filter must. Only the tape measure of this step answers correctness.

---

## Step 7 — The IMU

```
python kalman/imu_realsense.py
```

Camera connected. The script opens the SDK's `accel` and `gyro` streams,
measures the rest state for 5 s, then shows the orientation live.

Four things show up, in order:

1. **The IMU is read** — the raw values move when you move the camera.
2. **The measurements are sound** — at rest the accelerometer reads 9.81 m/s2.
   If not, the units are wrong and so is everything downstream.
3. **The maths work** — turn a quarter turn, the yaw reads 90 degrees.
4. **The drift is where it should be** — roll and pitch stay stable (the
   accelerometer holds them), yaw drifts. That is the visible demonstration of
   why the tags are necessary.

Without a camera to hand:

```
python kalman/imu_realsense.py --simulation
```

The same maths on a simulated unit. The truth being known, the error is
quantified: 0.03 deg on a real quarter turn.

### Two numbers to measure, vehicle AT REST

One minute without moving, then the standard deviation of the measurements.
Both are already measured and installed:

```python
GYRO_NOISE_DEG_S = 0.106   # deg/s
ACCEL_NOISE      = 0.015   # m/s2
```

### The limit to state honestly

The accelerometer has a slowly varying bias that **nothing here estimates**.
Double integration turns it into a quadratic error: 0.05 m/s2 becomes 2.5 cm
after one second, but **1 m after ten**.

The IMU is therefore for crossing a tag dropout of a few seconds, **not for
dead reckoning**. The tags remain the only drift-free source.

---

## Recap — a single pool session

| Step | What to do | Time |
|---|---|---|
| 1 | Calibrate `tube_water` on the checkerboard | 20 min |
| 2 | `set_mounting.py tube_water` on that machine | 1 min |
| 3 | Place the tags, build the map | 30 min |
| 4 | `measure_tag_noise.py` → `SIGMA_PIXEL` | 10 min |
| 5 | `world_frame_check.py`, drive 30 s → 2 lines | 10 min |
| 6 | `world_frame_check.py`, 15 tape measurements → verdict | 30 min |
| 7 | `imu_realsense.py` → already done, re-check if in doubt | 5 min |

Everything measured lands in **one file**: `kalman/kalman_filter.py`, block
"THE NUMBERS TO MEASURE". The scripts print the lines to copy. Nothing else in
the repository needs touching.
