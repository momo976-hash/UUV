# Legacy notes — AprilTag localisation (Task 1)

These are the original Task 1 notes, kept for the record. They describe
`src/apriltag_pose.py` alone, before the calibration, mapping, IMU and Kalman
work that the rest of the repository now holds. **For anything current, read
the top-level `README.md` and `docs/kalman_protocol.md` instead.**

Locating an underwater vehicle (UUV) with an **Intel RealSense** camera and
**AprilTag** markers, in **pure Python** (no ROS 2 at this stage).

This covers **steps 1 → 4**: show the camera stream, retrieve the intrinsic
parameters, detect the AprilTags and estimate the **3D pose** (position +
orientation) live.

## Where to run the code

The script runs **in a terminal**, on the computer connected to the camera.
There are two comfortable options:

### Option A — Visual Studio Code (recommended)
1. Install **VS Code** + the **Python** extension (Microsoft).
2. Open the project folder: `File > Open Folder…` → choose `UUV`.
3. Open an integrated terminal: `Terminal > New Terminal`.
4. Follow the installation below, then run the script from that terminal
   (or with the ▶ "Run Python File" button at the top right).

> This is exactly the workflow discussed in the meeting: coding in VS Code,
> and later connecting to it remotely over **SSH** on the Raspberry Pi
> (the "Remote - SSH" extension) once we move to the embedded hardware.

### Option B — a terminal on its own
Any terminal (PowerShell, bash, the macOS terminal…) will do: you type the
same commands. VS Code is not required, just more convenient.

> ⚠️ The script opens a **video window** (`cv2.imshow`). So it needs an
> environment with a **graphical display** (your laptop, not a remote server
> with no screen). On a Raspberry Pi over SSH with no desktop, this will be
> adapted later (X11 forwarding, or writing to a file).

## Installation

```bash
# 1. Create a virtual environment (isolates the dependencies)
python3 -m venv venv

# 2. Activate it
#    Linux / macOS:
source venv/bin/activate
#    Windows (PowerShell):
#    venv\Scripts\Activate.ps1

# 3. Install the dependencies
pip install -r requirements.txt
```

## Preparing the markers

1. Generate/print AprilTags of the **`tag36h11`** family
   (official generator: https://github.com/AprilRobotics/apriltag-imgs).
2. Print them and **measure the black square's side precisely** (e.g. 10 cm).
   That measurement is a critical parameter of the distance calculation.

> The repository now has `tools/print_tag.py`, which generates a printable A4
> page at an exact physical size, with a checking ruler to catch a printer
> that has rescaled the page.

## Running it

```bash
# With the RealSense camera (10 cm tag):
python src/apriltag_pose.py --tag-size 0.10

# To TEST without the RealSense, using the PC's webcam:
# (the distances will not be reliable — the intrinsics are approximated — but
#  the detection and the axis display do work)
python src/apriltag_pose.py --source webcam --tag-size 0.10
```

On screen: a green outline + each tag's id, the 3D axes (X red, Y green,
Z blue), the distance and the roll/pitch/yaw angles. Press **`q`** or **Esc**
to quit.

### Checking that it works (success criteria)
- The 3D axes "stick" to the tag as you move the camera.
- The distance shown matches reality (check with a tape measure).
- A tag seen squarely head-on gives angles close to 0°.

## Good practice from the reference papers
- **Distance < 4 m**: beyond that the pose error explodes.
- **Avoid a perfectly head-on view**: a slight angle (roll/pitch) makes the
  depth Z better observable.
- **Several non-coplanar tags**: the error drops (≈40 cm → ≈8 cm at 5 m going
  from 1 tag to 4).

## Next steps (not included here)

All three have since been done, and are described in the top-level `README.md`:

- **Step 5**: frame transforms (the camera's pose in the *world* from a tag
  map) → `localization/world_frame_check.py`, `localization/auto_mapping.py`.
- **Step 6**: multi-tag fusion + a Kalman filter → `kalman/kalman_filter.py`.
- **Step 7**: CSV recording and error analysis for the report → every
  measurement script now appends to a CSV of its own.
