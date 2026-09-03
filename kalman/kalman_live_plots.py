"""kalman_live_plots.py — The course figures, drawn LIVE on real measurements.

===========================================================================
HOW TO USE IT
===========================================================================
This module does not run on its own. It opens from a real measurement run:

    python localization/world_frame_check.py --plots

A window opens next to the video and updates every frame with what the
camera actually measures. Closing the window does not stop the measurement;
quitting the measurement with 'q' closes everything and saves the figure to
kalman_live_plots_session.png.

To check the module itself without a camera:

    python kalman/kalman_live_plots.py

It replays the six panels on simulated data and saves a PNG.

Requires matplotlib:  python -m pip install matplotlib
Without it the measurement still runs, just with no plots. At the pool a
measurement is not redone because a plotting library is missing.

===========================================================================
WHY THIS MODULE EXISTS
===========================================================================
The project's reference document — Alex Becker, "Kalman Filter Explained
Through Examples", kalmanfilter.net — explains the filter through a handful
of figures. The repository could already produce them in SIMULATION
(demos/demo_kalman.py), which proves the maths are right but shows nothing
about the real system. Here the same figures are drawn on what the camera
measures at that very instant.

===========================================================================
THE SIX PANELS, AND WHAT EACH ONE PROVES
===========================================================================
  1. BAYESIAN UPDATE — prior, likelihood, posterior
     The three Gaussians of the document's central figure. The prior is what
     the filter believed BEFORE the measurement; the likelihood is what the
     tag just said; the posterior is the compromise. What to look for: the
     posterior is ALWAYS narrower than both others, and falls between them.
     That is the filter's defining property, and it can be checked by eye,
     frame by frame.

  2. ESTIMATE vs RAW MEASUREMENT over time, with the +/- 1 sigma band
     The document's tracking figure. The filtered curve must be smoother than
     the raw measurements, and the band must CONTAIN the measurements roughly
     two times out of three — that is what "1 sigma" means. A band that
     contains nothing is a filter lying to itself.

  3. UNCERTAINTY over time
     It falls when measurements arrive and RISES as soon as they stop. This
     is where the IMU shows its worth: without it the uncertainty explodes
     within the first second without a tag.

  4. KALMAN GAIN over time
     K = 0: the filter ignores the measurement and trusts only its prediction.
     K = 1: it throws its prediction away and takes the measurement at face
     value. In between is the trade-off. It should settle after a few
     seconds; if it stays pinned at 1, the filter is not filtering anything.

  5. QUALITY OF EACH TAG — this system's equivalent of the document's "SNR"
     The document illustrates measurement quality with a strong or weak radar
     echo in noise. Here the measurement comes from a tag, not an echo, and
     its quality is set by GEOMETRY: lateral sigma grows like the distance,
     DEPTH sigma like its SQUARE, and it degrades further when the tag is
     seen at an angle. The black dots are the tags actually visible at that
     instant. This is the numeric justification for the filter: one tag, far
     away or seen edge-on, is not enough.

  6. OUTLIER TEST — the document's "Outlier Treatment"
     Each measurement is compared with what the filter expected; the
     normalised gap follows a chi-square law with 3 degrees of freedom.
     Beyond the threshold the measurement is rejected. This is how a flipped
     tag is caught before it poisons the estimate.

===========================================================================
WHAT TO LOOK AT, AND WHAT NOT TO BELIEVE
===========================================================================
None of these figures says whether the position is CORRECT — there is no
ground truth in a real run. They say whether the filter behaves the way a
Kalman filter must behave. To know whether it tells the truth, it has to be
confronted with a tape measure: that is step 6 of docs/kalman_protocol.md.
"""
import sys
from collections import deque
from pathlib import Path

import numpy as np

# matplotlib is optional: without it the run must continue with no plots
# rather than refuse to start. A pool measurement is not redone because a
# display library is missing.
try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:                                    # pragma: no cover
    matplotlib = plt = None

HISTORY = 600                # points kept in the curves (20 s at 30 Hz)
REDRAW_EVERY = 5             # frames between two redraws
AXIS_NAMES = ("x", "y", "z")


def available():
    """Is matplotlib installed?"""
    return plt is not None


class KalmanLivePlots:
    """The six course figures, fed by a real PoseFilter.

    Usage:
        plots = KalmanLivePlots(axis=0)
        ...
        plots.add(timestamp, pose_filter, raw_measurement, tags=[...])
        plots.refresh()                      # same cadence, throttled
        ...
        plots.save(path)
        plots.close()
    """

    def __init__(self, axis=0, title="Kalman filter, live",
                 subtitle="REAL measurements from the camera"):
        if not available():
            raise RuntimeError("matplotlib is not installed")
        self.axis = int(axis)
        self.frame_count = 0
        self.t = deque(maxlen=HISTORY)
        self.raw = deque(maxlen=HISTORY)
        self.filtered = deque(maxlen=HISTORY)
        self.sigma = deque(maxlen=HISTORY)
        self.gain = deque(maxlen=HISTORY)
        self.t_gain = deque(maxlen=HISTORY)
        self.t_raw = deque(maxlen=HISTORY)
        self.rejected_t = deque(maxlen=HISTORY)
        self.rejected_v = deque(maxlen=HISTORY)
        self.mahalanobis = deque(maxlen=HISTORY)
        self.t_mahalanobis = deque(maxlen=HISTORY)
        self.threshold = None
        self.visible_tags = []      # [(id, distance_m, incidence_deg)] now
        self.last_update = None

        plt.ion()
        self.figure, grid = plt.subplots(2, 3, figsize=(17, 8))
        self.figure.canvas.manager.set_window_title(title)
        self.figure.suptitle(f"{title} — axis {AXIS_NAMES[self.axis]} — {subtitle}",
                             fontsize=12, fontweight="bold")
        ((self.ax_bayes, self.ax_track, self.ax_quality),
         (self.ax_sigma, self.ax_gain, self.ax_outlier)) = grid
        self._label_axes()
        self.figure.tight_layout()
        self.figure.canvas.draw()
        plt.show(block=False)

    def _label_axes(self):
        a = AXIS_NAMES[self.axis]
        self.ax_bayes.set_title("1. Bayesian update (current instant)")
        self.ax_bayes.set_xlabel(f"position {a} (m)")
        self.ax_bayes.set_ylabel("probability density")

        self.ax_track.set_title("2. Estimate vs raw measurement")
        self.ax_track.set_xlabel("time (s)")
        self.ax_track.set_ylabel(f"position {a} (m)")

        self.ax_quality.set_title("5. Quality of each tag (this system's \"SNR\")")
        self.ax_quality.set_xlabel("tag distance (m)")
        self.ax_quality.set_ylabel("measurement sigma (mm)")

        self.ax_sigma.set_title("3. Reported uncertainty (1 sigma)")
        self.ax_sigma.set_xlabel("time (s)")
        self.ax_sigma.set_ylabel("sigma (mm)")

        self.ax_gain.set_title("4. Kalman gain")
        self.ax_gain.set_xlabel("time (s)")
        self.ax_gain.set_ylabel("K (0 = ignore measurement, 1 = trust it)")
        self.ax_gain.set_ylim(-0.05, 1.05)

        self.ax_outlier.set_title("6. Outlier test")
        self.ax_outlier.set_xlabel("time (s)")
        self.ax_outlier.set_ylabel("Mahalanobis distance^2")

    # -- collection ---------------------------------------------------------
    def add(self, timestamp, pose_filter, raw_measurement=None, tags=None):
        """Record the filter state at this instant. Call it every frame.

        tags: [(identifier, distance_m, incidence_deg)] of the visible tags,
        for panel 5. Optional: without it that panel stays empty.
        """
        position = pose_filter.position
        if tags is not None:
            self.visible_tags = list(tags)
        if not position.started:
            return
        self.t.append(timestamp)
        self.filtered.append(float(position.x[self.axis]))
        self.sigma.append(1000.0 * float(np.sqrt(position.P[self.axis, self.axis])))
        if raw_measurement is not None:
            self.t_raw.append(timestamp)
            self.raw.append(float(np.asarray(raw_measurement).ravel()[self.axis]))

        # `last_update` PERSISTS between frames: without this identity check a
        # single rejected measurement would be counted again on every frame
        # until the next one arrives. Since the filter builds a fresh dict on
        # each update, `is not` is enough and cannot get it wrong.
        update = position.last_update
        if (update is not None and "x_posterior" in update
                and update is not self.last_update):
            self.last_update = update
            self.gain.append(float(update["K"][self.axis, self.axis]))
            self.t_gain.append(timestamp)
            self.mahalanobis.append(float(update["mahalanobis"]))
            self.t_mahalanobis.append(timestamp)
            self.threshold = float(update["threshold"])
            if not update.get("accepted", True):
                self.rejected_t.append(timestamp)
                self.rejected_v.append(float(update["z"][self.axis]))

    # -- drawing ------------------------------------------------------------
    def refresh(self, force=False):
        """Redraw, at most once every REDRAW_EVERY frames."""
        self.frame_count += 1
        if not force and self.frame_count % REDRAW_EVERY:
            return
        if not self.t:
            return
        try:
            self._draw_bayes()
            self._draw_track()
            self._draw_sigma()
            self._draw_gain()
            self._draw_quality()
            self._draw_outlier()
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
        except Exception:
            # A window closed by hand must not bring the measurement down.
            pass

    def _draw_bayes(self):
        self.ax_bayes.clear()
        a = AXIS_NAMES[self.axis]
        self.ax_bayes.set_title("1. Bayesian update (current instant)")
        self.ax_bayes.set_xlabel(f"position {a} (m)")
        self.ax_bayes.set_ylabel("probability density")
        update = self.last_update
        if update is None:
            self.ax_bayes.text(0.5, 0.5, "waiting for a measurement",
                               ha="center", transform=self.ax_bayes.transAxes)
            return
        i = self.axis
        laws = [
            ("prior P(x)", update["x_prior"][i],
             np.sqrt(update["P_prior"][i, i]), "green"),
            ("likelihood P(z|x)", update["z"][i],
             np.sqrt(update["R"][i, i]), "red"),
            ("posterior P(x|z)", update["x_posterior"][i],
             np.sqrt(update["P_posterior"][i, i]), "blue"),
        ]
        spread = max(s for _, _, s, _ in laws)
        centre = np.mean([m for _, m, _, _ in laws])
        grid = np.linspace(centre - 4 * spread, centre + 4 * spread, 400)
        for name, mean, sd, colour in laws:
            sd = max(float(sd), 1e-9)
            density = (np.exp(-0.5 * ((grid - mean) / sd) ** 2)
                       / (sd * np.sqrt(2 * np.pi)))
            self.ax_bayes.plot(grid, density, color=colour, lw=2, label=name)
            if colour == "blue":
                self.ax_bayes.fill_between(grid, density, alpha=0.2, color="blue")
        if not update.get("accepted", True):
            self.ax_bayes.set_title("1. Bayesian update — MEASUREMENT REJECTED",
                                    color="crimson")
        self.ax_bayes.legend(fontsize=8)
        self.ax_bayes.grid(alpha=0.3)

    def _draw_track(self):
        self.ax_track.clear()
        a = AXIS_NAMES[self.axis]
        self.ax_track.set_title("2. Estimate vs raw measurement")
        self.ax_track.set_xlabel("time (s)")
        self.ax_track.set_ylabel(f"position {a} (m)")
        t = np.fromiter(self.t, float)
        f = np.fromiter(self.filtered, float)
        s = np.fromiter(self.sigma, float) / 1000.0
        if len(self.raw):
            self.ax_track.plot(np.fromiter(self.t_raw, float),
                               np.fromiter(self.raw, float), ".",
                               color="orange", ms=4, alpha=0.6,
                               label="raw measurement (tags)")
        self.ax_track.plot(t, f, "-", color="green", lw=1.8,
                           label="filter output")
        self.ax_track.fill_between(t, f - s, f + s, color="green", alpha=0.18,
                                   label="+/- 1 sigma reported")
        if len(self.rejected_t):
            self.ax_track.plot(np.fromiter(self.rejected_t, float),
                               np.fromiter(self.rejected_v, float), "x",
                               color="crimson", ms=7, label="rejected measurement")
        self.ax_track.legend(fontsize=8, loc="best")
        self.ax_track.grid(alpha=0.3)

    def _draw_sigma(self):
        self.ax_sigma.clear()
        self.ax_sigma.set_title("3. Reported uncertainty (1 sigma)")
        self.ax_sigma.set_xlabel("time (s)")
        self.ax_sigma.set_ylabel("sigma (mm)")
        self.ax_sigma.plot(np.fromiter(self.t, float),
                           np.fromiter(self.sigma, float),
                           "-", color="steelblue", lw=1.8)
        self.ax_sigma.grid(alpha=0.3)
        self.ax_sigma.text(0.02, 0.92, "falls when tags arrive, rises without them",
                           transform=self.ax_sigma.transAxes, fontsize=8,
                           color="gray")

    def _draw_gain(self):
        self.ax_gain.clear()
        self.ax_gain.set_title("4. Kalman gain")
        self.ax_gain.set_xlabel("time (s)")
        self.ax_gain.set_ylabel("K (0 = ignore measurement, 1 = trust it)")
        self.ax_gain.set_ylim(-0.05, 1.05)
        if len(self.gain):
            self.ax_gain.plot(np.fromiter(self.t_gain, float),
                              np.fromiter(self.gain, float),
                              "-", color="purple", lw=1.8)
        self.ax_gain.grid(alpha=0.3)

    def _draw_quality(self):
        """Panel 5 — this system's equivalent of the document's "SNR".

        The document illustrates measurement quality with a strong or weak
        radar echo in noise. Here the measurement comes from a tag, not an
        echo, and its quality is read off the GEOMETRY. That is what
        tag_position_covariance computes, and the two curves are its two
        terms:

            lateral sigma = d . sigma_px / f                grows like d
            depth sigma   = d^2 . sigma_px / (f.T.cos(i).2) grows like d^2

        Hence the counter-intuitive fact that depth degrades MUCH faster than
        lateral: at 3 m it is already an order of magnitude worse. That is
        the filter's reason to exist — one tag is not enough.
        """
        self.ax_quality.clear()
        self.ax_quality.set_title("5. Quality of each tag (this system's \"SNR\")")
        self.ax_quality.set_xlabel("tag distance (m)")
        self.ax_quality.set_ylabel("measurement sigma (mm)")
        try:
            from kalman_filter import (WATER_FOCAL_LENGTH, TAG_SIZE, SIGMA_PIXEL,
                                       CORNERS_PER_TAG)
        except Exception:
            return
        distances = np.linspace(0.3, 4.0, 120)
        lateral = 1000 * distances * SIGMA_PIXEL / WATER_FOCAL_LENGTH
        self.ax_quality.plot(distances, lateral, "-", color="seagreen", lw=2,
                             label="lateral (grows like d)")
        for incidence, style in ((0.0, "-"), (45.0, "--")):
            cos_i = max(np.cos(np.radians(incidence)), 0.20)
            depth = (1000 * distances ** 2 * SIGMA_PIXEL
                     / (WATER_FOCAL_LENGTH * TAG_SIZE * cos_i
                        * np.sqrt(CORNERS_PER_TAG)))
            self.ax_quality.plot(distances, depth, style, color="indianred",
                                 lw=2, label=f"depth, {incidence:.0f} deg off-axis")
        # The tags ACTUALLY seen right now, placed on those curves.
        for identifier, distance, incidence in self.visible_tags:
            cos_i = max(np.cos(np.radians(incidence)), 0.20)
            sigma = (1000 * distance ** 2 * SIGMA_PIXEL
                     / (WATER_FOCAL_LENGTH * TAG_SIZE * cos_i
                        * np.sqrt(CORNERS_PER_TAG)))
            self.ax_quality.plot([distance], [sigma], "o", color="black", ms=8,
                                 zorder=5)
            self.ax_quality.annotate(f" tag {identifier}", (distance, sigma),
                                     fontsize=8, va="bottom")
        self.ax_quality.set_yscale("log")
        self.ax_quality.legend(fontsize=7, loc="upper left")
        self.ax_quality.grid(alpha=0.3, which="both")

    def _draw_outlier(self):
        """Panel 6 — the document's "Outlier Treatment", live.

        Every measurement is compared with what the filter expected. The gap,
        normalised by the uncertainty of both (Mahalanobis distance), follows
        a chi-square law with 3 degrees of freedom: beyond the threshold the
        measurement is too improbable to be true and gets rejected. This is
        how a flipped tag is caught before it poisons the estimate.
        """
        self.ax_outlier.clear()
        self.ax_outlier.set_title("6. Outlier test")
        self.ax_outlier.set_xlabel("time (s)")
        self.ax_outlier.set_ylabel("Mahalanobis distance^2")
        if len(self.mahalanobis):
            t = np.fromiter(self.t_mahalanobis, float)
            d = np.fromiter(self.mahalanobis, float)
            self.ax_outlier.plot(t, d, ".-", color="darkorange", lw=1, ms=4,
                                 label="measurement vs prediction gap")
            if self.threshold:
                self.ax_outlier.axhline(
                    self.threshold, color="crimson", ls="--", lw=1.5,
                    label=f"threshold ({self.threshold:.1f})")
                over = d > self.threshold
                if over.any():
                    self.ax_outlier.plot(t[over], d[over], "x", color="crimson",
                                         ms=8, label="rejected")
            self.ax_outlier.set_yscale("symlog")
            self.ax_outlier.legend(fontsize=7, loc="upper left")
        self.ax_outlier.grid(alpha=0.3)

    # -- end ----------------------------------------------------------------
    def save(self, path):
        """Save the current figure, to attach it to a report."""
        try:
            self.figure.savefig(path, dpi=130, bbox_inches="tight")
            return True
        except Exception:
            return False

    def close(self):
        try:
            plt.close(self.figure)
        except Exception:
            pass


def _self_check():
    """Replay the module on simulated data, with no camera.

    Checks that the six panels draw and update — the real run needs the
    camera, this check does not.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from kalman_filter import PoseFilter, tag_position_covariance

    if not available():
        print("matplotlib is not installed:")
        print("    python -m pip install matplotlib")
        return 1
    matplotlib.use("Agg")          # no window: we only check the drawing

    rng = np.random.default_rng(4)
    pose_filter = PoseFilter()
    plots = KalmanLivePlots(axis=0, title="Self-check",
                            subtitle="SIMULATED DATA, not a real measurement")
    dt = 1 / 30
    p = np.zeros(3)
    tag = np.array([2.0, 0.0, 0.0])
    # The injected noise is DRAWN FROM THE MODEL's covariance, not picked at
    # random. A first attempt injected 8 mm on all three axes when the model
    # expects 3.2 in depth and 0.7 laterally: 51 measurements out of 80 were
    # rejected, which made the filter look paranoid when it was the
    # simulation that was lying.
    #
    # It also travels ALONGSIDE the tag, not into it: a first attempt moved
    # 2 m towards a tag placed at 2 m, so the camera ended up INSIDE the tag,
    # where the covariance degenerates and the gain runs to 1.000.
    for k in range(240):
        p = p + np.array([0.0, 0.25, 0.0]) * dt
        pose_filter.predict(dt)
        measurement = None
        if k % 3 == 0:
            R = tag_position_covariance(p, tag, 15.0)
            measurement = p + rng.multivariate_normal(np.zeros(3), R)
            if k == 120:                      # one outlier, to see it caught
                measurement = measurement + np.array([0.4, 0.0, 0.0])
            pose_filter.add_tag(measurement, tag, 15.0)
            pose_filter.apply()
        plots.add(k * dt, pose_filter, measurement,
                  tags=[(0, float(np.linalg.norm(tag - p)), 15.0)])
        plots.refresh()
    plots.refresh(force=True)

    output = Path(__file__).resolve().with_name("kalman_live_plots_demo.png")
    saved = plots.save(output)
    plots.close()
    print("The six panels drew correctly over 240 simulated frames.")
    print(f"  curve points       : {len(plots.t)}")
    print(f"  measurements rejected: {len(plots.rejected_t)}  (1 was injected)")
    print(f"  final gain         : {plots.gain[-1]:.3f}")
    print(f"  final sigma        : {plots.sigma[-1]:.1f} mm")
    if saved:
        print(f"  figure saved       : {output}")
    print("\nTo see them LIVE on the real camera:")
    print("    python localization/world_frame_check.py --plots")
    return 0


if __name__ == "__main__":
    sys.exit(_self_check())
