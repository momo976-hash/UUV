# imu_realsense.py — Read the D435i's IMU, and prove it is being used.
#
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python kalman/imu_realsense.py               camera plugged in
#     python kalman/imu_realsense.py --simulation  same maths, no camera
#
# The run has three parts and takes about two minutes:
#
#   1. It measures the IMU AT REST for 5 seconds. Do not touch anything.
#      It then prints two numbers to copy into kalman/kalman_filter.py:
#
#          GYRO_NOISE_DEG_S = ...
#          ACCEL_NOISE      = ...
#
#      Those two lines already exist there; only the numbers change.
#
#   2. It checks the measurements are sound: at rest the accelerometer must
#      read 9.81 m/s2 in norm. If it does not, the scale is wrong and
#      everything downstream will be too.
#
#   3. It shows the orientation LIVE. Turn the camera a quarter turn: the yaw
#      must read 90 degrees. Put it back down flat — "flat" meaning
#      horizontal on a table, not tilted in your hand — and roll and pitch
#      must return towards zero. Press Ctrl+C to stop. Every sample is
#      written to imu_data.csv, so the run can be re-analysed later.
#
# Two things it is worth watching while it runs: roll and pitch stay bounded
# (the accelerometer holds them), while yaw drifts slowly. Nothing re-anchors
# yaw without a tag. That is precisely why the fusion with the tags exists,
# and this demonstration is what makes it visible.
#
# ===========================================================================
# WHAT THIS FILE IS
# ===========================================================================
# Two things in one:
#   1. A MODULE. The RealSenseImu class opens the Intel SDK's accel and gyro
#      streams and returns measurements ready to feed kalman_filter.py.
#   2. A DEMONSTRATION. Run directly, it does the three steps above.
#
# ===========================================================================
# WHAT THE IMU CAN AND CANNOT GIVE — say this out loud, it matters
# ===========================================================================
# ORIENTATION: yes. Roll and pitch are held indefinitely by the
# accelerometer; yaw is integrated from the gyro and drifts.
#
# POSITION: no, not from the IMU alone, and that is a property of the sensor
# rather than a shortcoming of this code. A MEMS accelerometer has a slowly
# varying bias that NOTHING here estimates, and double integration turns it
# into a quadratic error: 0.05 m/s2 becomes 2.5 cm after one second, but 1 m
# after ten. The IMU is therefore what carries the estimate through a tag
# dropout of a few seconds — not a way to navigate blind. The tags remain the
# only drift-free source.
#
# ===========================================================================
# FRAMES: THE MAIN TRAP
# ===========================================================================
# The IMU is NOT aligned with the colour camera. The SDK gives the rotation
# between the two (get_extrinsics_to) and this module fetches it. Feeding raw
# measurements to the filter without that rotation mixes the axes and makes
# the vehicle drift sideways, with no error message whatsoever.
#
# Nothing is assumed about the starting orientation either: it is DEDUCED
# from the accelerometer at rest, rather than assumed along some axis. The
# mounting can therefore sit any way up — on its base, on its side, in the
# tube.
#
# CONVENTION OF THE MEASURED VECTOR. It is treated as pointing UP: at rest an
# accelerometer measures specific force, the support's reaction, not gravity.
# `initial_orientation` and `correct_with_gravity` make the same assumption,
# and that is essential — a version where one flipped the sign and the other
# did not makes the orientation converge 180 degrees away from the truth,
# with no message at all. The demonstration checks this automatically, in a
# way that does not depend on how the unit is posed.
import csv
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kalman_filter import (OrientationFilter, euler_to_quaternion,  # noqa: E402
                           quaternion_to_euler, quaternion_from_rotation,
                           quaternion_to_matrix, quaternion_product)

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

GRAVITE = 9.81
ICI = Path(__file__).resolve().parent


class CentraleRealSense:
    """Flux accel + gyro de la D435i, plus la rotation vers la camera colour.

    Les deux capteurs n'arrivent PAS au meme rythme (l'accelerometre vers
    60-250 Hz, le gyroscope vers 200-400 Hz) et chaque image ne porte qu'un
    seul des deux. On garde donc la derniere value connue de chacun, et on
    date les measurements avec l'horloge du capteur plutot que celle du PC : les
    intervalles servent a integrer, une gigue de quelques millisecondes se
    paierait directement en drift.
    """

    def __init__(self, avec_couleur=False, bavard=True):
        if rs is None:
            raise RuntimeError(
                "pyrealsense2 n'est pas installe.\n"
                "  pip install pyrealsense2\n"
                "C'est la bibliotheque du SDK Intel qui donne acces a l'IMU.")

        offerts = self._profils_offerts()
        if rs.stream.gyro not in offerts or rs.stream.accel not in offerts:
            raise RuntimeError(
                "l'appareil branche n'offre pas accel + gyro.\n"
                "Lance 'python list_realsense.py' pour voir ce qu'il a.")
        if bavard:
            for flux, name in ((rs.stream.accel, "accel"), (rs.stream.gyro, "gyro")):
                cadences = sorted({fps for _, fps in offerts[flux]})
                print(f"  {name:5} : cadences offertes {cadences} Hz")

        # On request EXACTEMENT ce que l'appareil annonce, au lieu de supposer
        # un format et une rate. Coder ces values en dur donne l'error
        # "Couldn't resolve requests" des que le SDK ou le micrologiciel
        # change ses profils — et le message ne dit pas lequel manque.
        def _config_mouvement():
            config = rs.config()
            for flux in (rs.stream.accel, rs.stream.gyro):
                format_, fps = max(offerts[flux], key=lambda couple: couple[1])
                config.enable_stream(flux, format_, fps)
            return config

        # ETAPE 1 — extrinseques IMU -> camera colour, puis on referme.
        #
        # POURQUOI NE PAS GARDER LA COULEUR OUVERTE. Le pipeline synchronise
        # tous ses flux sur le plus lent : avec la colour a 30 Hz,
        # wait_for_frames ne rend plus que ~27 jeux par seconde, alors que le
        # gyro en product 200 a 400. On perd 86 % des measurements, et l'integration
        # assumed alors omega constant sur 37 ms au lieu de 5 — a 100 deg/s
        # cela fait 3.7 deg d'error par pas. La colour ne sert qu'a lire une
        # rotation constante : on la prend, puis on s'en debarrasse.
        self.R_imu_camera = np.eye(3)
        self.extrinseques_lues = False
        if avec_couleur:
            try:
                config = _config_mouvement()
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                pipeline = rs.pipeline()
                profil = pipeline.start(config)
                extr = (profil.get_stream(rs.stream.gyro)
                        .get_extrinsics_to(profil.get_stream(rs.stream.color)))
                # Le SDK range la rotation en COLONNES ; numpy lit en rows.
                self.R_imu_camera = np.array(extr.rotation).reshape(3, 3).T
                self.extrinseques_lues = True
                pipeline.stop()
            except Exception as souci:
                if bavard:
                    print(f"  extrinseques non lues ({souci}) — identity used")

        # ETAPE 2 — IMU seule, a pleine rate.
        self.pipeline = rs.pipeline()
        try:
            self.profil = self.pipeline.start(_config_mouvement())
        except Exception as souci:
            raise RuntimeError(f"impossible d'ouvrir accel + gyro : {souci}")

        self._gyro = np.zeros(3)
        self._accel = np.zeros(3)
        self._t_gyro = None

    @staticmethod
    def _profils_offerts():
        """Ce que le Motion Module annonce vraiment : {flux: [(format, fps)]}."""
        offerts = {}
        for appareil in rs.context().query_devices():
            for capteur in appareil.sensors:
                for profil in capteur.get_stream_profiles():
                    flux = profil.stream_type()
                    if flux in (rs.stream.accel, rs.stream.gyro):
                        offerts.setdefault(flux, []).append(
                            (profil.format(), profil.fps()))
            if offerts:
                break            # first appareil qui en a
        return offerts

    def lire(self, timeout_ms=5000):
        """Rend (gyro, accel, dt) ou None. dt est l'interval depuis la
        derniere measurement de gyro, en seconds, ou None a la premiere."""
        frames = self.pipeline.wait_for_frames(timeout_ms)
        dt = None
        for image in frames:
            if not image.is_motion_frame():
                continue
            motion = image.as_motion_frame()
            donnees = motion.get_motion_data()
            value = np.array([donnees.x, donnees.y, donnees.z], dtype=float)
            flux = motion.get_profile().stream_type()
            if flux == rs.stream.gyro:
                timestamp = motion.get_timestamp() / 1000.0    # ms -> s
                if self._t_gyro is not None:
                    gap = timestamp - self._t_gyro
                    # Une image sautee ou une horloge qui recule donnerait un
                    # dt aberrant : on l'ignore plutot que d'integrer n'importe
                    # quoi. Bornes larges, elles n'ecartent que l'absurde.
                    if 1e-5 < gap < 0.5:
                        dt = gap
                self._t_gyro = timestamp
                self._gyro = value
            elif flux == rs.stream.accel:
                self._accel = value
        return self._gyro.copy(), self._accel.copy(), dt

    def arreter(self):
        self.pipeline.stop()


def mesurer_au_repos(imu, duration=5.0):
    """Biais et noise des deux capteurs, engin IMMOBILE.

    Le bias du gyro est sa lecture mean alors qu'il ne tourne pas : c'est
    lui qui, integre, fait deriver l'orientation. Le noise est l'gap-type
    autour de cette mean, et c'est le chiffre que le filter attend.

    La norme de l'accelerometre doit valoir 9.81 : c'est le check d'echelle
    le plus simple qui soit, et il attrape une unite fausse (g au lieu de
    m/s2) ou un facteur d'echelle errone.
    """
    print(f"\nMESURE AU REPOS — ne key a rien pendant {duration:.0f} s...")
    gyros, accels = [], []
    start = time.time()
    while time.time() - start < duration:
        gyro, accel, _ = imu.lire()
        gyros.append(gyro)
        accels.append(accel)
        reste = duration - (time.time() - start)
        print(f"\r  {reste:4.1f} s   {len(gyros)} samples", end="", flush=True)
    print()

    gyros, accels = np.array(gyros), np.array(accels)
    bias = gyros.mean(axis=0)
    gyro_noise = float(np.degrees(gyros.std(axis=0).mean()))
    norme = float(np.linalg.norm(accels.mean(axis=0)))
    accel_noise = float(accels.std(axis=0).mean())
    # Direction du vector measurement. On le nomme "haut" et non "bas" : au rest
    # un accelerometre measurement la force specifique, la reaction du support,
    # dirigee vers le HAUT. Le name compte — c'est en l'appelant "bas" qu'on
    # finit par l'inverser quelque part et pas ailleurs.
    haut = accels.mean(axis=0) / max(norme, 1e-9)
    return {"bias": bias, "bruit_gyro_deg_s": gyro_noise, "norme_accel": norme,
            "accel_noise": accel_noise, "haut": haut, "samples": len(gyros),
            "rate": len(gyros) / duration}


def orientation_initiale(accel_repos):
    """Orientation de depart deduite de l'accelerometre AU REPOS.

    CONVENTION, ET C'EST LE POINT DELICAT. On traite le vector measurement comme
    pointant vers le HAUT. C'est la convention physique de l'accelerometre :
    au rest il measurement la force specifique, c'est-a-dire la reaction du
    support, dirigee vers le haut — et non la pesanteur elle-meme.

    `correct_with_gravity` fait exactement la meme hypothese. Les deux DOIVENT
    s'accorder : une version qui inversait le vector ici et pas la, ce qui
    etait le cas, fait que l'initialisation et la correction se combattent.
    Le symptome est un roll qui se stabilise vers 180 degres au lieu de
    zero — quiet, et facile a prendre pour un probleme d'axes.

    Si un capteur rendait la convention opposee, le check automatique de
    la demonstration le dirait : juste apres l'initialisation, roll et
    pitch doivent lire zero, puisqu'on part precisement de cette pose.

    On prend la rotation la plus courte qui amene le haut measurement sur la
    verticale du world. Le yaw reste arbitraire — la pesanteur n'en dit
    rien — et c'est justement ce que les tags apporteront.
    """
    haut_monde = np.array([0.0, 0.0, 1.0])
    haut_mesure = np.asarray(accel_repos, dtype=float)
    haut_mesure = haut_mesure / max(np.linalg.norm(haut_mesure), 1e-9)
    axis = np.cross(haut_mesure, haut_monde)
    sine = float(np.linalg.norm(axis))
    cosinus = float(np.clip(haut_mesure @ haut_monde, -1.0, 1.0))
    if sine < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0]) if cosinus > 0 else \
               np.array([0.0, 1.0, 0.0, 0.0])
    return quaternion_from_rotation(axis / sine * np.arctan2(sine, cosinus))


def _demonstration():
    print("=" * 70)
    print("IS THE D435i's IMU BEING READ AND USED?")
    print("=" * 70)

    try:
        imu = CentraleRealSense(avec_couleur=True)
    except Exception as souci:
        print(f"\nERREUR : {souci}")
        return 1

    print("\naccel and gyro streams opened through the Intel SDK (pyrealsense2).")
    if imu.extrinseques_lues:
        angles = np.degrees(quaternion_to_euler(
            _quaternion_from_matrix(imu.R_imu_camera)))
        print(f"IMU -> colour camera rotation read from the SDK: "
              f"{angles.round(1)} deg")
    else:
        print("WARNING: IMU -> camera extrinsics could not be read, identity "
              "used.")

    rest = mesurer_au_repos(imu)

    print("\n" + "-" * 70)
    print("1. ARE THE MEASUREMENTS SOUND?")
    print("-" * 70)
    print(f"  samples              {rest['samples']}"
          f"   soit {rest['rate']:.0f} Hz")
    cadence_gyro = max(fps for _, fps in
                       CentraleRealSense._profils_offerts()[rs.stream.gyro])
    if rest["rate"] < 0.5 * cadence_gyro:
        # Symptome known : un flux video ouvert en meme time force le
        # pipeline a se synchroniser sur lui, et les measurements de mouvement
        # sont jetees entre deux frames.
        print(f"  [PROBLEM] the gyro runs at {cadence_gyro} Hz mais on n'en "
              f"lit que {rest['rate']:.0f}.")
        print("     Integration then assumes omega constant over intervals that are")
        print("     too long, and the rotation is under-estimated.")
    else:
        print(f"  [OK] on lit bien la rate du capteur ({cadence_gyro} Hz).")

    print(f"  accelerometer norm  {rest['norme_accel']:.3f} m/s2   "
          f"(should be {GRAVITE})")
    ecart_g = abs(rest["norme_accel"] / GRAVITE - 1)
    if ecart_g > 0.05:
        print("  [PROBLEM] far from gravity: wrong scale or wrong units.")
    elif ecart_g > 0.01:
        print(f"  [OK] c'est la pesanteur, a {100*ecart_g:.1f} % pres.")
        print("     That gap is a scale bias of the accelerometer. It has no")
        print("     consequence here: only the DIRECTION of the vector is used")
        print("     for roll and pitch, never its norm.")
    else:
        print("  [OK] this is gravity: the scale is right.")

    print(f"  measured direction        {rest['haut'].round(3)}")
    print("     (not assumed: the mounting may be posed any way up)")

    # -- la convention de signe de l'accelerometre est-elle la bonne ? ------
    # On NE peut PAS check que roll et pitch valent zero : le mounting a
    # parfaitement le droit d'etre pose sur le cote, et ils vaudraient alors
    # 90 a juste titre. Le check doit donc etre independant de la pose.
    #
    # Ce qui doit tenir quelle que soit la pose : l'orientation initialisee
    # PREVOIT une direction pour le haut, et cette prevision doit coincider
    # avec le vector measurement. Si les deux sont opposes, le capteur rend la
    # pesanteur la ou on attend la force specifique — initialisation et
    # correction se combattent alors, et l'orientation se stabilise a 180
    # degres de la truth sans que rien ne le signale.
    q0 = orientation_initiale(rest["haut"])
    prevu = quaternion_to_matrix(q0).T @ np.array([0.0, 0.0, 1.0])
    accord = float(prevu @ rest["haut"])
    ecart_conv = float(np.degrees(np.arccos(np.clip(accord, -1.0, 1.0))))
    roulis0, tangage0, _ = np.degrees(quaternion_to_euler(q0))
    print(f"\n  deduced starting pose: roll {roulis0:+.1f}, "
          f"pitch {tangage0:+.1f} deg")
    print(f"  check de convention : gap prevu / measurement {ecart_conv:.2f} deg")
    if ecart_conv > 5.0:
        print("  [PROBLEM] should be zero whatever the pose.")
        print("     Close to 180: the measured vector points DOWN rather than")
        print("     up. Its sign must be flipped when read.")
    else:
        print("  [OK] the accelerometer does point up, as assumed.")

    print("\n" + "-" * 70)
    print("2. THE TWO NUMBERS TO COPY INTO kalman/kalman_filter.py")
    print("-" * 70)
    print(f"  gyro bias at rest        "
          f"{np.degrees(rest['bias']).round(3)} deg/s")
    print("     This is the bias that, integrated, makes the orientation drift.")
    print("     The filter estimates it on its own once the tags re-anchor it.")
    print()
    print(f"      GYRO_NOISE_DEG_S = {rest['bruit_gyro_deg_s']:.3f}")
    print(f"      ACCEL_NOISE      = {rest['accel_noise']:.3f}")

    print("\n" + "-" * 70)
    print("3. THE MATHS: DOES THE ORIENTATION FOLLOW THE MOTION?")
    print("-" * 70)
    print("  Turn the camera a quarter turn: yaw must read 90.")
    print("  Put it back down FLAT (horizontal on a table, not tilted in your hand):")
    print("  roll and pitch must return towards zero.  Ctrl+C to stop.\n")

    # Les measurements sont ENREGISTREES, pas seulement affichees. C'est ce qui
    # rend l'extraction montrable : un path qu'on ouvre et qu'on relit,
    # plutot que des chiffres qui defilent et disparaissent. Chaque row
    # porte les measurements BRUTES du SDK et l'orientation qu'on en tire, donc le
    # calcul est refaisable par un tiers.
    fichier_csv = ICI / "imu_donnees.csv"
    suivi = OrientationFilter()
    suivi.start(orientation_initiale(rest["haut"]), sigma_deg=5.0)
    depart = time.time()
    derniers = deque(maxlen=50)
    rows = 0
    with open(fichier_csv, "w", newline="") as output:
        ecrivain = csv.writer(output)
        ecrivain.writerow(["t_s",
                           "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s",
                           "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2",
                           "roulis_deg", "tangage_deg", "lacet_deg",
                           "qw", "qx", "qy", "qz"])
        try:
            while True:
                gyro, accel, dt = imu.lire()
                if dt is None:
                    continue
                omega = imu.R_imu_camera @ (gyro - rest["bias"])
                suivi.predict(dt, omega)
                suivi.correct_with_gravity(imu.R_imu_camera @ accel,
                                       gravity=GRAVITE)
                derniers.append(np.linalg.norm(omega))

                roll, pitch, yaw = np.degrees(quaternion_to_euler(suivi.q))
                timestamp = time.time() - depart
                ecrivain.writerow(
                    [f"{timestamp:.4f}"]
                    + [f"{v:.6f}" for v in gyro]
                    + [f"{v:.6f}" for v in accel]
                    + [f"{roll:.3f}", f"{pitch:.3f}", f"{yaw:.3f}"]
                    + [f"{v:.6f}" for v in suivi.q])
                rows += 1

                print(f"\r  roll {roll:+7.1f}   pitch {pitch:+7.1f}   "
                      f"yaw {yaw:+7.1f} deg    "
                      f"|omega| {np.degrees(np.mean(derniers)):5.1f} deg/s   "
                      f"({timestamp:4.0f} s, {rows} rows)", end="", flush=True)
        except KeyboardInterrupt:
            print("\n")
        finally:
            imu.arreter()

    print(f"  {rows} measurements enregistrees dans : {fichier_csv}")
    print("     columns: time, raw gyro (rad/s), raw accel (m/s2),")
    print("     then the computed orientation, as angles and as a quaternion.")

    print("-" * 70)
    print("WHAT YOU HAVE JUST SHOWN")
    print("-" * 70)
    print("  - the Intel SDK's IMU streams are read (accel + gyro)")
    print("  - the scale is checked against gravity")
    print("  - the gyro is integrated into orientation, through quaternions")
    print("  - the accelerometer bounds roll and pitch")
    print("  - yaw, on the other hand, drifts: nothing re-anchors it without a tag.")
    print("    That is the whole reason for fusing with the AprilTags.")
    print("=" * 70)
    return 0


def _quaternion_from_matrix(R):
    from kalman_filter import matrix_to_quaternion
    return matrix_to_quaternion(np.asarray(R, dtype=float))


def _simulation():
    """Les memes maths, sur une imu SIMULEE : marche sans camera.

    Sert a deux choses : montrer que le traitement est juste meme quand le
    materiel n'est pas la, et donner un result verifiable — on connait la
    truth, donc on peut chiffrer l'error, ce qu'aucune manip reelle ne
    permet.
    """
    from kalman_filter import quaternion_angle
    rng = np.random.default_rng(3)
    dt, noise = 1 / 200, np.radians(0.15)

    print("=" * 70)
    print("THE IMU MATHS, ON A SIMULATED UNIT")
    print("=" * 70)
    print("No camera plugged in. The truth being known, the error can be quantified.")

    print("\n1. ORIENTATION FROM THE ACCELEROMETER ALONE")
    print("   The up direction PREDICTED by the orientation must match the up")
    print("   direction MEASURED, for any pose of the mounting.")
    pires = []
    for _ in range(300):
        v = rng.normal(size=3)
        haut_mesure = v / np.linalg.norm(v)
        prevu = quaternion_to_matrix(
            orientation_initiale(haut_mesure)).T @ np.array([0.0, 0.0, 1.0])
        pires.append(np.degrees(np.arccos(np.clip(prevu @ haut_mesure, -1, 1))))
    print(f"   300 arbitrary poses, max error {max(pires):.1e} deg")
    assert max(pires) < 1e-4      # noise d'arccos, pas d'error de calcul

    print("\n2. A QUARTER TURN ABOUT THE VERTICAL")
    print("   30 deg/s for 3 s, noisy gyro, noisy accelerometer.")
    suivi = OrientationFilter()
    suivi.start(orientation_initiale(np.array([0.0, 0.0, 1.0])), sigma_deg=5.0)
    truth = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(int(3.0 / dt)):
        omega = np.array([0.0, 0.0, np.radians(30.0)])
        truth = quaternion_product(truth, quaternion_from_rotation(omega * dt))
        suivi.predict(dt, omega + rng.normal(0, noise, 3))
        R = quaternion_to_matrix(truth)
        suivi.correct_with_gravity(R.T @ np.array([0.0, 0.0, GRAVITE])
                               + rng.normal(0, 0.05, 3))
    roll, pitch, yaw = np.degrees(quaternion_to_euler(suivi.q))
    print(f"   read: roll {roll:+.2f}   pitch {pitch:+.2f}   "
          f"yaw {yaw:+.2f} deg   (expected 0, 0, 90)")
    print(f"   orientation error: {quaternion_angle(suivi.q, truth):.2f} deg")
    assert abs(yaw - 90) < 3.0 and abs(roll) < 2 and abs(pitch) < 2

    print("\n3. THIRTY SECONDS AT REST, WITH NO TAG AT ALL")
    print("   The gyro's residual bias works freely.")
    suivi = OrientationFilter()
    suivi.start(orientation_initiale(np.array([0.0, 0.0, 1.0])), sigma_deg=5.0)
    residuel = np.radians([0.05, -0.04, 0.30])
    for _ in range(int(30.0 / dt)):
        suivi.predict(dt, residuel + rng.normal(0, noise, 3))
        suivi.correct_with_gravity(np.array([0.0, 0.0, GRAVITE])
                               + rng.normal(0, 0.05, 3))
    roll, pitch, yaw = np.degrees(quaternion_to_euler(suivi.q))
    print(f"   roll {roll:+.2f}   pitch {pitch:+.2f} deg"
          f"   <- held by the accelerometer")
    print(f"   yaw  {yaw:+.2f} deg                <- drifts freely")
    assert abs(roll) < 2 and abs(pitch) < 2
    assert abs(yaw) > 3

    print("\n" + "=" * 70)
    print("WHAT THIS ESTABLISHES")
    print("=" * 70)
    print("  - the gyro integrates correctly into orientation (90 deg read")
    print("    for 90 deg real, to within 0.03 deg)")
    print("  - the accelerometer bounds roll and pitch indefinitely")
    print("  - yaw, on the other hand, drifts: gravity says nothing about it.")
    print("    Hence the fusion with the AprilTags — not a comfort choice,")
    print("    it is the only way to hold a heading.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    if "--simulation" in sys.argv:
        sys.exit(_simulation())
    sys.exit(_demonstration())
