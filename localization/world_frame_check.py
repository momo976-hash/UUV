# world_frame_check.py — Camera motion in a WORLD FRAME, and the pool protocol.
#
# ===========================================================================
# HOW TO USE IT — THIS IS THE SCRIPT FOR STEPS 5 AND 6 OF THE PROTOCOL
# ===========================================================================
#     python localization/world_frame_check.py
#     python localization/world_frame_check.py --plots     with live figures
#
# Keys:  o = set the reference tag (origin)   m = distance/rotation mode
#        f = filter on/off                    r = reset everything
#        0-9 and '.' = type a real value      BACKSPACE = erase
#        s = save that measurement            q = quit
#
# ---------------------------------------------------------------------------
# STEP 5 — measure the vehicle's real dynamics  (10 minutes, once)
# ---------------------------------------------------------------------------
#   1. Vehicle in the water, camera seeing the tags.
#   2. Aim at a tag and press  o . It becomes the world origin.
#   3. Drive ~30 seconds LIKE A REAL MISSION. Usual speeds — neither parked,
#      nor deliberately shaken. Keep going after the "tag linked" message:
#      that message is a confirmation, not a signal to stop.
#   4. Press  q . The script prints two ready-made lines to copy into
#      kalman/kalman_filter.py.
#
#   You do NOT need the m / s keys for this. They belong to step 6.
#
# ---------------------------------------------------------------------------
# STEP 6 — check the filter actually improves things  (tape measure needed)
# ---------------------------------------------------------------------------
#   1. Press  o  on the reference tag.
#   2. Check the display reads  filter : ON  (key  f  toggles it).
#   3. Move the camera by a distance MEASURED WITH A TAPE.
#   4. Type that real value on the keyboard, then press  s  to record it.
#   5. Repeat about FIFTEEN times, at varied distances.
#   6. Press  q . The script prints the verdict on its own: how much the
#      filter reduces the error, and — more important — whether the filter
#      tells the truth about its own precision.
#
#   Do not announce an expected gain in advance. The self-tests show 33x, but
#   that is a simulation in which the filter's assumptions are true by
#   construction. Expect 1.5-2x in reality. The only defensible number is the
#   one measured here.
#
# ===========================================================================
# WHAT THE SCRIPT DOES
# ===========================================================================
# Aim: measure the camera's motion WITHOUT having to keep the same tag in
# view. The tags are first linked into a single world frame (a tag is seen at
# the same time as an already-known tag), after which the camera pose is
# computed in that common frame, whichever tag is being looked at.
#
# The linking happens BY ITSELF along the way: it is enough for two tags to be
# visible together for a moment. The reference tag can then leave the field of
# view and the measurement continues.
#
# KALMAN FILTER (key 'f')
#   On every frame, ALL known visible tags feed the filter
#   (kalman/kalman_filter.py): their estimates fuse and smooth over time. The
#   screen shows the raw position AND the filtered position one under the
#   other, with the uncertainty the filter reports, to compare live.
#
# ===========================================================================
# ONE TRAP WORTH KNOWING ABOUT
# ===========================================================================
# Two consecutive poses are only comparable if they come from THE SAME map of
# THE SAME reference tag. The map of a tag keeps being refined as long as that
# tag stays co-visible with another known one, and the reference tag itself is
# re-chosen every frame as whichever known tag looks largest.
#
# Differencing across a change of either produces a jump that is not motion —
# it is the gap between two maps — and divided by a frame interval it turns a
# few millimetres into hundreds of deg/s. Measured on real pool data before
# this was handled: 283 deg/s and 470 m/s2 as MEDIANS, from a hand-held
# camera. Both cases are now treated exactly like a lost tag.
import csv
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kalman"))
import optics  # noqa: E402

from kalman_filter import (PoseFilter, remind_missing_measurements,
                           SIGMA_ACCELERATION, GYRO_DRIFT_DEG_S)

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None          # pas de imu : le filter tourne sans elle

CAMERA_INDEX = None
RESOLUTION = optics.RESOLUTION

TAG_SIZE = optics.LARGE_TAG_SIZE   # measurement au pied a coulisse, pas 223 mm nominal
MIN_LIAISON = 6      # co-visibilites avant d'utiliser un tag (liaison rapide)
MAX_LIAISON = 60     # on garde ce count d'observations pour affiner la liaison
LISSAGE = 15

MONTAGE = optics.ACTIVE_MOUNTING
# L'optics vient de optics.py : camera, tube, hublot, milieu. Le mounting
# n'est ecrit nulle part dans le code : optics.py le lit dans le path
# montage_local.txt propre a CETTE machine, et le demande une fois s'il
# n'existe pas encore. Pour le changer :
#     python calibration/set_mounting.py
# Pour une seule commande, sans rien deregler :
#     UUV_MONTAGE=nue_air python ce_script.py
# Tant qu'il n'est pas calibre, optics.py retombe sur la camera nue en le
# disant.
K_CALIB, DIST_CALIB = optics.load(MONTAGE)
LARGEUR_CALIB, HAUTEUR_CALIB = optics.RESOLUTION


def transformation(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).flatten()
    return T


def inverse(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def angle_entre(R1, R2):
    cos = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


# ===========================================================================
# Source d'frames : colour SEULE, ou colour + imu inertielle
#
# POURQUOI UNE SEULE CONNEXION. La D435i ne se laisse pas ouvrir deux fois :
# si OpenCV tient le flux colour, pyrealsense2 ne peut plus atteindre le
# module de mouvement, et l'IMU reste muette sans qu'aucune error ne le
# dise. On prend donc TOUT par pyrealsense2 quand il est la, et on retombe
# sur OpenCV sans IMU sinon — le filter fonctionne dans les deux cas, avec ou
# sans imu.
#
# CADENCE DE L'IMU. Le pipeline se cale sur son flux le plus lent, ici la
# colour a 30 Hz. On ne lit donc qu'une measurement de gyro par image. Ce n'est
# pas une perte : le filter avance d'un pas par image, et integrer omega sur
# les 33 ms de ce pas est exactement ce qu'il faut. La haute rate ne
# servirait qu'a capter des transitoires plus rapides que les frames.
# ===========================================================================
class SourceRealSense:
    """Couleur et imu inertielle, depuis une seule connexion."""

    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, RESOLUTION[0], RESOLUTION[1],
                             rs.format.bgr8, 30)
        # On demande les profils de mouvement que l'appareil ANNONCE, plutot
        # qu'un format assumed : c'est ce qui evite "Couldn't resolve requests"
        # quand le SDK ou le micrologiciel change ses profils.
        offerts = {}
        for appareil in rs.context().query_devices():
            for capteur in appareil.sensors:
                for profil in capteur.get_stream_profiles():
                    flux = profil.stream_type()
                    if flux in (rs.stream.accel, rs.stream.gyro):
                        offerts.setdefault(flux, []).append(
                            (profil.format(), profil.fps()))
            if offerts:
                break
        self.with_imu = (rs.stream.accel in offerts and rs.stream.gyro in offerts)
        if self.with_imu:
            for flux in (rs.stream.accel, rs.stream.gyro):
                format_, fps = max(offerts[flux], key=lambda couple: couple[1])
                config.enable_stream(flux, format_, fps)
        self.profil = self.pipeline.start(config)

        # Rotation imu -> camera colour. La D435i ne les aligne pas, et
        # passer les measurements brutes sans elle fait deriver l'engin de travers
        # sans aucun message d'error.
        self.R_imu_camera = np.eye(3)
        if self.with_imu:
            try:
                extr = (self.profil.get_stream(rs.stream.gyro)
                        .get_extrinsics_to(self.profil.get_stream(rs.stream.color)))
                self.R_imu_camera = np.array(extr.rotation).reshape(3, 3).T
            except Exception:
                pass
        self._gyro = np.zeros(3)
        self._accel = np.zeros(3)
        self._imu_vue = False

    def read(self):
        frames = self.pipeline.wait_for_frames()
        for image in frames:
            if not image.is_motion_frame():
                continue
            motion = image.as_motion_frame()
            d = motion.get_motion_data()
            value = np.array([d.x, d.y, d.z], dtype=float)
            if motion.get_profile().stream_type() == rs.stream.gyro:
                self._gyro = value
                self._imu_vue = True
            else:
                self._accel = value
        colour = frames.get_color_frame()
        if not colour:
            return False, None
        return True, np.asanyarray(colour.get_data())

    def imu(self):
        """(gyro, accel) dans le frame CAMERA, ou (None, None)."""
        if not (self.with_imu and self._imu_vue):
            return None, None
        return self.R_imu_camera @ self._gyro, self.R_imu_camera @ self._accel

    def release(self):
        self.pipeline.stop()


class SourceOpenCV:
    """Couleur seule : le filter tourne, sans apport inertiel."""

    with_imu = False

    def __init__(self, cap):
        self.cap = cap

    def read(self):
        return self.cap.read()

    def imu(self):
        return None, None

    def release(self):
        self.cap.release()


def ouvrir_camera():
    if rs is not None:
        try:
            source = SourceRealSense()
            ok, img = source.read()
            if ok and img is not None:
                hh, ww = img.shape[:2]
                state = "AVEC imu inertielle" if source.with_imu else "sans IMU"
                print(f"Camera : RealSense {ww}x{hh}, {state}")
                return source, ww, hh
            source.release()
        except Exception as souci:
            print(f"RealSense unavailable ({souci}) — trying OpenCV, no IMU")

    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "AUTO")]
    indices = [CAMERA_INDEX] if CAMERA_INDEX is not None else range(4)
    for index in indices:
        for backend, name in backends:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
                ok, img = cap.read()
                if ok and img is not None:
                    hh, ww = img.shape[:2]
                    print(f"Camera : OpenCV index={index}, {name}, {ww}x{hh}, no IMU")
                    return SourceOpenCV(cap), ww, hh
            cap.release()
    return None, 0, 0


cam, L, H = ouvrir_camera()
if cam is None:
    print("ERROR: no camera could be opened.")
    raise SystemExit

K, dist = K_CALIB.copy(), DIST_CALIB.copy()
Lc, Hc = LARGEUR_CALIB, HAUTEUR_CALIB
try:
    path = np.load("calibration_camera.npz")
    K, dist = path["K"].astype(np.float64), path["dist"].ravel()
    Lc, Hc = int(path["width"]), int(path["height"])
    print("Calibration loaded from calibration_camera.npz")
except Exception:
    print("Using the calibration built into the script")
if (L, H) != (Lc, Hc):
    print(f"  >>> WARNING: capture is {L}x{H} but the calibration is {Lc}x{Hc}.")

h = TAG_SIZE / 2
coins_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(dictionary, params)

tag_map = {}                       # id -> T_monde_tag (frame commun)
candidats = defaultdict(lambda: deque(maxlen=MAX_LIAISON))
MODES = ["deplacement camera (m)", "rotation camera (deg)"]
mode = 0
origin = None                   # tag chosen comme origin du world (au 'o')
ref_p = ref_R = None
lissage = deque(maxlen=LISSAGE)
saisie = ""

# --- filter de Kalman (key 'f' pour l'activer / le couper) --------------
filter = PoseFilter()
filtre_actif = True
dernier_temps = None
ref_p_filtre = ref_R_filtre = None
lissage_filtre = deque(maxlen=LISSAGE)

# --- measurement des vitesses reelles de l'engin --------------------------------
# Le filter a besoin de deux chiffres qui decrivent ce que l'engin fait sans
# qu'il le sache : sigma_acceleration et derive_gyro. Plutot que de les
# supposer, on les lit ici sur le mouvement reel. La pose BRUTE sert de
# source (pas la filtered : le filter lisse justement ce qu'on veut mesurer).
MEMOIRE_DYNAMIQUE = 900          # 30 s a 30 Hz
vitesses_angulaires = deque(maxlen=MEMOIRE_DYNAMIQUE)   # deg/s
accelerations = deque(maxlen=MEMOIRE_DYNAMIQUE)         # m/s^2
precedent_p = precedent_R = precedent_t = precedent_ref = precedent_carte_ref = None
precedente_vitesse = None


def centile(values, part):
    return float(np.percentile(np.fromiter(values, dtype=float), part)) if values else 0.0


def incidence_du_tag(pose_camera_tag):
    """Angle en degres sous lequel la camera voit ce tag (0 = pile en face).

    La normale du tag dans le frame camera est sa 3e column ; le tag est seen
    d'autant plus de bias que cette normale s'ecarte de l'axis camera->tag."""
    normale = pose_camera_tag[:3, 2]
    vers_tag = pose_camera_tag[:3, 3]
    distance = np.linalg.norm(vers_tag)
    if distance < 1e-6:
        return 0.0
    cos = abs(float(normale @ vers_tag) / distance)
    return float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))

CSV = os.path.abspath("world_frame_check.csv")
# On enregistre le BRUT ET LE FILTRE sur la meme row, au meme timestamp. Les
# consigner separement (une serie filter ON, une serie filter OFF) obligerait
# a refaire exactement le meme geste deux fois : impossible a la main, et
# c'est le geste qui domine l'gap. Ici la comparaison porte sur la meme
# measurement, donc elle ne measurement que le filter.
#
# sigma_filtre_mm est l'uncertainty que le filter ANNONCE. C'est elle qui
# permet de repondre a la seule question qui compte vraiment : le filter
# dit-il la truth sur sa propre precision ?
ENTETE = ["mode", "valeur_reelle", "raw", "erreur_brut",
          "filter", "erreur_filtre", "sigma_filtre_mm", "nb_tags"]
if os.path.exists(CSV):
    with open(CSV, newline="") as fic:
        ancienne = next(csv.reader(fic), [])
    if ancienne != ENTETE:
        # Un path a l'old format (4 colonnes) : y ajouter des rows a 8
        # colonnes produirait un tableau illisible et un bilan faux. On le met
        # de cote plutot que d'y toucher.
        archive = CSV.replace(".csv", "_ancien_format.csv")
        os.replace(CSV, archive)
        print(f"Previous measurement file moved to: {archive}")
if not os.path.exists(CSV):
    with open(CSV, "w", newline="") as fic:
        csv.writer(fic).writerow(ENTETE)

print("=" * 66)
optics.announce_mounting("MONTAGE :")
print("WORLD-FRAME CHECK (move freely between tags)")
print("  1. look at the reference tag, press 'o'")
print("  2. move towards the 2nd tag: linking happens BY ITSELF on the way")
print("     (the 2 tags only need to be visible together for a moment)")
print("  'm' mode | 'r' reset | 's' record | 'q' quit")
if "--plots" not in sys.argv:
    print("  To SEE the filter working (6 course figures, live):")
    print("      re-run with  --plots")
print("=" * 66)

# Le rappel est affiche AVANT la session, pas seulement apres : c'est
# maintenant que la personne a l'engin dans l'water sous la main. Le lui dire
# une fois la manip terminee l'obligerait a tout recommencer.
remind_missing_measurements(with_imu=cam.with_imu)

# --- les figures du cours, en direct (option --graphiques) -----------------
# Facultatif et sans consequence si matplotlib manque : au bassin, une measurement
# ne se refait pas parce qu'une bibliotheque d'display n'est pas installee.
windows = None
debut_session = time.time()
if "--plots" in sys.argv:
    try:
        from kalman_live_plots import FenetresKalman, disponible
        if disponible():
            windows = FenetresKalman(axis=0)
            print("Filter plots: window open (6 course figures).")
        else:
            print("--plots was requested but matplotlib is not installed:")
            print("    python -m pip install matplotlib")
            print("The measurement continues without plots.")
    except Exception as souci:
        print(f"--plots unavailable ({souci}) — the measurement continues without it.")

while True:
    ok, image = cam.read()
    if not ok:
        continue
    # Garde-fou : l'image contredit-elle le mounting declare ? Ne se declenche
    # qu'une fois, et seulement quand le doute n'est pas permis (voir
    # optics.check_image_matches_mounting).
    alerte = optics.check_image_matches_mounting(image, MONTAGE)
    if alerte:
        print(f"\n*** SUSPICIOUS MOUNTING: {alerte}\n")

    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gris)

    # pose de chaque tag dans le frame camera
    poses, surfaces = {}, {}
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        for c, tid in zip(corners, ids.flatten()):
            pts = c.reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(coins_3d, pts, K, dist,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok2:
                # Le hublot courbe deplace le point de vue apparent : toutes
                # les distances sortent 16 mm trop courtes sous l'water, measurement
                # au bassin. La direction, elle, est juste — on allonge sans
                # tourner. Vaut 0 hors mounting immerge.
                tvec = optics.correct_window_offset(tvec, MONTAGE)
                poses[int(tid)] = transformation(cv2.Rodrigues(rvec)[0], tvec)
                surfaces[int(tid)] = cv2.contourArea(pts.astype(np.float32))

    # liaison automatique et continue des tags (par paires).
    # Des qu'un tag inconnu B est seen en meme time qu'un tag known A, on
    # accumule sa pose dans le frame world et on l'utilise tres vite
    # (>= MIN_LIAISON co-visibilites), tout en continuant a l'affiner.
    if origin is not None:
        for B in list(poses):
            known = [A for A in poses if A in tag_map and A != B]
            if not known:
                continue
            A = max(known, key=lambda i: surfaces[i])
            candidats[B].append(tag_map[A] @ inverse(poses[A]) @ poses[B])
            if len(candidats[B]) >= MIN_LIAISON:
                obs = np.array(candidats[B])
                T = np.median(obs, axis=0)
                T[:3, :3] = obs[len(obs) // 2][:3, :3]
                new = B not in tag_map
                tag_map[B] = T
                if new:
                    print(f"Tag {B} linked automatically. World: {sorted(tag_map)}")

    # pose de la camera dans le frame world (meilleur tag known visible)
    cam_p = cam_R = ref = carte_ref = None
    known_seen = [i for i in poses if i in tag_map]
    if known_seen:
        ref = max(known_seen, key=lambda i: surfaces[i])
        carte_ref = tag_map[ref]
        T_monde_cam = carte_ref @ inverse(poses[ref])
        cam_p = T_monde_cam[:3, 3]
        cam_R = T_monde_cam[:3, :3]

    # --- ce que l'engin fait vraiment : velocity de rotation et acceleration -
    # Mesure sur la pose BRUTE, entre deux frames consecutives.
    #
    # Deux poses consecutives ne sont comparables QUE si elles viennent de LA
    # MEME tag_map du MEME tag de reference. Un changement de tag (7 -> 2) ne
    # suffit pas a le detecter : tag_map[B] continue d'etre affinee en continu
    # tant que B reste co-visible avec un autre tag known (cf. plus haut), et
    # cela vaut aussi pour le tag origin — des que le 2e tag est assez lie
    # pour servir a son tour de reference, il peut re-affiner tag_map[origin].
    # Verifie par simulation : une camera IMMOBILE, meme tag de reference
    # d'un bout a l'autre mais dont la tag_map se raffine chaque image, rend
    # deja des dizaines de deg/s et m/s2 de dynamique fantome — la seule
    # comparaison d'ID (premiere version de ce correctif) ne voit rien venir
    # puisque l'ID de reference, lui, ne change pas.
    #
    # On compare donc l'OBJET tag_map[ref] par IDENTITE (`is`), pas sa value :
    # chaque affinage fait `tag_map[B] = T` avec un tableau tout neuf (issu de
    # np.median(...)), donc `is` detecte un affinage meme infime, ce qu'une
    # comparaison numerique a tolerance fixe pourrait manquer. Un changement
    # de reference OU un affinage de la tag_map entre deux frames est donc
    # traite exactement comme une perte de tag : on ne differencie pas a
    # travers deux etats de tag_map differents, aussi proches soient-ils.
    if cam_p is not None:
        timestamp = time.time()
        if (precedent_t is not None and ref == precedent_ref
                and carte_ref is precedent_carte_ref):
            interval = timestamp - precedent_t
            if 1e-3 < interval < 0.5:      # on ignore les trous (tag perdu)
                vitesses_angulaires.append(angle_entre(precedent_R, cam_R) / interval)
                velocity = (cam_p - precedent_p) / interval
                if precedente_vitesse is not None:
                    accelerations.append(
                        float(np.linalg.norm(velocity - precedente_vitesse) / interval))
                precedente_vitesse = velocity
            else:
                precedente_vitesse = None
        else:
            precedente_vitesse = None
        precedent_p, precedent_R, precedent_t, precedent_ref, precedent_carte_ref = (
            cam_p.copy(), cam_R.copy(), timestamp, ref, carte_ref)
    else:
        precedent_t = precedent_ref = precedent_carte_ref = None
        precedente_vitesse = None

    # --- filter de Kalman : nourri par TOUS les tags known visible --------
    # Chaque tag donne sa propre estimation de la pose camera dans le world ;
    # le filter les fusionne (les incertitudes s'additionnent) et lisse dans
    # le time. La key 'f' permet de comparer avec/sans en direct.
    cam_p_filtre = cam_R_filtre = None
    maintenant = time.time()
    dt = 0.0 if dernier_temps is None else maintenant - dernier_temps
    dernier_temps = maintenant
    gyro_mesure, accel_mesure = cam.imu()
    if filtre_actif and known_seen:
        # Le gyro propage l'orientation entre deux tags, l'accelerometre tient
        # le roll et le pitch. Sans imu les deux valent None, et la
        # prediction retombe sur l'hypothese "velocity constante" d'avant.
        filter.predict(dt, gyro=gyro_mesure, accel=accel_mesure)
        for i in known_seen:
            T_i = tag_map[i] @ inverse(poses[i])           # pose camera vue par le tag i
            filter.add_tag(T_i[:3, 3], tag_map[i][:3, 3],
                               incidence_du_tag(poses[i]),
                               rotation_mesuree=T_i[:3, :3],
                               distance=float(np.linalg.norm(poses[i][:3, 3])),
                               identifiant=i)
        filter.apply()
        # Les quatre figures du cours, tracees sur CETTE measurement-ci. La measurement
        # raw passee en reference est celle du meilleur tag visible, la meme
        # que la position raw affichee a l'ecran.
        if windows is not None:
            windows.ajouter(
                maintenant - debut_session, filter, cam_p,
                tags=[(i, float(np.linalg.norm(poses[i][:3, 3])),
                       incidence_du_tag(poses[i])) for i in known_seen])
            windows.rafraichir()
        if filter.position.started:
            cam_p_filtre = filter.position.position
            cam_R_filtre = filter.orientation.matrix
            # la reference filtered est la 1ere pose stable apres un 'o'.
            if ref_p is not None and ref_p_filtre is None:
                ref_p_filtre = cam_p_filtre.copy()
                ref_R_filtre = cam_R_filtre.copy()

    # measurement du mouvement depuis la reference (raw, puis filtered)
    measurement = None
    if cam_p is not None and ref_p is not None:
        measurement = (float(np.linalg.norm(cam_p - ref_p)) if mode == 0
                  else angle_entre(ref_R, cam_R))
    if measurement is not None:
        lissage.append(measurement)
    else:
        lissage.clear()
    d = sum(lissage) / len(lissage) if lissage else None

    mesure_filtre = None
    if cam_p_filtre is not None and ref_p_filtre is not None:
        mesure_filtre = (float(np.linalg.norm(cam_p_filtre - ref_p_filtre)) if mode == 0
                         else angle_entre(ref_R_filtre, cam_R_filtre))
    if mesure_filtre is not None:
        lissage_filtre.append(mesure_filtre)
    else:
        lissage_filtre.clear()
    d_filtre = sum(lissage_filtre) / len(lissage_filtre) if lissage_filtre else None

    # --- display ---
    unite = "m" if mode == 0 else "deg"
    etat_filtre = "ON" if filtre_actif else "OFF"
    # L'state de l'IMU est affiche en permanence : une imu absente ou
    # muette n'empeche rien de tourner, et sans ce temoin on croirait la
    # fusion active alors qu'elle ne l'est pas.
    etat_imu = "IMU" if gyro_mesure is not None else "sans IMU"
    cv2.putText(image, f"MODE : {MODES[mode]}   world : {sorted(tag_map)}   "
                       f"filter(f) : {etat_filtre}   {etat_imu}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    y = 52
    for B in list(candidats):
        if B in tag_map:
            continue
        pct = min(100, int(100 * len(candidats[B]) / MIN_LIAISON))
        cv2.putText(image, f"liaison tag {B} : {pct}%", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 170, 255), 2)
        y += 22
    if cam_p is not None:
        cv2.putText(image, f"CAMERA raw  : x={cam_p[0]:+.2f} y={cam_p[1]:+.2f} "
                           f"z={cam_p[2]:+.2f} m  ({len(known_seen)} tag)", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        y += 24
    elif not known_seen:
        cv2.putText(image, "Aucun tag known visible", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y += 24
    if cam_p_filtre is not None:
        sigma = filter.position.position_uncertainty
        cv2.putText(image, f"CAMERA filter : x={cam_p_filtre[0]:+.2f} "
                           f"y={cam_p_filtre[1]:+.2f} z={cam_p_filtre[2]:+.2f} m  "
                           f"(+/- {sigma*1000:.0f} mm)", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2)
        y += 24
    if len(vitesses_angulaires) > 30:
        cv2.putText(image, f"dynamique : rotation {centile(vitesses_angulaires, 50):.1f} "
                           f"deg/s (95e {centile(vitesses_angulaires, 95):.1f})   "
                           f"accel {centile(accelerations, 95):.2f} m/s2", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        y += 22

    if ref_p is None:
        cv2.putText(image, "Regarde le tag de reference et appuie sur 'o'", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 170, 255), 2)
    elif d is not None:
        cv2.putText(image, f"mouvement measurement : {d:.3f} {unite}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 26
        if d_filtre is not None:
            cv2.putText(image, f"      (filter)   : {d_filtre:.3f} {unite}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)
            y += 26
        if saisie:
            try:
                reel = float(saisie)
                e = d - reel
                fe = f"{e*100:+.1f} cm" if mode == 0 else f"{e:+.2f} deg"
                row = f"gap raw : {fe}"
                if d_filtre is not None:
                    ef = d_filtre - reel
                    fef = f"{ef*100:+.1f} cm" if mode == 0 else f"{ef:+.2f} deg"
                    row += f"   filter : {fef}"
                cv2.putText(image, row, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except ValueError:
                pass

    cv2.putText(image, f"value reelle ({unite}) : {saisie or '...'}", (10, H - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(image, "m=mode o=reference r=zero f=filter s=save q=quit", (10, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow("Verification frame world (q pour quitter)", image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("m"):
        mode = 1 - mode
        lissage.clear(); lissage_filtre.clear(); saisie = ""
        print(f"Mode : {MODES[mode]}")
    if key == ord("f"):
        filtre_actif = not filtre_actif
        lissage_filtre.clear()
        print(f"Kalman filter: {'ON' if filtre_actif else 'OFF'}")
    if key == ord("o"):
        if poses:
            # le tag regarde devient l'origin du world ET la reference.
            origin = max(poses, key=lambda i: surfaces[i])
            tag_map.clear(); candidats.clear()
            tag_map[origin] = np.eye(4)
            T_monde_cam = tag_map[origin] @ inverse(poses[origin])
            ref_p, ref_R = T_monde_cam[:3, 3].copy(), T_monde_cam[:3, :3].copy()
            lissage.clear()
            # on repart aussi le filter depuis cette reference.
            filter = PoseFilter()
            ref_p_filtre = ref_R_filtre = None
            lissage_filtre.clear()
            print(f"Reference = tag {origin}. Move towards the 2nd tag: "
                  "liaison se fait toute seule quand les 2 tags se croisent.")
        else:
            print("No tag visible: cannot set the reference.")
    if key == ord("r"):
        tag_map.clear(); candidats.clear()
        origin = None
        ref_p = ref_R = None
        lissage.clear()
        filter = PoseFilter()
        ref_p_filtre = ref_R_filtre = None
        lissage_filtre.clear()
        print("Reset: look at the reference tag and press 'o'.")
    if ord("0") <= key <= ord("9") or key == ord("."):
        saisie += chr(key)
    if key == 8 and saisie:
        saisie = saisie[:-1]
    if key == ord("s") and saisie and d is not None:
        try:
            reel = float(saisie)
        except ValueError:
            print("Invalid value.")
            continue
        e = d - reel
        ef = None if d_filtre is None else d_filtre - reel
        sigma_mm = (filter.position.position_uncertainty * 1000
                    if filter.position.started else None)
        with open(CSV, "a", newline="") as fic:
            csv.writer(fic).writerow([
                MODES[mode], f"{reel:.3f}", f"{d:.3f}", f"{e:+.3f}",
                "" if d_filtre is None else f"{d_filtre:.3f}",
                "" if ef is None else f"{ef:+.3f}",
                "" if sigma_mm is None else f"{sigma_mm:.1f}",
                len(known_seen)])
        if mode == 0:
            row = (f"[deplacement] reel {reel:.3f} m | raw {d:.3f} m "
                     f"({e*100:+.1f} cm)")
            if ef is not None:
                row += f" | filter {d_filtre:.3f} m ({ef*100:+.1f} cm)"
        else:
            row = (f"[rotation] reel {reel:.2f} deg | raw {d:.2f} deg "
                     f"({e:+.2f} deg)")
            if ef is not None:
                row += f" | filter {d_filtre:.2f} deg ({ef:+.2f} deg)"
        print(row)

cam.release()
cv2.destroyAllWindows()
print(f"\nDone. Measurements in: {CSV}")

# La figure est ENREGISTREE avant d'etre fermee : sans cela, tout ce que les
# quatre graphiques ont montre pendant la session disparait a la fermeture de
# la window, et il faut refaire la manip pour en garder une trace.
if windows is not None:
    windows.rafraichir(force=True)
    image_figures = os.path.abspath("graphiques_kalman_session.png")
    if windows.enregistrer(image_figures):
        print(f"Filter figures saved to: {image_figures}")
    windows.fermer()


# --- le filter fait-il son travail ? ---------------------------------------
def bilan_filtre():
    """Verdict lu sur les measurements a distance connue enregistrees.

    DEUX questions distinctes, et la seconde est la plus importante.

    1. Le filter REDUIT-IL l'error ? Se lit sur le report des RMS. C'est la
       question qu'on pose spontanement, et la plus facile.

    2. Le filter DIT-IL LA VERITE sur sa propre precision ? Un filter qui
       annonce +/- 2 mm alors qu'il se trompe de 20 est plus dangereux qu'un
       filter qui ne lisse rien : tout ce qui consomme sa output -- une
       commande, une tag_map, un report -- le croit sur parole. Cette
       question-la ne se voit pas a l'oeil sur l'ecran, seulement ici.

    Reserve a garder en tete pour le point 2 : l'error enregistree porte sur
    une DISTANCE entre deux poses, quand sigma porte sur UNE position. Les
    deux ne sont pas la meme grandeur (facteur ~racine de 2 au pire), et
    l'error du metre a ruban s'y ajoute. Le report ci-dessous se lit donc
    en ordre de grandeur : il attrape un filter qui ment d'un facteur 3, pas
    un gap de 20 %.
    """
    try:
        with open(CSV, newline="") as fic:
            rows = [l for l in csv.DictReader(fic) if l["mode"] == MODES[0]]
    except OSError:
        return
    if len(rows) < 3:
        print("\n(Fewer than 3 displacement measurements: no filter verdict.)")
        return

    def column(name):
        values = []
        for l in rows:
            try:
                values.append(float(l[name]))
            except (ValueError, KeyError, TypeError):
                values.append(None)
        return values

    bruts = [e for e in column("erreur_brut") if e is not None]
    apparies = [(b, f) for b, f in zip(column("erreur_brut"), column("erreur_filtre"))
                if b is not None and f is not None]

    print("\n" + "=" * 66)
    print(f"IS THE FILTER DOING ITS JOB?   ({len(rows)} displacement measurements)")
    print("=" * 66)

    if not apparies:
        print("  No measurement was taken with the filter ON (key 'f').")
        print("  Redo a series with the filter ON to be able to conclude.")
        print("=" * 66)
        return

    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    rms_raw = rms([b for b, _ in apparies])
    rms_filtered = rms([f for _, f in apparies])
    print(f"  RMS error   raw    {rms_raw*1000:7.1f} mm")
    print(f"               filter {rms_filtered*1000:7.1f} mm", end="")
    if rms_filtered > 0:
        print(f"     -> gain {rms_raw/rms_filtered:.2f}x")
    else:
        print()
    gain = rms_raw / rms_filtered if rms_filtered > 0 else float("inf")
    if gain >= 1.2:
        print("  [OK] the filter reduces the error.")
    elif gain > 1.0:
        # Sur une dizaine de measurements, un gain de quelques pourcents ne se
        # distingue pas du hasard. L'annoncer comme un succes serait se
        # mentir : autant dire qu'on ne sait pas encore.
        print("  [INCONCLUSIVE] gain too small to be told apart from chance on so")
        print("       few measurements. Take about twenty, or check")
        print("       sigma_acceleration (protocol step 5).")
    else:
        print("  [NO] the filter does not improve things. Most common cause:")
        print("       sigma_acceleration badly set (protocol step 5).")

    # -- le filter est-il honnete sur son uncertainty ? ----------------------
    couples = [(abs(f), s) for (_, f), s in zip(apparies, column("sigma_filtre_mm"))
               if s is not None and s > 0]
    if len(couples) >= 3:
        reel = float(np.median([f * 1000 for f, _ in couples]))
        annonce = float(np.median([s for _, s in couples]))
        report = reel / annonce
        print(f"\n  uncertainty reported by the filter: {annonce:6.1f} mm (median)")
        print(f"  error actually observed            : {reel:6.1f} mm (median)")
        print(f"  actual / reported ratio: {report:.1f}")
        if report < 0.5:
            print("  [OK] the filter is cautious: it reports more error than it makes.")
            print("       Harmless, but it under-rates itself.")
        elif report <= 2.0:
            print("  [OK] the filter tells the truth about its precision.")
        elif report <= 4.0:
            print("  [WARNING] the filter believes itself more precise than it is.")
            print("       Do not take the displayed +/- at face value.")
        else:
            print("  [NO] the filter LIES about its precision. Do not use its +/- to")
            print("       decide anything. Check SIGMA_PIXEL first (does it really")
            print("       measure the pool's noise?), then the tag positions in the map.")
            print("")

    # -- rejections et recoveries --------------------------------------------------
    total_rejets = filter.position.rejections + filter.orientation.rejections
    recoveries = filter.position.recoveries + filter.orientation.recoveries
    print(f"\n  measurements rejected: {total_rejets}   recoveries after lock-out: {recoveries}")
    if recoveries > 3:
        print("  [WARNING] many recoveries: the filter locks out then re-anchors.")
        print("       Often a sign of tags misplaced in the map.")

    suspects = filter.watchdog.report()
    if "aucun tag suspect" not in suspects:
        print("\n  SUPPORTS THAT HAVE MOVED")
        print(suspects)
    print("=" * 66)


bilan_filtre()

# --- les deux reglages du filter, lus sur le mouvement reel ----------------
if len(vitesses_angulaires) > 100:
    rotation_95 = centile(vitesses_angulaires, 95)
    accel_95 = centile(accelerations, 95)
    print("\n" + "=" * 66)
    print("OBSERVED DYNAMICS")
    print("=" * 66)
    print(f"  rotation    median {centile(vitesses_angulaires, 50):6.1f} deg/s"
          f"   95e centile {rotation_95:6.1f} deg/s")
    print(f"  acceleration median {centile(accelerations, 50):5.2f} m/s2"
          f"   95e centile {accel_95:6.2f} m/s2")
    print("-" * 66)
    # Le noise de model doit couvrir ce que l'engin fait REELLEMENT sans que
    # le filter le sache. Le 95e centile evite a la fois de sous-estimer, ce
    # qui ferait retarder le filter, et de se caler sur un pic isole.
    print("  HERE ARE THE TWO NUMBERS. What to do with them:")
    print()
    print("   1. Open the file   kalman/kalman_filter.py")
    print("      (Notepad, VS Code, any text editor will do)")
    print("   2. Search (Ctrl+F) for:  SIGMA_ACCELERATION")
    print("   3. Two lines ALREADY exist near the top of the file. They look")
    print("      like this:")
    print()
    print(f"          SIGMA_ACCELERATION = {SIGMA_ACCELERATION}")
    print(f"          GYRO_DRIFT_DEG_S = {GYRO_DRIFT_DEG_S}")
    print()
    print("   4. Replace ONLY the numbers, to get:")
    print()
    print(f"          SIGMA_ACCELERATION = {accel_95:.1f}")
    print(f"          GYRO_DRIFT_DEG_S = {rotation_95:.0f}")
    print()
    print("   5. Save the file. That is all — nothing else to change anywhere,")
    print("      and the reminder at startup will disappear by itself.")
    print()
    print("")
    print("  find the two lines starting with SIGMA_ACCELERATION and")
    print("")
    print("")
    print("=" * 66)
    print("  Valable si ce que tu viens de faire ressemble a une true mission.")
    print("  A session with the camera left sitting still measures nothing useful.")
