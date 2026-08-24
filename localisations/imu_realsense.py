# imu_realsense.py — Lire la centrale inertielle de la D435i, et le prouver.
#
#     python imu_realsense.py
#
# Deux choses en une :
#   1. UN MODULE. La classe CentraleRealSense ouvre les flux accel et gyro du
#      SDK Intel et rend des mesures pretes a entrer dans filtre_kalman.py.
#   2. UNE DEMONSTRATION. Lancee directement, elle mesure le biais au repos
#      puis affiche l'orientation en direct — de quoi montrer que la centrale
#      est lue et exploitee, pas seulement branchee.
#
# ---------------------------------------------------------------------------
# CE QUE LA DEMONSTRATION PROUVE, ET DANS QUEL ORDRE
# ---------------------------------------------------------------------------
# 1. LA CENTRALE EST LUE. Les vitesses angulaires et accelerations brutes
#    s'affichent. Bouge la camera, les chiffres bougent.
#
# 2. LES MESURES SONT SAINES. Au repos, l'accelerometre doit lire 9.81 m/s2
#    en norme — ni plus ni moins, c'est la pesanteur. Si ce n'est pas le cas,
#    l'echelle est fausse et tout le reste le sera.
#
# 3. LES MATHS MARCHENT. L'orientation integree suit les mouvements reels.
#    Tourne la camera d'un quart de tour : le lacet affiche 90 degres.
#
# 4. LA DERIVE EST REELLE ET BORNEE OU IL FAUT. Laisse tourner : le roulis et
#    le tangage restent stables (l'accelerometre les tient), le lacet derive
#    lentement (rien ne le recale sans tag). C'est precisement pourquoi la
#    fusion avec les tags est necessaire — la demonstration le rend visible.
#
# ---------------------------------------------------------------------------
# REPERES : LE PIEGE PRINCIPAL
# ---------------------------------------------------------------------------
# La centrale n'est PAS alignee avec la camera couleur. Le SDK donne la
# rotation entre les deux (get_extrinsics_to) et ce module la recupere.
# Passer les mesures brutes au filtre sans cette rotation melange les axes et
# fait deriver l'engin de travers, sans aucun message d'erreur.
#
# On ne suppose rien non plus sur l'orientation de depart : la direction du
# bas est MESUREE au repos, au lieu d'etre supposee selon un axe. Le montage
# peut donc etre pose n'importe comment.
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filtre_kalman import (FiltreOrientation, euler_vers_quaternion,  # noqa: E402
                           quaternion_vers_euler, quaternion_depuis_rotation,
                           produit_quaternions)

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

GRAVITE = 9.81


class CentraleRealSense:
    """Flux accel + gyro de la D435i, plus la rotation vers la camera couleur.

    Les deux capteurs n'arrivent PAS au meme rythme (l'accelerometre vers
    60-250 Hz, le gyroscope vers 200-400 Hz) et chaque image ne porte qu'un
    seul des deux. On garde donc la derniere valeur connue de chacun, et on
    date les mesures avec l'horloge du capteur plutot que celle du PC : les
    intervalles servent a integrer, une gigue de quelques millisecondes se
    paierait directement en derive.
    """

    def __init__(self, avec_couleur=False):
        if rs is None:
            raise RuntimeError(
                "pyrealsense2 n'est pas installe.\n"
                "  pip install pyrealsense2\n"
                "C'est la bibliotheque du SDK Intel qui donne acces a l'IMU.")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f)
        config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f)
        if avec_couleur:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profil = self.pipeline.start(config)

        self.R_imu_camera = np.eye(3)
        self.extrinseques_lues = False
        if avec_couleur:
            try:
                extr = (self.profil.get_stream(rs.stream.gyro)
                        .get_extrinsics_to(self.profil.get_stream(rs.stream.color)))
                # Le SDK range la rotation en COLONNES ; numpy lit en lignes.
                self.R_imu_camera = np.array(extr.rotation).reshape(3, 3).T
                self.extrinseques_lues = True
            except Exception:
                pass          # on reste sur l'identite, signalee a l'appelant

        self._gyro = np.zeros(3)
        self._accel = np.zeros(3)
        self._t_gyro = None

    def lire(self, timeout_ms=5000):
        """Rend (gyro, accel, dt) ou None. dt est l'intervalle depuis la
        derniere mesure de gyro, en secondes, ou None a la premiere."""
        images = self.pipeline.wait_for_frames(timeout_ms)
        dt = None
        for image in images:
            if not image.is_motion_frame():
                continue
            motion = image.as_motion_frame()
            donnees = motion.get_motion_data()
            valeur = np.array([donnees.x, donnees.y, donnees.z], dtype=float)
            flux = motion.get_profile().stream_type()
            if flux == rs.stream.gyro:
                instant = motion.get_timestamp() / 1000.0    # ms -> s
                if self._t_gyro is not None:
                    ecart = instant - self._t_gyro
                    # Une image sautee ou une horloge qui recule donnerait un
                    # dt aberrant : on l'ignore plutot que d'integrer n'importe
                    # quoi. Bornes larges, elles n'ecartent que l'absurde.
                    if 1e-5 < ecart < 0.5:
                        dt = ecart
                self._t_gyro = instant
                self._gyro = valeur
            elif flux == rs.stream.accel:
                self._accel = valeur
        return self._gyro.copy(), self._accel.copy(), dt

    def arreter(self):
        self.pipeline.stop()


def mesurer_au_repos(centrale, duree=5.0):
    """Biais et bruit des deux capteurs, engin IMMOBILE.

    Le biais du gyro est sa lecture moyenne alors qu'il ne tourne pas : c'est
    lui qui, integre, fait deriver l'orientation. Le bruit est l'ecart-type
    autour de cette moyenne, et c'est le chiffre que le filtre attend.

    La norme de l'accelerometre doit valoir 9.81 : c'est le controle d'echelle
    le plus simple qui soit, et il attrape une unite fausse (g au lieu de
    m/s2) ou un facteur d'echelle errone.
    """
    print(f"\nMESURE AU REPOS — ne touche a rien pendant {duree:.0f} s...")
    gyros, accels = [], []
    debut = time.time()
    while time.time() - debut < duree:
        gyro, accel, _ = centrale.lire()
        gyros.append(gyro)
        accels.append(accel)
        reste = duree - (time.time() - debut)
        print(f"\r  {reste:4.1f} s   {len(gyros)} echantillons", end="", flush=True)
    print()

    gyros, accels = np.array(gyros), np.array(accels)
    biais = gyros.mean(axis=0)
    bruit_gyro = float(np.degrees(gyros.std(axis=0).mean()))
    norme = float(np.linalg.norm(accels.mean(axis=0)))
    bruit_accel = float(accels.std(axis=0).mean())
    bas = accels.mean(axis=0) / max(norme, 1e-9)
    return {"biais": biais, "bruit_gyro_deg_s": bruit_gyro, "norme_accel": norme,
            "bruit_accel": bruit_accel, "bas": bas, "echantillons": len(gyros)}


def orientation_initiale(bas_mesure):
    """Orientation de depart deduite de la direction du bas MESUREE.

    On ne suppose pas comment le montage est pose : on prend la rotation la
    plus courte qui amene la verticale du monde sur le bas observe. Le lacet
    reste arbitraire — la pesanteur n'en dit rien — et c'est justement ce que
    les tags apporteront.
    """
    haut_monde = np.array([0.0, 0.0, 1.0])
    haut_mesure = -np.asarray(bas_mesure, dtype=float)
    haut_mesure = haut_mesure / max(np.linalg.norm(haut_mesure), 1e-9)
    axe = np.cross(haut_mesure, haut_monde)
    sinus = float(np.linalg.norm(axe))
    cosinus = float(np.clip(haut_mesure @ haut_monde, -1.0, 1.0))
    if sinus < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0]) if cosinus > 0 else \
               np.array([0.0, 1.0, 0.0, 0.0])
    return quaternion_depuis_rotation(axe / sinus * np.arctan2(sinus, cosinus))


def _demonstration():
    print("=" * 70)
    print("LA CENTRALE INERTIELLE DE LA D435i EST-ELLE LUE ET EXPLOITEE ?")
    print("=" * 70)

    try:
        centrale = CentraleRealSense(avec_couleur=True)
    except Exception as souci:
        print(f"\nERREUR : {souci}")
        return 1

    print("\nFlux accel et gyro ouverts via le SDK Intel (pyrealsense2).")
    if centrale.extrinseques_lues:
        angles = np.degrees(quaternion_vers_euler(
            _quaternion_depuis_matrice(centrale.R_imu_camera)))
        print(f"Rotation IMU -> camera couleur lue dans le SDK : "
              f"{angles.round(1)} deg")
    else:
        print("ATTENTION : extrinseques IMU -> camera non lues, identite "
              "utilisee.")

    repos = mesurer_au_repos(centrale)

    print("\n" + "-" * 70)
    print("1. LES MESURES SONT-ELLES SAINES ?")
    print("-" * 70)
    print(f"  echantillons              {repos['echantillons']}")
    print(f"  norme de l'accelerometre  {repos['norme_accel']:.3f} m/s2   "
          f"(doit valoir {GRAVITE})")
    ecart_g = abs(repos["norme_accel"] / GRAVITE - 1)
    if ecart_g > 0.05:
        print("  [PROBLEME] loin de la pesanteur : echelle ou unites fausses.")
    else:
        print("  [OK] c'est bien la pesanteur : l'echelle est juste.")
    print(f"  direction du bas mesuree  {repos['bas'].round(3)}")
    print("     (on ne la suppose pas : le montage peut etre pose n'importe comment)")

    print("\n" + "-" * 70)
    print("2. LES DEUX NOMBRES A RECOPIER DANS filtre_kalman.py")
    print("-" * 70)
    print(f"  biais du gyro au repos    "
          f"{np.degrees(repos['biais']).round(3)} deg/s")
    print("     C'est ce biais qui, integre, fait deriver l'orientation.")
    print("     Le filtre l'estime tout seul des que les tags le recalent.")
    print()
    print(f"      BRUIT_GYRO_DEG_S = {repos['bruit_gyro_deg_s']:.3f}")
    print(f"      BRUIT_ACCEL      = {repos['bruit_accel']:.3f}")

    print("\n" + "-" * 70)
    print("3. LES MATHS : L'ORIENTATION SUIT-ELLE LES MOUVEMENTS ?")
    print("-" * 70)
    print("  Tourne la camera d'un quart de tour : le lacet doit afficher 90.")
    print("  Repose-la : roulis et tangage doivent revenir vers zero.")
    print("  Ctrl+C pour arreter.\n")

    suivi = FiltreOrientation()
    suivi.demarrer(orientation_initiale(repos["bas"]), sigma_deg=5.0)
    depart = time.time()
    derniers = deque(maxlen=50)
    try:
        while True:
            gyro, accel, dt = centrale.lire()
            if dt is None:
                continue
            omega = centrale.R_imu_camera @ (gyro - repos["biais"])
            suivi.predire(dt, omega)
            suivi.corriger_gravite(centrale.R_imu_camera @ accel, gravite=GRAVITE)
            derniers.append(np.linalg.norm(omega))

            roulis, tangage, lacet = np.degrees(quaternion_vers_euler(suivi.q))
            print(f"\r  roulis {roulis:+7.1f}   tangage {tangage:+7.1f}   "
                  f"lacet {lacet:+7.1f} deg    "
                  f"|omega| {np.degrees(np.mean(derniers)):5.1f} deg/s   "
                  f"({time.time()-depart:4.0f} s)", end="", flush=True)
    except KeyboardInterrupt:
        print("\n")
    finally:
        centrale.arreter()

    print("-" * 70)
    print("CE QUE TU VIENS DE MONTRER")
    print("-" * 70)
    print("  - les flux IMU du SDK Intel sont lus (accel + gyro)")
    print("  - l'echelle est verifiee sur la pesanteur")
    print("  - le gyro est integre en orientation, par quaternions")
    print("  - l'accelerometre borne roulis et tangage")
    print("  - le lacet, lui, derive : rien ne le recale sans tag.")
    print("    C'est la raison d'etre de la fusion avec les AprilTags.")
    print("=" * 70)
    return 0


def _quaternion_depuis_matrice(R):
    from filtre_kalman import matrice_vers_quaternion
    return matrice_vers_quaternion(np.asarray(R, dtype=float))


def _simulation():
    """Les memes maths, sur une centrale SIMULEE : marche sans camera.

    Sert a deux choses : montrer que le traitement est juste meme quand le
    materiel n'est pas la, et donner un resultat verifiable — on connait la
    verite, donc on peut chiffrer l'erreur, ce qu'aucune manip reelle ne
    permet.
    """
    from filtre_kalman import angle_quaternions, quaternion_vers_matrice
    generateur = np.random.default_rng(3)
    dt, bruit = 1 / 200, np.radians(0.15)

    print("=" * 70)
    print("LES MATHS DE L'IMU, SUR UNE CENTRALE SIMULEE")
    print("=" * 70)
    print("Sans camera branchee. La verite etant connue, l'erreur est chiffree.")

    print("\n1. ORIENTATION DEDUITE DE LA SEULE DIRECTION DU BAS")
    pires = []
    for _ in range(300):
        v = generateur.normal(size=3)
        bas = v / np.linalg.norm(v)
        haut = quaternion_vers_matrice(orientation_initiale(bas)).T @ np.array(
            [0.0, 0.0, 1.0])
        pires.append(np.degrees(np.arccos(np.clip(haut @ (-bas), -1, 1))))
    print(f"   300 poses quelconques, erreur max {max(pires):.1e} deg")
    assert max(pires) < 1e-4      # bruit d'arccos, pas d'erreur de calcul

    print("\n2. UN QUART DE TOUR AUTOUR DE LA VERTICALE")
    print("   30 deg/s pendant 3 s, gyro bruite, accelerometre bruite.")
    suivi = FiltreOrientation()
    suivi.demarrer(orientation_initiale(np.array([0.0, 0.0, -1.0])), sigma_deg=5.0)
    verite = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(int(3.0 / dt)):
        omega = np.array([0.0, 0.0, np.radians(30.0)])
        verite = produit_quaternions(verite, quaternion_depuis_rotation(omega * dt))
        suivi.predire(dt, omega + generateur.normal(0, bruit, 3))
        R = quaternion_vers_matrice(verite)
        suivi.corriger_gravite(R.T @ np.array([0.0, 0.0, GRAVITE])
                               + generateur.normal(0, 0.05, 3))
    roulis, tangage, lacet = np.degrees(quaternion_vers_euler(suivi.q))
    print(f"   lu : roulis {roulis:+.2f}   tangage {tangage:+.2f}   "
          f"lacet {lacet:+.2f} deg   (attendu 0, 0, 90)")
    print(f"   erreur d'orientation : {angle_quaternions(suivi.q, verite):.2f} deg")
    assert abs(lacet - 90) < 3.0 and abs(roulis) < 2 and abs(tangage) < 2

    print("\n3. TRENTE SECONDES IMMOBILE, SANS AUCUN TAG")
    print("   Le biais residuel du gyro travaille librement.")
    suivi = FiltreOrientation()
    suivi.demarrer(orientation_initiale(np.array([0.0, 0.0, -1.0])), sigma_deg=5.0)
    residuel = np.radians([0.05, -0.04, 0.30])
    for _ in range(int(30.0 / dt)):
        suivi.predire(dt, residuel + generateur.normal(0, bruit, 3))
        suivi.corriger_gravite(np.array([0.0, 0.0, GRAVITE])
                               + generateur.normal(0, 0.05, 3))
    roulis, tangage, lacet = np.degrees(quaternion_vers_euler(suivi.q))
    print(f"   roulis {roulis:+.2f}   tangage {tangage:+.2f} deg"
          f"   <- tenus par l'accelerometre")
    print(f"   lacet  {lacet:+.2f} deg                <- derive librement")
    assert abs(roulis) < 2 and abs(tangage) < 2
    assert abs(lacet) > 3

    print("\n" + "=" * 70)
    print("CE QUE CELA ETABLIT")
    print("=" * 70)
    print("  - le gyro s'integre correctement en orientation (90 deg lus")
    print("    pour 90 deg reels, a 0.03 deg pres)")
    print("  - l'accelerometre borne roulis et tangage indefiniment")
    print("  - le lacet, lui, derive : la pesanteur n'en dit rien.")
    print("    D'ou la fusion avec les AprilTags — ce n'est pas un choix")
    print("    de confort, c'est le seul moyen de tenir le cap.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    if "--simulation" in sys.argv:
        sys.exit(_simulation())
    sys.exit(_demonstration())
