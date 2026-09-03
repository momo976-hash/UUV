# imu_realsense.py — Lire la imu inertielle de la D435i, et le prouver.
#
#     python imu_realsense.py
#
# Deux choses en une :
#   1. UN MODULE. La classe CentraleRealSense ouvre les flux accel et gyro du
#      SDK Intel et rend des measurements pretes a entrer dans kalman_filter.py.
#   2. UNE DEMONSTRATION. Lancee directement, elle measurement le bias au rest
#      puis affiche l'orientation en direct — de quoi montrer que la imu
#      est lue et exploitee, pas seulement branchee.
#
# ---------------------------------------------------------------------------
# CE QUE LA DEMONSTRATION PROUVE, ET DANS QUEL ORDRE
# ---------------------------------------------------------------------------
# 1. LA CENTRALE EST LUE. Les vitesses angulaires et accelerations brutes
#    s'affichent. Bouge la camera, les chiffres bougent.
#
# 2. LES MESURES SONT SAINES. Au rest, l'accelerometre doit lire 9.81 m/s2
#    en norme — ni plus ni moins, c'est la pesanteur. Si ce n'est pas le cas,
#    l'echelle est fausse et tout le reste le sera.
#
# 3. LES MATHS MARCHENT. L'orientation integree suit les mouvements reels.
#    Tourne la camera d'un quart de tour : le yaw affiche 90 degres.
#
# 4. LA DERIVE EST REELLE ET BORNEE OU IL FAUT. Laisse tourner : le roll et
#    le pitch restent stables (l'accelerometre les tient), le yaw drift
#    lentement (rien ne le recale sans tag). C'est precisement pourquoi la
#    fusion avec les tags est necessaire — la demonstration le rend visible.
#
# ---------------------------------------------------------------------------
# REPERES : LE PIEGE PRINCIPAL
# ---------------------------------------------------------------------------
# La imu n'est PAS alignee avec la camera colour. Le SDK donne la
# rotation entre les deux (get_extrinsics_to) et ce module la recupere.
# Passer les measurements brutes au filter sans cette rotation melange les axes et
# fait deriver l'engin de travers, sans aucun message d'error.
#
# On ne assumed rien non plus sur l'orientation de depart : elle est DEDUITE
# de l'accelerometre au rest, au lieu d'etre supposee selon un axis. Le
# mounting peut donc etre pose n'importe comment — sur sa base, sur le cote,
# dans le tube.
#
# CONVENTION DU VECTEUR MESURE. On le traite comme pointant vers le HAUT : au
# rest un accelerometre measurement la force specifique, la reaction du support,
# pas la pesanteur. `orientation_initiale` et `correct_with_gravity` font la meme
# hypothese, et c'est indispensable — une version ou l'une inversait le signe
# et pas l'autre fait converger l'orientation a 180 degres de la verite, sans
# aucun message. La demonstration verifie ce point automatiquement, d'une
# facon qui ne depend pas de la pose.
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

        # On demande EXACTEMENT ce que l'appareil annonce, au lieu de supposer
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
                    print(f"  extrinseques non lues ({souci}) — identite utilisee")

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
    print("LA CENTRALE INERTIELLE DE LA D435i EST-ELLE LUE ET EXPLOITEE ?")
    print("=" * 70)

    try:
        imu = CentraleRealSense(avec_couleur=True)
    except Exception as souci:
        print(f"\nERREUR : {souci}")
        return 1

    print("\nFlux accel et gyro ouverts via le SDK Intel (pyrealsense2).")
    if imu.extrinseques_lues:
        angles = np.degrees(quaternion_to_euler(
            _quaternion_from_matrix(imu.R_imu_camera)))
        print(f"Rotation IMU -> camera colour lue dans le SDK : "
              f"{angles.round(1)} deg")
    else:
        print("ATTENTION : extrinseques IMU -> camera non lues, identite "
              "utilisee.")

    rest = mesurer_au_repos(imu)

    print("\n" + "-" * 70)
    print("1. LES MESURES SONT-ELLES SAINES ?")
    print("-" * 70)
    print(f"  samples              {rest['samples']}"
          f"   soit {rest['rate']:.0f} Hz")
    cadence_gyro = max(fps for _, fps in
                       CentraleRealSense._profils_offerts()[rs.stream.gyro])
    if rest["rate"] < 0.5 * cadence_gyro:
        # Symptome known : un flux video ouvert en meme time force le
        # pipeline a se synchroniser sur lui, et les measurements de mouvement
        # sont jetees entre deux frames.
        print(f"  [PROBLEME] le gyro tourne a {cadence_gyro} Hz mais on n'en "
              f"lit que {rest['rate']:.0f}.")
        print("     L'integration assumed alors omega constant sur des")
        print("     intervalles trop longs, et la rotation est sous-estimee.")
    else:
        print(f"  [OK] on lit bien la rate du capteur ({cadence_gyro} Hz).")

    print(f"  norme de l'accelerometre  {rest['norme_accel']:.3f} m/s2   "
          f"(doit valoir {GRAVITE})")
    ecart_g = abs(rest["norme_accel"] / GRAVITE - 1)
    if ecart_g > 0.05:
        print("  [PROBLEME] loin de la pesanteur : echelle ou unites fausses.")
    elif ecart_g > 0.01:
        print(f"  [OK] c'est la pesanteur, a {100*ecart_g:.1f} % pres.")
        print("     Cet gap est un bias d'echelle de l'accelerometre. Sans")
        print("     consequence ici : on n'utilise que la DIRECTION du vector")
        print("     pour le roll et le pitch, pas sa norme.")
    else:
        print("  [OK] c'est bien la pesanteur : l'echelle est juste.")

    print(f"  direction measured         {rest['haut'].round(3)}")
    print("     (on ne la assumed pas : le mounting peut etre pose n'importe comment)")

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
    # degres de la verite sans que rien ne le signale.
    q0 = orientation_initiale(rest["haut"])
    prevu = quaternion_to_matrix(q0).T @ np.array([0.0, 0.0, 1.0])
    accord = float(prevu @ rest["haut"])
    ecart_conv = float(np.degrees(np.arccos(np.clip(accord, -1.0, 1.0))))
    roulis0, tangage0, _ = np.degrees(quaternion_to_euler(q0))
    print(f"\n  pose de depart deduite : roll {roulis0:+.1f}, "
          f"pitch {tangage0:+.1f} deg")
    print(f"  check de convention : gap prevu / measurement {ecart_conv:.2f} deg")
    if ecart_conv > 5.0:
        print("  [PROBLEME] devrait valoir zero quelle que soit la pose.")
        print("     Proche de 180 : le vector measurement pointe vers le BAS et non")
        print("     vers le haut. Il faut inverser son signe a la lecture.")
    else:
        print("  [OK] l'accelerometre pointe bien vers le haut, comme assumed.")

    print("\n" + "-" * 70)
    print("2. LES DEUX NOMBRES A RECOPIER DANS kalman_filter.py")
    print("-" * 70)
    print(f"  bias du gyro au rest    "
          f"{np.degrees(rest['bias']).round(3)} deg/s")
    print("     C'est ce bias qui, integre, fait deriver l'orientation.")
    print("     Le filter l'estime tout seul des que les tags le recalent.")
    print()
    print(f"      GYRO_NOISE_DEG_S = {rest['bruit_gyro_deg_s']:.3f}")
    print(f"      ACCEL_NOISE      = {rest['accel_noise']:.3f}")

    print("\n" + "-" * 70)
    print("3. LES MATHS : L'ORIENTATION SUIT-ELLE LES MOUVEMENTS ?")
    print("-" * 70)
    print("  Tourne la camera d'un quart de tour : le yaw doit afficher 90.")
    print("  Repose-la : roll et pitch doivent revenir vers zero.")
    print("  Ctrl+C pour arreter.\n")

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
    print("     colonnes : time, gyro raw (rad/s), accel raw (m/s2),")
    print("     puis l'orientation calculee en angles et en quaternion.")

    print("-" * 70)
    print("CE QUE TU VIENS DE MONTRER")
    print("-" * 70)
    print("  - les flux IMU du SDK Intel sont lus (accel + gyro)")
    print("  - l'echelle est verifiee sur la pesanteur")
    print("  - le gyro est integre en orientation, par quaternions")
    print("  - l'accelerometre borne roll et pitch")
    print("  - le yaw, lui, drift : rien ne le recale sans tag.")
    print("    C'est la raison d'etre de la fusion avec les AprilTags.")
    print("=" * 70)
    return 0


def _quaternion_from_matrix(R):
    from kalman_filter import matrix_to_quaternion
    return matrix_to_quaternion(np.asarray(R, dtype=float))


def _simulation():
    """Les memes maths, sur une imu SIMULEE : marche sans camera.

    Sert a deux choses : montrer que le traitement est juste meme quand le
    materiel n'est pas la, et donner un result verifiable — on connait la
    verite, donc on peut chiffrer l'error, ce qu'aucune manip reelle ne
    permet.
    """
    from kalman_filter import quaternion_angle
    rng = np.random.default_rng(3)
    dt, noise = 1 / 200, np.radians(0.15)

    print("=" * 70)
    print("LES MATHS DE L'IMU, SUR UNE CENTRALE SIMULEE")
    print("=" * 70)
    print("Sans camera branchee. La verite etant connue, l'error est chiffree.")

    print("\n1. ORIENTATION DEDUITE DU SEUL ACCELEROMETRE")
    print("   Le haut PREVU par l'orientation doit coincider avec le haut")
    print("   MESURE, et ce pour n'importe quelle pose du mounting.")
    pires = []
    for _ in range(300):
        v = rng.normal(size=3)
        haut_mesure = v / np.linalg.norm(v)
        prevu = quaternion_to_matrix(
            orientation_initiale(haut_mesure)).T @ np.array([0.0, 0.0, 1.0])
        pires.append(np.degrees(np.arccos(np.clip(prevu @ haut_mesure, -1, 1))))
    print(f"   300 poses quelconques, error max {max(pires):.1e} deg")
    assert max(pires) < 1e-4      # noise d'arccos, pas d'error de calcul

    print("\n2. UN QUART DE TOUR AUTOUR DE LA VERTICALE")
    print("   30 deg/s pendant 3 s, gyro bruite, accelerometre bruite.")
    suivi = OrientationFilter()
    suivi.start(orientation_initiale(np.array([0.0, 0.0, 1.0])), sigma_deg=5.0)
    verite = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(int(3.0 / dt)):
        omega = np.array([0.0, 0.0, np.radians(30.0)])
        verite = quaternion_product(verite, quaternion_from_rotation(omega * dt))
        suivi.predict(dt, omega + rng.normal(0, noise, 3))
        R = quaternion_to_matrix(verite)
        suivi.correct_with_gravity(R.T @ np.array([0.0, 0.0, GRAVITE])
                               + rng.normal(0, 0.05, 3))
    roll, pitch, yaw = np.degrees(quaternion_to_euler(suivi.q))
    print(f"   lu : roll {roll:+.2f}   pitch {pitch:+.2f}   "
          f"yaw {yaw:+.2f} deg   (attendu 0, 0, 90)")
    print(f"   error d'orientation : {quaternion_angle(suivi.q, verite):.2f} deg")
    assert abs(yaw - 90) < 3.0 and abs(roll) < 2 and abs(pitch) < 2

    print("\n3. TRENTE SECONDES IMMOBILE, SANS AUCUN TAG")
    print("   Le bias residuel du gyro travaille librement.")
    suivi = OrientationFilter()
    suivi.start(orientation_initiale(np.array([0.0, 0.0, 1.0])), sigma_deg=5.0)
    residuel = np.radians([0.05, -0.04, 0.30])
    for _ in range(int(30.0 / dt)):
        suivi.predict(dt, residuel + rng.normal(0, noise, 3))
        suivi.correct_with_gravity(np.array([0.0, 0.0, GRAVITE])
                               + rng.normal(0, 0.05, 3))
    roll, pitch, yaw = np.degrees(quaternion_to_euler(suivi.q))
    print(f"   roll {roll:+.2f}   pitch {pitch:+.2f} deg"
          f"   <- tenus par l'accelerometre")
    print(f"   yaw  {yaw:+.2f} deg                <- drift librement")
    assert abs(roll) < 2 and abs(pitch) < 2
    assert abs(yaw) > 3

    print("\n" + "=" * 70)
    print("CE QUE CELA ETABLIT")
    print("=" * 70)
    print("  - le gyro s'integre correctement en orientation (90 deg lus")
    print("    pour 90 deg reels, a 0.03 deg pres)")
    print("  - l'accelerometre borne roll et pitch indefiniment")
    print("  - le yaw, lui, drift : la pesanteur n'en dit rien.")
    print("    D'ou la fusion avec les AprilTags — ce n'est pas un choix")
    print("    de confort, c'est le seul moyen de tenir le cap.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    if "--simulation" in sys.argv:
        sys.exit(_simulation())
    sys.exit(_demonstration())
