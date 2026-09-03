# measure_tag_noise.py — Measure your camera's REAL detection noise.
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python calibration/measure_tag_noise.py
#
# Camera still, tag still. The script watches the detected corners dance and
# turns that into SIGMA_PIXEL, the number the Kalman filter uses to know how
# much to trust a tag. It prints the line to copy into kalman/kalman_filter.py.
#
# The value in the repository (0.215 px) was measured IN AIR. It must be
# redone underwater: murkier water and poorer contrast make it worse, and the
# filter would otherwise believe the tags more than it should.
# ===========================================================================
#
# POURQUOI CE SCRIPT
# Le filter de Kalman a besoin de savoir a quel point une measurement de tag est
# fiable. Jusqu'ici cette fiabilite etait SUPPOSEE (sigma_pixel = 0.5 px,
# value current en vision mais jamais verifiee sur ta camera). Ce script la
# MESURE, sur ta camera, avec tes tags, dans tes conditions d'eclairage.
#
# PRINCIPE
# Camera at_rest, tag at_rest. En theorie les 4 corners detectes devraient
# tomber au meme pixel a chaque image. En pratique ils dansent d'une fraction
# de pixel : noise du capteur, compression, eclairage. On enregistre 300
# frames sans rien toucher et on calcule l'gap-type reel.
#
# CE QUE CA VALIDE EN PLUS
# Le model dit que l'error de position d'un tag est ANISOTROPE :
#     laterale    ~ d   . sigma_px / f
#     depth  ~ d^2 . sigma_px / (f . T . cos incidence)
# En refaisant la measurement a plusieurs distances, on verifie si l'error croit
# bien comme d lateralement et comme d^2 en depth. Si oui, le model du
# filter est valide EXPERIMENTALEMENT et plus seulement assumed.
#
# MODE D'EMPLOI
#   1. Pose la camera sur un support stable (table, trepied). NE LA TIENS PAS
#      A LA MAIN : ta main tremble bien plus que le noise qu'on veut mesurer.
#   2. Place un tag devant, bien visible, a environ 50 cm.
#   3. Appuie sur 'c'. Ne key plus a rien pendant la capture.
#   4. Recommence a 1 m, 1.5 m, 2 m... (key 'c' a chaque fois)
#   5. REFAIS LA MEME CHOSE AVEC 'd', camera EN MAIN, en la deplacant
#      lentement. Camera posee, on measurement le meilleur cas absolu ; or dans
#      la piscine elle bougera, avec du flou de bouge. C'est cette
#      second value, plus grande, qu'il faut donner au filter.
#   6. 't' affiche le tableau recapitulatif et la check des lois.
#
# Touches : c = capture camera posee | d = capture camera qui bouge
#           t = tableau | e = effacer les measurements | q = quitter
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import optics  # noqa: E402

CAMERA_INDEX = None
RESOLUTION = optics.RESOLUTION
TAG_SIZE = optics.LARGE_TAG_SIZE   # measurement au pied a coulisse, pas 223 mm nominal
IMAGES_PAR_CAPTURE = 300
FREQUENCE_SUPPOSEE = 30.0

# Au-dela de cette velocity, l'image bouge de plus d'un pixel pendant le time
# de pose : le flou de bouge deforme les corners et la measurement ne veut plus rien
# dire. Repere : v_limite ~ d / (focal_length x temps_de_pose).
VITESSE_MAX_CONSEILLEE = 0.15    # m/s

# Le noise measurement vaut pour le milieu ou la manip est faite. Celui du depot a
# ete releve EN AIR ; sous l'water, la turbidite et la perte de contraste le
# degraderont, et il faut donc le remesurer une fois immerge — en basculant
# optics.ACTIVE_MOUNTING sur 'tube_eau'.
MONTAGE = optics.ACTIVE_MOUNTING
# L'optics vient de optics.py : camera, tube, hublot, milieu. Le mounting
# n'est ecrit dans aucun path de code : optics.py le lit dans
# calibration/montage_local.txt, propre a CETTE machine, et le demande une
# fois s'il n'existe pas encore. Pour le changer :
#     python calibration/set_mounting.py
# Pour une seule commande, sans rien deregler :
#     UUV_MONTAGE=nue_air python ce_script.py
# Tant qu'il n'est pas calibre, optics.py retombe sur la camera nue en le
# disant.
K_CALIB, DIST_CALIB = optics.load(MONTAGE)

CSV = Path(__file__).resolve().with_name("bruit_tag.csv")
COLONNES = ["mode", "distance_m", "incidence_deg", "frames", "vitesse_cm_s",
            "sigma_pixel", "sigma_lateral_mm", "sigma_profondeur_mm",
            "lateral_theorique_mm", "profondeur_theorique_mm"]

# Le facteur 2 vient de ce que l'echelle du tag, d'ou se deduit la distance,
# est lue sur QUATRE corners et non un seul : la mean divise le noise par
# racine de 4. Sans lui le model surestimait la depth d'un facteur 2.3
# sur les measurements reelles ; avec lui l'gap tombe sous 15 %.
CORNERS_PER_TAG = 4.0


def _poids_lissage(demi_fenetre, degre):
    """Poids d'un lissage polynomial local (Savitzky-Golay) et son bias.

    Renvoie aussi le facteur par lequel la variance des residus sous-estime
    la true variance du noise : le lissage absorbe une part du noise.
    """
    x = np.arange(-demi_fenetre, demi_fenetre + 1, dtype=float)
    A = np.vander(x, degre + 1, increasing=True)
    # la value lissee au centre est le terme constant de l'ajustement local
    w = np.linalg.pinv(A)[0]
    correction = 1.0 - 2.0 * w[demi_fenetre] + float(w @ w)
    return w, correction


def separer_bruit(values, demi_fenetre=7, degre=2):
    """Separe un mouvement LISSE d'un noise rapide.

    Camera at_rest, tout est noise. Camera qui bouge, il faut d'abord
    retirer le mouvement reel : on l'ajuste localement par un polynome et
    on ne garde que ce qui ne s'y ajuste pas.

    Renvoie (partie_lisse, noise) avec le noise deja corrige du bias
    d'absorption du lissage.
    """
    values = np.asarray(values, dtype=float)
    forme = values.shape
    plat = values.reshape(forme[0], -1)
    w, correction = _poids_lissage(demi_fenetre, degre)

    lisse = np.empty((plat.shape[0] - 2 * demi_fenetre, plat.shape[1]))
    for column in range(plat.shape[1]):
        lisse[:, column] = np.convolve(plat[:, column], w[::-1], mode="valid")
    utile = plat[demi_fenetre:plat.shape[0] - demi_fenetre]
    noise = (utile - lisse) / np.sqrt(correction)

    nouvelle_forme = (lisse.shape[0],) + forme[1:]
    return lisse.reshape(nouvelle_forme), noise.reshape(nouvelle_forme)


def analyser(corners, positions, focal_length, taille_tag, dynamique=False):
    """Coeur du calcul, isole de la camera pour pouvoir etre teste.

    corners     : (N, 4, 2) positions des 4 corners en pixels, sur N frames
    positions : (N, 3) position du tag dans le frame camera, sur N frames
    dynamique : True si la camera bougeait pendant la capture.
    """
    corners = np.asarray(corners, dtype=float)
    positions = np.asarray(positions, dtype=float)

    if dynamique:
        _, bruit_coins = separer_bruit(corners)
        lisse, bruit_positions = separer_bruit(positions)
        reference = lisse                      # la trajectoire, sans le noise
    else:
        bruit_coins = corners - corners.mean(axis=0)
        bruit_positions = positions - positions.mean(axis=0)
        reference = np.repeat(positions.mean(axis=0)[None], len(positions), axis=0)

    # --- noise des corners, en pixels ---------------------------------------
    sigma_pixel = float(np.sqrt(np.mean(np.square(bruit_coins))))

    # --- noise de position, decompose lateral / depth -----------------
    # l'axis de visee change quand la camera bouge : on le reprend a chaque image
    distances = np.linalg.norm(reference, axis=1)
    distance = float(np.mean(distances))
    u = reference / distances[:, None]
    le_long = np.einsum("ij,ij->i", bruit_positions, u)
    en_travers = bruit_positions - le_long[:, None] * u
    sigma_profondeur = float(np.sqrt(np.mean(le_long ** 2)))
    # deux degres de liberte lateraux : on ramene a un gap-type par axis
    sigma_lateral = float(np.sqrt(np.mean(np.sum(en_travers ** 2, axis=1)) / 2.0))

    return {
        "distance_m": distance,
        "frames": len(bruit_positions),
        "sigma_pixel": sigma_pixel,
        "sigma_lateral_mm": 1000 * sigma_lateral,
        "sigma_profondeur_mm": 1000 * sigma_profondeur,
        # predictions du model, a partir du sigma_pixel qu'on vient de mesurer
        "lateral_theorique_mm": 1000 * distance * sigma_pixel / focal_length,
        "profondeur_theorique_mm": 1000 * distance ** 2 * sigma_pixel
                                   / (focal_length * taille_tag * np.sqrt(CORNERS_PER_TAG)),
    }


def incidence(rvec, tvec):
    """Angle sous lequel la camera voit le tag (0 = pile en face)."""
    R = cv2.Rodrigues(rvec)[0]
    normale, vers = R[:, 2], tvec.flatten()
    distance = np.linalg.norm(vers)
    if distance < 1e-9:
        return 0.0
    cos = abs(float(normale @ vers) / distance)
    return float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))


def tableau(rows):
    if not rows:
        return "Aucune measurement. Appuie sur 'c' devant un tag."
    output = ["", "=" * 96, "MESURES DE BRUIT DE DETECTION", "=" * 96,
              f"{'mode':>7} {'dist':>6} {'incid':>6} {'img':>5} {'sigma_px':>9} "
              f"{'lat_mes':>9} {'lat_th':>8} {'prof_mes':>9} {'prof_th':>8}   unites",
              "-" * 96]
    for l in rows:
        output.append(
            f"{(l.get('mode') or 'pose'):>7} "
            f"{float(l['distance_m']):6.2f} {float(l['incidence_deg']):6.1f} "
            f"{int(float(l['frames'])):5d} {float(l['sigma_pixel']):9.3f} "
            f"{float(l['sigma_lateral_mm']):9.2f} {float(l['lateral_theorique_mm']):8.2f} "
            f"{float(l['sigma_profondeur_mm']):9.2f} "
            f"{float(l['profondeur_theorique_mm']):8.2f}   m/deg/px/mm")
    output.append("-" * 96)

    # --- les deux regimes se resument separement ---------------------------
    # On prend la MEDIANE et non la mean : une seule capture ratee (geste
    # trop brusque, tag mal eclaire) suffirait sinon a tirer le result.
    # `or "pose"` et pas `get(..., "pose")` : les rows ecrites avant que la
    # column mode existe ont la CLE presente mais VIDE, le default ne joue pas.
    poses = [l for l in rows if (l.get("mode") or "pose") == "pose"]
    bouges = [l for l in rows if l.get("mode") == "bouge"]
    sigmas = lambda ens: np.array([float(l["sigma_pixel"]) for l in ens])

    # --- captures suspectes ------------------------------------------------
    # Trop de noise par report aux autres, ou geste trop rapide : on les
    # signale ET on les retire de TOUT le depouillement, median comprise,
    # sinon elles le faussent.
    def fiables(ensemble):
        if len(ensemble) < 3:
            return ensemble, []
        med = float(np.median(sigmas(ensemble)))
        bons, ecartes = [], []
        for row in ensemble:
            value = float(row["sigma_pixel"])
            velocity = float(row.get("vitesse_cm_s") or 0.0)
            if value > 3 * med:
                ecartes.append((row, f"{value:.3f} px, soit "
                                       f"{value/med:.0f}x la median"))
            elif velocity > 100 * VITESSE_MAX_CONSEILLEE:
                ecartes.append((row, f"deplacee a {velocity:.0f} cm/s"))
            else:
                bons.append(row)
        return bons, ecartes

    poses_bons, poses_ecartes = fiables(poses)
    bouges_bons, bouges_ecartes = fiables(bouges)
    if poses_bons:
        output.append(f"sigma_pixel camera POSEE     : "
                      f"{np.median(sigmas(poses_bons)):.3f} px (median sur "
                      f"{len(poses_bons)})   <- plancher, meilleur cas absolu")
    if bouges_bons:
        med = float(np.median(sigmas(bouges_bons)))
        output.append(f"sigma_pixel camera QUI BOUGE : {med:.3f} px (median sur "
                      f"{len(bouges_bons)})   <- LA value a retenir")
        if poses_bons:
            output.append(f"                               le mouvement degrade d'un "
                          f"facteur {med/max(np.median(sigmas(poses_bons)), 1e-9):.1f}")
        output += [
            "",
            "A RECOPIER dans kalman/kalman_filter.py,",
            "bloc « LES TROIS NOMBRES A MESURER » (vers la row 190) :",
            "",
            f"    SIGMA_PIXEL = {med:.3f}",
            "",
            "Cette row existe deja : il n'y a qu'a changer le count.",
        ]
    else:
        output.append("Aucune capture en mouvement ('d'). Le filter a besoin du noise")
        output.append("EN CONDITIONS : camera posee, c'est le meilleur cas, pas l'usage.")
    for nom_regime, ecartes in (("posee", poses_ecartes), ("bouge", bouges_ecartes)):
        for row, raison in ecartes:
            output.append(f"  !! capture '{nom_regime}' a "
                          f"{float(row['distance_m']):.2f} m ecartee : {raison}")

    # --- check des lois en d et d^2 ---------------------------------
    for nom_regime, ensemble in (("camera posee", poses_bons),
                                 ("camera qui bouge", bouges_bons)):
        output.append("")
        if len(ensemble) < 3:
            output.append(f"VERIFICATION DU MODELE — {nom_regime} : "
                          f"{len(ensemble)} capture(s) fiable(s), il en faut 3.")
            continue
        d = np.array([float(l["distance_m"]) for l in ensemble])
        etendue = float(d.max() / max(d.min(), 1e-9))
        output.append(f"VERIFICATION DU MODELE — {nom_regime} (pente en log-log)")
        if etendue < 2.0:
            # Ajuster une loi de puissance demande un bras de levier suffisant :
            # sur une plage trop courte, le noise domine la pente.
            output.append(f"  distances de {d.min():.2f} a {d.max():.2f} m, soit un "
                          f"report de {etendue:.1f}x seulement.")
            output.append("  TROP ETROIT pour conclure : il faut au moins un report "
                          "de 3x (ex. 0.6 m a 2 m).")
            continue
        for name, cle, cle_th, attendu in (
                ("lateral", "sigma_lateral_mm", "lateral_theorique_mm", 1.0),
                ("depth", "sigma_profondeur_mm", "profondeur_theorique_mm", 2.0)):
            values = np.array([float(l[cle]) for l in ensemble])
            theorie = np.array([float(l[cle_th]) for l in ensemble])
            bons = values > 0
            if bons.sum() >= 3:
                pente = float(np.polyfit(np.log(d[bons]), np.log(values[bons]), 1)[0])
                # le report dit si le model vise juste EN AMPLITUDE ;
                # la pente dit s'il vise juste EN TENDANCE.
                report = float(np.median(values[bons] / np.maximum(theorie[bons], 1e-9)))
                verdict = ("conforme" if abs(pente - attendu) < 0.5
                           and 0.5 < report < 2.0 else "A REVOIR")
                output.append(f"  {name:<11} error ~ d^{pente:.2f} "
                              f"(model d^{attendu:.0f}),  amplitude measured = "
                              f"{report:.2f}x le model  -> {verdict}")
    output.append("=" * 96)
    return "\n".join(output)


def ouvrir_camera():
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
                    print(f"Camera : index={index}, backend={name}, {ww}x{hh}")
                    return cap, ww, hh
            cap.release()
    return None, 0, 0


def main():
    cam, L, H = ouvrir_camera()
    if cam is None:
        print("ERREUR : aucune camera ouverte.")
        return

    demi = TAG_SIZE / 2
    coins_3d = np.array([[-demi, demi, 0], [demi, demi, 0],
                         [demi, -demi, 0], [-demi, -demi, 0]], dtype=np.float64)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    rows = []
    if CSV.exists():
        with open(CSV, newline="") as fic:
            rows = list(csv.DictReader(fic))
        print(f"{len(rows)} measurement(s) rechargee(s) depuis {CSV.name}")

    print("=" * 70)
    print("MESURE DU BRUIT DE DETECTION")
    print("  Pose la camera sur un support STABLE. Ne la tiens pas a la main.")
    print("  'c' capturer | 't' tableau | 'e' effacer | 'q' quitter")
    print("=" * 70)

    capture = None
    while True:
        ok, image = cam.read()
        if not ok:
            continue
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detectes, ids, _ = detector.detectMarkers(gris)

        seen = None
        if ids is not None and len(ids):
            cv2.aruco.drawDetectedMarkers(image, detectes, ids)
            # on garde le plus gros tag visible
            aires = [cv2.contourArea(c.reshape(4, 2).astype(np.float32)) for c in detectes]
            meilleur = int(np.argmax(aires))
            pts = detectes[meilleur].reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(coins_3d, pts, K_CALIB, DIST_CALIB,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok2:
                seen = (int(ids.flatten()[meilleur]), pts, rvec, tvec)

        if capture is not None and seen is not None:
            capture["corners"].append(seen[1])
            capture["positions"].append(seen[3].flatten())
            capture["incidences"].append(incidence(seen[2], seen[3]))
            if len(capture["positions"]) >= 2:
                pas = np.linalg.norm(capture["positions"][-1] - capture["positions"][-2])
                capture["vitesses"].append(pas * FREQUENCE_SUPPOSEE)
            if len(capture["positions"]) >= IMAGES_PAR_CAPTURE:
                dynamique = capture["mode"] == "bouge"
                result = analyser(capture["corners"], capture["positions"],
                                    K_CALIB[0, 0], TAG_SIZE, dynamique=dynamique)
                result["incidence_deg"] = float(np.mean(capture["incidences"]))
                result["mode"] = capture["mode"]
                result["vitesse_cm_s"] = (100 * float(np.mean(capture["vitesses"]))
                                            if capture["vitesses"] else 0.0)
                row = {c: (f"{int(result[c])}" if c == "frames"
                             else result[c] if c == "mode"
                             else f"{result[c]:.4f}") for c in COLONNES}
                rows.append(row)
                with open(CSV, "w", newline="") as fic:
                    ecrivain = csv.DictWriter(fic, fieldnames=COLONNES)
                    ecrivain.writeheader()
                    ecrivain.writerows(rows)
                print(f"\nCapture '{capture['mode']}' terminee a "
                      f"{result['distance_m']:.2f} m :")
                print(f"  sigma_pixel      = {result['sigma_pixel']:.3f} px")
                print(f"  noise lateral    = {result['sigma_lateral_mm']:.2f} mm "
                      f"(model : {result['lateral_theorique_mm']:.2f} mm)")
                print(f"  noise depth = {result['sigma_profondeur_mm']:.2f} mm "
                      f"(model : {result['profondeur_theorique_mm']:.2f} mm)")
                if capture["vitesses"]:
                    rapide = float(np.mean(capture["vitesses"]))
                    if rapide > VITESSE_MAX_CONSEILLEE:
                        print(f"  !! velocity mean {rapide*100:.0f} cm/s, au-dessus "
                              f"des {VITESSE_MAX_CONSEILLEE*100:.0f} cm/s conseilles.")
                        print("     Le flou de bouge gonfle la measurement : capture a refaire "
                              "plus lentement.")
                capture = None

        # --- display ---------------------------------------------------
        if capture is not None:
            fait = len(capture["positions"])
            consigne = ("NE BOUGE PAS" if capture["mode"] == "pose"
                        else "BOUGE LENTEMENT ET REGULIEREMENT")
            cv2.putText(image, f"CAPTURE {fait}/{IMAGES_PAR_CAPTURE} — {consigne}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.rectangle(image, (10, 44), (10 + int(400 * fait / IMAGES_PAR_CAPTURE), 56),
                          (0, 0, 255), -1)
            if capture["vitesses"]:
                recentes = capture["vitesses"][-10:]
                velocity = float(np.mean(recentes))
                trop = velocity > VITESSE_MAX_CONSEILLEE
                cv2.putText(image, f"velocity {velocity*100:5.1f} cm/s   "
                                   f"{'>>> TROP VITE <<<' if trop else 'ok'}",
                            (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255) if trop else (0, 220, 0), 2)
        elif seen is not None:
            distance = float(np.linalg.norm(seen[3]))
            cv2.putText(image, f"tag {seen[0]} a {distance:.2f} m, "
                               f"incidence {incidence(seen[2], seen[3]):.0f} deg",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(image, "'c' camera posee   |   'd' camera qui bouge",
                        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        else:
            cv2.putText(image, "Aucun tag visible", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, f"{len(rows)} measurement(s)   c=posee d=bouge t=tableau "
                           f"e=effacer q=quitter", (10, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Mesure du noise de detection (q pour quitter)", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in (ord("c"), ord("d")) and capture is None:
            if seen is None:
                print("Aucun tag visible : impossible de capturer.")
            else:
                mode = "pose" if key == ord("c") else "bouge"
                capture = {"corners": [], "positions": [], "incidences": [],
                           "vitesses": [], "mode": mode}
                if mode == "pose":
                    print(f"Capture IMMOBILE... ne key a rien "
                          f"({IMAGES_PAR_CAPTURE} frames)")
                else:
                    print(f"Capture EN MOUVEMENT... deplace la camera "
                          f"LENTEMENT et REGULIEREMENT ({IMAGES_PAR_CAPTURE} frames)")
        if key == ord("t"):
            print(tableau(rows))
        if key == ord("e"):
            rows = []
            if CSV.exists():
                CSV.unlink()
            print("Mesures effacees.")

    cam.release()
    cv2.destroyAllWindows()
    print(tableau(rows))
    if rows:
        print(f"\nMesures enregistrees dans : {CSV}")


if __name__ == "__main__":
    main()
