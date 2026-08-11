from pathlib import Path
import sys
# comparaison_calibration.py — Compare l'approximation et la calibration damier.
#
# DEUX grandeurs mesurables, a ne pas confondre :
#   MODE 1  ecart entre DEUX TAGS   (distance entre les centres des tags)
#           -> facile a mesurer au ruban, independant de la position camera
#   MODE 2  distance CAMERA -> TAG  (profondeur ; le point de reference cote
#           camera est le centre optique, difficile a reperer physiquement)
#   MODE 3  ANGLE entre DEUX TAGS   (orientation relative, en degres)
#           -> si les 2 tags sont a plat sur la meme surface, la valeur exacte
#              est 0 deg : tout ecart mesure est de l'erreur, sans rapporteur
#
# Les deux sont calculees avec les DEUX jeux de parametres en meme temps :
#   A) approximation : focale = largeur x 0.95, sans distorsion
#   B) calibration par damier : fx, fy, cx, cy + distorsion
#
# Touches : m = changer de mode | 0-9 et '.' = saisir la mesure au ruban
#           o = fixer la reference d'orientation (mode 4)
#           RET. ARRIERE = effacer | s = enregistrer | q = quitter
import csv
import os
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import optique  # noqa: E402

CAMERA_INDEX = None          # None = detection automatique
RESOLUTION = (640, 480)      # doit etre identique a celle de la calibration

TAILLE_TAG = 0.223           # cote du carre noir, en metres
FACTEUR_APPROX = 0.95        # ancienne approximation
LISSAGE = 30                 # images moyennees pour stabiliser l'affichage

# Calibration par damier (5x7, 22 vues, RMS 0.169 px)
MONTAGE = "tube_air"
# L'optique vient de optique.py : camera, tube, hublot, milieu. Tant que le
# montage n'est pas calibre, optique.py retombe sur la camera nue en le disant.
K_CALIB, DIST_CALIB = optique.charger(MONTAGE)
LARGEUR_CALIB, HAUTEUR_CALIB = 640, 480


def ouvrir_camera():
    """Ouvre la camera en forcant toujours la meme resolution."""
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "AUTO")]
    indices = [CAMERA_INDEX] if CAMERA_INDEX is not None else range(4)
    for index in indices:
        for backend, nom in backends:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
                ok, img = cap.read()
                if ok and img is not None:
                    hh, ww = img.shape[:2]
                    print(f"Camera utilisee : index={index}, backend={nom}, {ww}x{hh}")
                    return cap, ww, hh
            cap.release()
    return None, 0, 0


def positions(pts_par_tag, K, dist):
    """Pose (position, rotation) de chaque tag dans le repere camera."""
    resultat = {}
    for tag_id, pts in pts_par_tag.items():
        ok, rvec, tvec = cv2.solvePnP(coins_3d, pts, K, dist,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            resultat[tag_id] = (tvec.flatten(), cv2.Rodrigues(rvec)[0])
    return resultat


def angle_entre(R1, R2):
    """Angle (degres) de la rotation qui amene le repere 1 sur le repere 2."""
    R_rel = R1.T @ R2
    cos = (np.trace(R_rel) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


cam, L, H = ouvrir_camera()
if cam is None:
    print("ERREUR : aucune camera ouverte.")
    raise SystemExit

# A) approximation
f = L * FACTEUR_APPROX
K_approx = np.array([[f, 0, L / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float64)
dist_approx = np.zeros(5)

# B) calibration
K_calib, dist_calib = K_CALIB.copy(), DIST_CALIB.copy()
Lc, Hc = LARGEUR_CALIB, HAUTEUR_CALIB
try:
    fichier = np.load("calibration_camera.npz")
    K_calib = fichier["K"].astype(np.float64)
    dist_calib = fichier["dist"].ravel()
    Lc, Hc = int(fichier["largeur"]), int(fichier["hauteur"])
    print("Calibration chargee depuis calibration_camera.npz")
except Exception:
    print("Calibration integree au script utilisee")
print(f"  calibration : {Lc}x{Hc} (fx = {K_calib[0, 0]:.1f})   capture : {L}x{H}")
if (L, H) != (Lc, Hc):
    print("  >>> ATTENTION : formats differents, la calibration n'est pas valable ici.")

h = TAILLE_TAG / 2
coins_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

dictionnaire = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detecteur = cv2.aruco.ArucoDetector(dictionnaire, params)

MODES = ["ecart entre 2 tags", "distance camera -> tag",
         "angle entre 2 tags (deg)", "rotation d'UN tag (deg)"]
mode = 0
ref_Ra = ref_Rb = None      # orientation de reference du mode 4
hist_a, hist_b = deque(maxlen=LISSAGE), deque(maxlen=LISSAGE)
saisie = ""
CSV = os.path.abspath("comparaison_calibration.csv")
if not os.path.exists(CSV):
    with open(CSV, "w", newline="") as fic:
        csv.writer(fic).writerow(
            ["mode", "reference", "approx", "erreur_approx", "erreur_approx_pct",
             "calib", "erreur_calib", "erreur_calib_pct"])

print("=" * 64)
print("MODE 1 (defaut) : ecart entre DEUX tags -> montre les 2 tags ensemble")
print("MODE 2          : distance camera -> tag")
print("MODE 3          : angle entre 2 tags coplanaires -> reference = 0 deg")
print("MODE 4          : rotation d'UN tag -> 'o' fixe la reference, puis")
print("                  fais tourner le tag d'un angle connu (ex. 90 deg)")
print("'m' change de mode | tape la mesure au ruban | 's' enregistre | 'q' quitte")
print(f"Resultats dans : {CSV}")
print("=" * 64)

while True:
    ok, image = cam.read()
    if not ok:
        continue
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coins, ids, _ = detecteur.detectMarkers(gris)

    pts_par_tag = {}
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, coins, ids)
        for c, tag_id in zip(coins, ids.flatten()):
            pts_par_tag[int(tag_id)] = c.reshape(4, 2).astype(np.float64)

    pos_a = positions(pts_par_tag, K_approx, dist_approx)
    pos_b = positions(pts_par_tag, K_calib, dist_calib)

    mesure_a = mesure_b = None
    detail = ""
    communs = sorted(set(pos_a) & set(pos_b))
    if mode == 0:                                   # ecart entre deux tags
        if len(communs) >= 2:
            t1, t2 = communs[0], communs[1]
            mesure_a = float(np.linalg.norm(pos_a[t1][0] - pos_a[t2][0]))
            mesure_b = float(np.linalg.norm(pos_b[t1][0] - pos_b[t2][0]))
            detail = f"tags {t1} et {t2}"
        else:
            detail = "montre DEUX tags en meme temps"
    elif mode == 1:                                 # distance camera -> tag
        if communs:
            t1 = communs[0]
            mesure_a = float(np.linalg.norm(pos_a[t1][0]))
            mesure_b = float(np.linalg.norm(pos_b[t1][0]))
            detail = f"tag {t1}"
        else:
            detail = "aucun tag detecte"
    elif mode == 2:                                 # angle entre deux tags
        if len(communs) >= 2:
            t1, t2 = communs[0], communs[1]
            mesure_a = angle_entre(pos_a[t1][1], pos_a[t2][1])
            mesure_b = angle_entre(pos_b[t1][1], pos_b[t2][1])
            detail = f"tags {t1} et {t2} (coplanaires -> attendu 0 deg)"
        else:
            detail = "montre DEUX tags en meme temps"
    else:                          # rotation d'UN SEUL tag depuis une reference
        if not communs:
            detail = "aucun tag detecte"
        elif ref_Ra is None:
            detail = "place le tag, puis 'o' pour fixer la reference"
        else:
            t1 = communs[0]
            mesure_a = angle_entre(ref_Ra, pos_a[t1][1])
            mesure_b = angle_entre(ref_Rb, pos_b[t1][1])
            detail = f"tag {t1} : rotation depuis la reference"

    if mesure_a is not None:
        hist_a.append(mesure_a)
        hist_b.append(mesure_b)
    else:
        hist_a.clear()
        hist_b.clear()
    d_a = sum(hist_a) / len(hist_a) if hist_a else None
    d_b = sum(hist_b) / len(hist_b) if hist_b else None

    # --- affichage ---
    cv2.putText(image, f"MODE : {MODES[mode]}  ({detail})", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    if d_a is not None:
        unite = "deg" if mode >= 2 else "m"
        cv2.putText(image, f"A) approximation : {d_a:.3f} {unite}", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(image, f"B) calibration   : {d_b:.3f} {unite}", (10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if saisie:
            try:
                ref = float(saisie)
                ea, eb = d_a - ref, d_b - ref
                if mode >= 2:
                    cv2.putText(image, f"ecart A : {ea:+.2f} deg", (10, 112),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
                    cv2.putText(image, f"ecart B : {eb:+.2f} deg", (10, 136),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                elif ref:
                    cv2.putText(image, f"ecart A : {ea*100:+.1f} cm ({ea/ref*100:+.1f} %)",
                                (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
                    cv2.putText(image, f"ecart B : {eb*100:+.1f} cm ({eb/ref*100:+.1f} %)",
                                (10, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    if d_b:
                        cv2.putText(image,
                                    f"TAILLE_TAG deduite : {TAILLE_TAG*ref/d_b*100:.1f} cm"
                                    f"  (declaree {TAILLE_TAG*100:.1f} cm)",
                                    (10, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                                    (255, 255, 0), 2)
            except ValueError:
                pass

    cv2.putText(image, f"reference ({'deg' if mode >= 2 else 'm'}) : {saisie or '...'}", (10, H - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(image, "m=mode  o=ref.orientation  chiffres=saisir  s=enregistrer  q=quitter",
                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    cv2.imshow("Comparaison des calibrations (q pour quitter)", image)

    touche = cv2.waitKey(1) & 0xFF
    if touche == ord("q"):
        break
    if touche == ord("m"):
        mode = (mode + 1) % len(MODES)
        hist_a.clear(); hist_b.clear()
        print(f"Mode : {MODES[mode]}")
    if touche == ord("o"):
        if communs:
            t1 = communs[0]
            ref_Ra = pos_a[t1][1].copy()
            ref_Rb = pos_b[t1][1].copy()
            hist_a.clear(); hist_b.clear()
            print(f"Reference d'orientation fixee sur le tag {t1}. "
                  f"Fais maintenant tourner le tag d'un angle connu.")
        else:
            print("Aucun tag visible : impossible de fixer la reference.")
    if ord("0") <= touche <= ord("9") or touche == ord("."):
        saisie += chr(touche)
    if touche == 8 and saisie:
        saisie = saisie[:-1]
    if touche == ord("s") and saisie and d_a is not None:
        try:
            ref = float(saisie)
        except ValueError:
            print("Mesure saisie invalide.")
            continue
        ea, eb = d_a - ref, d_b - ref
        # Le pourcentage n'a pas de sens si la reference est nulle
        # (cas du mode angle, ou l'angle attendu entre tags coplanaires est 0).
        pct_a = f"{ea / ref * 100:+.2f}" if ref else ""
        pct_b = f"{eb / ref * 100:+.2f}" if ref else ""
        with open(CSV, "a", newline="") as fic:
            csv.writer(fic).writerow([
                MODES[mode], f"{ref:.3f}", f"{d_a:.3f}", f"{ea:+.3f}", pct_a,
                f"{d_b:.3f}", f"{eb:+.3f}", pct_b])
        if mode >= 2:
            print(f"[{MODES[mode]}] reference {ref:.2f} deg | "
                  f"approx {d_a:.2f} ({ea:+.2f} deg) | calib {d_b:.2f} ({eb:+.2f} deg)")
        else:
            print(f"[{MODES[mode]}] ruban {ref:.3f} m | approx {d_a:.3f} ({ea*100:+.1f} cm) "
                  f"| calib {d_b:.3f} ({eb*100:+.1f} cm)")

cam.release()
cv2.destroyAllWindows()
print(f"\nTermine. Mesures dans : {CSV}")
