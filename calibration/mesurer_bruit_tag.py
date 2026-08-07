# mesurer_bruit_tag.py — Mesurer le VRAI bruit de detection de ta camera.
#
# POURQUOI CE SCRIPT
# Le filtre de Kalman a besoin de savoir a quel point une mesure de tag est
# fiable. Jusqu'ici cette fiabilite etait SUPPOSEE (sigma_pixel = 0.5 px,
# valeur courante en vision mais jamais verifiee sur ta camera). Ce script la
# MESURE, sur ta camera, avec tes tags, dans tes conditions d'eclairage.
#
# PRINCIPE
# Camera immobile, tag immobile. En theorie les 4 coins detectes devraient
# tomber au meme pixel a chaque image. En pratique ils dansent d'une fraction
# de pixel : bruit du capteur, compression, eclairage. On enregistre 300
# images sans rien toucher et on calcule l'ecart-type reel.
#
# CE QUE CA VALIDE EN PLUS
# Le modele dit que l'erreur de position d'un tag est ANISOTROPE :
#     laterale    ~ d   . sigma_px / f
#     profondeur  ~ d^2 . sigma_px / (f . T . cos incidence)
# En refaisant la mesure a plusieurs distances, on verifie si l'erreur croit
# bien comme d lateralement et comme d^2 en profondeur. Si oui, le modele du
# filtre est valide EXPERIMENTALEMENT et plus seulement suppose.
#
# MODE D'EMPLOI
#   1. Pose la camera sur un support stable (table, trepied). NE LA TIENS PAS
#      A LA MAIN : ta main tremble bien plus que le bruit qu'on veut mesurer.
#   2. Place un tag devant, bien visible, a environ 50 cm.
#   3. Appuie sur 'c'. Ne touche plus a rien pendant la capture.
#   4. Recommence a 1 m, 1.5 m, 2 m... (touche 'c' a chaque fois)
#   5. 't' affiche le tableau recapitulatif et la verification des lois.
#
# Touches : c = capturer | t = tableau | e = effacer les mesures | q = quitter
import csv
from pathlib import Path

import cv2
import numpy as np

CAMERA_INDEX = None
RESOLUTION = (640, 480)
TAILLE_TAG = 0.223
IMAGES_PAR_CAPTURE = 300

# Calibration EN AIR (ce script se fait en air ; sous l'eau, multiplier par 1.33)
K_CALIB = np.array([
    [604.1876, 0.0000, 326.1973],
    [0.0000, 602.3668, 242.8850],
    [0.0000, 0.0000, 1.0000],
], dtype=np.float64)
DIST_CALIB = np.array([0.013835, 0.733706, -0.002333, 0.001136, -2.707687],
                      dtype=np.float64)

CSV = Path(__file__).resolve().with_name("bruit_tag.csv")
COLONNES = ["distance_m", "incidence_deg", "images", "sigma_pixel",
            "sigma_lateral_mm", "sigma_profondeur_mm",
            "lateral_theorique_mm", "profondeur_theorique_mm"]


def analyser(coins, positions, focale, taille_tag):
    """Coeur du calcul, isole de la camera pour pouvoir etre teste.

    coins     : (N, 4, 2) positions des 4 coins en pixels, sur N images
    positions : (N, 3) position du tag dans le repere camera, sur N images
    """
    coins = np.asarray(coins, dtype=float)
    positions = np.asarray(positions, dtype=float)

    # --- bruit des coins, en pixels ---------------------------------------
    # ecart-type de chaque coordonnee de chaque coin, puis moyenne quadratique
    sigma_pixel = float(np.sqrt(np.mean(np.var(coins, axis=0))))

    # --- bruit de position, decompose lateral / profondeur -----------------
    centre = positions.mean(axis=0)
    distance = float(np.linalg.norm(centre))
    u = centre / distance                      # axe camera -> tag
    residus = positions - centre
    le_long = residus @ u                      # composante en profondeur
    en_travers = residus - np.outer(le_long, u)
    sigma_profondeur = float(np.std(le_long))
    # deux degres de liberte lateraux : on ramene a un ecart-type par axe
    sigma_lateral = float(np.sqrt(np.mean(np.sum(en_travers ** 2, axis=1)) / 2.0))

    return {
        "distance_m": distance,
        "images": len(positions),
        "sigma_pixel": sigma_pixel,
        "sigma_lateral_mm": 1000 * sigma_lateral,
        "sigma_profondeur_mm": 1000 * sigma_profondeur,
        # predictions du modele, a partir du sigma_pixel qu'on vient de mesurer
        "lateral_theorique_mm": 1000 * distance * sigma_pixel / focale,
        "profondeur_theorique_mm": 1000 * distance ** 2 * sigma_pixel
                                   / (focale * taille_tag),
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


def tableau(lignes):
    if not lignes:
        return "Aucune mesure. Appuie sur 'c' devant un tag."
    sortie = ["", "=" * 92,
              "MESURES DE BRUIT — camera immobile, tag immobile", "=" * 92,
              f"{'dist':>6} {'incid':>6} {'img':>5} {'sigma_px':>9} "
              f"{'lat_mes':>9} {'lat_th':>8} {'prof_mes':>9} {'prof_th':>8}   unites",
              "-" * 92]
    for l in lignes:
        sortie.append(
            f"{float(l['distance_m']):6.2f} {float(l['incidence_deg']):6.1f} "
            f"{int(float(l['images'])):5d} {float(l['sigma_pixel']):9.3f} "
            f"{float(l['sigma_lateral_mm']):9.2f} {float(l['lateral_theorique_mm']):8.2f} "
            f"{float(l['sigma_profondeur_mm']):9.2f} "
            f"{float(l['profondeur_theorique_mm']):8.2f}   m/deg/px/mm")
    sortie.append("-" * 92)
    sortie.append(f"sigma_pixel MOYEN : "
                  f"{np.mean([float(l['sigma_pixel']) for l in lignes]):.3f} px"
                  "     <-- la valeur a mettre dans filtre_kalman.py")

    # --- verification des lois en d et d^2 ---------------------------------
    if len(lignes) >= 3:
        d = np.array([float(l["distance_m"]) for l in lignes])
        lat = np.array([float(l["sigma_lateral_mm"]) for l in lignes])
        prof = np.array([float(l["sigma_profondeur_mm"]) for l in lignes])
        sortie.append("")
        sortie.append("VERIFICATION DU MODELE (pente en echelle log-log)")
        for nom, valeurs, attendu in (("lateral", lat, 1.0), ("profondeur", prof, 2.0)):
            bons = valeurs > 0
            if bons.sum() >= 3:
                pente = float(np.polyfit(np.log(d[bons]), np.log(valeurs[bons]), 1)[0])
                verdict = "conforme" if abs(pente - attendu) < 0.5 else "NON CONFORME"
                sortie.append(f"  {nom:<11} erreur ~ d^{pente:.2f}  "
                              f"(le modele predit d^{attendu:.0f})  -> {verdict}")
    else:
        sortie.append("")
        sortie.append("Fais au moins 3 captures a des distances differentes "
                      "pour verifier les lois en d et d^2.")
    sortie.append("=" * 92)
    return "\n".join(sortie)


def ouvrir_camera():
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
                    print(f"Camera : index={index}, backend={nom}, {ww}x{hh}")
                    return cap, ww, hh
            cap.release()
    return None, 0, 0


def main():
    cam, L, H = ouvrir_camera()
    if cam is None:
        print("ERREUR : aucune camera ouverte.")
        return

    demi = TAILLE_TAG / 2
    coins_3d = np.array([[-demi, demi, 0], [demi, demi, 0],
                         [demi, -demi, 0], [-demi, -demi, 0]], dtype=np.float64)
    dictionnaire = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detecteur = cv2.aruco.ArucoDetector(dictionnaire, params)

    lignes = []
    if CSV.exists():
        with open(CSV, newline="") as fic:
            lignes = list(csv.DictReader(fic))
        print(f"{len(lignes)} mesure(s) rechargee(s) depuis {CSV.name}")

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
        detectes, ids, _ = detecteur.detectMarkers(gris)

        vu = None
        if ids is not None and len(ids):
            cv2.aruco.drawDetectedMarkers(image, detectes, ids)
            # on garde le plus gros tag visible
            aires = [cv2.contourArea(c.reshape(4, 2).astype(np.float32)) for c in detectes]
            meilleur = int(np.argmax(aires))
            pts = detectes[meilleur].reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(coins_3d, pts, K_CALIB, DIST_CALIB,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok2:
                vu = (int(ids.flatten()[meilleur]), pts, rvec, tvec)

        if capture is not None and vu is not None:
            capture["coins"].append(vu[1])
            capture["positions"].append(vu[3].flatten())
            capture["incidences"].append(incidence(vu[2], vu[3]))
            if len(capture["positions"]) >= IMAGES_PAR_CAPTURE:
                resultat = analyser(capture["coins"], capture["positions"],
                                    K_CALIB[0, 0], TAILLE_TAG)
                resultat["incidence_deg"] = float(np.mean(capture["incidences"]))
                ligne = {c: (f"{int(resultat[c])}" if c == "images"
                             else f"{resultat[c]:.4f}") for c in COLONNES}
                lignes.append(ligne)
                with open(CSV, "w", newline="") as fic:
                    ecrivain = csv.DictWriter(fic, fieldnames=COLONNES)
                    ecrivain.writeheader()
                    ecrivain.writerows(lignes)
                print(f"\nCapture terminee a {resultat['distance_m']:.2f} m :")
                print(f"  sigma_pixel      = {resultat['sigma_pixel']:.3f} px")
                print(f"  bruit lateral    = {resultat['sigma_lateral_mm']:.2f} mm "
                      f"(modele : {resultat['lateral_theorique_mm']:.2f} mm)")
                print(f"  bruit profondeur = {resultat['sigma_profondeur_mm']:.2f} mm "
                      f"(modele : {resultat['profondeur_theorique_mm']:.2f} mm)")
                capture = None

        # --- affichage ---------------------------------------------------
        if capture is not None:
            fait = len(capture["positions"])
            cv2.putText(image, f"CAPTURE {fait}/{IMAGES_PAR_CAPTURE} — NE BOUGE PAS",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(image, (10, 44), (10 + int(400 * fait / IMAGES_PAR_CAPTURE), 56),
                          (0, 0, 255), -1)
        elif vu is not None:
            distance = float(np.linalg.norm(vu[3]))
            cv2.putText(image, f"tag {vu[0]} a {distance:.2f} m, "
                               f"incidence {incidence(vu[2], vu[3]):.0f} deg",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(image, "'c' pour capturer (camera posee, immobile)",
                        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        else:
            cv2.putText(image, "Aucun tag visible", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, f"{len(lignes)} mesure(s)   c=capturer t=tableau "
                           f"e=effacer q=quitter", (10, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Mesure du bruit de detection (q pour quitter)", image)
        touche = cv2.waitKey(1) & 0xFF
        if touche == ord("q"):
            break
        if touche == ord("c") and capture is None:
            if vu is None:
                print("Aucun tag visible : impossible de capturer.")
            else:
                capture = {"coins": [], "positions": [], "incidences": []}
                print(f"Capture en cours... ne touche a rien "
                      f"({IMAGES_PAR_CAPTURE} images)")
        if touche == ord("t"):
            print(tableau(lignes))
        if touche == ord("e"):
            lignes = []
            if CSV.exists():
                CSV.unlink()
            print("Mesures effacees.")

    cam.release()
    cv2.destroyAllWindows()
    print(tableau(lignes))
    if lignes:
        print(f"\nMesures enregistrees dans : {CSV}")


if __name__ == "__main__":
    main()
