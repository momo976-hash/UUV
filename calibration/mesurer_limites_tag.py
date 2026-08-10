# mesurer_limites_tag.py — Jusqu'ou un tag reste-t-il detectable ?
#
# POURQUOI CE SCRIPT
# Le plan de pose des tags s'appuie sur deux limites qui, jusqu'ici, venaient
# de regles empiriques lues dans la litterature AprilTag :
#     PIXELS_MIN    = 30 px   taille apparente minimale du tag dans l'image
#     INCIDENCE_MAX = 65 deg  angle au-dela duquel le tag est trop de biais
# Ces deux nombres decident de l'espacement des tags dans le bassin. Autant
# les mesurer sur le vrai materiel plutot que les croire sur parole.
#
# LA DIFFICULTE, ET COMMENT ON LA CONTOURNE
# Quand la detection echoue, on n'a plus de pose : on ne sait donc pas a
# quelle distance ni sous quel angle elle a echoue. On s'en sort avec une
# FENETRE GLISSANTE : sur les 30 dernieres images (une seconde), on compte
# la proportion d'images ou le tag a ete vu, et on lui associe la distance
# moyenne des images reussies. En une seconde la camera bouge a peine, donc
# cette distance vaut aussi pour les images ratees.
#
# On obtient ainsi un TAUX DE DETECTION en fonction de la taille apparente,
# puis en fonction de l'angle. La limite est l'endroit ou ce taux decroche.
#
# MODE D'EMPLOI
#   1. Un tag bien eclaire, pose contre un mur.
#   2. 'd' : balayage en DISTANCE. Garde le tag bien en face et recule
#      LENTEMENT jusqu'a le perdre completement, puis reviens. Fais deux ou
#      trois allers-retours. 'd' a nouveau pour arreter.
#   3. 'i' : balayage en INCIDENCE. Reste a distance constante (environ 1 m)
#      et tourne progressivement, jusqu'a perdre le tag. Idem, 'i' pour
#      arreter.
#   4. 'r' : le rapport, avec les deux limites mesurees.
#
# Touches : d = balayage distance | i = balayage incidence
#           r = rapport | e = effacer | q = quitter
import csv
from pathlib import Path

import cv2
import numpy as np

CAMERA_INDEX = None
RESOLUTION = (640, 480)
TAILLE_TAG = 0.223

FENETRE = 30          # images sur lesquelles on estime le taux de detection
TAUX_LIMITE = 0.95    # en dessous, on considere la detection non fiable

K_CALIB = np.array([
    [604.1876, 0.0000, 326.1973],
    [0.0000, 602.3668, 242.8850],
    [0.0000, 0.0000, 1.0000],
], dtype=np.float64)
DIST_CALIB = np.array([0.013835, 0.733706, -0.002333, 0.001136, -2.707687],
                      dtype=np.float64)

CSV = Path(__file__).resolve().with_name("limites_tag.csv")
COLONNES = ["balayage", "taux", "pixels", "incidence_deg", "distance_m"]


def limite_par_paliers(echantillons, cle, croissant, taux_limite=TAUX_LIMITE,
                       nb_paliers=12):
    """Cherche la valeur de `cle` ou le taux de detection decroche.

    `croissant` dit dans quel sens la difficulte augmente : l'incidence rend
    la detection plus dure quand elle MONTE, la taille apparente quand elle
    DESCEND. On parcourt donc du plus facile vers le plus difficile et on
    s'arrete au premier palier qui passe sous le seuil.
    """
    if len(echantillons) < nb_paliers:
        return None, []
    valeurs = np.array([e[cle] for e in echantillons])
    taux = np.array([e["taux"] for e in echantillons])

    bords = np.linspace(valeurs.min(), valeurs.max(), nb_paliers + 1)
    paliers = []
    for k in range(nb_paliers):
        dans = (valeurs >= bords[k]) & (valeurs <= bords[k + 1])
        if dans.sum() >= 5:
            paliers.append({"centre": float((bords[k] + bords[k + 1]) / 2),
                            "taux": float(taux[dans].mean()),
                            "n": int(dans.sum())})
    if not paliers:
        return None, []

    # du plus facile vers le plus difficile
    ordonnes = sorted(paliers, key=lambda p: p["centre"], reverse=not croissant)
    limite = None
    for precedent, suivant in zip(ordonnes, ordonnes[1:]):
        if precedent["taux"] >= taux_limite > suivant["taux"]:
            limite = suivant["centre"]
            break
    return limite, sorted(paliers, key=lambda p: p["centre"])


def rapport(lignes):
    if not lignes:
        return "Aucun balayage. 'd' pour la distance, 'i' pour l'incidence."
    sortie = ["", "=" * 78, "LIMITES DE DETECTION MESUREES", "=" * 78]

    distance = [{"taux": float(l["taux"]), "pixels": float(l["pixels"]),
                 "incidence_deg": float(l["incidence_deg"])}
                for l in lignes if l["balayage"] == "distance"]
    incidence = [{"taux": float(l["taux"]), "pixels": float(l["pixels"]),
                  "incidence_deg": float(l["incidence_deg"])}
                 for l in lignes if l["balayage"] == "incidence"]

    # --- taille apparente minimale ----------------------------------------
    sortie.append(f"\nBALAYAGE EN DISTANCE — {len(distance)} points")
    if len(distance) < 12:
        sortie.append("  Trop peu de points. Refais un aller-retour complet ('d').")
    else:
        limite, paliers = limite_par_paliers(distance, "pixels", croissant=False)
        sortie.append(f"  {'taille apparente':>18} {'taux de detection':>18}")
        for p in reversed(paliers):
            barre = "#" * int(round(20 * p["taux"]))
            sortie.append(f"  {p['centre']:>15.0f} px {100*p['taux']:>15.0f} %  {barre}")
        if limite is None:
            sortie.append("\n  Le taux n'est jamais descendu sous "
                          f"{100*TAUX_LIMITE:.0f} % : recule davantage,")
            sortie.append("  jusqu'a perdre completement le tag.")
        else:
            sortie.append(f"\n  PIXELS_MIN mesure = {limite:.0f} px "
                          f"(la valeur supposee etait 30 px)")
            portee = K_CALIB[0, 0] * TAILLE_TAG / limite
            sortie.append(f"  soit une portee de {portee:.2f} m en air, "
                          f"{portee*1.33:.2f} m sous l'eau")

    # --- incidence maximale ------------------------------------------------
    sortie.append(f"\nBALAYAGE EN INCIDENCE — {len(incidence)} points")
    if len(incidence) < 12:
        sortie.append("  Trop peu de points. Refais un balayage complet ('i').")
    else:
        limite, paliers = limite_par_paliers(incidence, "incidence_deg", croissant=True)
        sortie.append(f"  {'incidence':>18} {'taux de detection':>18}")
        for p in paliers:
            barre = "#" * int(round(20 * p["taux"]))
            sortie.append(f"  {p['centre']:>14.0f} deg {100*p['taux']:>15.0f} %  {barre}")
        if limite is None:
            sortie.append("\n  Le taux n'est jamais descendu sous "
                          f"{100*TAUX_LIMITE:.0f} % : tourne davantage le tag.")
        else:
            sortie.append(f"\n  INCIDENCE_MAX mesuree = {limite:.0f} deg "
                          f"(la valeur supposee etait 65 deg)")

    sortie.append("\n" + "=" * 78)
    sortie.append("Reporte ces deux valeurs dans plan_piscine_3d.py.")
    sortie.append("=" * 78)
    return "\n".join(sortie)


def incidence_du_tag(rvec, tvec):
    R = cv2.Rodrigues(rvec)[0]
    normale, vers = R[:, 2], tvec.flatten()
    distance = np.linalg.norm(vers)
    if distance < 1e-9:
        return 0.0
    cos = abs(float(normale @ vers) / distance)
    return float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))


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
        print(f"{len(lignes)} point(s) recharge(s) depuis {CSV.name}")

    print("=" * 70)
    print("MESURE DES LIMITES DE DETECTION")
    print("  'd' balayage en distance  : recule LENTEMENT jusqu'a perdre le tag")
    print("  'i' balayage en incidence : tourne le tag jusqu'a le perdre")
    print("  'r' rapport | 'e' effacer | 'q' quitter")
    print("=" * 70)

    balayage = None
    fenetre = []          # (vu, pixels, incidence, distance) des dernieres images

    while True:
        ok, image = cam.read()
        if not ok:
            continue
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detectes, ids, _ = detecteur.detectMarkers(gris)

        vu = None
        if ids is not None and len(ids):
            cv2.aruco.drawDetectedMarkers(image, detectes, ids)
            aires = [cv2.contourArea(c.reshape(4, 2).astype(np.float32)) for c in detectes]
            meilleur = int(np.argmax(aires))
            pts = detectes[meilleur].reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(coins_3d, pts, K_CALIB, DIST_CALIB,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok2:
                # taille apparente = cote moyen du carre detecte, en pixels
                cotes = [np.linalg.norm(pts[k] - pts[(k + 1) % 4]) for k in range(4)]
                vu = {"pixels": float(np.mean(cotes)),
                      "incidence": incidence_du_tag(rvec, tvec),
                      "distance": float(np.linalg.norm(tvec))}

        if balayage is not None:
            fenetre.append(vu)
            if len(fenetre) > FENETRE:
                fenetre.pop(0)
            reussies = [f for f in fenetre if f is not None]
            if len(fenetre) == FENETRE and reussies:
                lignes.append({
                    "balayage": balayage,
                    "taux": f"{len(reussies) / FENETRE:.4f}",
                    "pixels": f"{np.mean([f['pixels'] for f in reussies]):.4f}",
                    "incidence_deg": f"{np.mean([f['incidence'] for f in reussies]):.4f}",
                    "distance_m": f"{np.mean([f['distance'] for f in reussies]):.4f}",
                })

        # --- affichage ---------------------------------------------------
        if balayage is not None:
            reussies = [f for f in fenetre if f is not None]
            taux = len(reussies) / max(len(fenetre), 1)
            cv2.putText(image, f"BALAYAGE {balayage.upper()} — "
                               f"{len(lignes)} points", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            couleur = (0, 220, 0) if taux > 0.95 else (
                (0, 170, 255) if taux > 0.4 else (0, 0, 255))
            cv2.putText(image, f"taux de detection {100*taux:3.0f} %", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur, 2)
            if reussies:
                m = reussies[-1]
                cv2.putText(image, f"{m['pixels']:.0f} px   {m['incidence']:.0f} deg"
                                   f"   {m['distance']:.2f} m", (10, 84),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            consigne = ("RECULE lentement jusqu'a perdre le tag"
                        if balayage == "distance"
                        else "TOURNE le tag jusqu'a le perdre")
            cv2.putText(image, consigne, (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        elif vu is not None:
            cv2.putText(image, f"{vu['pixels']:.0f} px   {vu['incidence']:.0f} deg"
                               f"   {vu['distance']:.2f} m", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(image, "'d' balayage distance   |   'i' balayage incidence",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        else:
            cv2.putText(image, "Aucun tag visible", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, f"{len(lignes)} point(s)   d=distance i=incidence "
                           f"r=rapport e=effacer q=quitter", (10, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Limites de detection (q pour quitter)", image)
        touche = cv2.waitKey(1) & 0xFF
        if touche == ord("q"):
            break
        for nom, lettre in (("distance", "d"), ("incidence", "i")):
            if touche == ord(lettre):
                if balayage == nom:
                    balayage, fenetre = None, []
                    with open(CSV, "w", newline="") as fic:
                        ecrivain = csv.DictWriter(fic, fieldnames=COLONNES)
                        ecrivain.writeheader()
                        ecrivain.writerows(lignes)
                    print(f"Balayage {nom} arrete. {len(lignes)} points au total.")
                elif balayage is None:
                    balayage, fenetre = nom, []
                    print(f"Balayage {nom} en cours... ('{lettre}' pour arreter)")
        if touche == ord("r"):
            print(rapport(lignes))
        if touche == ord("e"):
            lignes, fenetre = [], []
            if CSV.exists():
                CSV.unlink()
            print("Points effaces.")

    cam.release()
    cv2.destroyAllWindows()
    if lignes:
        with open(CSV, "w", newline="") as fic:
            ecrivain = csv.DictWriter(fic, fieldnames=COLONNES)
            ecrivain.writeheader()
            ecrivain.writerows(lignes)
    print(rapport(lignes))


if __name__ == "__main__":
    main()
