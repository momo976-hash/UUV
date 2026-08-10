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
#   5. REFAIS LA MEME CHOSE AVEC 'd', camera EN MAIN, en la deplacant
#      lentement. Camera posee, on mesure le meilleur cas absolu ; or dans
#      la piscine elle bougera, avec du flou de bouge. C'est cette
#      deuxieme valeur, plus grande, qu'il faut donner au filtre.
#   6. 't' affiche le tableau recapitulatif et la verification des lois.
#
# Touches : c = capture camera posee | d = capture camera qui bouge
#           t = tableau | e = effacer les mesures | q = quitter
import csv
from pathlib import Path

import cv2
import numpy as np

CAMERA_INDEX = None
RESOLUTION = (640, 480)
TAILLE_TAG = 0.223
IMAGES_PAR_CAPTURE = 300
FREQUENCE_SUPPOSEE = 30.0

# Au-dela de cette vitesse, l'image bouge de plus d'un pixel pendant le temps
# de pose : le flou de bouge deforme les coins et la mesure ne veut plus rien
# dire. Repere : v_limite ~ d / (focale x temps_de_pose).
VITESSE_MAX_CONSEILLEE = 0.15    # m/s

# Calibration EN AIR (ce script se fait en air ; sous l'eau, multiplier par 1.33)
K_CALIB = np.array([
    [604.1876, 0.0000, 326.1973],
    [0.0000, 602.3668, 242.8850],
    [0.0000, 0.0000, 1.0000],
], dtype=np.float64)
DIST_CALIB = np.array([0.013835, 0.733706, -0.002333, 0.001136, -2.707687],
                      dtype=np.float64)

CSV = Path(__file__).resolve().with_name("bruit_tag.csv")
COLONNES = ["mode", "distance_m", "incidence_deg", "images", "vitesse_cm_s",
            "sigma_pixel", "sigma_lateral_mm", "sigma_profondeur_mm",
            "lateral_theorique_mm", "profondeur_theorique_mm"]

# Le facteur 2 vient de ce que l'echelle du tag, d'ou se deduit la distance,
# est lue sur QUATRE coins et non un seul : la moyenne divise le bruit par
# racine de 4. Sans lui le modele surestimait la profondeur d'un facteur 2.3
# sur les mesures reelles ; avec lui l'ecart tombe sous 15 %.
COINS_PAR_TAG = 4.0


def _poids_lissage(demi_fenetre, degre):
    """Poids d'un lissage polynomial local (Savitzky-Golay) et son biais.

    Renvoie aussi le facteur par lequel la variance des residus sous-estime
    la vraie variance du bruit : le lissage absorbe une part du bruit.
    """
    x = np.arange(-demi_fenetre, demi_fenetre + 1, dtype=float)
    A = np.vander(x, degre + 1, increasing=True)
    # la valeur lissee au centre est le terme constant de l'ajustement local
    w = np.linalg.pinv(A)[0]
    correction = 1.0 - 2.0 * w[demi_fenetre] + float(w @ w)
    return w, correction


def separer_bruit(valeurs, demi_fenetre=7, degre=2):
    """Separe un mouvement LISSE d'un bruit rapide.

    Camera immobile, tout est bruit. Camera qui bouge, il faut d'abord
    retirer le mouvement reel : on l'ajuste localement par un polynome et
    on ne garde que ce qui ne s'y ajuste pas.

    Renvoie (partie_lisse, bruit) avec le bruit deja corrige du biais
    d'absorption du lissage.
    """
    valeurs = np.asarray(valeurs, dtype=float)
    forme = valeurs.shape
    plat = valeurs.reshape(forme[0], -1)
    w, correction = _poids_lissage(demi_fenetre, degre)

    lisse = np.empty((plat.shape[0] - 2 * demi_fenetre, plat.shape[1]))
    for colonne in range(plat.shape[1]):
        lisse[:, colonne] = np.convolve(plat[:, colonne], w[::-1], mode="valid")
    utile = plat[demi_fenetre:plat.shape[0] - demi_fenetre]
    bruit = (utile - lisse) / np.sqrt(correction)

    nouvelle_forme = (lisse.shape[0],) + forme[1:]
    return lisse.reshape(nouvelle_forme), bruit.reshape(nouvelle_forme)


def analyser(coins, positions, focale, taille_tag, dynamique=False):
    """Coeur du calcul, isole de la camera pour pouvoir etre teste.

    coins     : (N, 4, 2) positions des 4 coins en pixels, sur N images
    positions : (N, 3) position du tag dans le repere camera, sur N images
    dynamique : True si la camera bougeait pendant la capture.
    """
    coins = np.asarray(coins, dtype=float)
    positions = np.asarray(positions, dtype=float)

    if dynamique:
        _, bruit_coins = separer_bruit(coins)
        lisse, bruit_positions = separer_bruit(positions)
        reference = lisse                      # la trajectoire, sans le bruit
    else:
        bruit_coins = coins - coins.mean(axis=0)
        bruit_positions = positions - positions.mean(axis=0)
        reference = np.repeat(positions.mean(axis=0)[None], len(positions), axis=0)

    # --- bruit des coins, en pixels ---------------------------------------
    sigma_pixel = float(np.sqrt(np.mean(np.square(bruit_coins))))

    # --- bruit de position, decompose lateral / profondeur -----------------
    # l'axe de visee change quand la camera bouge : on le reprend a chaque image
    distances = np.linalg.norm(reference, axis=1)
    distance = float(np.mean(distances))
    u = reference / distances[:, None]
    le_long = np.einsum("ij,ij->i", bruit_positions, u)
    en_travers = bruit_positions - le_long[:, None] * u
    sigma_profondeur = float(np.sqrt(np.mean(le_long ** 2)))
    # deux degres de liberte lateraux : on ramene a un ecart-type par axe
    sigma_lateral = float(np.sqrt(np.mean(np.sum(en_travers ** 2, axis=1)) / 2.0))

    return {
        "distance_m": distance,
        "images": len(bruit_positions),
        "sigma_pixel": sigma_pixel,
        "sigma_lateral_mm": 1000 * sigma_lateral,
        "sigma_profondeur_mm": 1000 * sigma_profondeur,
        # predictions du modele, a partir du sigma_pixel qu'on vient de mesurer
        "lateral_theorique_mm": 1000 * distance * sigma_pixel / focale,
        "profondeur_theorique_mm": 1000 * distance ** 2 * sigma_pixel
                                   / (focale * taille_tag * np.sqrt(COINS_PAR_TAG)),
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
    sortie = ["", "=" * 96, "MESURES DE BRUIT DE DETECTION", "=" * 96,
              f"{'mode':>7} {'dist':>6} {'incid':>6} {'img':>5} {'sigma_px':>9} "
              f"{'lat_mes':>9} {'lat_th':>8} {'prof_mes':>9} {'prof_th':>8}   unites",
              "-" * 96]
    for l in lignes:
        sortie.append(
            f"{(l.get('mode') or 'pose'):>7} "
            f"{float(l['distance_m']):6.2f} {float(l['incidence_deg']):6.1f} "
            f"{int(float(l['images'])):5d} {float(l['sigma_pixel']):9.3f} "
            f"{float(l['sigma_lateral_mm']):9.2f} {float(l['lateral_theorique_mm']):8.2f} "
            f"{float(l['sigma_profondeur_mm']):9.2f} "
            f"{float(l['profondeur_theorique_mm']):8.2f}   m/deg/px/mm")
    sortie.append("-" * 96)

    # --- les deux regimes se resument separement ---------------------------
    # On prend la MEDIANE et non la moyenne : une seule capture ratee (geste
    # trop brusque, tag mal eclaire) suffirait sinon a tirer le resultat.
    # `or "pose"` et pas `get(..., "pose")` : les lignes ecrites avant que la
    # colonne mode existe ont la CLE presente mais VIDE, le defaut ne joue pas.
    poses = [l for l in lignes if (l.get("mode") or "pose") == "pose"]
    bouges = [l for l in lignes if l.get("mode") == "bouge"]
    sigmas = lambda ens: np.array([float(l["sigma_pixel"]) for l in ens])

    # --- captures suspectes ------------------------------------------------
    # Trop de bruit par rapport aux autres, ou geste trop rapide : on les
    # signale ET on les retire de TOUT le depouillement, mediane comprise,
    # sinon elles le faussent.
    def fiables(ensemble):
        if len(ensemble) < 3:
            return ensemble, []
        med = float(np.median(sigmas(ensemble)))
        bons, ecartes = [], []
        for ligne in ensemble:
            valeur = float(ligne["sigma_pixel"])
            vitesse = float(ligne.get("vitesse_cm_s") or 0.0)
            if valeur > 3 * med:
                ecartes.append((ligne, f"{valeur:.3f} px, soit "
                                       f"{valeur/med:.0f}x la mediane"))
            elif vitesse > 100 * VITESSE_MAX_CONSEILLEE:
                ecartes.append((ligne, f"deplacee a {vitesse:.0f} cm/s"))
            else:
                bons.append(ligne)
        return bons, ecartes

    poses_bons, poses_ecartes = fiables(poses)
    bouges_bons, bouges_ecartes = fiables(bouges)
    if poses_bons:
        sortie.append(f"sigma_pixel camera POSEE     : "
                      f"{np.median(sigmas(poses_bons)):.3f} px (mediane sur "
                      f"{len(poses_bons)})   <- plancher, meilleur cas absolu")
    if bouges_bons:
        med = float(np.median(sigmas(bouges_bons)))
        sortie.append(f"sigma_pixel camera QUI BOUGE : {med:.3f} px (mediane sur "
                      f"{len(bouges_bons)})   <- LA valeur pour filtre_kalman.py")
        if poses_bons:
            sortie.append(f"                               le mouvement degrade d'un "
                          f"facteur {med/max(np.median(sigmas(poses_bons)), 1e-9):.1f}")
    else:
        sortie.append("Aucune capture en mouvement ('d'). Le filtre a besoin du bruit")
        sortie.append("EN CONDITIONS : camera posee, c'est le meilleur cas, pas l'usage.")
    for nom_regime, ecartes in (("posee", poses_ecartes), ("bouge", bouges_ecartes)):
        for ligne, raison in ecartes:
            sortie.append(f"  !! capture '{nom_regime}' a "
                          f"{float(ligne['distance_m']):.2f} m ecartee : {raison}")

    # --- verification des lois en d et d^2 ---------------------------------
    for nom_regime, ensemble in (("camera posee", poses_bons),
                                 ("camera qui bouge", bouges_bons)):
        sortie.append("")
        if len(ensemble) < 3:
            sortie.append(f"VERIFICATION DU MODELE — {nom_regime} : "
                          f"{len(ensemble)} capture(s) fiable(s), il en faut 3.")
            continue
        d = np.array([float(l["distance_m"]) for l in ensemble])
        etendue = float(d.max() / max(d.min(), 1e-9))
        sortie.append(f"VERIFICATION DU MODELE — {nom_regime} (pente en log-log)")
        if etendue < 2.0:
            # Ajuster une loi de puissance demande un bras de levier suffisant :
            # sur une plage trop courte, le bruit domine la pente.
            sortie.append(f"  distances de {d.min():.2f} a {d.max():.2f} m, soit un "
                          f"rapport de {etendue:.1f}x seulement.")
            sortie.append("  TROP ETROIT pour conclure : il faut au moins un rapport "
                          "de 3x (ex. 0.6 m a 2 m).")
            continue
        for nom, cle, cle_th, attendu in (
                ("lateral", "sigma_lateral_mm", "lateral_theorique_mm", 1.0),
                ("profondeur", "sigma_profondeur_mm", "profondeur_theorique_mm", 2.0)):
            valeurs = np.array([float(l[cle]) for l in ensemble])
            theorie = np.array([float(l[cle_th]) for l in ensemble])
            bons = valeurs > 0
            if bons.sum() >= 3:
                pente = float(np.polyfit(np.log(d[bons]), np.log(valeurs[bons]), 1)[0])
                # le rapport dit si le modele vise juste EN AMPLITUDE ;
                # la pente dit s'il vise juste EN TENDANCE.
                rapport = float(np.median(valeurs[bons] / np.maximum(theorie[bons], 1e-9)))
                verdict = ("conforme" if abs(pente - attendu) < 0.5
                           and 0.5 < rapport < 2.0 else "A REVOIR")
                sortie.append(f"  {nom:<11} erreur ~ d^{pente:.2f} "
                              f"(modele d^{attendu:.0f}),  amplitude mesuree = "
                              f"{rapport:.2f}x le modele  -> {verdict}")
    sortie.append("=" * 96)
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
            if len(capture["positions"]) >= 2:
                pas = np.linalg.norm(capture["positions"][-1] - capture["positions"][-2])
                capture["vitesses"].append(pas * FREQUENCE_SUPPOSEE)
            if len(capture["positions"]) >= IMAGES_PAR_CAPTURE:
                dynamique = capture["mode"] == "bouge"
                resultat = analyser(capture["coins"], capture["positions"],
                                    K_CALIB[0, 0], TAILLE_TAG, dynamique=dynamique)
                resultat["incidence_deg"] = float(np.mean(capture["incidences"]))
                resultat["mode"] = capture["mode"]
                resultat["vitesse_cm_s"] = (100 * float(np.mean(capture["vitesses"]))
                                            if capture["vitesses"] else 0.0)
                ligne = {c: (f"{int(resultat[c])}" if c == "images"
                             else resultat[c] if c == "mode"
                             else f"{resultat[c]:.4f}") for c in COLONNES}
                lignes.append(ligne)
                with open(CSV, "w", newline="") as fic:
                    ecrivain = csv.DictWriter(fic, fieldnames=COLONNES)
                    ecrivain.writeheader()
                    ecrivain.writerows(lignes)
                print(f"\nCapture '{capture['mode']}' terminee a "
                      f"{resultat['distance_m']:.2f} m :")
                print(f"  sigma_pixel      = {resultat['sigma_pixel']:.3f} px")
                print(f"  bruit lateral    = {resultat['sigma_lateral_mm']:.2f} mm "
                      f"(modele : {resultat['lateral_theorique_mm']:.2f} mm)")
                print(f"  bruit profondeur = {resultat['sigma_profondeur_mm']:.2f} mm "
                      f"(modele : {resultat['profondeur_theorique_mm']:.2f} mm)")
                if capture["vitesses"]:
                    rapide = float(np.mean(capture["vitesses"]))
                    if rapide > VITESSE_MAX_CONSEILLEE:
                        print(f"  !! vitesse moyenne {rapide*100:.0f} cm/s, au-dessus "
                              f"des {VITESSE_MAX_CONSEILLEE*100:.0f} cm/s conseilles.")
                        print("     Le flou de bouge gonfle la mesure : capture a refaire "
                              "plus lentement.")
                capture = None

        # --- affichage ---------------------------------------------------
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
                vitesse = float(np.mean(recentes))
                trop = vitesse > VITESSE_MAX_CONSEILLEE
                cv2.putText(image, f"vitesse {vitesse*100:5.1f} cm/s   "
                                   f"{'>>> TROP VITE <<<' if trop else 'ok'}",
                            (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255) if trop else (0, 220, 0), 2)
        elif vu is not None:
            distance = float(np.linalg.norm(vu[3]))
            cv2.putText(image, f"tag {vu[0]} a {distance:.2f} m, "
                               f"incidence {incidence(vu[2], vu[3]):.0f} deg",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(image, "'c' camera posee   |   'd' camera qui bouge",
                        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        else:
            cv2.putText(image, "Aucun tag visible", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, f"{len(lignes)} mesure(s)   c=posee d=bouge t=tableau "
                           f"e=effacer q=quitter", (10, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Mesure du bruit de detection (q pour quitter)", image)
        touche = cv2.waitKey(1) & 0xFF
        if touche == ord("q"):
            break
        if touche in (ord("c"), ord("d")) and capture is None:
            if vu is None:
                print("Aucun tag visible : impossible de capturer.")
            else:
                mode = "pose" if touche == ord("c") else "bouge"
                capture = {"coins": [], "positions": [], "incidences": [],
                           "vitesses": [], "mode": mode}
                if mode == "pose":
                    print(f"Capture IMMOBILE... ne touche a rien "
                          f"({IMAGES_PAR_CAPTURE} images)")
                else:
                    print(f"Capture EN MOUVEMENT... deplace la camera "
                          f"LENTEMENT et REGULIEREMENT ({IMAGES_PAR_CAPTURE} images)")
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
