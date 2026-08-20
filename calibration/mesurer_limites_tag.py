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
# LE TAG D'ESSAI PEUT ETRE PLUS PETIT QUE LE VRAI
# Le detecteur ne connait pas les metres : il ne voit qu'un carre de N pixels.
# Un tag de 5 cm a 1.5 m produit exactement la meme image qu'un tag de 22.3 cm
# a 6.7 m. On peut donc mesurer la limite dans un couloir de 2 m avec un petit
# tag imprime, puis la transposer au vrai tag du bassin.
#
#     taille apparente en pixels  =  focale x taille_tag / distance
#
# Avec les 22.3 cm du bassin, 30 px ne sont atteints qu'a 4.5 m : impossible
# avec une camera au bout d'un cable. Avec un tag de 5 cm, 30 px tombent a
# 1.0 m et 20 px a 1.5 m — tout le domaine utile tient sur un bureau.
# On passe la taille du tag d'essai avec --tag ; le rapport, lui, reconvertit
# toujours vers TAILLE_TAG_REELLE.
#
# MODE D'EMPLOI
#   1. Un tag bien eclaire, pose contre un mur.
#   2. 'd' : balayage en DISTANCE. Petit tag (--tag 0.05). Garde-le bien en
#      face et recule LENTEMENT jusqu'a le perdre completement, puis reviens.
#      Fais deux ou trois allers-retours. 'd' a nouveau pour arreter.
#   3. 'i' : balayage en INCIDENCE. Cette fois le GRAND tag, a environ 1 m,
#      pour que la taille apparente ne soit jamais le facteur limitant.
#      Tourne progressivement jusqu'a perdre le tag. Idem, 'i' pour arreter.
#      (relance le script avec --tag 0.223 entre les deux balayages)
#   4. 'r' : le rapport, avec les deux limites mesurees.
#
# Touches : d = balayage distance | i = balayage incidence
#           r = rapport | e = effacer | q = quitter
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import optique  # noqa: E402

CAMERA_INDEX = None

MONTAGE = optique.MONTAGE_ACTIF
# L'optique vient de optique.py : camera, tube, hublot, milieu. Le montage
# n'est plus ecrit ici : il se regle en UN seul endroit, optique.MONTAGE_ACTIF
# (ou pour une seule commande : UUV_MONTAGE=tube_eau python ce_script.py).
# Tant qu'il n'est pas calibre, optique.py retombe sur la camera nue en le
# disant.
K_CALIB, DIST_CALIB = optique.charger(MONTAGE)
RESOLUTION = optique.RESOLUTION

TAILLE_TAG_REELLE = 0.223   # les tags du bassin : c'est vers eux qu'on conclut
TAILLE_TAG = 0.223          # le tag d'essai devant la camera (option --tag)

FENETRE = 30          # images sur lesquelles on estime le taux de detection
TAUX_LIMITE = 0.95    # en dessous, on considere la detection non fiable
MARGE_BORD = 20       # px : plus pres du bord, le tag risque de sortir du cadre
PALIERS_CONFIRMATION = 2   # paliers consecutifs sous le seuil pour conclure

# Taille apparente qu'il faut atteindre pour esperer encadrer la limite. Un
# 36h11 fait huit cellules de large et il en faut environ deux pixels chacune
# pour decoder : la limite ne peut pas etre bien au-dessus de la quinzaine de
# pixels. Tant que le balayage s'arrete au-dessus, il ne prouve rien.
CIBLE_PIXELS = 15

# Le bassin, pour rapporter la mesure a ce qu'on en fera vraiment. Sa
# diagonale majore la distance camera-tag. On croit souvent que l'eau arrange
# les choses — a travers un hublot plat elle grossit l'image de 1.33. Dans ce
# montage-ci la camera est COUCHEE dans le tube : un seul des deux axes voit
# une lame plane, l'autre traverse un menisque qui retrecit. Et pour decoder
# un tag, c'est l'axe le moins grossi qui commande. `optique.focale_eau`
# renvoie donc celui-la, et le pire cas du bassin est plus severe que ne le
# laisserait croire le facteur 1.33.
BASSIN = (3.80, 1.67, 1.00)


def pixels_pire_cas():
    """Taille apparente du tag au point le plus eloigne possible du bassin.

    C'est la seule valeur qui compte pour le plan de pose : inutile de
    connaitre la limite absolue de detection si le bassin ne l'approche
    jamais. Il suffit d'avoir verifie la detection jusqu'en dessous.
    """
    diagonale = float(np.linalg.norm(BASSIN))
    return optique.focale_eau(MONTAGE) * TAILLE_TAG_REELLE / diagonale, diagonale


CSV = Path(__file__).resolve().with_name("limites_tag.csv")
COLONNES = ["balayage", "taux", "pixels", "incidence_deg", "distance_m", "bord_px"]


def limite_par_paliers(echantillons, cle, croissant, taux_limite=TAUX_LIMITE,
                       nb_paliers=12, logarithmique=False):
    """Cherche la valeur de `cle` ou le taux de detection decroche.

    `croissant` dit dans quel sens la difficulte augmente : l'incidence rend
    la detection plus dure quand elle MONTE, la taille apparente quand elle
    DESCEND. On parcourt donc du plus facile vers le plus difficile.

    Deux precautions, apprises a nos depens sur un balayage ou le taux
    sautait de 70 a 100 % sans rapport avec la taille du tag :

    1. On mesure d'abord un TAUX DE REFERENCE sur les paliers les plus
       faciles. S'il n'est pas proche de 100 %, c'est qu'une cause etrangere
       fait rater des images — tag hors cadre, image filee — et le balayage
       ne mesure plus ce qu'on croit. Les taux sont alors rapportes a cette
       reference, et l'appelant est prevenu.
    2. On ne conclut qu'apres PALIERS_CONFIRMATION paliers consecutifs sous
       le seuil. Un creux isole est du bruit, pas une limite : au-dela de la
       vraie limite, la detection ne revient jamais.

    `logarithmique` decoupe les paliers en proportions plutot qu'en ecarts.
    C'est ce qu'il faut pour la taille apparente : entre 20 et 210 px, des
    paliers reguliers en font un seul de 20 a 36 px, justement la ou tout se
    joue. En log, chaque palier vaut 21 % du precedent, et le bas du domaine
    est resolu aussi finement que le haut.
    """
    diagnostic = {"reference": None, "confirmee": False}
    if len(echantillons) < nb_paliers:
        return None, [], diagnostic
    valeurs = np.array([e[cle] for e in echantillons])
    taux = np.array([e["taux"] for e in echantillons])

    if logarithmique and valeurs.min() > 0:
        bords = np.geomspace(valeurs.min(), valeurs.max(), nb_paliers + 1)
        milieu = lambda a, b: float(np.sqrt(a * b))  # noqa: E731
    else:
        bords = np.linspace(valeurs.min(), valeurs.max(), nb_paliers + 1)
        milieu = lambda a, b: float((a + b) / 2)  # noqa: E731
    paliers = []
    for k in range(nb_paliers):
        dans = (valeurs >= bords[k]) & (valeurs <= bords[k + 1])
        if dans.sum() >= 5:
            paliers.append({"centre": milieu(bords[k], bords[k + 1]),
                            "taux": float(taux[dans].mean()),
                            "n": int(dans.sum())})
    if len(paliers) < 4:
        return None, sorted(paliers, key=lambda p: p["centre"]), diagnostic

    # du plus facile vers le plus difficile
    ordonnes = sorted(paliers, key=lambda p: p["centre"], reverse=not croissant)

    reference = float(np.median([p["taux"] for p in ordonnes[:3]]))
    diagnostic["reference"] = reference
    if reference < 0.85:
        return None, sorted(paliers, key=lambda p: p["centre"]), diagnostic

    # rapporte au regime facile : ce qui rate partout n'est pas du a la taille
    for palier in ordonnes:
        palier["taux_relatif"] = min(palier["taux"] / reference, 1.0)

    limite = None
    for k, palier in enumerate(ordonnes):
        if palier["taux_relatif"] >= taux_limite:
            continue
        suite = ordonnes[k:k + PALIERS_CONFIRMATION]
        if (len(suite) == PALIERS_CONFIRMATION
                and all(p["taux_relatif"] < taux_limite for p in suite)):
            limite, diagnostic["confirmee"] = palier["centre"], True
            break
    return limite, sorted(paliers, key=lambda p: p["centre"]), diagnostic


def lire_balayage(lignes, nom):
    """Les fenetres d'un balayage, celles au cadrage douteux mises de cote.

    Une fenetre ou le tag a frole le bord de l'image ne mesure rien
    d'exploitable : les images ratees le sont parce que le tag est sorti du
    champ, pas parce qu'il etait trop petit ou trop de biais.
    """
    gardees, ecartees = [], 0
    for ligne in lignes:
        if ligne["balayage"] != nom:
            continue
        # les anciens enregistrements n'ont pas la colonne : on les garde
        bord = float(ligne.get("bord_px") or MARGE_BORD)
        if bord < MARGE_BORD:
            ecartees += 1
            continue
        gardees.append({"taux": float(ligne["taux"]),
                        "pixels": float(ligne["pixels"]),
                        "incidence_deg": float(ligne["incidence_deg"])})
    return gardees, ecartees


def diagnostiquer(diagnostic, limite, sortie):
    """Dit si le balayage a mesure ce qu'on croit. Vrai s'il est exploitable."""
    reference = diagnostic["reference"]
    if reference is None:
        sortie.append("\n  Trop peu de paliers pour conclure. Balaye plus large.")
        return False
    if reference < 0.98:
        sortie.append(f"\n  BALAYAGE CONTAMINE — meme dans le regime le plus facile,")
        sortie.append(f"  {100*(1-reference):.0f} % des images ratent la detection. "
                      "Ce n'est donc pas")
        sortie.append("  la difficulte balayee qui les fait echouer, mais autre chose :")
        sortie.append("   - le tag sort du champ (garde-le bien au centre) ;")
        sortie.append("   - l'image est filee (avance par PALIERS : immobile 2 a 3 s,")
        sortie.append("     puis un pas, puis immobile a nouveau — ne marche pas en continu) ;")
        sortie.append("   - le tag gondole ou reflechit la lumiere.")
        if reference < 0.85:
            sortie.append("\n  Trop contamine pour en tirer quoi que ce soit. A refaire.")
            return False
        sortie.append(f"\n  Les taux ci-dessous sont rapportes a ce regime facile "
                      f"({100*reference:.0f} %),")
        sortie.append("  mais le resultat reste a confirmer par un balayage propre.")
    if limite is None:
        return False
    return True


def besoin_du_bassin(atteint, recul):
    """Ce que le balayage doit encore couvrir — et ce qu'il couvre deja.

    Deux lectures d'un meme balayage. La limite ABSOLUE de detection demande
    de descendre vers CIBLE_PIXELS, ce qui exige beaucoup de recul. Mais le
    plan de pose n'en a pas besoin : il lui suffit que la detection soit
    verifiee en dessous de ce que le bassin peut produire de plus petit.
    """
    pire, diagonale = pixels_pire_cas()
    lignes = ["", "  CE QUE LE BASSIN DEMANDE VRAIMENT"]
    lignes.append(f"  Sa diagonale fait {diagonale:.2f} m. A cette distance — le pire cas —")
    lignes.append(f"  un tag de {100*TAILLE_TAG_REELLE:.1f} cm paraitra {pire:.0f} px "
                  f"sous l'eau, dans l'axe")
    lignes.append(f"  le moins grossi par le tube (focale {optique.focale_eau(MONTAGE):.0f} px "
                  f"contre {max(optique.focales_eau(MONTAGE)):.0f} dans l'autre).")
    lignes.append("  C'est le plus petit que le bassin produise.")

    if atteint <= pire:
        lignes.append(f"\n  Tu es descendu a {atteint:.0f} px sans perdre le tag, "
                      f"donc en dessous des {pire:.0f} px")
        lignes.append("  du pire cas : la taille apparente ne sera JAMAIS le facteur")
        lignes.append("  limitant dans ce bassin. C'est la conclusion utile, et elle")
        lignes.append("  est acquise — la limite absolue n'a plus d'interet pratique.")
    else:
        lignes.append(f"\n  Ton balayage s'est arrete a {atteint:.0f} px, au-dessus de ces "
                      f"{pire:.0f} px.")
        lignes.append(f"  Il reste la tranche {pire:.0f}-{atteint:.0f} px a couvrir "
                      "pour conclure. Avec")
        lignes.append(f"  ce tag de {100*TAILLE_TAG:.1f} cm il faudrait reculer jusqu'a "
                      f"{K_CALIB[0, 0] * TAILLE_TAG / pire:.1f} m ;")
        besoin = pire * recul / K_CALIB[0, 0]
        lignes.append(f"  en restant a {recul:.1f} m, il faut un tag de "
                      f"{100*besoin:.0f} cm  (--tag {besoin:.3f}).")

    lignes.append(f"\n  Pour la limite ABSOLUE de detection il faudrait descendre vers")
    lignes.append(f"  {CIBLE_PIXELS:.0f} px — un 36h11 fait huit cellules de large et il "
                  "en faut deux")
    lignes.append("  pixels chacune pour decoder. Utile pour le rapport, pas pour poser")
    lignes.append(f"  les tags. Il faudrait un tag de "
                  f"{100 * CIBLE_PIXELS * recul / K_CALIB[0, 0]:.0f} cm a {recul:.1f} m.")
    return lignes


def rapport(lignes):
    if not lignes:
        return "Aucun balayage. 'd' pour la distance, 'i' pour l'incidence."
    sortie = ["", "=" * 78, "LIMITES DE DETECTION MESUREES", "=" * 78]

    distance, hors_cadre_d = lire_balayage(lignes, "distance")
    incidence, hors_cadre_i = lire_balayage(lignes, "incidence")

    # --- taille apparente minimale ----------------------------------------
    sortie.append(f"\nBALAYAGE EN DISTANCE — {len(distance)} points"
                  + (f", {hors_cadre_d} ecartes (tag au bord de l'image)"
                     if hors_cadre_d else ""))
    if len(distance) < 12:
        sortie.append("  Trop peu de points. Refais un aller-retour complet ('d').")
    else:
        limite, paliers, diagnostic = limite_par_paliers(distance, "pixels",
                                                         croissant=False,
                                                         logarithmique=True)
        sortie.append(f"  {'taille apparente':>18} {'taux de detection':>18}")
        for p in reversed(paliers):
            barre = "#" * int(round(20 * p["taux"]))
            sortie.append(f"  {p['centre']:>15.0f} px {100*p['taux']:>15.0f} %  {barre}")

        if diagnostiquer(diagnostic, limite, sortie):
            sortie.append(f"\n  PIXELS_MIN mesure = {limite:.0f} px "
                          f"(la valeur supposee etait 30 px)")
            if abs(TAILLE_TAG - TAILLE_TAG_REELLE) > 1e-6:
                sortie.append(f"  (mesure avec un tag d'essai de "
                              f"{100*TAILLE_TAG:.1f} cm ; la limite est en pixels,")
                sortie.append(f"   elle vaut donc aussi pour les "
                              f"{100*TAILLE_TAG_REELLE:.1f} cm du bassin)")
            portee = min(K_CALIB[0, 0], K_CALIB[1, 1]) * TAILLE_TAG_REELLE / limite
            sortie.append(f"  Pour un tag de {100*TAILLE_TAG_REELLE:.1f} cm, cela donne")
            sortie.append(f"  une portee de {portee:.2f} m en air, "
                          f"{optique.portee_eau(portee, MONTAGE):.2f} m sous l'eau")
            sortie.append("  (pas de « x 1.33 » ici : la camera est couchee dans le "
                          "tube, et")
            sortie.append("   c'est l'axe le MOINS grossi qui decide de la detection)")
        elif limite is None and paliers:
            atteint = min(p["centre"] for p in paliers)
            recul = K_CALIB[0, 0] * TAILLE_TAG / atteint
            sortie.append(f"\n  Aucune limite confirmee : a {atteint:.0f} px, le plus "
                          "petit atteint, le tag")
            sortie.append("  est encore detecte. La limite est en dessous.")
            sortie.extend(besoin_du_bassin(atteint, recul))

    # --- incidence maximale ------------------------------------------------
    sortie.append(f"\nBALAYAGE EN INCIDENCE — {len(incidence)} points"
                  + (f", {hors_cadre_i} ecartes (tag au bord de l'image)"
                     if hors_cadre_i else ""))
    if len(incidence) < 12:
        sortie.append("  Trop peu de points. Refais un balayage complet ('i').")
    else:
        limite, paliers, diagnostic = limite_par_paliers(incidence, "incidence_deg",
                                                         croissant=True)
        sortie.append(f"  {'incidence':>18} {'taux de detection':>18}")
        for p in paliers:
            barre = "#" * int(round(20 * p["taux"]))
            sortie.append(f"  {p['centre']:>14.0f} deg {100*p['taux']:>15.0f} %  {barre}")
        taille_mediane = float(np.median([e["pixels"] for e in incidence]))
        if taille_mediane < 60:
            sortie.append(f"\n  ATTENTION : le tag ne faisait que {taille_mediane:.0f} px "
                          "pendant ce balayage.")
            sortie.append("  A cette taille c'est peut-etre la resolution qui a lache,")
            sortie.append("  pas l'angle. Refais-le avec le grand tag, plus pres.")

        if diagnostiquer(diagnostic, limite, sortie):
            sortie.append(f"\n  INCIDENCE_MAX mesuree = {limite:.0f} deg "
                          f"(la valeur supposee etait 65 deg)")
        elif limite is None and paliers:
            atteint = max(p["centre"] for p in paliers)
            sortie.append(f"\n  Aucune limite confirmee : a {atteint:.0f} deg, le plus "
                          "oblique atteint,")
            sortie.append("  le tag est encore detecte. Tourne-le davantage.")

    sortie.append("\n" + "=" * 78)
    sortie.append("Reporte ces deux valeurs dans plan_piscine_3d.py.")
    sortie.append("=" * 78)
    return "\n".join(sortie)


def equivalent_reel(pixels):
    """A quelle distance le VRAI tag du bassin ferait-il cette taille ?

    Le detecteur ne voit que des pixels : un petit tag pres et un grand tag
    loin lui sont indiscernables. C'est ce qui autorise a mesurer la limite
    dans un couloir de 2 m et a la transposer au bassin.
    """
    return K_CALIB[0, 0] * TAILLE_TAG_REELLE / max(pixels, 1e-6)


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


def guide_de_portee(taille):
    """Rappelle, avant de commencer, quelle plage de pixels est atteignable."""
    f = K_CALIB[0, 0]
    lignes = ["", f"TAG D'ESSAI : {100*taille:.1f} cm",
              "  distance      taille apparente"]
    for d in (0.5, 1.0, 1.5, 2.0, 3.0):
        px = f * taille / d
        marque = "  <-- sous la limite supposee (30 px)" if px < 30 else ""
        lignes.append(f"  {d:>5.1f} m {px:>13.0f} px{marque}")
    d30 = f * taille / 30
    lignes.append(f"\n  30 px sont atteints a {d30:.2f} m, "
                  f"20 px a {f * taille / 20:.2f} m.")
    if d30 > 2.5:
        lignes.append("  C'est loin. Si tu ne peux pas reculer autant, imprime un tag")
        lignes.append(f"  plus petit : --tag 0.05 met la limite a "
                      f"{f * 0.05 / 30:.2f} m.")
    return "\n".join(lignes)


def main():
    global TAILLE_TAG
    analyseur = argparse.ArgumentParser(
        description="Mesure PIXELS_MIN et INCIDENCE_MAX sur la vraie camera.")
    analyseur.add_argument("--tag", type=float, default=TAILLE_TAG_REELLE,
                           metavar="METRES",
                           help="cote du tag d'essai en metres (defaut %(default)s). "
                                "Un petit tag rapproche la limite de detection : "
                                "0.05 la place vers 1 m au lieu de 4.5 m.")
    TAILLE_TAG = analyseur.parse_args().tag

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
    print(guide_de_portee(TAILLE_TAG))
    print("\n  'd' balayage en distance  : eloigne le tag jusqu'a le perdre")
    print("  'i' balayage en incidence : tourne le tag jusqu'a le perdre")
    print("  'r' rapport | 'e' effacer | 'q' quitter")
    print("\n  AVANCE PAR PALIERS : immobile 3 s, un pas, immobile 3 s...")
    print("  Marcher en continu file les images et fait rater la detection")
    print("  pour une raison qui n'a rien a voir avec ce qu'on mesure.")
    print("  Garde le tag BIEN AU CENTRE : s'il frole le bord, la fenetre")
    print("  est ecartee du depouillement.")
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
                      "distance": float(np.linalg.norm(tvec)),
                      # combien de pixels separent le tag du bord de l'image :
                      # s'il le frole, les images ratees sont des sorties de
                      # champ et ne disent rien de la limite cherchee
                      "bord": float(min(pts[:, 0].min(), pts[:, 1].min(),
                                        L - pts[:, 0].max(), H - pts[:, 1].max()))}

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
                    "bord_px": f"{min(f['bord'] for f in reussies):.1f}",
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
                cv2.putText(image, f"= tag {100*TAILLE_TAG_REELLE:.0f} cm vu de "
                                   f"{equivalent_reel(m['pixels']):.2f} m", (10, 136),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)
            consigne = ("PAR PALIERS : immobile 3 s, un pas en arriere, immobile"
                        if balayage == "distance"
                        else "PAR PALIERS : immobile 3 s, tourne un peu, immobile")
            cv2.putText(image, consigne, (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if reussies and reussies[-1]["bord"] < MARGE_BORD:
                cv2.rectangle(image, (2, 2), (L - 3, H - 3), (0, 0, 255), 3)
                cv2.putText(image, "TAG AU BORD — recentre-le, sinon la mesure "
                                   "est perdue", (10, 162),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif vu is not None:
            cv2.putText(image, f"{vu['pixels']:.0f} px   {vu['incidence']:.0f} deg"
                               f"   {vu['distance']:.2f} m", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(image, f"= tag {100*TAILLE_TAG_REELLE:.0f} cm vu de "
                               f"{equivalent_reel(vu['pixels']):.2f} m", (10, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)
            cv2.putText(image, "'d' balayage distance   |   'i' balayage incidence",
                        (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
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
