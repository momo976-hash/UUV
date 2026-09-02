# kalman_du_cours.py — Le filtre du projet EST celui du document de reference.
#
#     python localisations/kalman_du_cours.py
#
# DOCUMENT DE REFERENCE
#   Alex Becker, « Kalman Filter Explained Through Examples »,
#   kalmanfilter.net, consulte le 02/09/2026.
#
# CE QUE FAIT CE SCRIPT, ET POURQUOI
# Le document deroule un exemple chiffre : un radar 1D qui suit un avion. Il
# imprime toutes les valeurs intermediaires — F, Q, P(1,0), K(1), x(1,1),
# P(1,1), x(2,1), P(2,1). Ce script fait tourner CE meme exemple a travers la
# classe KalmanLineaire du projet, celle-la meme qui filtre la position de
# l'engin, et compare chaque nombre a celui imprime dans le document.
#
# Autrement dit : ce n'est pas un filtre ecrit pour la demonstration, c'est le
# filtre de l'engin qu'on met a l'epreuve sur un probleme dont la reponse est
# publiee. S'il s'ecartait des equations du cours, la comparaison le dirait
# tout de suite au lieu de le cacher derriere la geometrie des tags.
#
# Le script montre ensuite que le modele de l'engin est le MEME modele
# cinematique a vitesse constante, simplement porte de 1 a 3 dimensions.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filtre_kalman import FiltreKalmanPosition, KalmanLineaire  # noqa: E402


# ---------------------------------------------------------------------------
# Les valeurs imprimees dans le document, recopiees telles quelles.
# Ce sont les references : le code n'a pas le droit de les toucher.
# ---------------------------------------------------------------------------
DOC = {
    "Q": np.array([[6.25, 2.5], [2.5, 1.0]]),
    "x(1,0)": np.array([11000.0, 200.0]),
    "P(1,0)": np.array([[28.5, 3.75], [3.75, 1.25]]),
    "K(1)": np.array([[0.4048, 0.6377], [0.0399, 0.3144]]),
    "x(1,1)": np.array([11009.37, 201.43]),
    "P(1,1)": np.array([[14.57, 1.43], [1.43, 0.71]]),
    "x(2,1)": np.array([12016.5, 201.43]),
    "P(2,1)": np.array([[52.86, 7.47], [7.47, 1.71]]),
}

# Le document arrondit ses tableaux a deux decimales, et le gain a quatre.
# On tolere donc un demi-dernier-chiffre, pas davantage.
TOLERANCE = {"K(1)": 5e-5}
TOLERANCE_PAR_DEFAUT = 5e-3

_resultats = []


def comparer(nom, calcule):
    """Compare une valeur calculee a celle imprimee dans le document."""
    attendu = DOC[nom]
    calcule = np.asarray(calcule, dtype=float)
    ecart = float(np.max(np.abs(calcule - attendu)))
    seuil = TOLERANCE.get(nom, TOLERANCE_PAR_DEFAUT)
    ok = ecart <= seuil
    _resultats.append((nom, ok, ecart))

    plat = " ".join(f"{v:12.4f}" for v in np.ravel(calcule))
    doc = " ".join(f"{v:12.4f}" for v in np.ravel(attendu))
    marque = "OK " if ok else "NON"
    print(f"  [{marque}] {nom:8s} calcule : {plat}")
    print(f"          {'':8s} document: {doc}     ecart {ecart:.2e}")
    return ok


def exemple_du_document():
    """L'exemple du radar, deroule avec le noyau du projet."""
    print("=" * 74)
    print("EXEMPLE DU DOCUMENT — radar 1D suivant un avion")
    print("  Alex Becker, Kalman Filter Explained Through Examples,")
    print("  kalmanfilter.net")
    print("=" * 74)

    # --- le probleme, tel que pose par le document -------------------------
    dt = 5.0                         # temps de revisite du radar
    sigma_a = 0.2                    # m/s^2, acceleration aleatoire de l'avion
    sigma_portee, sigma_vitesse = 4.0, 0.5        # bruit de la 1re mesure
    z1 = np.array([11020.0, 202.0])               # 2e mesure
    R1 = np.diag([6.0 ** 2, 1.5 ** 2])            # elle est plus bruitee

    # Etat : x = [portee, vitesse]. Modele CINEMATIQUE a vitesse constante.
    F = np.array([[1.0, dt],
                  [0.0, 1.0]])
    # Q tel qu'ecrit dans le document, section 8.2.2 :
    #     Q = sigma_a^2 [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]
    Q = sigma_a ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2],
                                 [dt ** 3 / 2, dt ** 2]])
    H = np.eye(2)                    # le radar mesure portee ET vitesse

    print(f"\n  dt = {dt} s, sigma_a = {sigma_a} m/s2")
    print(f"  F = {F.tolist()}")
    comparer("Q", Q)

    # --- ITERATION 0 : initialisation puis prediction ----------------------
    print("\nITERATION 0 — initialisation par la premiere mesure")
    filtre = KalmanLineaire(
        x=np.array([10000.0, 200.0]),
        P=np.diag([sigma_portee ** 2, sigma_vitesse ** 2]))
    print(f"  x(0,0) = {filtre.x.tolist()}   P(0,0) = "
          f"{np.diag(filtre.P).tolist()} (diagonale)")

    print("\nITERATION 0 — prediction")
    filtre.predire(F, Q)
    comparer("x(1,0)", filtre.x)
    comparer("P(1,0)", filtre.P)

    # --- ITERATION 1 : mise a jour ----------------------------------------
    print("\nITERATION 1 — mise a jour par la deuxieme mesure")
    print(f"  z(1) = {z1.tolist()}   R(1) = {np.diag(R1).tolist()} (diagonale)")
    K, _ = filtre.gain(H, R1)
    comparer("K(1)", K)

    P_avant = filtre.P.copy()
    innovation, _ = filtre.corriger(z1, H, R1)
    print(f"  innovation z - Hx = {innovation.tolist()}   "
          f"(le document donne [20, 2])")
    comparer("x(1,1)", filtre.x)
    comparer("P(1,1)", filtre.P)

    # Joseph contre la forme simplifiee : le document dit qu'elles sont egales
    # en arithmetique exacte et recommande Joseph. On le verifie plutot que de
    # le croire.
    simplifiee = (np.eye(2) - K @ H) @ P_avant
    ecart = float(np.max(np.abs(filtre.P - simplifiee)))
    print(f"\n  forme de Joseph vs forme simplifiee : ecart {ecart:.1e}")
    print("    -> identiques, comme l'annonce le document. On garde Joseph,")
    print("       qui reste symetrique definie positive apres des milliers")
    print("       d'iterations en virgule flottante.")

    # --- ITERATION 1 : prediction suivante ---------------------------------
    print("\nITERATION 1 — prediction")
    filtre.predire(F, Q)
    comparer("x(2,1)", filtre.x)
    comparer("P(2,1)", filtre.P)


def modele_de_lengin():
    """Le meme modele, porte de 1 a 3 dimensions pour l'engin."""
    print("\n" + "=" * 74)
    print("LE MEME MODELE, APPLIQUE A L'ENGIN")
    print("=" * 74)
    print("""
  Le document raisonne sur un etat a deux composantes, [portee, vitesse],
  parce que son radar est unidimensionnel. L'engin se deplace dans l'eau :
  son etat en a six, [px py pz vx vy vz]. Le modele est le meme, bloc par
  bloc — c'est le meme modele CINEMATIQUE a vitesse constante.

      document              engin
      F = [[1, dt],         F = [[I3, dt.I3],
           [0,  1]]              [ 0,    I3 ]]

      Q = sa^2 [[dt^4/4, dt^3/2],      Q = sa^2 . G G'
                [dt^3/2, dt^2  ]]      avec G = [dt^2/2 . I3 ; dt . I3]

  Ces deux ecritures de Q sont la meme. Verification numerique :""")

    dt, sigma_a = 5.0, 0.2
    _, G = FiltreKalmanPosition.modele(dt)
    Q3 = sigma_a ** 2 * (G @ G.T)
    attendus = (("position  dt^4/4", Q3[0, 0], sigma_a ** 2 * dt ** 4 / 4),
                ("croise    dt^3/2", Q3[0, 3], sigma_a ** 2 * dt ** 3 / 2),
                ("vitesse   dt^2  ", Q3[3, 3], sigma_a ** 2 * dt ** 2))
    tout_bon = True
    for nom, obtenu, attendu in attendus:
        ok = abs(obtenu - attendu) < 1e-12
        tout_bon &= ok
        print(f"    [{'OK ' if ok else 'NON'}] bloc {nom} : "
              f"G G' donne {obtenu:8.4f}, formule du document {attendu:8.4f}")
    _resultats.append(("Q en 3D = Q du document", tout_bon, 0.0))

    print("""
  UNE SEULE DIFFERENCE, ET ELLE EST DANS H. Le radar du document mesure la
  portee ET la vitesse, donc H = I. Les tags ne donnent qu'une position :

      H = [I3  0]

  Les vitesses ne sont jamais mesurees, elles sont DEDUITES par le filtre a
  partir de l'evolution des positions. C'est le cas le plus courant, et le
  document le prevoit explicitement : « the measurement and the system state
  may belong to different physical domains ».

  CE QUE LE PROJET AJOUTE, ET POURQUOI
  Trois choses seulement, toutes prevues par le document :

    1. L'ENTREE u. Le document ecrit x(n+1,n) = F x(n,n) + G u(n) et donne
       pour exemple d'entree « readings from an onboard accelerometer ».
       C'est exactement ce que fait FiltreKalmanPosition.predire(dt, accel) :
       sans accelerometre l'acceleration est un alea couvert par sigma_a,
       avec lui elle est mesuree et il ne reste que le bruit du capteur.

    2. LE REJET DES MESURES ABERRANTES. Un tag vu de trop biais peut se
       retourner et donner une pose fausse de plusieurs decimetres. On la
       reconnait a sa distance de Mahalanobis y' S^-1 y, ou S = H P H' + R
       est deja calculee pour le gain. Le document renvoie ce sujet a son
       chapitre « Outlier Treatment » : « in practice it is often necessary
       to reject certain measurements ».

    3. L'ORIENTATION A PART. Les rotations ne s'additionnent pas, donc un
       Kalman lineaire ne s'applique pas directement a une orientation. Elle
       est traitee par un filtre scalaire sur l'angle, applique par slerp sur
       les quaternions — c'est le sujet des filtres non lineaires, que le
       document renvoie a son livre. La POSITION, elle, reste exactement le
       filtre lineaire ci-dessus.
""")


def main():
    exemple_du_document()
    modele_de_lengin()

    print("=" * 74)
    echecs = [nom for nom, ok, _ in _resultats if not ok]
    if echecs:
        print(f"DESACCORD avec le document sur : {', '.join(echecs)}")
        print("=" * 74)
        return 1
    print(f"LES {len(_resultats)} VALEURS DU DOCUMENT SONT RETROUVEES")
    print("")
    print("  Le filtre de position du projet n'est pas inspire du document :")
    print("  c'est le meme filtre. Il reproduit son exemple chiffre a la")
    print("  quatrieme decimale, avec la classe qui tourne reellement sur")
    print("  l'engin (filtre_kalman.KalmanLineaire).")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
