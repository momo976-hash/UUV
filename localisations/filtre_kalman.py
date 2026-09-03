# filtre_kalman.py — Filtrage de la pose de la camera estimee par les tags.
#
# Ce module ne depend QUE de numpy : ni OpenCV, ni ROS. Il peut donc etre
# appele aussi bien depuis verification_monde.py que depuis un noeud ROS2 qui
# ecoute le topic /tf publie par Josiah.
#
# ---------------------------------------------------------------------------
# CE QUE FAIT UN FILTRE DE KALMAN, EN UNE PHRASE
# ---------------------------------------------------------------------------
# Il combine deux sources d'information qui disent toutes les deux ou se
# trouve l'engin, et dont aucune n'est exacte :
#   - la PREDICTION : ou l'engin devrait etre, sachant ou il etait et a
#     quelle vitesse il allait. Fiable a court terme, derive a long terme.
#   - la MESURE : ou les tags disent qu'il est. Sans derive, mais bruitee.
# Le filtre les moyenne en donnant plus de poids a celle dont l'incertitude
# est la plus faible. Ce poids, c'est le gain K.
#
# ---------------------------------------------------------------------------
# 1. LE MODELE D'ETAT (vitesse constante)
# ---------------------------------------------------------------------------
# Etat :  x = [px, py, pz, vx, vy, vz]^T        (6 composantes)
#
# Entre deux images separees de dt, on suppose la vitesse constante :
#       p(t+dt) = p(t) + v(t)·dt
#       v(t+dt) = v(t)
# soit, sous forme matricielle,  x(t+dt) = F·x(t)  avec
#
#       F = [ I3   dt·I3 ]
#           [ 0     I3   ]
#
# Cette hypothese est fausse : un UUV accelere. On l'assume en injectant du
# bruit de modele. On suppose qu'une acceleration aleatoire a, d'ecart-type
# sigma_a, agit pendant dt. Elle deplace l'engin de :
#       delta_p = 1/2·a·dt^2        delta_v = a·dt
# soit  delta_x = G·a  avec  G = [1/2·dt^2·I3 ; dt·I3]   (6x3)
#
# La covariance de ce bruit est donc, directement :
#       Q = sigma_a^2 · G·G^T
#
# C'est tout : Q n'est pas une constante a regler au hasard, elle se derive
# de la seule question "de combien l'engin peut-il accelerer sans que je le
# sache ?". sigma_a se lit sur les capacites du propulseur.
#
# ---------------------------------------------------------------------------
# 2. LA MESURE
# ---------------------------------------------------------------------------
# Les tags donnent une position, pas une vitesse :
#       z = H·x + bruit        avec   H = [ I3  0 ]
#
# ---------------------------------------------------------------------------
# 3. LE COEUR : D'OU VIENT R, LA CONFIANCE DANS LA MESURE
# ---------------------------------------------------------------------------
# C'est la partie qui merite d'etre derivee plutot que devinee, parce que
# l'erreur d'un tag N'EST PAS ISOTROPE : un tag dit tres bien ou il est
# lateralement, et tres mal a quelle distance il est.
#
#   LATERAL. Un point a la distance d se projette en u = f·X/d, donc
#            X = u·d/f. Une erreur de sigma_px pixels sur le coin donne
#                  sigma_lat = d·sigma_px / f                    (~ en d)
#
#   PROFONDEUR. La distance se deduit de la TAILLE APPARENTE du tag,
#            s = f·T/d, donc d = f·T/s. En derivant, |dd/ds| = d^2/(f·T) :
#                  sigma_prof = d^2·sigma_px / (f·T_apparent)    (~ en d^2)
#            avec T_apparent = T·cos(incidence) : vu de biais, le tag est
#            plus etroit, donc sa taille apparente est moins informative.
#
# Ordre de grandeur avec TES chiffres (f_eau = 803.6 px, T = 0.223 m,
# sigma_px = 0.5 px, d = 1.6 m) :
#       sigma_lat  = 1.0 mm          sigma_prof = 7.1 mm
# Le 1.0 mm lateral est exactement l'erreur que tu as mesuree a la main sur
# verification_monde.py. Le modele est donc cale sur le reel.
#
# On construit R dans le repere monde en placant la grande incertitude le
# long de l'axe de visee u (vecteur unitaire camera -> tag) :
#       R = sigma_lat^2·(I - u·u^T)  +  sigma_prof^2·u·u^T
#
# ---------------------------------------------------------------------------
# 4. LA REPONSE A JOSIAH : POURQUOI PLUSIEURS TAGS VALENT MIEUX QU'UN
# ---------------------------------------------------------------------------
# L'inverse d'une covariance s'appelle une matrice d'information. Pour des
# mesures independantes, LES INFORMATIONS S'ADDITIONNENT :
#       R_total^-1 = somme( R_i^-1 )
#
# Deux consequences concretes :
#   - deux tags vus ensemble reduisent l'incertitude, meme s'ils sont tous
#     les deux mediocres ;
#   - surtout, deux tags places sur des MURS DIFFERENTS ont des axes de
#     visee u differents. Le premier est mauvais en profondeur la ou le
#     second est bon lateralement. Leurs faiblesses ne se superposent pas,
#     et l'ellipsoide d'incertitude s'effondre dans toutes les directions.
# C'est la justification chiffree de la remarque de Josiah.
#
# ---------------------------------------------------------------------------
# 5. LE REJET DES ABERRATIONS
# ---------------------------------------------------------------------------
# L'ambiguite de retournement d'un tag plan (flip) produit de temps en temps
# une pose completement fausse. On la detecte avec la distance de
# Mahalanobis de l'innovation y = z - H·x :
#       d2 = y^T·S^-1·y      avec   S = H·P·H^T + R
# d2 suit une loi du chi2 a 3 degres de liberte. Au-dela de ~16, il y a
# moins de 0.1 % de chances que la mesure soit legitime : on la jette.
# Sans ce test, une seule aberration decale le filtre pendant des secondes.
#
# ---------------------------------------------------------------------------
# 6. LE PIEGE DU TEST DE MAHALANOBIS : LE VERROUILLAGE
# ---------------------------------------------------------------------------
# Ce test est indispensable, mais il se retourne contre le filtre. Si une
# aberration passe AVANT que P se soit resserree, l'etat part sur une
# position fausse. P continue ensuite de retrecir sur les predictions, si
# bien que les mesures CORRECTES deviennent a leur tour incompatibles avec
# un etat faux mais tres "sur" de lui. Le filtre les rejette toutes et ne
# revient jamais.
#
# Mesure sur la simulation de demo_kalman.py, sans garde-fou : 1198 rejets
# sur 1200 images, et une erreur finale de 38 metres dans un bassin de 3.80.
#
# Le garde-fou : compter les rejets CONSECUTIFS. Au-dela de quelques-uns,
# la conclusion qui s'impose n'est pas "toutes les mesures sont fausses"
# mais "mon etat est faux". Le filtre se recale alors sur la mesure et
# reouvre son incertitude. C'est ce que fait `max_rejets_consecutifs`.
#
# ---------------------------------------------------------------------------
# 7. LA LIMITE QUE LE FILTRE NE PEUT PAS FRANCHIR : UN TAG QUI BOUGE
# ---------------------------------------------------------------------------
# Tout ce qui precede suppose les tags a des positions FIXES et CONNUES.
# Montes sur des boites en acrylique lestees posees au fond, et non scelles
# dans du beton, ils ne le sont qu'a peu pres : le souffle des propulseurs,
# un cable qui accroche, et une boite se decale.
#
# Or si un tag se deplace de delta, la position de camera qu'on en deduit
# se decale de delta AUSSI, exactement, et dans la meme direction. C'est un
# BIAIS, pas un bruit. Un filtre de Kalman ne sait traiter que du bruit
# centre : il moyenne le bruit, mais il SUIT le biais.
#
# L'ordre de grandeur est brutal. L'erreur mediane apres filtrage vaut
# 2.1 mm ; une boite decalee de 1 cm produit a elle seule cinq fois tout le
# reste du budget d'erreur. Autrement dit, des lors que les tags sont sur
# des supports libres, LA PRECISION DU SYSTEME N'EST PLUS LIMITEE PAR
# L'OPTIQUE NI PAR LE FILTRE, MAIS PAR LA STABILITE MECANIQUE DES SUPPORTS.
#
# La parade est SurveillanceTags, plus bas. Un tag bien enregistre produit
# une innovation centree sur zero. Un tag qui a bouge produit une
# innovation dont la MOYENNE derive vers une constante, qui est justement
# son deplacement. On surveille donc la moyenne glissante, tag par tag.
#
# Limite d'observabilite a annoncer honnetement : si un seul tag est
# visible, rien ne distingue "la camera a bouge" de "le tag a bouge". La
# detection exige que le tag suspect soit vu, au moins par moments, en meme
# temps que d'autres.
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
import optique  # noqa: E402

# L'optique vient de optique.py : camera, tube, paroi, milieu.
#
# TROIS RESERVES sur ce nombre unique, toutes trois dues au tube.
#
# 1. Le systeme est ANAMORPHIQUE sous l'eau. La camera est couchee dans le
#    tube : l'axe horizontal de l'image suit l'axe du tube et traverse une
#    lame a faces paralleles (focale x 1.33) ; l'axe vertical est
#    circonferentiel et traverse un menisque, qui RETRECIT la focale quand la
#    pupille est en retrait de l'axe. Les deux focales finissent a 1.4 l'une
#    de l'autre. On retient ici la plus PETITE, donc la plus defavorable : la
#    covariance annoncee est majorante dans un sens et juste dans l'autre,
#    jamais optimiste.
#
# 2. Meme dans la direction « lame plane », le facteur 1.33 n'est exact qu'au
#    voisinage de l'axe optique — plus de 16 px d'ecart a 20 deg, quand le
#    bruit de detection vaut 0.215 px. Cette erreur-la est SYSTEMATIQUE : le
#    filtre la suit au lieu de la moyenner. Seule une calibration faite SOUS
#    L'EAU la corrige.
#
# 3. La focale verticale depend de la position de la camera DANS son support :
#    un millimetre de glissement apres calibration, et c'est 1 % sur toutes
#    les distances — 30 mm a 3 m, soit vingt fois le bruit de mesure. Aucun
#    filtre ne rattrape cela ; seule la mecanique le peut.
#
# `python optique.py` chiffre les trois.
FOCALE_EAU = optique.focale_eau()
TAILLE_TAG = optique.TAILLE_TAG_GRAND   # mesure au pied a coulisse, pas 223 mm nominal

# ===========================================================================
# LES NOMBRES A MESURER
#
# C'est LE SEUL bloc a modifier apres une session de bassin. Tout le reste du
# depot vient y puiser : les classes ci-dessous en font leurs valeurs par
# defaut, et verification_monde.py construit son filtre sans rien preciser.
# Auparavant ces chiffres etaient recopies a cinq endroits dans deux fichiers,
# et en corriger quatre sur cinq ne produisait aucun message d'erreur.
#
# Le protocole (documents/protocole_kalman.md) dit comment mesurer chacun.
#
# Les trois premiers gouvernent le filtre alimente par les seuls tags. Les
# deux derniers ne servent que si la centrale inertielle de la D435i est
# branchee — et dans ce cas ils remplacent avantageusement SIGMA_ACCELERATION
# et DERIVE_GYRO_DEG_S, qui decrivent une ignorance plutot qu'une mesure.
# ===========================================================================

# Bruit de detection d'un coin de tag, en pixels.
# MESURE, plus suppose : 14 captures camera en main a 0.72 - 1.52 m
# (calibration/mesurer_bruit_tag.py, mode 'd'). La valeur precedente, 0.5 px,
# etait une valeur d'usage courante en vision, jamais verifiee ici.
#   camera posee    0.049 px  <- plancher, sans flou de bouge
#   camera qui bouge 0.215 px <- valeur d'usage, retenue ici
# Ces mesures sont faites EN AIR. A refaire dans le bassin : l'eau trouble et
# le moindre contraste degraderont ce chiffre.
SIGMA_PIXEL = 0.215

# De combien l'engin peut accelerer sans que le filtre le sache, en m/s2.
# Trop petit : le filtre retarde dans les virages. Trop grand : il ne lisse
# plus rien. SUPPOSE — a remplacer par le 95e centile qu'affiche
# verification_monde.py en fin de session.
SIGMA_ACCELERATION = 0.4

# A quelle vitesse l'orientation peut changer entre deux images sans mesure,
# en deg/s. SUPPOSE — meme source que ci-dessus.
#
# Ne sert QUE si aucun gyroscope n'alimente le filtre. Des que la centrale de
# la D435i est branchee, on sait de combien l'engin a tourne et c'est
# BRUIT_GYRO_DEG_S qui gouverne, deux ordres de grandeur plus bas.
DERIVE_GYRO_DEG_S = 10.0

# Bruit du gyroscope de la D435i, en deg/s (marche aleatoire angulaire).
# MESURE, plus suppose : centrale immobile, 5 s a 400 Hz (3963 echantillons),
# ecart-type des vitesses angulaires — imu_realsense.py du 02/09. La valeur
# precedente, 0.15, etait un ordre de grandeur pour un MEMS de cette classe.
BRUIT_GYRO_DEG_S = 0.106

# Bruit de l'accelerometre, en m/s2. Sert quand il alimente la prediction de
# position a la place de l'hypothese "vitesse constante". MESURE dans la meme
# session que ci-dessus (precedemment 0.05, suppose).
BRUIT_ACCEL = 0.015

# L'echelle du tag, d'ou se deduit la distance, est lue sur QUATRE coins et
# non un seul : moyenner divise le bruit par racine de 4. Sans ce facteur, le
# modele surestimait l'erreur de profondeur d'un facteur 2.3 face aux mesures
# reelles ; avec lui l'ecart tombe a 0.85x, soit 15 %.
COINS_PAR_TAG = 4.0


# ===========================================================================
# Quaternions (l'orientation ne vit pas dans un espace vectoriel : on ne
# peut pas faire la moyenne de deux matrices de rotation)
# ===========================================================================
def matrice_vers_quaternion(R):
    """Matrice de rotation 3x3 -> quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = np.array([0.25 / s, (R[2, 1] - R[1, 2]) * s,
                      (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def quaternion_vers_matrice(q):
    """Quaternion [w, x, y, z] -> matrice de rotation 3x3."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def slerp(q0, q1, t):
    """Interpolation sur la sphere des quaternions : la 'moyenne ponderee'
    correcte entre deux orientations. t=0 rend q0, t=1 rend q1."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    produit = float(q0 @ q1)
    if produit < 0.0:          # q et -q sont la meme rotation : on recolle
        q1, produit = -q1, -produit
    if produit > 0.9995:       # quasi confondus : interpolation lineaire
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(np.clip(produit, -1.0, 1.0))
    sinus = np.sin(theta)
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / sinus


def angle_quaternions(q0, q1):
    """Angle en degres entre deux orientations."""
    produit = abs(float(q0 @ q1) / (np.linalg.norm(q0) * np.linalg.norm(q1)))
    return float(np.degrees(2.0 * np.arccos(np.clip(produit, -1.0, 1.0))))


def produit_quaternions(a, b):
    """Compose deux rotations : a PUIS b se lit produit(a, b)."""
    w0, v0 = a[0], a[1:]
    w1, v1 = b[0], b[1:]
    return np.concatenate([[w0 * w1 - v0 @ v1],
                           w0 * v1 + w1 * v0 + np.cross(v0, v1)])


def quaternion_depuis_rotation(vecteur):
    """Vecteur de rotation (axe x angle, en radians) -> quaternion.

    C'est la brique qui transforme une vitesse angulaire mesuree en increment
    d'orientation : omega * dt donne exactement un tel vecteur.
    """
    angle = float(np.linalg.norm(vecteur))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axe = np.asarray(vecteur, dtype=float) / angle
    return np.concatenate([[np.cos(angle / 2)], axe * np.sin(angle / 2)])


# ===========================================================================
# Passer d'une representation d'orientation a l'autre
#
# POURQUOI CES CONVERSIONS SONT NECESSAIRES
# Aucune bibliotheque ne rend l'orientation dans le meme format. L'IMU de la
# D435i donne des vitesses angulaires ; certaines piles IMU donnent un
# quaternion, d'autres des angles d'Euler ; les detecteurs d'AprilTag rendent
# soit un rvec (vecteur de Rodrigues), soit une transformation homogene. Il
# faut savoir naviguer entre les quatre sans se tromper de convention, sinon
# les erreurs sont silencieuses et l'engin part de travers.
#
# CONVENTION RETENUE POUR EULER : Z-Y-X intrinseque, dite lacet-tangage-roulis
# (yaw-pitch-roll). C'est celle de la robotique et de ROS. On tourne d'abord
# de `lacet` autour de Z, puis de `tangage` autour du nouveau Y, puis de
# `roulis` autour du nouveau X. Une autre convention donnerait d'autres
# nombres pour la MEME rotation : c'est la source d'erreur classique.
# ===========================================================================
def quaternion_vers_euler(q):
    """Quaternion [w,x,y,z] -> (roulis, tangage, lacet) en radians, Z-Y-X."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    roulis = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    # Le tangage passe par un arcsin : a +/-90 deg les deux autres angles
    # deviennent indistinguables (blocage de cardan). On borne l'argument
    # plutot que de laisser sortir un NaN.
    sinus = np.clip(2 * (w * y - z * x), -1.0, 1.0)
    tangage = np.arcsin(sinus)
    lacet = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return float(roulis), float(tangage), float(lacet)


def euler_vers_quaternion(roulis, tangage, lacet):
    """(roulis, tangage, lacet) en radians, Z-Y-X -> quaternion [w,x,y,z]."""
    cr, sr = np.cos(roulis / 2), np.sin(roulis / 2)
    cp, sp = np.cos(tangage / 2), np.sin(tangage / 2)
    cy, sy = np.cos(lacet / 2), np.sin(lacet / 2)
    return np.array([cr * cp * cy + sr * sp * sy,
                     sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy])


def transformation_homogene(rotation, translation):
    """Assemble la matrice 4x4 : bloc R 3x3, bloc t 3x1, derniere ligne
    (0,0,0,1). `rotation` accepte une matrice 3x3 ou un quaternion."""
    R = np.asarray(rotation, dtype=float)
    if R.shape != (3, 3):
        R = quaternion_vers_matrice(R)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(translation, dtype=float).ravel()
    return T


def decomposer_homogene(T):
    """Matrice 4x4 -> (rotation 3x3, translation 3)."""
    T = np.asarray(T, dtype=float)
    return T[:3, :3].copy(), T[:3, 3].copy()


def inverser_homogene(T):
    """Inverse d'une transformation rigide, sans passer par une inversion
    numerique : R^-1 = R^T pour une rotation, ce qui est exact et rapide."""
    R, t = decomposer_homogene(T)
    return transformation_homogene(R.T, -R.T @ t)


# ===========================================================================
# Bruit de mesure deduit de la geometrie du tag
# ===========================================================================
def covariance_position_tag(position_camera, position_tag, incidence_deg,
                            focale=FOCALE_EAU, taille_tag=TAILLE_TAG,
                            sigma_pixel=SIGMA_PIXEL):
    """Covariance 3x3, dans le repere monde, de la position de la camera
    estimee a partir d'UN tag. Anisotrope : mauvaise le long de la visee."""
    v = np.asarray(position_tag, dtype=float) - np.asarray(position_camera, dtype=float)
    d = float(np.linalg.norm(v))
    if d < 1e-6:
        return np.eye(3) * 1e-6
    u = v / d

    sigma_lat = d * sigma_pixel / focale
    # vu de biais, le tag parait plus etroit : sa taille apparente, d'ou l'on
    # tire la distance, porte moins d'information.
    cos_incidence = max(np.cos(np.radians(incidence_deg)), 0.20)
    sigma_prof = (d * d * sigma_pixel
                  / (focale * taille_tag * cos_incidence * np.sqrt(COINS_PAR_TAG)))

    return (sigma_lat ** 2 * (np.eye(3) - np.outer(u, u))
            + sigma_prof ** 2 * np.outer(u, u))


def ecart_type_angle_tag(distance, incidence_deg, focale=FOCALE_EAU,
                         taille_tag=TAILLE_TAG, sigma_pixel=SIGMA_PIXEL):
    """Ecart-type, en radians, de l'orientation donnee par UN tag.

    Le demi-cote du tag mesure s = f·T/(2d) pixels dans l'image. Un coin
    deplace de sigma_px fait donc tourner le tag d'environ sigma_px/s.
    Le facteur 1/sin(incidence) traduit l'ambiguite de retournement : vu
    parfaitement de face, un tag plan renseigne tres mal son inclinaison.
    """
    demi_cote_px = focale * taille_tag / (2.0 * max(distance, 1e-6))
    base = sigma_pixel / max(demi_cote_px, 1e-6)
    return float(base / max(np.sin(np.radians(incidence_deg)), 0.25))


def fusionner_positions(mesures):
    """Fusionne plusieurs estimations de position par addition des
    informations. `mesures` : liste de (position, covariance).

    C'est la reponse chiffree a Josiah : R_total^-1 = somme(R_i^-1).
    Retourne (position_fusionnee, covariance_fusionnee).
    """
    if not mesures:
        raise ValueError("aucune mesure a fusionner")
    if len(mesures) == 1:
        return np.asarray(mesures[0][0], dtype=float), np.asarray(mesures[0][1], dtype=float)

    information = np.zeros((3, 3))
    vecteur = np.zeros(3)
    for position, covariance in mesures:
        inverse = np.linalg.inv(covariance)
        information += inverse
        vecteur += inverse @ np.asarray(position, dtype=float)
    covariance_totale = np.linalg.inv(information)
    return covariance_totale @ vecteur, covariance_totale


# ===========================================================================
# Le noyau : les cinq equations du Kalman lineaire, et rien d'autre
# ===========================================================================
class KalmanLineaire:
    """Les cinq equations du filtre de Kalman lineaire, telles quelles.

    NOTATION. Celle du document de reference du projet — Alex Becker,
    « Kalman Filter Explained Through Examples », kalmanfilter.net, 2026.
    Les symboles sont les siens, exactement :

        PREDICTION
            x(n+1,n) = F x(n,n) + G u(n)          equation d'extrapolation
            P(n+1,n) = F P(n,n) F' + Q            covariance extrapolee

        MISE A JOUR
            K(n)   = P(n,n-1) H' [H P(n,n-1) H' + R(n)]^-1        gain
            x(n,n) = x(n,n-1) + K(n) [z(n) - H x(n,n-1)]          etat
            P(n,n) = (I-KH) P(n,n-1) (I-KH)' + K R K'             covariance

    POURQUOI CETTE CLASSE EXISTE SEPAREMENT. Elle ne connait ni tag, ni tube,
    ni centrale inertielle : elle ne sait faire que ces cinq lignes. Tout ce
    qui est propre a l'engin — quel etat, quel modele de mouvement, quelle
    mesure, quelle confiance — vit dans les classes qui l'utilisent.

    Cette separation n'est pas une coquetterie : elle rend le coeur du filtre
    VERIFIABLE sur l'exemple chiffre du document lui-meme (un radar qui suit
    un avion, etat [portee, vitesse]). C'est ce que fait kalman_du_cours.py,
    qui retrouve les valeurs imprimees dans le document a la quatrieme
    decimale. Un desaccord la-dessus se verrait tout de suite, au lieu de se
    cacher derriere la geometrie des tags.

    FORME DE JOSEPH. Le document donne deux ecritures de la mise a jour de P :
    la simplifiee (I-KH)P, et celle de Joseph. Elles sont egales en arithmetique
    exacte — kalman_du_cours.py le verifie, l'ecart vaut 2e-15 sur son exemple.
    On garde Joseph, que le document recommande : elle reste symetrique et
    definie positive apres des milliers d'iterations en virgule flottante, la
    simplifiee non.
    """

    def __init__(self, x, P):
        self.x = np.asarray(x, dtype=float).ravel()
        self.P = np.asarray(P, dtype=float)

    def predire(self, F, Q, G=None, u=None):
        """x(n+1,n) = F x + G u   et   P(n+1,n) = F P F' + Q.

        G et u sont l'entree connue du document (« input variable »), dont il
        donne pour exemple les lectures d'un accelerometre embarque. C'est
        exactement l'usage qu'on en fait ici.
        """
        self.x = F @ self.x
        if G is not None and u is not None:
            self.x = self.x + G @ np.asarray(u, dtype=float).ravel()
        self.P = F @ self.P @ F.T + Q

    def innovation(self, z, H):
        """z(n) - H x(n,n-1) : l'information neuve apportee par la mesure."""
        return np.asarray(z, dtype=float).ravel() - H @ self.x

    def gain(self, H, R):
        """K = P H' (H P H' + R)^-1, et S = H P H' + R au passage.

        S est la covariance de l'innovation. Le document ne s'en sert pas,
        mais c'est elle qui permet de reconnaitre une mesure aberrante — le
        « Outlier Treatment » qu'il renvoie a son chapitre dedie.
        """
        S = H @ self.P @ H.T + R
        return self.P @ H.T @ np.linalg.inv(S), S

    def corriger(self, z, H, R):
        """Les trois equations de mise a jour. Retourne (innovation, K)."""
        y = self.innovation(z, H)
        K, _ = self.gain(H, R)
        self.x = self.x + K @ y
        I_KH = np.eye(len(self.x)) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return y, K


# ===========================================================================
# Filtre de position : le noyau ci-dessus, avec F, Q et H de l'engin
# ===========================================================================
class FiltreKalmanPosition:
    """Modele CINEMATIQUE a vitesse constante, mesure de position seule.

    C'est le modele du document de reference, porte de 1 a 3 dimensions :

        etat      x = [px, py, pz, vx, vy, vz]'
        modele    F = [[I3, dt.I3], [0, I3]]        vitesse constante
        bruit     Q = sigma_a^2 . G G'   avec G = [dt^2/2 . I3 ; dt . I3]
        mesure    H = [I3, 0]                        les tags donnent p, pas v

    Le Q ci-dessus EST celui du document. Il ecrit, en 1D :
        Q = sigma_a^2 [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]
    et G G' vaut exactement ces quatre blocs. C'est verifie chiffre par
    chiffre dans kalman_du_cours.py.
    """

    H = np.hstack([np.eye(3), np.zeros((3, 3))])

    def __init__(self, sigma_acceleration=None, seuil_chi2=16.27,
                 max_rejets_consecutifs=5, sigma_vitesse_reprise=0.5,
                 bruit_accel=None):
        if sigma_acceleration is None:
            sigma_acceleration = SIGMA_ACCELERATION
        if bruit_accel is None:
            bruit_accel = BRUIT_ACCEL
        self.sigma_a = float(sigma_acceleration)   # m/s^2 d'acceleration non modelisee
        self.bruit_accel = float(bruit_accel)      # m/s^2 de bruit du capteur
        self.accel_utilise = False
        self.seuil = float(seuil_chi2)             # chi2 a 3 ddl, seuil 99.9 %
        self.max_rejets_consecutifs = int(max_rejets_consecutifs)
        self.sigma_v_reprise = float(sigma_vitesse_reprise)
        self.noyau = KalmanLineaire(np.zeros(6), np.eye(6) * 1e3)
        self.demarre = False
        self.rejets = 0
        self.rejets_consecutifs = 0
        self.reprises = 0

    # x et P vivent dans le noyau ; on les expose tels quels pour que le reste
    # du fichier — et les scripts qui lisent filtre.x — ne change pas.
    @property
    def x(self):
        return self.noyau.x

    @x.setter
    def x(self, valeur):
        self.noyau.x = np.asarray(valeur, dtype=float).ravel()

    @property
    def P(self):
        return self.noyau.P

    @P.setter
    def P(self, valeur):
        self.noyau.P = np.asarray(valeur, dtype=float)

    @staticmethod
    def modele(dt):
        """F et G du modele a vitesse constante, pour un pas dt."""
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        # G : effet d'une acceleration pendant dt, sur la position et la vitesse
        G = np.vstack([0.5 * dt * dt * np.eye(3), dt * np.eye(3)])
        return F, G

    def demarrer(self, position, sigma_position=0.05, sigma_vitesse=0.5):
        self.x = np.concatenate([np.asarray(position, dtype=float), np.zeros(3)])
        self.P = np.diag([sigma_position ** 2] * 3 + [sigma_vitesse ** 2] * 3)
        self.demarre = True

    def predire(self, dt, acceleration=None):
        """Fait avancer l'etat de dt secondes.

        acceleration : celle MESUREE par l'accelerometre, exprimee dans le
        repere MONDE et debarrassee de la pesanteur. Si elle est fournie, elle
        entre dans la prediction comme une commande connue au lieu d'etre
        traitee comme un alea.

        CE QUE L'ACCELEROMETRE CHANGE. Sans lui, on suppose la vitesse
        constante et on couvre l'ecart par sigma_a, l'acceleration que l'engin
        peut avoir sans qu'on le sache. Avec lui, cette acceleration est
        MESUREE : il ne reste que le bruit du capteur, bien plus petit. La
        prediction suit alors les manoeuvres au lieu de retarder dessus.

        RESERVE HONNETE. Un accelerometre MEMS a un biais lentement variable
        que rien ici n'estime, et une double integration transforme ce biais
        en erreur de position quadratique : un biais de 0.05 m/s2 fait 2.5 cm
        au bout d'une seconde, 1 m au bout de dix. C'est utile pour traverser
        une perte de tags de quelques instants, pas pour naviguer a l'estime.
        Les tags restent la seule source sans derive.
        """
        if not self.demarre or dt <= 0:
            return
        F, G = self.modele(dt)
        if acceleration is None:
            incertitude = self.sigma_a          # acceleration inconnue
            entree = None
        else:
            incertitude = self.bruit_accel      # acceleration mesuree
            entree = acceleration
            self.accel_utilise = True
        # Q = sigma^2 . G G' — le Q du document, ecrit en 3D.
        Q = incertitude ** 2 * (G @ G.T)
        self.noyau.predire(F, Q, G=G, u=entree)

    def corriger(self, position_mesuree, covariance):
        """Integre une mesure. Retourne (acceptee, distance_mahalanobis)."""
        if not self.demarre:
            self.demarrer(position_mesuree)
            return True, 0.0

        z = np.asarray(position_mesuree, dtype=float)
        R = np.asarray(covariance, dtype=float)
        y = self.noyau.innovation(z, self.H)
        S = self.H @ self.P @ self.H.T + R
        distance = float(y @ np.linalg.solve(S, y))

        if distance > self.seuil:      # aberration probable (flip d'un tag)
            self.rejets_consecutifs += 1
            if self.rejets_consecutifs < self.max_rejets_consecutifs:
                self.rejets += 1
                return False, distance
            # Verrouillage : autant de refus d'affilee ne s'explique plus par
            # des mesures aberrantes, mais par un etat faux. On se recale sur
            # la mesure et on rouvre l'incertitude.
            self.x[:3] = z
            self.P[:3, :3] = 4.0 * R
            self.P[3:, 3:] = np.eye(3) * self.sigma_v_reprise ** 2
            self.P[:3, 3:] = 0.0
            self.P[3:, :3] = 0.0
            self.reprises += 1
            self.rejets_consecutifs = 0
            return True, distance

        self.rejets_consecutifs = 0
        # Les trois equations de mise a jour du document, forme de Joseph.
        self.noyau.corriger(z, self.H, R)
        return True, distance

    @property
    def position(self):
        return self.x[:3].copy()

    @property
    def vitesse(self):
        return self.x[3:].copy()

    @property
    def incertitude_position(self):
        """Rayon a 1 sigma, en metres."""
        return float(np.sqrt(np.trace(self.P[:3, :3]) / 3.0))


# ===========================================================================
# Filtre d'orientation : Kalman scalaire sur l'angle, applique par slerp
# ===========================================================================
class FiltreOrientation:
    """On suit un quaternion et UNE variance angulaire scalaire.

    Approximation assumee : l'incertitude d'orientation est supposee
    isotrope (la meme autour des trois axes). C'est faux dans le detail --
    le lacet est mieux contraint que le tangage quand on regarde un mur de
    face -- mais ca evite un filtre a erreur d'etat complet tant qu'on n'a
    pas fusionne la centrale inertielle de la D435i.
    """

    def __init__(self, derive_gyro_deg_s=None, seuil_saut_deg=25.0,
                 max_rejets_consecutifs=5, bruit_gyro_deg_s=None,
                 tau_biais=20.0):
        if derive_gyro_deg_s is None:
            derive_gyro_deg_s = DERIVE_GYRO_DEG_S
        if bruit_gyro_deg_s is None:
            bruit_gyro_deg_s = BRUIT_GYRO_DEG_S
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.variance = np.radians(180.0) ** 2
        self.derive = np.radians(derive_gyro_deg_s)   # rad/s d'errance non modelisee
        self.bruit_gyro = np.radians(bruit_gyro_deg_s)  # rad/s de bruit du gyro
        self.seuil_saut = float(seuil_saut_deg)
        self.max_rejets_consecutifs = int(max_rejets_consecutifs)
        self.demarre = False
        self.rejets = 0
        self.rejets_consecutifs = 0
        self.reprises = 0
        # Biais du gyro, en rad/s, dans le repere de la centrale. Un gyro MEMS
        # ne mesure jamais zero au repos : ce petit decalage, integre, fait
        # deriver l'orientation. On l'estime sur les corrections que les tags
        # apportent, et on le retranche des mesures suivantes.
        self.biais = np.zeros(3)
        self.tau_biais = float(tau_biais)   # constante de temps de l'estimation
        self._temps_depuis_correction = 0.0
        self._rotation_gyro = np.zeros(3)   # rotation integree depuis la derniere
        self.gyro_utilise = False

    def demarrer(self, R_ou_q, sigma_deg=5.0):
        q = np.asarray(R_ou_q, dtype=float)
        self.q = matrice_vers_quaternion(q) if q.shape == (3, 3) else q / np.linalg.norm(q)
        self.variance = np.radians(sigma_deg) ** 2
        self.demarre = True

    def predire(self, dt, omega=None):
        """Fait avancer l'orientation de dt secondes.

        omega : vitesse angulaire mesuree par le GYROSCOPE, en rad/s, dans le
        repere de la camera. Si elle est fournie, l'orientation est reellement
        propagee au lieu d'etre supposee constante.

        CE QUE LE GYRO CHANGE. Sans lui, on suppose l'engin immobile en
        rotation et on gonfle l'incertitude de `derive` par seconde, soit
        10 deg/s dans nos reglages : au bout d'une seconde sans tag, on ne
        sait plus rien. Avec lui, on SAIT de combien l'engin a tourne, et
        l'incertitude ne croit plus que du bruit du gyro — deux ordres de
        grandeur en dessous. C'est ce qui permet de traverser une perte de
        tags sans perdre le cap.
        """
        if not self.demarre or dt <= 0:
            return
        if omega is None:
            self.variance += (self.derive * dt) ** 2
            return

        self.gyro_utilise = True
        vitesse = np.asarray(omega, dtype=float).ravel() - self.biais
        rotation = vitesse * dt
        # q PUIS la petite rotation, exprimee dans le repere du corps :
        # l'increment se compose a DROITE.
        self.q = produit_quaternions(self.q, quaternion_depuis_rotation(rotation))
        self.q /= np.linalg.norm(self.q)
        self.variance += (self.bruit_gyro * dt) ** 2
        self._temps_depuis_correction += dt
        self._rotation_gyro = self._rotation_gyro + rotation

    def corriger_gravite(self, acceleration, sigma_deg=8.0,
                         tolerance_g=0.15, gravite=9.81):
        """Recale le ROULIS et le TANGAGE sur la verticale vue par l'accelerometre.

        Au repos, un accelerometre mesure la reaction a la pesanteur : sa
        direction donne le haut. En comparant cette direction a celle que
        l'orientation courante predit, on corrige les deux axes horizontaux —
        et EUX SEULS. Le lacet reste inobservable : tourner autour de la
        verticale ne change pas la direction de la pesanteur. C'est pour cela
        que l'axe de correction, obtenu par produit vectoriel, est
        automatiquement perpendiculaire a la verticale.

        Interet : sans aucun tag, le roulis et le tangage restent bornes
        indefiniment. Seul le lacet derive, et c'est lui que les tags recalent.

        L'accelerometre ne distingue pas la pesanteur d'une acceleration de
        l'engin. On ne s'en sert donc que quand la norme mesuree est proche de
        g : sinon l'engin manoeuvre et la mesure ne dit plus ou est le bas.
        Retourne (utilisee, correction_en_degres).
        """
        if not self.demarre:
            return False, 0.0
        a = np.asarray(acceleration, dtype=float).ravel()
        norme = float(np.linalg.norm(a))
        if norme < 1e-6 or abs(norme / gravite - 1.0) > tolerance_g:
            return False, 0.0        # l'engin accelere : mesure inexploitable

        mesuree = a / norme
        # Direction du "haut" telle que l'orientation courante la prevoit,
        # ramenee dans le repere du corps.
        R = quaternion_vers_matrice(self.q)
        attendue = R.T @ np.array([0.0, 0.0, 1.0])
        axe = np.cross(attendue, mesuree)
        sinus = float(np.linalg.norm(axe))
        cosinus = float(np.clip(attendue @ mesuree, -1.0, 1.0))
        angle = float(np.arctan2(sinus, cosinus))
        if sinus < 1e-9:
            return True, 0.0                       # deja aligne
        axe = axe / sinus

        # Gain de Kalman scalaire, comme pour la correction par les tags.
        r = np.radians(sigma_deg) ** 2
        gain = self.variance / (self.variance + r)
        # SIGNE. `axe, angle` decrit la rotation Delta qui amene la direction
        # PREVUE sur la direction MESUREE, toutes deux dans le repere du corps.
        # L'orientation q, elle, va du corps vers le monde : pour que sa
        # prevision R'^T.ez vaille `mesuree`, il faut R' = R.Delta^T, donc
        # composer a droite par l'INVERSE de Delta — d'ou le signe moins.
        # Avec le signe oppose, la correction s'eloigne de la cible et
        # l'orientation converge vers le point fixe a 180 degres.
        self.q = produit_quaternions(
            self.q, quaternion_depuis_rotation(-axe * angle * gain))
        self.q /= np.linalg.norm(self.q)
        # L'accelerometre ne renseigne que deux axes sur trois : il ne peut
        # donc pas resserrer l'incertitude autant qu'une mesure complete.
        self.variance = (1.0 - gain * 2.0 / 3.0) * self.variance
        return True, float(np.degrees(angle))

    def corriger(self, R_ou_q_mesure, sigma_mesure_rad):
        """Retourne (acceptee, ecart_en_degres)."""
        q = np.asarray(R_ou_q_mesure, dtype=float)
        q = matrice_vers_quaternion(q) if q.shape == (3, 3) else q / np.linalg.norm(q)
        if not self.demarre:
            self.demarrer(q, np.degrees(sigma_mesure_rad))
            return True, 0.0

        ecart = angle_quaternions(self.q, q)
        # un tag retourne produit un saut brutal : on le refuse tant que le
        # filtre est encore confiant dans ce qu'il tient.
        if ecart > self.seuil_saut and np.degrees(np.sqrt(self.variance)) < self.seuil_saut:
            self.rejets_consecutifs += 1
            if self.rejets_consecutifs < self.max_rejets_consecutifs:
                self.rejets += 1
                return False, ecart
            # meme verrouillage que pour la position : trop de refus d'affilee
            # signifie que c'est l'orientation gardee qui est fausse.
            self.demarrer(q, max(np.degrees(sigma_mesure_rad) * 2.0, 10.0))
            self.reprises += 1
            self.rejets_consecutifs = 0
            return True, ecart

        self.rejets_consecutifs = 0
        # gain de Kalman scalaire sur l'angle
        r = float(sigma_mesure_rad) ** 2
        gain = self.variance / (self.variance + r)
        avant = self.q.copy()
        self.q = slerp(self.q, q, gain)
        self.variance = (1.0 - gain) * self.variance
        if self.gyro_utilise:
            # On passe la MESURE, pas l'etat corrige. Le gain de Kalman
            # n'applique qu'une fraction de l'ecart : estimer le biais sur la
            # correction appliquee le sous-estimerait d'autant, et d'autant
            # plus que le filtre est confiant. L'ecart complet — l'innovation —
            # est la vraie mesure de la derive accumulee depuis le dernier tag.
            self._reestimer_biais(avant, q)
        return True, ecart

    def _reestimer_biais(self, avant, mesure):
        """Attribue au biais du gyro la part systematique de l'innovation.

        Entre deux tags, l'orientation n'avance que par integration du gyro.
        Si le gyro a un biais b, l'orientation derive de b*dt, et le tag la
        trouve systematiquement decalee du meme cote : cet ecart, divise par
        le temps ecoule, EST une mesure du biais.

        On la moyenne lentement (constante de temps tau_biais) parce qu'une
        correction isolee melange le biais et le bruit du tag. Un biais reel
        est constant, le bruit ne l'est pas : seul le premier survit au
        moyennage.
        """
        dt = self._temps_depuis_correction
        self._temps_depuis_correction = 0.0
        rotation_gyro = self._rotation_gyro
        self._rotation_gyro = np.zeros(3)
        if dt < 0.05:
            return                              # trop court pour separer quoi que ce soit

        # Rotation apportee par la correction, exprimee dans le repere du corps.
        delta = produit_quaternions(np.array([avant[0], -avant[1], -avant[2],
                                              -avant[3]]), mesure)
        angle = 2.0 * np.arctan2(float(np.linalg.norm(delta[1:])),
                                 float(abs(delta[0])))
        if angle < 1e-9:
            return
        axe = delta[1:] / np.linalg.norm(delta[1:])
        if delta[0] < 0:
            axe = -axe
        correction = axe * angle

        # Le gyro a trop tourne de `-correction` pendant dt : c'est un biais
        # apparent de -correction/dt.
        mesure = -correction / dt
        poids = min(dt / self.tau_biais, 0.5)    # jamais plus de la moitie d'un coup
        self.biais = (1.0 - poids) * self.biais + poids * mesure

    @property
    def biais_deg_s(self):
        """Biais estime du gyro, en deg/s sur les trois axes."""
        return np.degrees(self.biais)

    @property
    def matrice(self):
        return quaternion_vers_matrice(self.q)

    @property
    def incertitude_deg(self):
        return float(np.degrees(np.sqrt(self.variance)))


# ===========================================================================
# Surveillance des tags : detecter une boite qui a bouge
# ===========================================================================
class SurveillanceTags:
    """Suit, tag par tag, la moyenne glissante de l'ecart entre la position
    que CE tag annonce et celle qu'annoncent les autres.

    Tag bien enregistre  -> moyenne qui tend vers zero.
    Tag qui a bouge      -> moyenne qui tend vers son deplacement.

    Le module donne donc non seulement QUEL support a bouge, mais DE
    COMBIEN et DANS QUELLE DIRECTION : de quoi corriger la carte sans tout
    reenregistrer.

    Un tag est confronte aux AUTRES TAGS DE LA MEME IMAGE, jamais a la
    sortie du filtre. La raison est subtile mais decisive : la sortie du
    filtre retarde sur le mouvement reel, et ce retard depend de quel tag
    est visible. Le confronter au filtre fabrique donc de faux coupables.
    Deux mesures prises au meme instant, elles, n'ont aucun retard relatif.

    Consequence assumee : un tag vu SEUL n'est jamais mis en defaut. C'est
    la limite d'observabilite, pas un defaut d'implementation -- rien ne
    distingue alors "la camera a bouge" de "le tag a bouge".
    """

    def __init__(self, fenetre=60, seuil_mm=8.0, minimum_observations=25):
        # Fenetre courte volontairement : elle doit se vider de l'ancien
        # regime en quelques secondes de co-visibilite, sinon un deplacement
        # recent reste dilue par les observations d'avant et l'amplitude
        # annoncee est sous-estimee. Le bruit residuel apres moyenne sur 60
        # vaut environ 1 mm, tres en dessous du seuil de 8 mm.
        self.fenetre = int(fenetre)
        self.seuil = float(seuil_mm) / 1000.0
        self.minimum = int(minimum_observations)
        self.ecarts = defaultdict(lambda: deque(maxlen=self.fenetre))

    def observer_groupe(self, mesures):
        """`mesures` : [(identifiant, position, covariance)] d'une meme image.

        On enregistre l'ecart PAR PAIRE. Dans ce bassin on ne voit jamais
        plus de deux tags a la fois : un ecart de paire dit qu'un des deux a
        bouge, sans dire lequel. C'est en recoupant plusieurs partenaires
        qu'on tranche.
        """
        valides = [(i, p, C) for i, p, C in mesures if i is not None]
        for rang_a in range(len(valides)):
            for rang_b in range(rang_a + 1, len(valides)):
                ia, pa, Ca = valides[rang_a]
                ib, pb, Cb = valides[rang_b]
                pa, pb = np.asarray(pa, dtype=float), np.asarray(pb, dtype=float)
                # Poids inverse de la variance de la difference : une paire
                # vue de tres loin ou tres de biais ne doit pas peser autant
                # qu'une paire vue de pres et de face.
                poids = 1.0 / max(np.trace(np.asarray(Ca) + np.asarray(Cb)), 1e-12)
                if ia < ib:
                    self.ecarts[(ia, ib)].append((pa - pb, poids))
                else:
                    self.ecarts[(ib, ia)].append((pb - pa, poids))

    def _moyenne_ponderee(self, observations):
        vecteurs = np.array([v for v, _ in observations])
        poids = np.array([w for _, w in observations])
        return (vecteurs * poids[:, None]).sum(axis=0) / poids.sum()

    def _ecarts_par_partenaire(self):
        """{tag: {partenaire: ecart moyen de la position deduite de tag}}."""
        resultat = defaultdict(dict)
        for (i, j), observations in self.ecarts.items():
            if len(observations) < self.minimum:
                continue
            moyenne = self._moyenne_ponderee(observations)
            resultat[i][j] = moyenne
            resultat[j][i] = -moyenne
        return resultat

    def suspects(self):
        """Tags convaincus : {identifiant: (norme, vecteur_deplacement)}.

        Un tag est retenu s'il contredit AU MOINS DEUX partenaires distincts,
        et toujours dans le meme sens. Contredire un seul voisin ne suffit
        pas : c'est peut-etre le voisin qui a bouge.

        Si la boite a bouge de d, la position deduite de ce tag se decale de
        -d : on part de la position supposee du tag, restee celle d'avant.
        Le deplacement est donc l'oppose de l'ecart moyen.
        """
        convaincus = {}
        for tag, partenaires in self._ecarts_par_partenaire().items():
            grands = [v for v in partenaires.values() if np.linalg.norm(v) > self.seuil]
            if len(grands) < 2:
                continue
            coherent = all(
                float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)) > 0.5
                for k, a in enumerate(grands) for b in grands[k + 1:])
            if coherent:
                moyen = np.mean(np.array(grands), axis=0)
                convaincus[tag] = (float(np.linalg.norm(moyen)), -moyen)
        return convaincus

    def paires_douteuses(self):
        """Paires en desaccord dont aucun membre n'est formellement convaincu."""
        convaincus = set(self.suspects())
        douteuses = {}
        for (i, j), observations in self.ecarts.items():
            if len(observations) < self.minimum:
                continue
            norme = float(np.linalg.norm(self._moyenne_ponderee(observations)))
            if norme > self.seuil and i not in convaincus and j not in convaincus:
                douteuses[(i, j)] = norme
        return douteuses

    def suspect_principal(self):
        """Tag commun a plusieurs paires en desaccord.

        Indice plus faible qu'une conviction, mais souvent suffisant : si
        toutes les paires qui se disputent contiennent le meme tag, c'est
        le denominateur commun qu'il faut aller regarder. Utile quand les
        donnees manquent pour trancher par coherence de direction.
        """
        douteuses = self.paires_douteuses()
        if len(douteuses) < 2:
            return None
        # On cumule l'AMPLITUDE des desaccords plutot que leur nombre : une
        # paire qui se dispute de 21 mm accuse davantage qu'une paire a 8 mm.
        scores = defaultdict(float)
        paires = defaultdict(int)
        for (i, j), norme in douteuses.items():
            for tag in (i, j):
                scores[tag] += norme
                paires[tag] += 1
        ordre = sorted(scores.items(), key=lambda couple: -couple[1])
        meilleur = ordre[0][0]
        if paires[meilleur] < 2:
            return None
        if len(ordre) > 1 and ordre[0][1] < 1.3 * ordre[1][1]:
            return None            # trop serre pour designer qui que ce soit
        return meilleur

    def rapport(self):
        lignes = []
        for identifiant, (norme, vecteur) in sorted(self.suspects().items()):
            lignes.append(f"  tag {identifiant} : boite deplacee de {norme*1000:.0f} mm "
                          f"({vecteur[0]*1000:+.0f}, {vecteur[1]*1000:+.0f}, "
                          f"{vecteur[2]*1000:+.0f}) mm  [confirme par plusieurs voisins]")
        for (i, j), norme in sorted(self.paires_douteuses().items()):
            lignes.append(f"  paire {i}-{j} : desaccord de {norme*1000:.0f} mm, "
                          "aucun des deux n'est formellement en cause")
        principal = self.suspect_principal()
        if principal is not None:
            lignes.append(f"  -> tag {principal} present dans toutes les paires en "
                          "desaccord : c'est la boite a verifier en premier")
        return "\n".join(lignes) if lignes else "  aucun tag suspect"


# ===========================================================================
# Facade : les deux filtres cote a cote
# ===========================================================================
class FiltrePose:
    """Enveloppe pratique : une pose complete (position + orientation).

    Utilisation type, a chaque image :
        filtre.predire(dt, gyro=omega, accel=a)     # IMU facultative
        for tag in tags_vus:
            filtre.ajouter_tag(position_estimee, position_tag, incidence, R_mesuree)
        filtre.appliquer()

    ---------------------------------------------------------------------
    CE QUE LA CENTRALE INERTIELLE APPORTE, ET CE QU'ELLE N'APPORTE PAS
    ---------------------------------------------------------------------
    Les tags et l'IMU ont des defauts opposes, et c'est ce qui rend leur
    fusion interessante :

      TAGS     sans derive, mais bruites, et surtout INTERMITTENTS. Des qu'on
               ne voit plus de tag, plus aucune information.
      GYRO     tres precis a court terme, mais son petit biais integre fait
               deriver l'orientation sans limite.
      ACCEL    donne la direction du bas en permanence, donc borne le roulis
               et le tangage pour toujours — mais ne dit RIEN du lacet, et sa
               double integration derive trop vite pour naviguer a l'estime.

    D'ou le partage : le gyro propage entre deux tags, l'accelerometre tient
    deux axes d'orientation sur trois, les tags recalent le lacet et la
    position et servent a estimer le biais du gyro. Chaque capteur couvre le
    trou de l'autre.

    REPERE DE L'IMU — piege a ne pas negliger. Sur la D435i la centrale n'est
    pas alignee avec la camera couleur : il existe une rotation constante
    entre les deux, que pyrealsense2 fournit
    (get_extrinsics_to). Passer les mesures brutes sans cette rotation
    melange les axes et fait deriver l'engin de travers, sans message
    d'erreur. `rotation_imu_camera` est la pour ca.
    """

    def __init__(self, sigma_acceleration=None, derive_gyro_deg_s=None,
                 seuil_deplacement_mm=8.0, rotation_imu_camera=None,
                 gravite=9.81):
        self.position = FiltreKalmanPosition(sigma_acceleration)
        self.orientation = FiltreOrientation(derive_gyro_deg_s)
        self.surveillance = SurveillanceTags(seuil_mm=seuil_deplacement_mm)
        # Rotation qui amene un vecteur du repere IMU vers le repere camera.
        # Identite par defaut : vrai seulement si les deux sont alignes.
        self.R_imu_camera = (np.eye(3) if rotation_imu_camera is None
                             else np.asarray(rotation_imu_camera, dtype=float))
        self.gravite = float(gravite)
        self._mesures = []
        self._orientations = []

    def predire(self, dt, gyro=None, accel=None):
        """Fait avancer la pose de dt secondes, avec l'IMU si elle est la.

        gyro  : vitesse angulaire, rad/s, repere IMU.
        accel : acceleration specifique, m/s2, repere IMU — pesanteur
                COMPRISE, telle que le capteur la rend.

        L'ordre compte : on propage d'abord l'orientation avec le gyro, puis
        on s'en sert pour retirer la pesanteur de l'accelerometre et exprimer
        le reste dans le repere monde. Utiliser l'ancienne orientation
        introduirait une erreur proportionnelle a la rotation faite pendant dt.
        """
        omega = None if gyro is None else self.R_imu_camera @ np.asarray(
            gyro, dtype=float).ravel()
        self.orientation.predire(dt, omega)

        acceleration_monde = None
        if accel is not None and self.orientation.demarre:
            a_camera = self.R_imu_camera @ np.asarray(accel, dtype=float).ravel()
            # Vers le repere monde, puis on retranche la pesanteur : ce qui
            # reste est l'acceleration propre de l'engin.
            a_monde = quaternion_vers_matrice(self.orientation.q) @ a_camera
            acceleration_monde = a_monde - np.array([0.0, 0.0, self.gravite])
        self.position.predire(dt, acceleration_monde)

        # L'accelerometre recale le roulis et le tangage, meme sans tag.
        if accel is not None:
            self.orientation.corriger_gravite(
                self.R_imu_camera @ np.asarray(accel, dtype=float).ravel(),
                gravite=self.gravite)

    def ajouter_tag(self, position_camera_estimee, position_tag, incidence_deg,
                    rotation_mesuree=None, distance=None, identifiant=None):
        """Empile la contribution d'un tag pour l'image courante."""
        covariance = covariance_position_tag(position_camera_estimee, position_tag,
                                             incidence_deg)
        self._mesures.append((np.asarray(position_camera_estimee, dtype=float),
                              covariance, identifiant))
        if rotation_mesuree is not None:
            if distance is None:
                distance = float(np.linalg.norm(np.asarray(position_tag, dtype=float)
                                                - np.asarray(position_camera_estimee, dtype=float)))
            sigma = ecart_type_angle_tag(distance, incidence_deg)
            self._orientations.append((rotation_mesuree, sigma))

    def appliquer(self):
        """Fusionne les tags empiles puis corrige. Retourne (acceptee, nb_tags)."""
        nombre = len(self._mesures)
        acceptee = False
        if nombre:
            z, R = fusionner_positions([(p, C) for p, C, _ in self._mesures])
            acceptee, _ = self.position.corriger(z, R)
            # chaque tag est confronte aux autres tags de la MEME image :
            # celui qui s'en ecarte systematiquement a bouge.
            self.surveillance.observer_groupe(
                [(identifiant, p, C) for p, C, identifiant in self._mesures])
        # l'orientation la mieux informee est celle du tag au plus petit sigma
        if self._orientations:
            rotation, sigma = min(self._orientations, key=lambda couple: couple[1])
            self.orientation.corriger(rotation, sigma)
        self._mesures.clear()
        self._orientations.clear()
        return acceptee, nombre


# ===========================================================================
# Auto-test : lance `python filtre_kalman.py`
# ===========================================================================
def _auto_test():
    generateur = np.random.default_rng(12345)
    print("=" * 68)
    print("AUTO-TEST DU FILTRE")
    print("=" * 68)

    # -- quaternions : aller-retour matrice <-> quaternion -------------------
    angle = 0.7
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0], [0, 0, 1.0]])
    erreur = np.abs(quaternion_vers_matrice(matrice_vers_quaternion(R)) - R).max()
    print(f"aller-retour matrice <-> quaternion : erreur max {erreur:.2e}")
    assert erreur < 1e-9

    # -- anisotropie de R ---------------------------------------------------
    C = covariance_position_tag([0.0, 0.0, 0.0], [1.6, 0.0, 0.0], 20.0)
    valeurs = np.sqrt(np.sort(np.linalg.eigvalsh(C)))
    print(f"un tag a 1.6 m : sigma lateral {valeurs[0]*1000:.2f} mm, "
          f"profondeur {valeurs[-1]*1000:.2f} mm "
          f"(rapport {valeurs[-1]/valeurs[0]:.1f}x)")
    assert valeurs[-1] > 3 * valeurs[0], "la profondeur doit etre nettement pire"

    # -- deux tags sur des murs differents ----------------------------------
    seul = covariance_position_tag([1.0, 0.8, 0.5], [1.0, 0.0, 0.35], 10.0)
    autre = covariance_position_tag([1.0, 0.8, 0.5], [0.0, 0.835, 0.65], 10.0)
    _, fusion = fusionner_positions([([1.0, 0.8, 0.5], seul), ([1.0, 0.8, 0.5], autre)])
    pire_seul = np.sqrt(np.linalg.eigvalsh(seul)).max()
    pire_fusion = np.sqrt(np.linalg.eigvalsh(fusion)).max()
    # Deux mesures independantes de meme qualite gagnent deja un facteur
    # racine de 2 par simple moyennage. Depasser ce seuil prouve que c'est la
    # GEOMETRIE qui travaille : la ou un tag est aveugle (sa profondeur),
    # l'autre est precis (son lateral).
    gain = pire_seul / pire_fusion
    print(f"pire direction : 1 tag {pire_seul*1000:.2f} mm -> "
          f"2 tags sur murs perpendiculaires {pire_fusion*1000:.2f} mm "
          f"(gain {gain:.2f}x, moyennage seul : 1.41x)")
    assert gain > np.sqrt(2), "les murs perpendiculaires doivent faire mieux que moyenner"

    # -- le filtre reduit-il vraiment le bruit ? ----------------------------
    dt, n = 1 / 30, 900
    filtre = FiltreKalmanPosition(sigma_acceleration=0.3)
    vraie = np.array([1.0, 0.8, 0.5])
    vitesse = np.array([0.25, 0.0, 0.0])
    brut, filtre_rms, rejets_injectes = [], [], 0
    for i in range(n):
        vraie = vraie + vitesse * dt
        C = covariance_position_tag(vraie, [vraie[0] + 1.6, vraie[1], vraie[2]], 15.0)
        bruit = generateur.multivariate_normal(np.zeros(3), C)
        mesure = vraie + bruit
        if i % 97 == 96:                    # aberration type flip
            mesure = mesure + np.array([0.35, -0.25, 0.15])
            rejets_injectes += 1
        filtre.predire(dt)
        filtre.corriger(mesure, C)
        brut.append(np.linalg.norm(mesure - vraie))
        filtre_rms.append(np.linalg.norm(filtre.position - vraie))

    rms_brut = float(np.sqrt(np.mean(np.square(brut))))
    rms_filtre = float(np.sqrt(np.mean(np.square(filtre_rms))))
    print(f"RMS brut {rms_brut*1000:.2f} mm -> filtre {rms_filtre*1000:.2f} mm "
          f"({rms_brut/rms_filtre:.1f}x mieux)")
    print(f"aberrations injectees {rejets_injectes}, rejetees {filtre.rejets}")
    assert rms_filtre < rms_brut, "le filtre doit reduire l'erreur"
    assert filtre.rejets >= rejets_injectes * 0.8, "le rejet doit attraper les flips"

    # -- orientation --------------------------------------------------------
    orientation = FiltreOrientation()
    q_vrai = matrice_vers_quaternion(R)
    orientation.demarrer(q_vrai, sigma_deg=5.0)
    sigma = ecart_type_angle_tag(1.6, 20.0)
    ecarts = []
    for _ in range(300):
        perturbation = generateur.normal(0.0, sigma, 3)
        norme = np.linalg.norm(perturbation)
        axe = perturbation / norme if norme > 1e-12 else np.array([1.0, 0, 0])
        dq = np.concatenate([[np.cos(norme / 2)], axe * np.sin(norme / 2)])
        w0, v0 = dq[0], dq[1:]
        w1, v1 = q_vrai[0], q_vrai[1:]
        q_mesure = np.concatenate([[w0 * w1 - v0 @ v1], w0 * v1 + w1 * v0 + np.cross(v0, v1)])
        orientation.predire(1 / 30)
        orientation.corriger(q_mesure, sigma)
        ecarts.append(angle_quaternions(orientation.q, q_vrai))
    print(f"orientation : bruit tag {np.degrees(sigma):.2f} deg -> "
          f"apres filtrage {np.mean(ecarts[-100:]):.2f} deg")
    assert np.mean(ecarts[-100:]) < np.degrees(sigma)

    # -- detection d'une boite deplacee -------------------------------------
    surveillance = SurveillanceTags(seuil_mm=5.0, minimum_observations=20)
    supports = {10: np.array([1.5, 0.0, 0.35]),
                11: np.array([2.4, 0.0, 0.65]),
                12: np.array([0.0, 0.8, 0.50])}
    pousse = np.array([0.018, -0.006, 0.0])       # 19 mm sur la boite 11
    camera = np.array([1.2, 1.4, 0.5])
    for _ in range(150):
        groupe = []
        for tid, endroit in supports.items():
            C = covariance_position_tag(camera, endroit, 20.0)
            biais = -pousse if tid == 11 else np.zeros(3)
            z = camera + biais + generateur.multivariate_normal(np.zeros(3), C)
            groupe.append((tid, z, C))
        surveillance.observer_groupe(groupe)
    convaincus = surveillance.suspects()
    assert 11 in convaincus, f"la boite 11 doit etre detectee, obtenu {sorted(convaincus)}"
    assert set(convaincus) == {11}, f"aucune autre ne doit l'etre : {sorted(convaincus)}"
    estime = convaincus[11][1]
    erreur = float(np.linalg.norm(estime - pousse))
    print(f"boite deplacee de {1000*np.linalg.norm(pousse):.0f} mm -> detectee a "
          f"{1000*convaincus[11][0]:.0f} mm (erreur {1000*erreur:.1f} mm)")
    assert erreur < 0.004, "le deplacement estime doit etre juste a 4 mm pres"

    # -- conversions entre representations d'orientation --------------------
    for essai in range(200):
        angles = generateur.uniform(-np.pi, np.pi, 3)
        angles[1] = generateur.uniform(-1.4, 1.4)      # hors blocage de cardan
        q = euler_vers_quaternion(*angles)
        retour = np.array(quaternion_vers_euler(q))
        # on compare les ROTATIONS, pas les triplets : deux triplets
        # differents peuvent decrire la meme orientation.
        #
        # Seuil a 1e-4 deg et non zero : `angle_quaternions` passe par un
        # arccos, dont la derivee explose au voisinage de 1. Deux quaternions
        # identiques au dernier bit y donnent quelques 1e-6 deg d'ecart
        # apparent. C'est du bruit de calcul, pas une erreur de conversion —
        # verifie sur des cas ronds, l'aller-retour rend les memes angles.
        assert angle_quaternions(q, euler_vers_quaternion(*retour)) < 1e-4
    print("aller-retour Euler <-> quaternion : 200 orientations, "
          "ecart max < 1e-4 deg")

    T = transformation_homogene(quaternion_vers_matrice(q_vrai), [1.0, -2.0, 0.5])
    assert T.shape == (4, 4) and np.allclose(T[3], [0, 0, 0, 1])
    identite = T @ inverser_homogene(T)
    assert np.abs(identite - np.eye(4)).max() < 1e-12
    R_lu, t_lu = decomposer_homogene(T)
    assert np.allclose(t_lu, [1.0, -2.0, 0.5])
    print(f"transformation homogene 4x4 : T . T^-1 = I a "
          f"{np.abs(identite - np.eye(4)).max():.1e} pres")

    # -- le gyroscope tient-il le cap quand les tags disparaissent ? --------
    # 6 secondes sans aucun tag, l'engin tournant a 20 deg/s.
    dt, duree = 1 / 200, 6.0
    vitesse_vraie = np.radians([3.0, -5.0, 20.0])
    biais_vrai = np.radians([0.4, -0.3, 0.6])
    for avec_gyro in (False, True):
        suivi = FiltreOrientation()
        suivi.demarrer(np.array([1.0, 0.0, 0.0, 0.0]), sigma_deg=2.0)
        verite = np.array([1.0, 0.0, 0.0, 0.0])
        for _ in range(int(duree / dt)):
            verite = produit_quaternions(
                verite, quaternion_depuis_rotation(vitesse_vraie * dt))
            mesure = (vitesse_vraie + biais_vrai
                      + generateur.normal(0, np.radians(0.15), 3))
            suivi.predire(dt, mesure if avec_gyro else None)
        ecart = angle_quaternions(suivi.q, verite)
        etiquette = "avec gyro " if avec_gyro else "sans gyro "
        print(f"{etiquette}: apres {duree:.0f} s sans tag, erreur de cap "
              f"{ecart:6.1f} deg   (incertitude annoncee "
              f"{suivi.incertitude_deg:5.1f} deg)")
        if avec_gyro:
            # le biais non estime domine : 0.6 deg/s pendant 6 s = 3.6 deg
            assert ecart < 8.0, f"le gyro doit tenir le cap, obtenu {ecart:.1f} deg"
        else:
            assert ecart > 100.0, "sans gyro on doit avoir tout perdu"

    # -- l'accelerometre borne-t-il roulis et tangage sans aucun tag ? ------
    suivi = FiltreOrientation()
    suivi.demarrer(euler_vers_quaternion(np.radians(12.0), np.radians(-9.0), 0.0),
                   sigma_deg=15.0)
    for _ in range(400):
        suivi.predire(1 / 100, np.zeros(3))
        # engin immobile et horizontal : l'accelerometre voit le haut
        suivi.corriger_gravite(np.array([0.0, 0.0, 9.81])
                               + generateur.normal(0, 0.05, 3))
    roulis, tangage, _ = quaternion_vers_euler(suivi.q)
    print(f"accelerometre seul : roulis {np.degrees(roulis):+.2f} deg, "
          f"tangage {np.degrees(tangage):+.2f} deg  (partis de +12 et -9)")
    assert abs(np.degrees(roulis)) < 2.0 and abs(np.degrees(tangage)) < 2.0

    # une acceleration franche ne doit PAS etre prise pour la pesanteur
    utilisee, _ = suivi.corriger_gravite(np.array([6.0, 0.0, 9.81]))
    assert not utilisee, "une mesure loin de g doit etre refusee"
    print("accelerometre : mesure a 1.2 g refusee, comme attendu")

    # -- le biais du gyro est-il retrouve sur les corrections des tags ? ----
    pose = FiltrePose()
    pose.orientation.demarrer(np.array([1.0, 0.0, 0.0, 0.0]), sigma_deg=2.0)
    verite = np.array([1.0, 0.0, 0.0, 0.0])
    dt = 1 / 100
    for pas in range(6000):
        verite = produit_quaternions(
            verite, quaternion_depuis_rotation(vitesse_vraie * dt))
        pose.orientation.predire(
            dt, vitesse_vraie + biais_vrai + generateur.normal(0, np.radians(0.15), 3))
        if pas % 50 == 0:                      # un tag toutes les 0.5 s
            pose.orientation.corriger(verite, np.radians(1.0))
    erreur_biais = np.degrees(np.linalg.norm(pose.orientation.biais - biais_vrai))
    print(f"biais du gyro : vrai {np.degrees(biais_vrai).round(2)} deg/s, "
          f"estime {pose.orientation.biais_deg_s.round(2)} deg/s "
          f"(erreur {erreur_biais:.2f} deg/s)")
    assert erreur_biais < 0.35, f"le biais doit etre approche, erreur {erreur_biais:.2f}"

    # -- l'accelerometre aide-t-il la position pendant une perte de tags ? --
    dt, duree = 1 / 100, 1.5
    resultats = {}
    for avec_accel in (False, True):
        suivi = FiltreKalmanPosition()
        suivi.demarrer(np.zeros(3), sigma_position=0.01, sigma_vitesse=0.05)
        suivi.x[3:] = [0.25, 0.0, 0.0]
        vraie_p, vraie_v = np.zeros(3), np.array([0.25, 0.0, 0.0])
        # l'engin accelere : c'est le cas ou l'hypothese "vitesse constante"
        # se trompe, et ou l'accelerometre a quelque chose a apporter.
        a = np.array([0.30, -0.15, 0.0])
        for _ in range(int(duree / dt)):
            vraie_p = vraie_p + vraie_v * dt + 0.5 * a * dt * dt
            vraie_v = vraie_v + a * dt
            suivi.predire(dt, (a + generateur.normal(0, 0.05, 3))
                          if avec_accel else None)
        resultats[avec_accel] = float(np.linalg.norm(suivi.position - vraie_p))
    print(f"perte de tags de {duree:.1f} s en pleine acceleration : "
          f"sans accel {1000*resultats[False]:.0f} mm, "
          f"avec accel {1000*resultats[True]:.0f} mm")
    assert resultats[True] < resultats[False] / 3

    # --- accord avec le document de reference ------------------------------
    # L'exemple chiffre de Becker (radar 1D, kalmanfilter.net), passe par le
    # noyau du projet. Ce test protege les cinq equations : si quelqu'un
    # touche a la prediction, au gain ou a la forme de Joseph, l'ecart avec
    # les valeurs publiees le dit immediatement. Le detail commente vit dans
    # kalman_du_cours.py ; ici on garde juste le verrou.
    dt_doc, sigma_doc = 5.0, 0.2
    F_doc = np.array([[1.0, dt_doc], [0.0, 1.0]])
    Q_doc = sigma_doc ** 2 * np.array(
        [[dt_doc ** 4 / 4, dt_doc ** 3 / 2], [dt_doc ** 3 / 2, dt_doc ** 2]])
    ref = KalmanLineaire(np.array([10000.0, 200.0]), np.diag([16.0, 0.25]))
    ref.predire(F_doc, Q_doc)
    assert np.allclose(ref.x, [11000.0, 200.0])
    assert np.allclose(ref.P, [[28.5, 3.75], [3.75, 1.25]])
    K_doc, _ = ref.gain(np.eye(2), np.diag([36.0, 2.25]))
    assert np.allclose(K_doc, [[0.4048, 0.6377], [0.0399, 0.3144]], atol=5e-5)
    ref.corriger(np.array([11020.0, 202.0]), np.eye(2), np.diag([36.0, 2.25]))
    assert np.allclose(ref.x, [11009.37, 201.43], atol=5e-3)
    assert np.allclose(ref.P, [[14.57, 1.43], [1.43, 0.71]], atol=5e-3)
    # Le Q 3D de l'engin est le Q 1D du document, bloc par bloc.
    _, G_doc = FiltreKalmanPosition.modele(dt_doc)
    Q3 = sigma_doc ** 2 * (G_doc @ G_doc.T)
    assert np.isclose(Q3[0, 0], Q_doc[0, 0]) and np.isclose(Q3[0, 3], Q_doc[0, 1])
    assert np.isclose(Q3[3, 3], Q_doc[1, 1])
    print("accord avec Becker (kalmanfilter.net) : les 8 valeurs publiees de "
          "son exemple sont retrouvees")

    print("=" * 68)
    print("TOUS LES TESTS PASSENT")
    print("=" * 68)


if __name__ == "__main__":
    _auto_test()
