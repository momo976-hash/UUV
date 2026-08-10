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
from collections import defaultdict, deque

import numpy as np

# Valeurs par defaut calees sur la D435i a 640x480 derriere un hublot plat.
FOCALE_EAU = 604.1876 * 1.33
TAILLE_TAG = 0.223

# MESURE, plus supposee : 14 captures camera en main a 0.72 - 1.52 m
# (calibration/mesurer_bruit_tag.py, mode 'd'). La valeur precedente, 0.5 px,
# etait une valeur d'usage courante en vision, jamais verifiee ici.
#   camera posee    0.049 px  <- plancher, sans flou de bouge
#   camera qui bouge 0.215 px <- valeur d'usage, retenue ici
# Ces mesures sont faites EN AIR. A refaire dans le bassin : l'eau trouble et
# le moindre contraste degraderont ce chiffre.
SIGMA_PIXEL = 0.215

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
# Filtre de position : lineaire, exactement le Kalman du cours
# ===========================================================================
class FiltreKalmanPosition:
    """Modele a vitesse constante, mesure de position seule."""

    H = np.hstack([np.eye(3), np.zeros((3, 3))])

    def __init__(self, sigma_acceleration=0.5, seuil_chi2=16.27,
                 max_rejets_consecutifs=5, sigma_vitesse_reprise=0.5):
        self.sigma_a = float(sigma_acceleration)   # m/s^2 d'acceleration non modelisee
        self.seuil = float(seuil_chi2)             # chi2 a 3 ddl, seuil 99.9 %
        self.max_rejets_consecutifs = int(max_rejets_consecutifs)
        self.sigma_v_reprise = float(sigma_vitesse_reprise)
        self.x = np.zeros(6)
        self.P = np.eye(6) * 1e3
        self.demarre = False
        self.rejets = 0
        self.rejets_consecutifs = 0
        self.reprises = 0

    def demarrer(self, position, sigma_position=0.05, sigma_vitesse=0.5):
        self.x = np.concatenate([np.asarray(position, dtype=float), np.zeros(3)])
        self.P = np.diag([sigma_position ** 2] * 3 + [sigma_vitesse ** 2] * 3)
        self.demarre = True

    def predire(self, dt):
        """Fait avancer l'etat de dt secondes, sans mesure."""
        if not self.demarre or dt <= 0:
            return
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        # G : effet d'une acceleration inconnue pendant dt (cf. en-tete)
        G = np.vstack([0.5 * dt * dt * np.eye(3), dt * np.eye(3)])
        Q = self.sigma_a ** 2 * (G @ G.T)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def corriger(self, position_mesuree, covariance):
        """Integre une mesure. Retourne (acceptee, distance_mahalanobis)."""
        if not self.demarre:
            self.demarrer(position_mesuree)
            return True, 0.0

        z = np.asarray(position_mesuree, dtype=float)
        R = np.asarray(covariance, dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        S_inv = np.linalg.inv(S)
        distance = float(y @ S_inv @ y)

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
        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ y
        # forme de Joseph : garde P symetrique et definie positive meme
        # apres des milliers d'iterations en virgule flottante.
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
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

    def __init__(self, derive_gyro_deg_s=2.0, seuil_saut_deg=25.0,
                 max_rejets_consecutifs=5):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.variance = np.radians(180.0) ** 2
        self.derive = np.radians(derive_gyro_deg_s)   # rad/s d'errance non modelisee
        self.seuil_saut = float(seuil_saut_deg)
        self.max_rejets_consecutifs = int(max_rejets_consecutifs)
        self.demarre = False
        self.rejets = 0
        self.rejets_consecutifs = 0
        self.reprises = 0

    def demarrer(self, R_ou_q, sigma_deg=5.0):
        q = np.asarray(R_ou_q, dtype=float)
        self.q = matrice_vers_quaternion(q) if q.shape == (3, 3) else q / np.linalg.norm(q)
        self.variance = np.radians(sigma_deg) ** 2
        self.demarre = True

    def predire(self, dt):
        """Sans gyroscope, l'orientation est supposee constante et son
        incertitude grandit avec le temps."""
        if not self.demarre or dt <= 0:
            return
        self.variance += (self.derive * dt) ** 2

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
        self.q = slerp(self.q, q, gain)
        self.variance = (1.0 - gain) * self.variance
        return True, ecart

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
        filtre.predire(dt)
        for tag in tags_vus:
            filtre.ajouter_tag(position_estimee, position_tag, incidence, R_mesuree)
        filtre.appliquer()
    """

    def __init__(self, sigma_acceleration=0.5, derive_gyro_deg_s=2.0,
                 seuil_deplacement_mm=8.0):
        self.position = FiltreKalmanPosition(sigma_acceleration)
        self.orientation = FiltreOrientation(derive_gyro_deg_s)
        self.surveillance = SurveillanceTags(seuil_mm=seuil_deplacement_mm)
        self._mesures = []
        self._orientations = []

    def predire(self, dt):
        self.position.predire(dt)
        self.orientation.predire(dt)

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

    print("=" * 68)
    print("TOUS LES TESTS PASSENT")
    print("=" * 68)


if __name__ == "__main__":
    _auto_test()
