"""graphiques_kalman.py — Les figures du cours de Kalman, tracees EN DIRECT.

    Ce module ne se lance pas seul. Il s'ouvre depuis une manip reelle :

        python localisations/verification_monde.py --graphiques

    Une fenetre s'ouvre a cote de la video et se met a jour a chaque image,
    avec les VRAIES mesures de la camera. Fermer la fenetre n'arrete pas la
    mesure ; quitter la mesure par 'q' ferme tout.

POURQUOI CE MODULE
Le document de reference du projet — Alex Becker, « Kalman Filter Explained
Through Examples », kalmanfilter.net — explique le filtre par une poignee de
figures. Le depot savait deja les produire en SIMULATION
(demos/demo_kalman.py), ce qui prouve que les maths sont justes mais ne
montre rien du systeme reel. Ici les memes figures sont tracees sur ce que la
camera mesure a l'instant meme.

LES SIX FIGURES, ET CE QUE CHACUNE PROUVE

  1. MISE A JOUR BAYESIENNE — a priori, vraisemblance, a posteriori
     Les trois gaussiennes de la figure centrale du document. L'a priori est
     ce que le filtre croyait AVANT la mesure ; la vraisemblance est ce que
     le tag vient de dire ; l'a posteriori est le compromis. Ce qu'il faut y
     voir : l'a posteriori est TOUJOURS plus etroit que les deux autres, et
     tombe entre eux. C'est la propriete fondamentale du filtre, et elle se
     verifie a l'oeil, image par image.

  2. ESTIMEE vs MESURE dans le temps, avec le couloir +/- 1 sigma
     La figure de suivi du document. La courbe filtree doit etre plus lisse
     que les mesures brutes, et le couloir doit CONTENIR les mesures a peu
     pres deux fois sur trois — c'est ce que « 1 sigma » veut dire. Un
     couloir qui ne contient rien est un filtre qui se ment a lui-meme.

  3. INCERTITUDE dans le temps
     Elle diminue quand les mesures arrivent, et REMONTE des qu'elles
     s'arretent. C'est la que se voit ce qu'apporte la centrale : sans elle
     l'incertitude explose des la premiere seconde sans tag.

  4. GAIN DE KALMAN dans le temps
     K = 0 : le filtre ignore la mesure et ne croit que sa prediction.
     K = 1 : il jette sa prediction et croit la mesure sur parole.
     Entre les deux, c'est l'arbitrage. Il doit se stabiliser apres quelques
     secondes ; s'il reste colle a 1, le filtre ne filtre rien.

  5. QUALITE DE CHAQUE TAG — l'equivalent du « SNR » du document
     Le document illustre la qualite d'une mesure par un echo radar fort ou
     faible dans le bruit. Ici la mesure ne vient pas d'un echo mais d'un
     tag, et sa qualite se lit sur la GEOMETRIE : sigma lateral croit comme
     la distance, sigma en PROFONDEUR comme son CARRE, et se degrade encore
     quand le tag est vu de biais. Les points noirs sont les tags reellement
     visibles a cet instant. C'est la justification chiffree du filtre : un
     seul tag, loin ou de biais, ne suffit pas.

  6. TEST D'ABERRATION — le « Outlier Treatment » du document
     Chaque mesure est comparee a ce que le filtre attendait ; l'ecart
     normalise suit une loi du chi2 a 3 degres de liberte. Au-dela du seuil
     la mesure est rejetee. C'est ainsi qu'un retournement de tag est attrape
     avant d'empoisonner l'estimee.

CE QU'IL FAUT REGARDER, ET CE QU'IL NE FAUT PAS CROIRE
Aucune de ces figures ne dit si la position est JUSTE — il n'y a pas de
verite terrain dans une manip reelle. Elles disent si le filtre se comporte
comme un filtre de Kalman doit se comporter. Pour savoir s'il dit vrai, il
faut le confronter au metre a ruban : c'est l'etape 6 du protocole.
"""
import sys
from collections import deque

import numpy as np

# matplotlib est facultatif : sans lui la manip doit continuer sans graphique
# plutot que de refuser de demarrer. Une mesure au bassin ne se refait pas
# parce qu'il manque une bibliotheque d'affichage.
try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:                                    # pragma: no cover
    matplotlib = plt = None

MEMOIRE = 600            # points gardes dans les courbes (20 s a 30 Hz)
PERIODE_RAFRAICHISSEMENT = 5     # images entre deux redessins
AXES = ("x", "y", "z")


def disponible():
    """matplotlib est-il installe ?"""
    return plt is not None


class FenetresKalman:
    """Les quatre figures du cours, alimentees par un FiltrePose reel.

    Usage :
        fenetres = FenetresKalman(axe=0)
        ...
        fenetres.ajouter(instant, filtre, mesure_brute)   # a chaque image
        fenetres.rafraichir()                             # idem, throttle
        ...
        fenetres.fermer()
    """

    def __init__(self, axe=0, titre="Filtre de Kalman en direct",
                 sous_titre="mesures REELLES de la camera"):
        if not disponible():
            raise RuntimeError("matplotlib n'est pas installe")
        self.axe = int(axe)
        self.compteur = 0
        self.t = deque(maxlen=MEMOIRE)
        self.brut = deque(maxlen=MEMOIRE)
        self.filtre = deque(maxlen=MEMOIRE)
        self.sigma = deque(maxlen=MEMOIRE)
        self.gain = deque(maxlen=MEMOIRE)
        self.t_gain = deque(maxlen=MEMOIRE)
        self.t_mesure = deque(maxlen=MEMOIRE)
        self.rejets_t = deque(maxlen=MEMOIRE)
        self.rejets_v = deque(maxlen=MEMOIRE)
        self.maha = deque(maxlen=MEMOIRE)
        self.t_maha = deque(maxlen=MEMOIRE)
        self.seuil_maha = None
        self.tags_vus = []          # [(id, distance_m, incidence_deg)] courant
        self.dernier = None

        plt.ion()
        self.figure, grille = plt.subplots(2, 3, figsize=(17, 8))
        self.figure.canvas.manager.set_window_title(titre)
        self.figure.suptitle(f"{titre} — axe {AXES[self.axe]} — {sous_titre}",
                             fontsize=12, fontweight="bold")
        ((self.ax_bayes, self.ax_suivi, self.ax_qualite),
         (self.ax_sigma, self.ax_gain, self.ax_maha)) = grille

        self.ax_bayes.set_title("1. Mise a jour bayesienne (instant courant)")
        self.ax_bayes.set_xlabel(f"position {AXES[self.axe]} (m)")
        self.ax_bayes.set_ylabel("densite de probabilite")

        self.ax_suivi.set_title("2. Estimee vs mesure brute")
        self.ax_suivi.set_xlabel("temps (s)")
        self.ax_suivi.set_ylabel(f"position {AXES[self.axe]} (m)")

        self.ax_sigma.set_title("3. Incertitude annoncee (1 sigma)")
        self.ax_sigma.set_xlabel("temps (s)")
        self.ax_sigma.set_ylabel("sigma (mm)")

        self.ax_gain.set_title("4. Gain de Kalman")
        self.ax_gain.set_xlabel("temps (s)")
        self.ax_gain.set_ylabel("K (0 = ignore la mesure, 1 = la croit)")
        self.ax_gain.set_ylim(-0.05, 1.05)

        self.ax_qualite.set_title("5. Qualite de chaque tag (le « SNR » du cours)")
        self.ax_qualite.set_xlabel("distance du tag (m)")
        self.ax_qualite.set_ylabel("sigma de la mesure (mm)")

        self.ax_maha.set_title("6. Test d'aberration")
        self.ax_maha.set_xlabel("temps (s)")
        self.ax_maha.set_ylabel("distance de Mahalanobis^2")

        self.figure.tight_layout()
        self.figure.canvas.draw()
        plt.show(block=False)

    # -- collecte -----------------------------------------------------------
    def ajouter(self, instant, filtre_pose, mesure_brute=None, tags=None):
        """Enregistre l'etat du filtre a cet instant. A appeler chaque image.

        tags : [(identifiant, distance_m, incidence_deg)] des tags visibles,
        pour le panneau 5. Facultatif : sans lui ce panneau reste vide.
        """
        position = filtre_pose.position
        if tags is not None:
            self.tags_vus = list(tags)
        if not position.demarre:
            return
        self.t.append(instant)
        self.filtre.append(float(position.x[self.axe]))
        self.sigma.append(1000.0 * float(np.sqrt(position.P[self.axe, self.axe])))
        if mesure_brute is not None:
            self.t_mesure.append(instant)
            self.brut.append(float(np.asarray(mesure_brute).ravel()[self.axe]))

        # `dernier_calcul` PERSISTE entre deux images : sans cette comparaison
        # d'identite, une seule mesure rejetee serait recomptee a chaque image
        # suivante jusqu'a la mesure d'apres. Le filtre en creant un dict neuf
        # a chaque mise a jour, `is not` suffit et ne peut pas se tromper.
        calcul = position.dernier_calcul
        if (calcul is not None and "x_apres" in calcul
                and calcul is not self.dernier):
            self.dernier = calcul
            self.gain.append(float(calcul["K"][self.axe, self.axe]))
            self.t_gain.append(instant)
            self.maha.append(float(calcul["distance"]))
            self.t_maha.append(instant)
            self.seuil_maha = float(calcul["seuil"])
            if not calcul.get("acceptee", True):
                self.rejets_t.append(instant)
                self.rejets_v.append(float(calcul["z"][self.axe]))

    # -- affichage ----------------------------------------------------------
    def rafraichir(self, force=False):
        """Redessine, au plus une fois toutes les PERIODE_RAFRAICHISSEMENT."""
        self.compteur += 1
        if not force and self.compteur % PERIODE_RAFRAICHISSEMENT:
            return
        if not self.t:
            return
        try:
            self._tracer_bayes()
            self._tracer_suivi()
            self._tracer_sigma()
            self._tracer_gain()
            self._tracer_qualite()
            self._tracer_maha()
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
        except Exception:
            # Une fenetre fermee a la main ne doit pas faire tomber la mesure.
            pass

    def _tracer_bayes(self):
        self.ax_bayes.clear()
        self.ax_bayes.set_title("1. Mise a jour bayesienne (instant courant)")
        self.ax_bayes.set_xlabel(f"position {AXES[self.axe]} (m)")
        self.ax_bayes.set_ylabel("densite de probabilite")
        calcul = self.dernier
        if calcul is None:
            self.ax_bayes.text(0.5, 0.5, "en attente d'une mesure",
                               ha="center", transform=self.ax_bayes.transAxes)
            return
        i = self.axe
        lois = [
            ("a priori P(x)", calcul["x_avant"][i],
             np.sqrt(calcul["P_avant"][i, i]), "green"),
            ("vraisemblance P(z|x)", calcul["z"][i],
             np.sqrt(calcul["R"][i, i]), "red"),
            ("a posteriori P(x|z)", calcul["x_apres"][i],
             np.sqrt(calcul["P_apres"][i, i]), "blue"),
        ]
        etendue = max(s for _, _, s, _ in lois)
        centre = np.mean([m for _, m, _, _ in lois])
        grille = np.linspace(centre - 4 * etendue, centre + 4 * etendue, 400)
        for nom, moyenne, ecart, couleur in lois:
            ecart = max(float(ecart), 1e-9)
            densite = (np.exp(-0.5 * ((grille - moyenne) / ecart) ** 2)
                       / (ecart * np.sqrt(2 * np.pi)))
            self.ax_bayes.plot(grille, densite, color=couleur, lw=2, label=nom)
            if couleur == "blue":
                self.ax_bayes.fill_between(grille, densite, alpha=0.2,
                                           color="blue")
        if not calcul.get("acceptee", True):
            self.ax_bayes.set_title(
                "1. Mise a jour bayesienne — MESURE REJETEE (aberration)",
                color="crimson")
        self.ax_bayes.legend(fontsize=8)
        self.ax_bayes.grid(alpha=0.3)

    def _tracer_suivi(self):
        self.ax_suivi.clear()
        self.ax_suivi.set_title("2. Estimee vs mesure brute")
        self.ax_suivi.set_xlabel("temps (s)")
        self.ax_suivi.set_ylabel(f"position {AXES[self.axe]} (m)")
        t = np.fromiter(self.t, float)
        f = np.fromiter(self.filtre, float)
        s = np.fromiter(self.sigma, float) / 1000.0
        if len(self.brut):
            self.ax_suivi.plot(np.fromiter(self.t_mesure, float),
                               np.fromiter(self.brut, float), ".",
                               color="orange", ms=4, alpha=0.6,
                               label="mesure brute (tags)")
        self.ax_suivi.plot(t, f, "-", color="green", lw=1.8,
                           label="sortie du filtre")
        self.ax_suivi.fill_between(t, f - s, f + s, color="green", alpha=0.18,
                                   label="+/- 1 sigma annonce")
        if len(self.rejets_t):
            self.ax_suivi.plot(np.fromiter(self.rejets_t, float),
                               np.fromiter(self.rejets_v, float), "x",
                               color="crimson", ms=7, label="mesure rejetee")
        self.ax_suivi.legend(fontsize=8, loc="best")
        self.ax_suivi.grid(alpha=0.3)

    def _tracer_sigma(self):
        self.ax_sigma.clear()
        self.ax_sigma.set_title("3. Incertitude annoncee (1 sigma)")
        self.ax_sigma.set_xlabel("temps (s)")
        self.ax_sigma.set_ylabel("sigma (mm)")
        self.ax_sigma.plot(np.fromiter(self.t, float),
                           np.fromiter(self.sigma, float),
                           "-", color="steelblue", lw=1.8)
        self.ax_sigma.grid(alpha=0.3)
        self.ax_sigma.text(
            0.02, 0.92, "descend quand les tags arrivent, remonte sans eux",
            transform=self.ax_sigma.transAxes, fontsize=8, color="gray")

    def _tracer_gain(self):
        self.ax_gain.clear()
        self.ax_gain.set_title("4. Gain de Kalman")
        self.ax_gain.set_xlabel("temps (s)")
        self.ax_gain.set_ylabel("K (0 = ignore la mesure, 1 = la croit)")
        self.ax_gain.set_ylim(-0.05, 1.05)
        if len(self.gain):
            self.ax_gain.plot(np.fromiter(self.t_gain, float),
                              np.fromiter(self.gain, float),
                              "-", color="purple", lw=1.8)
        self.ax_gain.grid(alpha=0.3)

    def _tracer_qualite(self):
        """Panneau 5 — l'equivalent, pour ce systeme, du « SNR » du cours.

        Le document illustre la qualite d'une mesure par un echo radar fort ou
        faible dans le bruit. Ici la mesure ne vient pas d'un echo mais d'un
        tag, et sa qualite se lit sur la GEOMETRIE : un tag loin ou vu de biais
        renseigne mal. C'est ce que calcule covariance_position_tag, et les
        deux courbes tracees sont ses deux termes :

            sigma lateral    = d . sigma_px / f            croit comme d
            sigma profondeur = d^2 . sigma_px / (f.T.cos(i).2)   croit comme d^2

        D'ou le fait, contre-intuitif, que la profondeur se degrade BEAUCOUP
        plus vite que le laterale : a 3 m elle est deja des dizaines de fois
        pire. C'est la raison d'etre du filtre — un seul tag ne suffit pas.
        """
        self.ax_qualite.clear()
        self.ax_qualite.set_title("5. Qualite de chaque tag (le « SNR » du cours)")
        self.ax_qualite.set_xlabel("distance du tag (m)")
        self.ax_qualite.set_ylabel("sigma de la mesure (mm)")
        try:
            from filtre_kalman import (FOCALE_EAU, TAILLE_TAG, SIGMA_PIXEL,
                                       COINS_PAR_TAG)
        except Exception:
            return
        distances = np.linspace(0.3, 4.0, 120)
        lateral = 1000 * distances * SIGMA_PIXEL / FOCALE_EAU
        self.ax_qualite.plot(distances, lateral, "-", color="seagreen", lw=2,
                             label="lateral (croit en d)")
        for incidence, style in ((0.0, "-"), (45.0, "--")):
            cos_i = max(np.cos(np.radians(incidence)), 0.20)
            profondeur = (1000 * distances ** 2 * SIGMA_PIXEL
                          / (FOCALE_EAU * TAILLE_TAG * cos_i
                             * np.sqrt(COINS_PAR_TAG)))
            self.ax_qualite.plot(distances, profondeur, style, color="indianred",
                                 lw=2, label=f"profondeur, biais {incidence:.0f} deg")
        # Les tags REELLEMENT vus a cet instant, poses sur ces courbes.
        for identifiant, distance, incidence in self.tags_vus:
            cos_i = max(np.cos(np.radians(incidence)), 0.20)
            sigma = (1000 * distance ** 2 * SIGMA_PIXEL
                     / (FOCALE_EAU * TAILLE_TAG * cos_i * np.sqrt(COINS_PAR_TAG)))
            self.ax_qualite.plot([distance], [sigma], "o", color="black", ms=8,
                                 zorder=5)
            self.ax_qualite.annotate(f" tag {identifiant}", (distance, sigma),
                                     fontsize=8, va="bottom")
        self.ax_qualite.set_yscale("log")
        self.ax_qualite.legend(fontsize=7, loc="upper left")
        self.ax_qualite.grid(alpha=0.3, which="both")

    def _tracer_maha(self):
        """Panneau 6 — le « Outlier Treatment » du cours, en direct.

        Chaque mesure est comparee a ce que le filtre attendait. L'ecart,
        normalise par l'incertitude des deux (distance de Mahalanobis), suit
        une loi du chi2 a 3 degres de liberte : au-dela du seuil, la mesure est
        trop improbable pour etre vraie et se fait rejeter. C'est ainsi qu'un
        retournement de tag est attrape avant d'empoisonner l'estimee.
        """
        self.ax_maha.clear()
        self.ax_maha.set_title("6. Test d'aberration")
        self.ax_maha.set_xlabel("temps (s)")
        self.ax_maha.set_ylabel("distance de Mahalanobis^2")
        if len(self.maha):
            t = np.fromiter(self.t_maha, float)
            d = np.fromiter(self.maha, float)
            self.ax_maha.plot(t, d, ".-", color="darkorange", lw=1, ms=4,
                              label="ecart mesure / prediction")
            if self.seuil_maha:
                self.ax_maha.axhline(self.seuil_maha, color="crimson", ls="--",
                                     lw=1.5, label=f"seuil ({self.seuil_maha:.1f})")
                depasse = d > self.seuil_maha
                if depasse.any():
                    self.ax_maha.plot(t[depasse], d[depasse], "x",
                                      color="crimson", ms=8, label="rejetee")
            self.ax_maha.set_yscale("symlog")
            self.ax_maha.legend(fontsize=7, loc="upper left")
        self.ax_maha.grid(alpha=0.3)

    # -- fin ----------------------------------------------------------------
    def enregistrer(self, chemin):
        """Sauve la figure courante, pour la joindre a un rapport."""
        try:
            self.figure.savefig(chemin, dpi=130, bbox_inches="tight")
            return True
        except Exception:
            return False

    def fermer(self):
        try:
            plt.close(self.figure)
        except Exception:
            pass


def _demonstration():
    """Rejoue le module sur des donnees simulees, sans camera.

    Sert a verifier que les quatre figures se tracent et se mettent a jour —
    la manip reelle demande la camera, ce controle-la non.
    """
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    from filtre_kalman import FiltrePose, covariance_position_tag

    if not disponible():
        print("matplotlib n'est pas installe :")
        print("    python -m pip install matplotlib")
        return 1
    matplotlib.use("Agg")          # pas de fenetre : on verifie le trace seul

    generateur = np.random.default_rng(4)
    filtre = FiltrePose()
    fenetres = FenetresKalman(
        axe=0, titre="Demonstration",
        sous_titre="DONNEES SIMULEES, pas une mesure reelle")
    dt = 1 / 30
    p = np.zeros(3)
    tag = np.array([2.0, 0.0, 0.0])
    # Le bruit injecte est TIRE DE LA COVARIANCE DU MODELE, pas choisi au
    # hasard. Un premier essai injectait 8 mm sur les trois axes alors que le
    # modele en attend 3.2 en profondeur et 0.7 lateralement : 51 mesures sur
    # 80 partaient au rejet, ce qui donnait l'illusion d'un filtre paranoiaque
    # alors que c'etait la simulation qui mentait.
    # On longe le tag au lieu d'aller dessus : un premier essai avancait de
    # 2 m vers un tag place a 2 m, la camera finissait donc DANS le tag, ou
    # la covariance degenere et le gain part a 1.000. La distance doit rester
    # realiste d'un bout a l'autre.
    for k in range(240):
        p = p + np.array([0.0, 0.25, 0.0]) * dt
        filtre.predire(dt)
        mesure = None
        if k % 3 == 0:
            R = covariance_position_tag(p, tag, 15.0)
            mesure = p + generateur.multivariate_normal(np.zeros(3), R)
            if k == 120:                      # une aberration, pour la voir
                mesure = mesure + np.array([0.4, 0.0, 0.0])
            filtre.ajouter_tag(mesure, tag, 15.0)
            filtre.appliquer()
        fenetres.ajouter(k * dt, filtre, mesure,
                         tags=[(0, float(np.linalg.norm(tag - p)), 15.0)])
        fenetres.rafraichir()
    fenetres.rafraichir(force=True)

    sortie = __import__("pathlib").Path(__file__).resolve().with_name(
        "graphiques_kalman_demo.png")
    ok = fenetres.enregistrer(sortie)
    fenetres.fermer()
    print("Les quatre figures se sont tracees sur 240 images simulees.")
    print(f"  points de courbe    : {len(fenetres.t)}")
    print(f"  mesures rejetees    : {len(fenetres.rejets_t)}")
    print(f"  gain final          : {fenetres.gain[-1]:.3f}")
    print(f"  sigma final         : {fenetres.sigma[-1]:.1f} mm")
    if ok:
        print(f"  image enregistree   : {sortie}")
    print("\nPour les voir EN DIRECT sur la vraie camera :")
    print("    python localisations/verification_monde.py --graphiques")
    return 0


if __name__ == "__main__":
    sys.exit(_demonstration())
