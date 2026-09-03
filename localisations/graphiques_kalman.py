"""graphiques_kalman.py — Les figures du cours de Kalman, tracees EN DIRECT.

    Ce module ne se lance pas seul. Il s'ouvre depuis une manip reelle :

        python localisations/verification_monde.py --graphiques

    Une fenetre s'ouvre a cote de la video et se met a jour a chaque image,
    avec les VRAIES mesures de la camera. Fermer la fenetre n'arrete pas la
    mesure ; quitter la mesure par 'q' ferme tout.

POURQUOI CE MODULE
Le document de reference du projet — Alex Becker, « Kalman Filter Explained
Through Examples » — explique le filtre par quatre figures. Le depot savait
deja les produire en SIMULATION (demos/demo_kalman.py), ce qui prouve que
les maths sont justes mais ne montre rien du systeme reel. Ici les memes
figures sont tracees sur ce que la camera mesure a l'instant meme.

LES QUATRE FIGURES, ET CE QUE CHACUNE PROUVE

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
        self.dernier = None

        plt.ion()
        self.figure, grille = plt.subplots(2, 2, figsize=(13, 7.5))
        self.figure.canvas.manager.set_window_title(titre)
        self.figure.suptitle(f"{titre} — axe {AXES[self.axe]} — {sous_titre}",
                             fontsize=12, fontweight="bold")
        (self.ax_bayes, self.ax_suivi), (self.ax_sigma, self.ax_gain) = grille

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

        self.figure.tight_layout()
        self.figure.canvas.draw()
        plt.show(block=False)

    # -- collecte -----------------------------------------------------------
    def ajouter(self, instant, filtre_pose, mesure_brute=None):
        """Enregistre l'etat du filtre a cet instant. A appeler chaque image."""
        position = filtre_pose.position
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
        fenetres.ajouter(k * dt, filtre, mesure)
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
