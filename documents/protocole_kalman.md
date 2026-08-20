# Protocole — du bassin au filtre de Kalman

Ce qu'il reste a faire APRES la calibration `tube_eau`, dans l'ordre.

Le filtre est deja ecrit et teste (`localisations/filtre_kalman.py`, lancer
`python localisations/filtre_kalman.py` passe 6 auto-tests). Il n'y a donc
rien a coder. Ce qui manque, ce sont **quatre nombres mesures** que le filtre
attend, et qui sont pour l'instant supposes ou mesures en air.

| Nombre | Valeur actuelle | Mesure comment | Etape |
|---|---|---|---|
| focale sous l'eau | **predite** : 792 / 625 px | calibration `tube_eau` | 1 et 2 |
| `SIGMA_PIXEL` | 0.215 px, **mesure en AIR** | `mesurer_bruit_tag.py` | 4 |
| `sigma_acceleration` | 0.4 m/s2, **suppose** | `verification_monde.py` | 5 |
| `derive_gyro_deg_s` | 10 deg/s, **suppose** | `verification_monde.py` | 5 |

---

## Etape 1 — Calibrer sous l'eau

```
python calibration/calibration.py --montage tube_eau
```

Meme deroule que `tube_air` : le damier de 50 mm, une trentaine de vues,
le damier bien present dans les COINS de l'image (c'est la que se lisent les
coefficients de distorsion).

Ce que ca ecrit :
- `calibration/montages/tube_eau.npz` — les 9 parametres
- `calibration/montages/tube_eau_ros.yaml` — pour le noeud de Josiah

Verification immediate :

```
python calibration/optique.py
```

La section CALIBRATIONS ENREGISTREES doit maintenant lister `tube_eau` avec
ses fx et fy mesures. Compare-les aux valeurs **predites** par le modele
optique (fx 792, fy 625, anamorphose 1.27). Un ecart de quelques pourcents
est normal ; un ecart de 30 % veut dire qu'on a mal compris le montage, et
il faut comprendre pourquoi avant d'aller plus loin.

---

## Etape 2 — Basculer sur `tube_eau`

**Une seule ligne**, dans `calibration/optique.py` :

```python
MONTAGE_ACTIF = os.environ.get("UUV_MONTAGE", "tube_air")
                                              ^^^^^^^^^^
                                              mettre "tube_eau"
```

Tous les scripts lisent cette valeur. Il n'y a rien d'autre a editer.

Variante sans rien modifier, pratique au bord du bassin ou pour comparer
deux montages sur la meme manip :

```
UUV_MONTAGE=tube_eau python localisations/verification_monde.py
```

Verifier tout de suite que la bascule a pris :

```
python calibration/optique.py
```

La ligne `MONTAGE ACTIF` en tete du rapport doit afficher `tube_eau`, et
`source` doit valoir `tube_eau` (et non `nue_air`, qui voudrait dire que la
calibration n'a pas ete trouvee).

**Pourquoi ca compte a ce point.** Un script reste sur la focale de l'air
sans jamais le dire : les distances sont fausses de pres d'un tiers et rien
ne clignote. C'est le genre d'erreur qu'on ne decouvre qu'apres des semaines
de mesures. Avant ce reglage il fallait editer dix fichiers et n'en oublier
aucun.

Deux appels sont concernes sans en avoir l'air, et se reglent tout seuls
desormais : `filtre_kalman.py` ligne 184 et `plan_piscine_3d.py` ligne 94
appellent `optique.focale_eau()`. Sur `tube_air`, cette fonction **predit**
la focale sous l'eau par le modele optique ; sur `tube_eau`, elle rend la
valeur **mesuree**. C'est exactement ce qu'on veut, mais seulement si la
bascule a bien eu lieu.

> Deux endroits gardent `tube_air` en dur, et c'est VOULU :
> `calibration.py` et `imprimer_tag.py` comparent deliberement le resultat
> immerge a la reference en air. Ne pas y toucher.

---

## Etape 3 — Poser les tags et faire la carte

Les tags doivent etre **fixes** et leurs positions **connues**. Le filtre
suppose exactement cela ; s'il est trompe la-dessus, aucun reglage ne le
rattrape (voir la section 7 de l'en-tete de `filtre_kalman.py`).

Deux options :
- positions mesurees au metre et entrees a la main ;
- ou auto-enregistrement : `localisations/verification_monde.py`, touche
  `o` sur le tag d'origine, puis se deplacer — les autres tags se relient
  tout seuls des qu'ils sont vus en meme temps qu'un tag deja connu.

Regle qui vient du filtre : **des tags sur des murs DIFFERENTS valent bien
mieux que des tags sur le meme mur.** Un tag est precis lateralement et
mauvais en profondeur ; deux tags perpendiculaires se couvrent mutuellement.
C'est chiffre dans l'auto-test : sur la pire direction, 1 tag donne 0.52 mm,
deux tags perpendiculaires 0.29 mm — gain 1.79x, la ou un simple moyennage
de deux mesures ne donnerait que 1.41x. Le surplus vient de la GEOMETRIE.

---

## Etape 4 — Remesurer le bruit de detection SOUS L'EAU

```
python calibration/mesurer_bruit_tag.py
```

- camera sur un support **stable**
- touche `c` = capture camera immobile, touche `d` = capture en mouvement
- touche `t` = tableau des resultats

La valeur a retenir est celle en **mouvement** (mode `d`) : c'est le regime
reel de l'engin. En air on avait 0.215 px. Sous l'eau ce sera **moins bon**
(turbidite, contraste plus faible, particules).

Reporter le resultat dans `localisations/filtre_kalman.py` ligne 194 :

```python
SIGMA_PIXEL = 0.215   # <- mettre la valeur mesuree dans l'eau
```

---

## Etape 5 — Mesurer la dynamique reelle de l'engin

```
python localisations/verification_monde.py
```

Faire un parcours qui **ressemble a une vraie mission** : les vitesses et
accelerations habituelles, pas une camera posee, pas des mouvements brusques
artificiels. Une trentaine de secondes suffit (la memoire est de 900 images).

Puis `q`. Le script affiche a la fin :

```
DYNAMIQUE OBSERVEE  (a reporter dans filtre_kalman.py)
  derive_gyro_deg_s   = ...
  sigma_acceleration  = ...
```

Ce sont les 95e centiles — assez larges pour couvrir ce que l'engin fait
vraiment, sans se caler sur un pic isole.

Reporter dans `localisations/verification_monde.py` lignes 129, 383 et 395 :

```python
filtre = FiltrePose(sigma_acceleration=0.4, derive_gyro_deg_s=10.0)
```

et dans les valeurs par defaut de `FiltrePose` (`filtre_kalman.py` lignes
658-659) pour que le noeud ROS herite des memes.

**Sens physique**, pour l'expliquer a Thein :
- `sigma_acceleration` = de combien l'engin peut accelerer sans que le filtre
  le sache. Trop petit -> le filtre retarde sur les virages. Trop grand ->
  il ne lisse plus rien.
- `derive_gyro_deg_s` = a quelle vitesse l'orientation peut changer entre
  deux images sans mesure.

---

## Etape 6 — Valider que le filtre ameliore vraiment

Toujours dans `verification_monde.py` : la touche `f` allume et coupe le
filtre, et l'ecran affiche la position **brute** et la position **filtree**
l'une sous l'autre.

Le test : deplacer la camera d'une distance connue au metre, taper la valeur
reelle au clavier, `s` pour enregistrer. Refaire une dizaine de fois, filtre
allume puis filtre coupe. Tout est journalise dans `verification_monde.csv`.

**Ce qu'on veut voir** : l'erreur RMS filtree nettement en dessous de
l'erreur RMS brute. Si le filtre n'ameliore pas, c'est que
`sigma_acceleration` est mal regle — c'est le symptome typique.

N'annonce PAS de gain chiffre a l'avance. L'auto-test affiche 33x, mais
c'est une simulation ou le bruit est exactement celui que le filtre suppose
et ou la trajectoire est a vitesse constante — les deux hypotheses du filtre
sont vraies par construction. Dans le bassin ce sera nettement moins. Le
chiffre honnete est celui que TU mesureras a cette etape.

Verifier aussi le compteur de rejets. Quelques rejets sont sains (le filtre
attrape les retournements de tag). Des centaines veulent dire que le modele
de bruit est trop optimiste.

---

## Etape 7 — Surveiller que les tags ne bougent pas

Ca tourne tout seul dans `FiltrePose.surveillance`. A la fin d'une session :

```python
print(filtre.surveillance.rapport())
```

Le module dit **quel** support a bouge, **de combien** et **dans quelle
direction** — de quoi corriger la carte sans tout re-enregistrer.

Pourquoi ca compte : l'erreur du systeme apres filtrage est de l'ordre de
2 mm. Une boite lestee decalee de 1 cm pese a elle seule cinq fois tout le
reste du budget d'erreur. Un tag qui bouge produit un **biais**, et un filtre
de Kalman ne sait traiter que du bruit centre : il moyenne le bruit, mais il
**suit** le biais.

Limite a annoncer honnetement : un tag vu **seul** n'est jamais mis en
defaut. Rien ne distingue alors « la camera a bouge » de « le tag a bouge ».

---

## Recapitulatif — une seule session de bassin

Les etapes 1, 3, 4, 5 et 6 se font toutes dans l'eau. Ordre conseille sur
place :

1. Calibration `tube_eau` avec le damier (etape 1)
2. Poser les tags, faire la carte (etape 3)
3. Bruit de detection, mode `d` (etape 4)
4. Parcours type mission, relever les deux chiffres (etape 5)
5. Une dizaine de mesures brut / filtre a distance connue (etape 6)

L'etape 2 (basculer les scripts) se fait **entre** 1 et 3, au bord du bassin,
sur le portable. Sans elle tout le reste est mesure avec la mauvaise focale.
