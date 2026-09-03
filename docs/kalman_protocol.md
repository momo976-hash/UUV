# Protocole — du bassin au filtre de Kalman

Ce qu'il reste a faire APRES la calibration `tube_eau`, dans l'ordre.

## Document de reference

Le filtre suit **Alex Becker, « Kalman Filter Explained Through Examples »**,
kalmanfilter.net — modele **cinematique a vitesse constante**, c'est celui
demande par Thein.

Ce n'est pas une inspiration lointaine : c'est le meme filtre. Pour le
verifier, lancer

```
python kalman/kalman_reference_check.py
```

Ce script fait passer l'exemple chiffre du document (un radar 1D qui suit un
avion) dans la classe qui filtre reellement la position de l'engin, et
compare les 9 valeurs publiees — `Q`, `x(1,0)`, `P(1,0)`, `K(1)`, `x(1,1)`,
`P(1,1)`, `x(2,1)`, `P(2,1)` — a celles calculees. Elles sont retrouvees a la
quatrieme decimale. Le meme controle tourne en verrou dans les auto-tests de
`kalman_filter.py` : toucher aux equations le fait echouer immediatement.

Le filtre est deja ecrit et teste (`kalman/kalman_filter.py`, lancer
`python kalman/kalman_filter.py` passe 6 auto-tests). Il n'y a donc
rien a coder. Ce qui manque, ce sont **quatre nombres mesures** que le filtre
attend, et qui sont pour l'instant supposes ou mesures en air.

| Nombre | Valeur actuelle | Mesure comment | Etape |
|---|---|---|---|
| focale sous l'eau | **predite** : 792 / 625 px | calibration `tube_eau` | 1 et 2 |
| `SIGMA_PIXEL` | 0.215 px, **mesure en AIR** | `measure_tag_noise.py` | 4 |
| `SIGMA_ACCELERATION` | 0.4 m/s2, **suppose** | `world_frame_check.py` | 5 |
| `DERIVE_GYRO_DEG_S` | 10 deg/s, **suppose** | `world_frame_check.py` | 5 |

## En tout : 3 lignes a changer, dans 1 seul fichier

Rien d'autre. Les scripts qui les utilisent vont tous y puiser.

**Le montage (etape 2) ne se change plus a la main du tout.** Il se regle
une fois par ordinateur, avec une commande :

```
python calibration/set_mounting.py
```

**`kalman/kalman_filter.py`** — bloc « LES TROIS NOMBRES A
MESURER », vers la ligne 190 (etapes 4 et 5)

```python
SIGMA_PIXEL        = 0.215
SIGMA_ACCELERATION = 0.4
DERIVE_GYRO_DEG_S  = 10.0
```

Les scripts qui mesurent ces valeurs **affichent la ligne exacte a recopier**
en fin de session. Tu n'as pas a chercher ou ca va.

---

## Etape 1 — Calibrer sous l'eau

```
python calibration/calibrate.py --montage tube_eau
```

Meme deroule que `tube_air` : le damier de 50 mm, une trentaine de vues,
le damier bien present dans les COINS de l'image (c'est la que se lisent les
coefficients de distorsion).

Ce que ca ecrit :
- `calibration/montages/tube_eau.npz` — les 9 parametres
- `calibration/montages/tube_eau_ros.yaml` — pour le noeud de Josiah

Verification immediate :

```
python calibration/optics.py
```

La section CALIBRATIONS ENREGISTREES doit maintenant lister `tube_eau` avec
ses fx et fy mesures. Compare-les aux valeurs **predites** par le modele
optics (fx 792, fy 625, anamorphose 1.27). Un ecart de quelques pourcents
est normal ; un ecart de 30 % veut dire qu'on a mal compris le montage, et
il faut comprendre pourquoi avant d'aller plus loin.

---

## Etape 2 — Basculer sur `tube_eau`

**Aucun fichier Python a editer.** Sur l'ordinateur qui va mesurer :

```
python calibration/set_mounting.py tube_eau
```

C'est a faire **une fois par machine**, pas une fois par manip. Le reglage
est ecrit dans `calibration/montage_local.txt`, qui n'est **pas** versionne :
le PC du bord du bassin reste sur `tube_eau` et le portable de bureau sur
`nue_air`, sans que l'un vienne deregler l'autre au prochain `git pull`.

Si personne n'a encore repondu sur cette machine, le premier script lance
pose la question tout seul et retient la reponse. Il n'y a donc rien a se
transmettre par message.

Pour verifier a tout moment :

```
python calibration/set_mounting.py --montrer
```

Variante le temps d'une seule commande, pour comparer deux montages sur la
meme manip sans rien deregler :

```
UUV_MONTAGE=tube_air python localization/world_frame_check.py
```

Verifier tout de suite que la bascule a pris :

```
python calibration/optics.py
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
desormais : `kalman_filter.py` ligne 184 et `pool_layout_3d.py` ligne 94
appellent `optics.focale_eau()`. Sur `tube_air`, cette fonction **predit**
la focale sous l'eau par le modele optics ; sur `tube_eau`, elle rend la
valeur **mesuree**. C'est exactement ce qu'on veut, mais seulement si la
bascule a bien eu lieu.

> Deux endroits gardent `tube_air` en dur, et c'est VOULU :
> `calibrate.py` et `print_tag.py` comparent deliberement le resultat
> immerge a la reference en air. Ne pas y toucher.

---

## Etape 3 — Poser les tags et faire la carte

Les tags doivent etre **fixes** et leurs positions **connues**. Le filtre
suppose exactement cela ; s'il est trompe la-dessus, aucun reglage ne le
rattrape (voir la section 7 de l'en-tete de `kalman_filter.py`).

Deux options :
- positions mesurees au metre et entrees a la main ;
- ou auto-enregistrement : `localization/world_frame_check.py`, touche
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
python calibration/measure_tag_noise.py
```

- camera sur un support **stable**
- touche `c` = capture camera immobile, touche `d` = capture en mouvement
- touche `t` = tableau des resultats

La valeur a retenir est celle en **mouvement** (mode `d`) : c'est le regime
reel de l'engin. En air on avait 0.215 px. Sous l'eau ce sera **moins bon**
(turbidite, contraste plus faible, particules).

En quittant (`q`), le script imprime la ligne exacte a recopier :

```
A RECOPIER dans kalman/kalman_filter.py,
bloc « LES TROIS NOMBRES A MESURER » (vers la ligne 190) :

    SIGMA_PIXEL = 0.312

Cette ligne existe deja : il n'y a qu'a changer le nombre.
```

**Une seule ligne a changer.** C'est tout pour cette etape.

---

## Etape 5 — Mesurer la dynamique reelle de l'engin

> **CETTE ETAPE N'EST PAS FAITE A CE JOUR.** Elle demande l'engin reel en
> mouvement dans l'eau, et n'a pas pu etre realisee avant le depart de la
> personne qui a ecrit ces scripts. Les deux nombres concernes valent encore
> leur valeur **supposee** (`SIGMA_ACCELERATION = 0.4`,
> `DERIVE_GYRO_DEG_S = 10.0`).
>
> **Faut-il s'en inquieter tout de suite ?** Non, tant que la centrale
> inertielle de la D435i est branchee : dans ce cas le filtre ne lit jamais
> ces deux reglages. C'est demontre chiffres en main par
> `python kalman/settings_sensitivity.py` (les faire varier d'un
> facteur 4572 ne change pas le resultat d'un millimetre). Ce sont alors
> `BRUIT_GYRO_DEG_S` et `BRUIT_ACCEL` qui gouvernent, et ceux-la **sont
> mesures**.
>
> **Quand cela devient necessaire :** le jour ou le filtre tourne SANS la
> centrale — panne, cable debranche, manip ou elle n'est pas utilisee. Ces
> deux reglages gouvernent alors tout, et une valeur devinee degrade la
> localisation sans que rien ne le signale.
>
> Les scripts affichent d'eux-memes un rappel detaille tant que la mesure
> n'est pas faite, et ce rappel s'eteint tout seul une fois les deux nombres
> remplaces. Il n'y a rien a desactiver a la main.

```
python localization/world_frame_check.py
```

Faire un parcours qui **ressemble a une vraie mission** : les vitesses et
accelerations habituelles, pas une camera posee, pas des mouvements brusques
artificiels. Une trentaine de secondes suffit (la memoire est de 900 images).

Puis `q`. Le script imprime les deux lignes exactes a recopier :

```
DYNAMIQUE OBSERVEE
  rotation    mediane   12.4 deg/s   95e centile   31.0 deg/s
  acceleration mediane   0.18 m/s2   95e centile     0.62 m/s2
------------------------------------------------------------------
  A RECOPIER dans kalman/kalman_filter.py,
  bloc « LES TROIS NOMBRES A MESURER » (vers la ligne 190) :

      SIGMA_ACCELERATION = 0.6
      DERIVE_GYRO_DEG_S  = 31

  Ces deux lignes existent deja : il n'y a qu'a changer les nombres.
  Tout le depot lit ce bloc, il n'y a rien d'autre a modifier.
```

**Deux lignes a changer**, dans le meme bloc qu'a l'etape 4.

Ce sont les 95e centiles — assez larges pour couvrir ce que l'engin fait
vraiment, sans se caler sur un pic isole.

**Sens physique**, pour l'expliquer a Thein :
- `SIGMA_ACCELERATION` = de combien l'engin peut accelerer sans que le filtre
  le sache. Trop petit -> le filtre retarde sur les virages. Trop grand ->
  il ne lisse plus rien.
- `DERIVE_GYRO_DEG_S` = a quelle vitesse l'orientation peut changer entre
  deux images sans mesure.

---

## Etape 6 — Valider que le filtre ameliore vraiment

Toujours dans `world_frame_check.py`. La manip :

1. `o` sur le tag de reference
2. deplacer la camera d'une distance **mesuree au metre a ruban**
3. taper la valeur reelle au clavier, `s` pour enregistrer
4. recommencer une **quinzaine** de fois, a des distances variees

Le filtre doit rester **allume** (touche `f`, indicateur `filtre : ON`) :
chaque `s` enregistre le brut ET le filtre du meme instant, sur la meme
ligne. Pas besoin de refaire la serie filtre coupe — comparer deux series
obligerait a refaire exactement le meme geste deux fois, et c'est le geste
qui dominerait l'ecart.

En quittant avec `q`, le script imprime le verdict tout seul.

### Les deux questions du verdict

**Question 1 — le filtre reduit-il l'erreur ?**

```
  erreur RMS   brut      16.1 mm
               filtre     8.6 mm     -> gain 1.88x
  [OK] le filtre reduit l'erreur.
```

Seuil : gain >= 1.2. En dessous de 1.2 le script repond `[PEU CONCLUANT]` —
sur quinze mesures, quelques pourcents ne se distinguent pas du hasard.
Un gain < 1.0 pointe presque toujours `sigma_acceleration` (etape 5).

**N'annonce aucun gain chiffre a l'avance.** L'auto-test affiche 33x, mais
c'est une simulation ou le bruit est exactement celui que le filtre suppose
et ou la trajectoire est a vitesse constante : les deux hypotheses du filtre
y sont vraies par construction. Dans le bassin ce sera bien moins. Le seul
chiffre defendable est celui que TU mesures ici.

**Question 2 — le filtre dit-il la verite sur sa precision ?**

C'est la question la plus importante, et elle ne se voit pas a l'ecran.

```
  incertitude annoncee par le filtre :    7.0 mm (mediane)
  erreur reellement constatee        :    6.5 mm (mediane)
  rapport reel / annonce : 0.9
  [OK] le filtre dit la verite sur sa precision.
```

| Rapport | Verdict |
|---|---|
| < 0.5 | prudent — il annonce plus d'erreur qu'il n'en fait, sans danger |
| 0.5 a 2 | honnete |
| 2 a 4 | il se croit plus precis qu'il n'est, ne pas se fier au `+/-` |
| > 4 | **il ment** — verifier `SIGMA_PIXEL`, puis la carte des tags |

Pourquoi ca compte plus que le gain : un filtre qui annonce +/- 2 mm en se
trompant de 20 est **plus dangereux** qu'un filtre qui ne lisse rien. Tout
ce qui consomme sa sortie — une commande, une carte, un rapport — le croit
sur parole.

Reserve honnete a connaitre : l'erreur enregistree porte sur une **distance
entre deux poses**, quand sigma porte sur **une position**. Ce ne sont pas
tout a fait les memes grandeurs, et l'erreur du metre a ruban s'y ajoute. Ce
rapport se lit en ordre de grandeur : il attrape un filtre qui ment d'un
facteur 3, pas un ecart de 20 %.

**Question 3 — les compteurs**

```
  mesures rejetees : 2   reprises apres blocage : 0
```

Quelques rejets sont **sains** : le filtre attrape les retournements de tag.
Des centaines veulent dire que le modele de bruit est trop optimiste. Plus
de 3 reprises signale en general des tags mal places dans la carte.

---

## Etape 6bis — La centrale inertielle (IMU)

La D435i porte une centrale : un gyroscope (vitesses angulaires) et un
accelerometre. Le filtre sait desormais s'en servir. Rien n'est obligatoire —
sans IMU il fonctionne comme avant — mais ce qu'elle apporte est mesurable.

### Ce que chaque capteur apporte, et ce qu'il n'apporte pas

| | sans derive | permanent | ce qu'il ne sait pas |
|---|---|---|---|
| **Tags** | oui | **non** — intermittents | rien quand aucun tag n'est vu |
| **Gyro** | non — biais integre | oui | derive sans limite si rien ne le recale |
| **Accel** | oui pour le bas | oui | **rien du lacet** ; derive trop vite en position |

Leurs defauts sont opposes, et c'est tout l'interet de les fusionner : le
gyro propage entre deux tags, l'accelerometre tient deux axes d'orientation
sur trois indefiniment, les tags recalent le lacet et la position — et
servent au passage a estimer le biais du gyro.

### Ce que ca change, chiffres de l'auto-test

```
sans gyro : apres 6 s sans tag, erreur de cap  125.0 deg
avec gyro : apres 6 s sans tag, erreur de cap    4.6 deg

accelerometre seul : roulis -0.01 deg, tangage +0.01 deg  (partis de +12 et -9)

biais du gyro : vrai [0.4 -0.3 0.6] deg/s, estime [0.38 -0.34 0.57] (erreur 0.05)

perte de tags de 1.5 s en pleine acceleration :
   sans accel 377 mm  ->  avec accel 9 mm
```

Sans gyro, une seconde sans tag et le cap est perdu. Avec, on traverse
plusieurs secondes.

### Comment l'utiliser

```python
filtre.predire(dt, gyro=omega, accel=a)
```

- `gyro` : vitesse angulaire en **rad/s**, repere IMU
- `accel` : acceleration en **m/s2**, repere IMU, **pesanteur comprise** —
  telle que le capteur la rend, sans rien retrancher

Les deux sont facultatifs : `filtre.predire(dt)` reste valable.

### Le piege du repere — a ne pas negliger

Sur la D435i, **la centrale n'est pas alignee avec la camera couleur**. Il
existe une rotation constante entre les deux, que `pyrealsense2` donne :

```python
extr = profil.get_stream(rs.stream.gyro).get_extrinsics_to(
           profil.get_stream(rs.stream.color))
R = np.array(extr.rotation).reshape(3, 3)
filtre = FiltrePose(rotation_imu_camera=R)
```

Sans cette rotation, les axes sont melanges et l'engin derive de travers —
**sans aucun message d'erreur**. C'est le genre de faute qu'on ne voit qu'au
bout de plusieurs jours.

### Montrer que l'IMU est lue et exploitee

C'est ce que Thein a demande : *extraire les donnees IMU du SDK Intel, puis
appliquer les maths pour en tirer position et orientation.*

```
python kalman/imu_realsense.py
```

Camera branchee. Le programme ouvre les flux `accel` et `gyro` du SDK, mesure
le repos pendant 5 s, puis affiche l'orientation en direct. Quatre choses s'y
montrent, dans l'ordre :

1. **La centrale est lue** — les mesures brutes bougent quand tu bouges.
2. **L'echelle est juste** — au repos l'accelerometre lit 9.81 m/s2. Si ce
   n'est pas le cas, les unites sont fausses et tout le reste aussi.
3. **Les maths marchent** — tourne d'un quart de tour, le lacet affiche 90.
4. **La derive est la ou on l'attend** — roulis et tangage restent stables
   (l'accelerometre les tient), le lacet derive. C'est la demonstration
   visible de pourquoi les tags sont necessaires.

**Sans camera sous la main :**

```
python kalman/imu_realsense.py --simulation
```

Les memes maths sur une centrale simulee. La verite etant connue, l'erreur
est chiffree — ce qu'aucune manip reelle ne permet :

```
2. UN QUART DE TOUR AUTOUR DE LA VERTICALE
   lu : roulis +0.00   tangage -0.00   lacet +90.03 deg   (attendu 0, 0, 90)
   erreur d'orientation : 0.03 deg

3. TRENTE SECONDES IMMOBILE, SANS AUCUN TAG
   roulis +0.53   tangage -0.46 deg   <- tenus par l'accelerometre
   lacet  +9.04 deg                   <- derive librement
```

### Deux nombres a mesurer, engin IMMOBILE

Une minute sans bouger, puis l'ecart-type des mesures :

```python
BRUIT_GYRO_DEG_S = 0.15   # ecart-type des vitesses angulaires, deg/s
BRUIT_ACCEL      = 0.05   # ecart-type des accelerations, m/s2
```

Ils sont dans le meme bloc que les autres, en tete de `kalman_filter.py`. Les
valeurs actuelles sont des ordres de grandeur pour un MEMS de cette classe,
pas des mesures.

### La limite a annoncer honnetement

L'accelerometre a un biais lentement variable que **rien ici n'estime**. La
double integration le transforme en erreur quadratique : 0.05 m/s2 font 2.5 cm
au bout d'une seconde, mais **1 m au bout de dix**.

L'IMU sert donc a traverser une perte de tags de quelques secondes, **pas a
naviguer a l'estime**. Les tags restent la seule source sans derive.

---

## Etape 7 — Surveiller que les tags ne bougent pas

**Rien a faire** : c'est inclus dans le verdict de l'etape 6. Si un support a
bouge, le bilan l'ajoute tout seul :

```
  SUPPORTS QUI ONT BOUGE
  tag 11 : boite deplacee de 19 mm (+18, -6, +0) mm  [confirme par plusieurs voisins]
```

Il dit **quel** support a bouge, **de combien** et **dans quelle direction** —
de quoi corriger la carte sans tout re-enregistrer. Si rien n'apparait sous ce
titre, c'est qu'aucun tag n'est suspect.

Pourquoi ca compte : l'erreur du systeme apres filtrage est de l'ordre de
2 mm. Une boite lestee decalee de 1 cm pese a elle seule cinq fois tout le
reste du budget d'erreur. Un tag qui bouge produit un **biais**, et un filtre
de Kalman ne sait traiter que du bruit centre : il moyenne le bruit, mais il
**suit** le biais.

Limite a annoncer honnetement : un tag vu **seul** n'est jamais mis en
defaut. Rien ne distingue alors « la camera a bouge » de « le tag a bouge ».

---

## Recapitulatif — une seule session de bassin

Tout se fait dans l'eau, dans cet ordre. Seules les etapes 4 et 5 demandent
d'ouvrir un fichier ; les autres, rien du tout.

| # | Sur place | A editer ensuite |
|---|---|---|
| 1 | `calibrate.py --montage tube_eau` — damier, 30 vues | — |
| 2 | `set_mounting.py tube_eau` — **une fois par machine** | — (aucun fichier) |
| 3 | Poser les tags, `o` puis se deplacer pour la carte | — |
| 4 | `measure_tag_noise.py` — captures mode `d` | `kalman_filter.py` : **1 ligne** |
| 5 | `world_frame_check.py` — parcours type mission, `q` | `kalman_filter.py` : **2 lignes** |
| 6 | `world_frame_check.py` — 15 mesures au metre, `q` | — (le verdict s'affiche) |
| 7 | — | — (inclus dans le verdict de 6) |

**Total : 3 lignes, dans 1 seul fichier.** Et les scripts des etapes 4 et 5
affichent la ligne exacte a recopier.

L'etape 2 ne peut pas attendre : sans elle, les etapes 3 a 6 sont mesurees
avec la focale de l'air, et le bilan de l'etape 6 ne veut plus rien dire.
A la fin de l'etape 1, `calibrate.py` propose d'ailleurs de la faire tout
seul — repondre « oui » suffit. Et si personne n'a rien regle sur cette
machine, le premier script lance pose la question et retient la reponse :
rien a se transmettre entre les deux PC.
