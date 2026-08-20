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
| `SIGMA_ACCELERATION` | 0.4 m/s2, **suppose** | `verification_monde.py` | 5 |
| `DERIVE_GYRO_DEG_S` | 10 deg/s, **suppose** | `verification_monde.py` | 5 |

## En tout : 4 lignes a changer, dans 2 fichiers

Rien d'autre. Les scripts qui les utilisent vont tous puiser a ces deux
endroits.

**1. `calibration/optique.py`** — le montage (etape 2)

```python
MONTAGE_ACTIF = os.environ.get("UUV_MONTAGE", "tube_air")
```

**2. `localisations/filtre_kalman.py`** — bloc « LES TROIS NOMBRES A
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

En quittant (`q`), le script imprime la ligne exacte a recopier :

```
A RECOPIER dans localisations/filtre_kalman.py,
bloc « LES TROIS NOMBRES A MESURER » (vers la ligne 190) :

    SIGMA_PIXEL = 0.312

Cette ligne existe deja : il n'y a qu'a changer le nombre.
```

**Une seule ligne a changer.** C'est tout pour cette etape.

---

## Etape 5 — Mesurer la dynamique reelle de l'engin

```
python localisations/verification_monde.py
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
  A RECOPIER dans localisations/filtre_kalman.py,
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

Toujours dans `verification_monde.py`. La manip :

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

Tout se fait dans l'eau, dans cet ordre. Les etapes 2, 4 et 5 demandent
d'ouvrir un fichier ; les autres, rien du tout.

| # | Sur place | A editer ensuite |
|---|---|---|
| 1 | `calibration.py --montage tube_eau` — damier, 30 vues | — |
| 2 | *(au bord du bassin, sur le portable)* | `optique.py` : **1 ligne** |
| 3 | Poser les tags, `o` puis se deplacer pour la carte | — |
| 4 | `mesurer_bruit_tag.py` — captures mode `d` | `filtre_kalman.py` : **1 ligne** |
| 5 | `verification_monde.py` — parcours type mission, `q` | `filtre_kalman.py` : **2 lignes** |
| 6 | `verification_monde.py` — 15 mesures au metre, `q` | — (le verdict s'affiche) |
| 7 | — | — (inclus dans le verdict de 6) |

**Total : 4 lignes, dans 2 fichiers.** Et les scripts des etapes 4 et 5
affichent la ligne exacte a recopier.

L'etape 2 ne peut pas attendre : sans elle, les etapes 3 a 6 sont mesurees
avec la focale de l'air, et le bilan de l'etape 6 ne veut plus rien dire.
