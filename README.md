# UUV — localisation par AprilTags, centrale inertielle et filtre de Kalman

Localiser un engin sous-marin avec une caméra RealSense D435i qui regarde des
AprilTags, à travers la paroi d'un tube étanche, sous l'eau.

---

## START HERE — one command

```
python kalman/proof_imu_kalman.py
```

It re-runs the verifications live and answers the two requests that were
made: **IMU read from the Intel SDK, maths applied to it**, and a Kalman
filter **based on a kinematic model**. Nothing is read from a saved results
file — every number is recomputed as it prints.

No camera needed for that one. Two demonstrations do need the camera plugged
in, and the script lists them at the end.

To *see* the filter working rather than read that it does — one figure, no
camera and no water needed (requires `matplotlib`):

```
python demos/demo_kalman.py
```

**One honest reserve, stated up front:** position cannot come from the IMU
alone. A MEMS bias double-integrates into 2.5 cm after 1 s but 1 m after
10 s. The IMU is what carries the estimate through a **tag dropout of a few
seconds** (9 mm error over 1.5 s, against 377 mm without it). The tags remain
the only drift-free source. That is why the two requests are one system, not
two.

---

## Ce qui est fait, et ce qui ne l'est pas

| | État | Preuve / ce qu'il reste |
|---|---|---|
| Calibration sous l'eau | ✅ | `fx = 838.45`, `fy = 652.10` — vérifiées sur le terrain |
| IMU lue du SDK Intel | ✅ | `python kalman/imu_realsense.py` |
| Maths IMU → orientation | ✅ | 0.03° sur un quart de tour connu |
| Filtre de Kalman cinématique | ✅ | `python kalman/kalman_reference_check.py` — les 9 valeurs publiées de Becker, à la 4ᵉ décimale |
| Bruit gyro / accéléromètre | ✅ mesuré | engin immobile, 400 Hz |
| **Dynamique réelle de l'engin** | ⏳ **à faire** | demande l'engin **en mouvement dans l'eau** — voir plus bas |
| `SIGMA_PIXEL` sous l'eau | ⏳ à faire | mesuré en air (0.215 px) ; `calibration/measure_tag_noise.py` |

### La mesure qui reste — étape 5

Deux réglages (`SIGMA_ACCELERATION`, `DERIVE_GYRO_DEG_S`) valent encore leur
valeur **supposée**. Ils décrivent à quelle vitesse l'engin accélère et
tourne : cela dépend de sa masse, de ses propulseurs et de l'eau, donc aucun
calcul ni fiche technique ne peut les donner.

**Ce n'est pas bloquant tant que la centrale inertielle est branchée** — dans
ce cas le filtre ne les lit jamais. Démontré chiffres en main :

```
python kalman/settings_sensitivity.py
```

Les faire varier d'un facteur 4572 ne change pas le résultat d'un millimètre.
Mais le jour où le filtre tourne **sans** la centrale, ils gouvernent tout.

Les scripts affichent d'eux-mêmes la marche à suivre, et le rappel s'éteint
tout seul une fois la mesure faite. Rien à désactiver à la main.

---

## Par où entrer, selon ce qu'on veut faire

| Je veux… | Commande |
|---|---|
| prouver que l'IMU et le filtre sont faits | `python kalman/proof_imu_kalman.py` |
| **voir le filtre travailler, en image** | `python demos/demo_kalman.py` |
| **les figures du cours EN DIRECT, vraies mesures** | `python localization/world_frame_check.py --graphiques` |
| montrer que la calibration donne la bonne distance | `python calibration/demo_distance.py --montage tube_eau` |
| vérifier une distance connue, au mètre | `python calibration/check_distance.py --reel 1.5 --tag 0.22389` |
| **faire l'étape 5 (engin dans l'eau)** | `python localization/world_frame_check.py` |
| voir l'IMU tourner en direct | `python kalman/imu_realsense.py` |
| les mêmes maths sans caméra | `python kalman/imu_realsense.py --simulation` |
| lancer les auto-tests du filtre | `python kalman/kalman_filter.py` |
| régler le montage de cette machine | `python calibration/set_mounting.py` |

---

## À faire une fois par ordinateur

Le montage physique (caméra nue à l'air / dans le tube / dans l'eau) est une
propriété de **la machine**, pas du code — le PC du bureau et celui du bassin
n'ont pas la même réponse. Il se règle une fois :

```
python calibration/set_mounting.py tube_eau
```

C'est écrit dans `calibration/montage_local.txt`, qui n'est **pas** versionné.
Se tromper là-dessus ne fait rien planter : les distances sont simplement
fausses de dizaines de pourcents, en silence. Les scripts préviennent quand
l'image contredit le montage déclaré, mais ils ne peuvent pas tout attraper.

---

## Le dépôt

| Dossier | Contenu |
|---|---|
| `calibration/` | Optique, calibration, vérifications de distance. `optics.py` fait foi pour toutes les constantes. |
| `localization/` | Filtre de Kalman, IMU, localisation dans un repère monde. |
| `docs/` | `kalman_protocol.md` — le protocole complet, du bassin au filtre. |
| `demos/` | Démonstrations autonomes, focale devinée : à ne pas confondre avec la chaîne calibrée. |
| `src/`, `autres/` | Utilitaires et travaux antérieurs. |

**Le document de référence du filtre** est Alex Becker, *Kalman Filter
Explained Through Examples* (kalmanfilter.net), modèle cinématique à vitesse
constante. Ce n'est pas une inspiration lointaine : `kalman_reference_check.py` fait
passer son exemple chiffré dans la classe qui tourne réellement sur l'engin
et retrouve ses 9 valeurs publiées.

Chaque script porte en tête un commentaire qui explique **pourquoi** il
existe et ce qu'il ne peut pas prouver. C'est là qu'il faut lire avant de
modifier quoi que ce soit.
