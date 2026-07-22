# UUV — Localisation par AprilTag (Tâche 1)

Localisation d'un véhicule sous-marin (UUV) à l'aide d'une caméra **Intel RealSense**
et de marqueurs **AprilTag**, en **Python pur** (sans ROS 2 pour l'instant).

Ce dépôt couvre les **étapes 1 → 4** : afficher le flux caméra, récupérer les
paramètres intrinsèques, détecter les AprilTags et estimer la **pose 3D**
(position + orientation) en direct.

## Où exécuter le code ?

Le script s'exécute **dans un terminal**, sur l'ordinateur relié à la caméra.
Concrètement, tu as deux options confortables :

### Option A — Visual Studio Code (recommandé)
1. Installe **VS Code** + l'extension **Python** (Microsoft).
2. Ouvre le dossier du projet : `File > Open Folder…` → choisis `UUV`.
3. Ouvre un terminal intégré : `Terminal > New Terminal` (ou `Ctrl+ù`).
4. Suis l'installation ci-dessous, puis lance le script depuis ce terminal
   (ou avec le bouton ▶ "Run Python File" en haut à droite).

> C'est exactement le workflow évoqué en réunion : coder dans VS Code, et plus
> tard s'y connecter à distance en **SSH** sur le Raspberry Pi (extension
> "Remote - SSH") quand on passera sur le matériel embarqué.

### Option B — Terminal seul
N'importe quel terminal (PowerShell, bash, le terminal macOS…) suffit :
tu tapes les mêmes commandes. VS Code n'est pas obligatoire, juste plus pratique.

> ⚠️ Le script ouvre une **fenêtre vidéo** (`cv2.imshow`). Il faut donc un
> environnement avec **affichage graphique** (ton PC portable, pas un serveur
> distant sans écran). Sur un Raspberry Pi en SSH sans bureau, on adaptera plus
> tard (X11 forwarding ou enregistrement dans un fichier).

## Installation

```bash
# 1. Créer un environnement virtuel (isole les dépendances)
python3 -m venv venv

# 2. L'activer
#    Linux / macOS :
source venv/bin/activate
#    Windows (PowerShell) :
#    venv\Scripts\Activate.ps1

# 3. Installer les dépendances
pip install -r requirements.txt
```

## Préparer les marqueurs

1. Génère/imprime des AprilTags de la famille **`tag36h11`**
   (générateur officiel : https://github.com/AprilRobotics/apriltag-imgs).
2. Imprime-les et **mesure précisément le côté du carré noir** (ex. 10 cm).
   Cette mesure est un paramètre critique du calcul de distance.

## Lancer

```bash
# Avec la caméra RealSense (tag de 10 cm) :
python src/apriltag_pose.py --tag-size 0.10

# Pour TESTER sans la RealSense, avec la webcam du PC :
# (les distances ne seront pas fiables — intrinsèques approximés — mais la
#  détection et l'affichage des axes fonctionnent)
python src/apriltag_pose.py --source webcam --tag-size 0.10
```

À l'écran : contour vert + ID de chaque tag, axes 3D (X rouge, Y vert, Z bleu),
distance et angles roll/pitch/yaw. Appuie sur **`q`** ou **Échap** pour quitter.

### Vérifier que ça marche (critères de réussite)
- Les axes 3D "collent" au tag quand tu bouges la caméra.
- La distance affichée correspond à la réalité (vérifie au mètre ruban).
- Le tag vu bien de face donne des angles proches de 0°.

## Bonnes pratiques issues des articles de référence
- **Distance < 4 m** : au-delà l'erreur de pose explose.
- **Éviter le face-à-face parfait** : un léger angle (roll/pitch) rend la
  profondeur Z mieux observable.
- **Plusieurs tags non coplanaires** : l'erreur chute (≈40 cm → ≈8 cm à 5 m en
  passant de 1 à 4 tags).

## Prochaines étapes (non incluses ici)
- **Étape 5** : transformations de repères (pose de la caméra dans le *monde*
  à partir d'une carte des tags).
- **Étape 6** : fusion multi-tags + filtre de Kalman.
- **Étape 7** : enregistrement CSV et analyse d'erreur pour le rapport de stage.
