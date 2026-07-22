# validation.py — Mesure la precision de la distance estimee par AprilTag.
# Affiche une distance STABILISEE (moyenne des dernieres mesures) et l'enregistre
# dans validation.csv quand on appuie sur 's'. Comparer ensuite au metre ruban.
import csv
import os
from collections import deque

import cv2
import numpy as np

# >>> METS ICI LA VRAIE TAILLE DU CARRE NOIR DE TON TAG, EN METRES <<<
TAILLE_TAG = 0.10  # ex. 0.16 pour un tag de 16 cm


def ouvrir_camera():
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "AUTO")]
    for index in range(4):
        for backend, nom in backends:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap.isOpened():
                ok, img = cap.read()
                if ok and img is not None:
                    h, w = img.shape[:2]
                    print(f"Camera trouvee : index={index}, backend={nom}, {w}x{h}")
                    return cap, w, h
            cap.release()
    return None, 0, 0


cam, L, H = ouvrir_camera()
if cam is None:
    print("ERREUR : aucune camera ouverte.")
    raise SystemExit

K = np.array([[L, 0, L / 2], [0, L, H / 2], [0, 0, 1]], dtype=np.float64)
dist = np.zeros(5)
h = TAILLE_TAG / 2
coins_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

dictionnaire = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
detecteur = cv2.aruco.ArucoDetector(dictionnaire, cv2.aruco.DetectorParameters())

historique = deque(maxlen=30)   # pour lisser la distance
fichier = os.path.abspath("validation.csv")
if not os.path.exists(fichier):
    with open(fichier, "w", newline="") as f:
        csv.writer(f).writerow(["n", "distance_mesuree_m"])
compteur = 0

print("=" * 55)
print("VALIDATION. Place le tag, garde-le stable et bien de face.")
print("  's' = enregistrer la mesure stabilisee")
print("  'q' = quitter")
print(f"Les mesures sont enregistrees dans : {fichier}")
print("=" * 55)

while True:
    ok, image = cam.read()
    if not ok:
        continue
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coins, ids, _ = detecteur.detectMarkers(gris)

    distance_stable = None
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, coins, ids)
        pts = coins[0].reshape(4, 2).astype(np.float64)  # 1er tag detecte
        ok2, rvec, tvec = cv2.solvePnP(coins_3d, pts, K, dist,
                                       flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok2:
            cv2.drawFrameAxes(image, K, dist, rvec, tvec, TAILLE_TAG / 2, 2)
            historique.append(float(np.linalg.norm(tvec)))
            distance_stable = sum(historique) / len(historique)
    else:
        historique.clear()

    if distance_stable is not None:
        cv2.putText(image, f"distance = {distance_stable:.3f} m",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(image, "'s' = enregistrer   'q' = quitter",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        cv2.putText(image, "Aucun tag detecte", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Validation precision (q pour quitter)", image)
    touche = cv2.waitKey(1) & 0xFF
    if touche == ord("q"):
        break
    if touche == ord("s") and distance_stable is not None:
        compteur += 1
        with open(fichier, "a", newline="") as f:
            csv.writer(f).writerow([compteur, f"{distance_stable:.3f}"])
        print(f"[{compteur}] enregistre : distance mesuree = {distance_stable:.3f} m")

cam.release()
cv2.destroyAllWindows()
print(f"\nTermine. Mesures dans : {fichier}")
