# demo_image.py — Detection AprilTag SANS camera.
# Le programme genere lui-meme une image contenant un AprilTag, la detecte,
# calcule la pose et affiche le resultat. Aucune webcam necessaire.
import os
import cv2
import numpy as np

TAILLE_TAG = 0.10  # cote du tag en metres

# --- 1) Fabriquer une image contenant un AprilTag (id 0) ---
dictionnaire = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
taille_px = 300
marqueur = cv2.aruco.generateImageMarker(dictionnaire, 0, taille_px)

# On pose le tag au centre d'une grande image blanche (bordure blanche = obligatoire)
canvas = np.full((600, 600), 255, dtype=np.uint8)
d = (600 - taille_px) // 2
canvas[d:d + taille_px, d:d + taille_px] = marqueur
image = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

# --- 2) Detecter le tag ---
detecteur = cv2.aruco.ArucoDetector(dictionnaire, cv2.aruco.DetectorParameters())
coins, ids, _ = detecteur.detectMarkers(canvas)

# --- 3) Parametres approx de la "camera" + coins 3D du tag ---
L, H = 600, 600
K = np.array([[L, 0, L / 2], [0, L, H / 2], [0, 0, 1]], dtype=np.float64)
dist = np.zeros(5)
h = TAILLE_TAG / 2
coins_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

# --- 4) Dessiner le resultat ---
if ids is not None:
    print(f"OK : {len(ids)} tag(s) detecte(s), id = {ids.flatten().tolist()}")
    cv2.aruco.drawDetectedMarkers(image, coins, ids)
    for c, tag_id in zip(coins, ids.flatten()):
        pts = c.reshape(4, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            coins_3d, pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if ok:
            cv2.drawFrameAxes(image, K, dist, rvec, tvec, TAILLE_TAG / 2, 3)
            distance = float(np.linalg.norm(tvec))
            print(f"  tag {tag_id} : distance estimee = {distance:.2f} m")
            cv2.putText(image, f"id={tag_id} d={distance:.2f}m", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
else:
    print("Aucun tag detecte (ne devrait pas arriver ici).")

# --- 5) Enregistrer ET afficher ---
chemin = os.path.abspath("resultat.png")
cv2.imwrite(chemin, image)
print(f"Image resultat enregistree ici : {chemin}")

cv2.imshow("Resultat (appuie sur une touche pour fermer)", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
