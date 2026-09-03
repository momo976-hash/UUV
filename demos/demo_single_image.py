# demo_single_image.py — Read the tags in a single image file.
# Le programme genere lui-meme une image contenant un AprilTag, la detecte,
# calcule la pose et affiche le result. Aucune webcam necessaire.
import os
import cv2
import numpy as np

TAG_SIZE = 0.10  # cote du tag en metres

# --- 1) Fabriquer une image contenant un AprilTag (id 0) ---
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
taille_px = 300
marqueur = cv2.aruco.generateImageMarker(dictionary, 0, taille_px)

# On pose le tag au centre d'une grande image blanche (bordure blanche = obligatoire)
canvas = np.full((600, 600), 255, dtype=np.uint8)
d = (600 - taille_px) // 2
canvas[d:d + taille_px, d:d + taille_px] = marqueur
image = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

# --- 2) Detecter le tag ---
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
corners, ids, _ = detector.detectMarkers(canvas)

# --- 3) Parametres approx de la "camera" + corners 3D du tag ---
L, H = 600, 600
K = np.array([[L, 0, L / 2], [0, L, H / 2], [0, 0, 1]], dtype=np.float64)
dist = np.zeros(5)
h = TAG_SIZE / 2
coins_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

# --- 4) Dessiner le result ---
if ids is not None:
    print(f"OK : {len(ids)} tag(s) detecte(s), id = {ids.flatten().tolist()}")
    cv2.aruco.drawDetectedMarkers(image, corners, ids)
    for c, tag_id in zip(corners, ids.flatten()):
        pts = c.reshape(4, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            coins_3d, pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if ok:
            cv2.drawFrameAxes(image, K, dist, rvec, tvec, TAG_SIZE / 2, 3)
            distance = float(np.linalg.norm(tvec))
            print(f"  tag {tag_id} : distance estimee = {distance:.2f} m")
            cv2.putText(image, f"id={tag_id} d={distance:.2f}m", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
else:
    print("Aucun tag detecte (ne devrait pas arriver ici).")

# --- 5) Enregistrer ET afficher ---
path = os.path.abspath("result.png")
cv2.imwrite(path, image)
print(f"Image result enregistree ici : {path}")

cv2.imshow("Resultat (appuie sur une key pour fermer)", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
