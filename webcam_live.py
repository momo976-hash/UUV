# webcam_live.py — Lecture d'AprilTags en direct, avec recherche auto de la camera.
# Essaie plusieurs index (0..3) et backends Windows jusqu'a trouver une camera
# qui renvoie vraiment une image. Puis detecte les AprilTags en temps reel.
import cv2
import numpy as np

TAILLE_TAG = 0.10  # cote reel de ton tag imprime, en metres. MESURE-LE et ajuste !


def ouvrir_camera():
    """Cherche une camera qui fonctionne et renvoie (capture, largeur, hauteur)."""
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "AUTO")]
    for index in range(4):
        for backend, nom in backends:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap.isOpened():
                ok, img = cap.read()  # verifie qu'on obtient VRAIMENT une image
                if ok and img is not None:
                    h, w = img.shape[:2]
                    print(f"Camera trouvee : index={index}, backend={nom}, {w}x{h}")
                    return cap, w, h
            cap.release()
    return None, 0, 0


cam, L, H = ouvrir_camera()
if cam is None:
    print("ERREUR : aucune camera n'a pu etre ouverte (index 0 a 3).")
    print(" -> Verifie : Parametres Windows > Confidentialite > Camera >")
    print("    'Autoriser les applications de bureau a acceder a la camera'.")
    print(" -> Ferme toute appli qui utilise la camera (Teams, Zoom, app Camera).")
    print(" -> Teste d'abord l'app 'Camera' de Windows : vois-tu l'image ?")
    raise SystemExit

# Parametres approx de la camera (suffisant pour un test ; a calibrer plus tard)
K = np.array([[L, 0, L / 2], [0, L, H / 2], [0, 0, 1]], dtype=np.float64)
dist = np.zeros(5)
h = TAILLE_TAG / 2
coins_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

dictionnaire = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
detecteur = cv2.aruco.ArucoDetector(dictionnaire, cv2.aruco.DetectorParameters())

print("En direct. Montre un tag a la camera. Appuie sur 'q' pour quitter.")

while True:
    ok, image = cam.read()
    if not ok:
        continue
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    coins, ids, _ = detecteur.detectMarkers(gris)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, coins, ids)
        for c, tag_id in zip(coins, ids.flatten()):
            pts = c.reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(
                coins_3d, pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if ok2:
                cv2.drawFrameAxes(image, K, dist, rvec, tvec, TAILLE_TAG / 2, 2)
                d = float(np.linalg.norm(tvec))
                cx, cy = pts.mean(axis=0).astype(int)
                cv2.putText(image, f"id={tag_id} d={d:.2f}m", (cx - 40, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imshow("AprilTag en direct (q pour quitter)", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
