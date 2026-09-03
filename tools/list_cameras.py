# list_cameras.py — List every camera this computer can open.
#
# Indispensable quand plusieurs cameras sont branchees (webcam du PC +
# RealSense) : calibrer une camera et mesurer avec une autre fausse tout.
#
# Le programme ouvre chaque index, affiche l'image et estime le champ de vision
# a partir de la resolution, pour t'aider a reconnaitre laquelle est laquelle.
#
# Keys: n = camera suivante | q = quitter
import cv2
import numpy as np

BACKENDS = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "AUTO")]


def cameras_disponibles(nb_index=5):
    """Renvoie la liste des (index, backend, name, width, height) qui marchent."""
    trouvees = []
    for index in range(nb_index):
        for backend, name in BACKENDS:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap.isOpened():
                ok, img = cap.read()
                if ok and img is not None:
                    h, w = img.shape[:2]
                    trouvees.append((index, backend, name, w, h))
                    cap.release()
                    break          # un backend qui marche suffit pour cet index
            cap.release()
    return trouvees


liste = cameras_disponibles()
if not liste:
    print("Aucune camera detectee.")
    raise SystemExit

print("=" * 58)
print("CAMERAS DETECTEES")
for index, _, name, w, h in liste:
    print(f"  index={index}  backend={name}  resolution={w}x{h}")
print("=" * 58)
print("Regarde chaque image et note l'index de celle que tu veux utiliser.")
print("Keys: 'n' = suivante | 'q' = quitter")

position = 0
while True:
    index, backend, name, w, h = liste[position]
    cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
    if not cap.isOpened():
        position = (position + 1) % len(liste)
        continue

    while True:
        ok, image = cap.read()
        if not ok:
            break
        cv2.putText(image, f"index = {index}   backend = {name}", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"resolution = {w}x{h}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(image, "'n' = camera suivante   'q' = quitter", (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Identification des cameras", image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            print("\nNote l'index chosen et mets-le dans CAMERA_INDEX "
                  "en haut de tes programmes.")
            raise SystemExit
        if key == ord("n"):
            break

    cap.release()
    position = (position + 1) % len(liste)
