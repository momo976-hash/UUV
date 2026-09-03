# list_realsense.py — List the RealSense devices and say if they have an IMU.
#
#     python list_realsense.py
#
# Repond a une error precise du SDK : "Couldn't resolve requests". Elle veut
# dire que les flux demandes n'existent pas sur l'appareil trouve, sans dire
# lesquels manquent ni pourquoi. Les causes possibles se ressemblent toutes a
# l'ecran :
#
#   - c'est une D435 et non une D435i : le model SANS "i" n'a pas d'IMU.
#     C'est de loin le cas le plus frequent, et rien ne le signale autrement
#     que par cette error.
#   - deux cameras sont branchees et le SDK a pris celle qui n'a pas d'IMU.
#   - un autre programme tient deja la camera.
#
# Ce script enumere les appareils, leur numero de serie, leurs capteurs et
# leurs flux, puis dit franchement si une imu inertielle est disponible.
import sys

try:
    import pyrealsense2 as rs
except ImportError:
    print("ERREUR : pyrealsense2 n'est pas installe.")
    print("  python -m pip install pyrealsense2")
    sys.exit(1)


def main():
    contexte = rs.context()
    appareils = list(contexte.query_devices())

    print("=" * 68)
    print("APPAREILS REALSENSE BRANCHES")
    print("=" * 68)
    if not appareils:
        print("\nAUCUN appareil trouve.")
        print("  - la camera est-elle branchee ?")
        print("  - un autre programme la tient-il deja ? (ferme-le)")
        print("  - essaie un autre port USB, de preference USB 3")
        return 1

    with_imu = []
    for numero, appareil in enumerate(appareils):
        name = appareil.get_info(rs.camera_info.name)
        serie = appareil.get_info(rs.camera_info.serial_number)
        print(f"\n[{numero}] {name}")
        print(f"     numero de serie : {serie}")
        try:
            print(f"     micrologiciel   : "
                  f"{appareil.get_info(rs.camera_info.firmware_version)}")
        except Exception:
            pass

        flux = {}
        for capteur in appareil.sensors:
            nom_capteur = capteur.get_info(rs.camera_info.name)
            types = sorted({p.stream_name() for p in capteur.get_stream_profiles()})
            flux[nom_capteur] = types
            print(f"     capteur : {nom_capteur}")
            print(f"        flux : {', '.join(types)}")

        tous = {t.lower() for types in flux.values() for t in types}
        gyro = any("gyro" in t for t in tous)
        accel = any("accel" in t for t in tous)
        if gyro and accel:
            print("     -> CENTRALE INERTIELLE PRESENTE (accel + gyro)")
            with_imu.append((numero, name, serie))
        else:
            manque = [n for n, present in (("gyro", gyro), ("accel", accel))
                      if not present]
            print(f"     -> PAS d'IMU utilisable (manque : {', '.join(manque)})")

    print("\n" + "=" * 68)
    print("CONCLUSION")
    print("=" * 68)
    if not with_imu:
        print("Aucun appareil branche n'a de imu inertielle.")
        print("\nLe model D435 (sans 'i') n'en a PAS ; seul le D435i en porte")
        print("une. Verifie le name exact affiche plus haut : c'est la seule")
        print("facon de les distinguer, ils sont physiquement identiques.")
        print("\nimu_realsense.py ne peut donc pas fonctionner avec celui-ci.")
        print("En attendant, la demonstration des maths tourne sans materiel :")
        print("  python imu_realsense.py --simulation")
        return 1

    print(f"{len(with_imu)} appareil(s) avec imu inertielle :")
    for numero, name, serie in with_imu:
        print(f"  [{numero}] {name}   serie {serie}")
    if len(appareils) > 1:
        print("\nATTENTION : plusieurs appareils sont branches. Le SDK prend le")
        print("first qu'il trouve, et rien ne dit lequel. Pour toute measurement")
        print("qui compte — bias du gyro, noise — DEBRANCHE les autres :")
        print("le bias est propre a un exemplaire, comme une calibration.")
    print("\nTu peux lancer :  python imu_realsense.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
