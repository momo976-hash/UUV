# list_realsense.py — List the RealSense devices and say if they have an IMU.
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python tools/list_realsense.py
#
# Run it when the SDK says "Couldn't resolve requests", or before any
# measurement that depends on the IMU. It lists the devices, their serial
# numbers and their streams, and says plainly whether an inertial IMU is
# available.
# ===========================================================================
#
#     python list_realsense.py
#
# It answers one precise SDK error: "Couldn't resolve requests". That means
# the streams requested do not exist on the device found, without saying which
# are missing or why. The possible causes all look alike from the outside:
#
#   - it is a D435 and not a D435i: the model WITHOUT the "i" has no IMU.
#     By far the most frequent case, and nothing signals it other than that
#     error.
#   - two cameras are plugged in and the SDK took the one without an IMU.
#   - another program already holds the camera.
#
# This script enumerates the devices, their serial numbers, their sensors and
# their streams, then says plainly whether an inertial IMU is available.
import sys

try:
    import pyrealsense2 as rs
except ImportError:
    print("ERROR: pyrealsense2 is not installed.")
    print("  python -m pip install pyrealsense2")
    sys.exit(1)


def main():
    context = rs.context()
    devices = list(context.query_devices())

    print("=" * 68)
    print("REALSENSE DEVICES PLUGGED IN")
    print("=" * 68)
    if not devices:
        print("\nNO device found.")
        print("  - is the camera plugged in?")
        print("  - does another program already hold it? (close it)")
        print("  - try another USB port, preferably USB 3")
        return 1

    with_imu = []
    for number, device in enumerate(devices):
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        print(f"\n[{number}] {name}")
        print(f"     serial number   : {serial}")
        try:
            print(f"     firmware        : "
                  f"{device.get_info(rs.camera_info.firmware_version)}")
        except Exception:
            pass

        streams = {}
        for sensor in device.sensors:
            sensor_name = sensor.get_info(rs.camera_info.name)
            types = sorted({p.stream_name() for p in sensor.get_stream_profiles()})
            streams[sensor_name] = types
            print(f"     sensor  : {sensor_name}")
            print(f"        streams : {', '.join(types)}")

        every_type = {t.lower() for types in streams.values() for t in types}
        gyro = any("gyro" in t for t in every_type)
        accel = any("accel" in t for t in every_type)
        if gyro and accel:
            print("     -> INERTIAL IMU PRESENT (accel + gyro)")
            with_imu.append((number, name, serial))
        else:
            missing = [n for n, present in (("gyro", gyro), ("accel", accel))
                      if not present]
            print(f"     -> NO usable IMU (missing: {', '.join(missing)})")

    print("\n" + "=" * 68)
    print("CONCLUSION")
    print("=" * 68)
    if not with_imu:
        print("No device plugged in has an inertial IMU.")
        print("\nThe D435 model (without the 'i') does NOT have one; only the")
        print("D435i carries one. Check the exact name shown above: it is the")
        print("only way to tell them apart, they are physically identical.")
        print("\nSo imu_realsense.py cannot work with this one.")
        print("In the meantime the maths demonstration runs with no hardware:")
        print("  python imu_realsense.py --simulation")
        return 1

    print(f"{len(with_imu)} device(s) with an inertial IMU:")
    for number, name, serial in with_imu:
        print(f"  [{number}] {name}   serial {serial}")
    if len(devices) > 1:
        print("\nWARNING: several devices are plugged in. The SDK takes the")
        print("first it finds, and nothing says which. For any measurement that")
        print("matters — gyro bias, noise — UNPLUG the others: the bias belongs")
        print("to one individual unit, like a calibration.")
    print("\nYou can now run:  python kalman/imu_realsense.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
