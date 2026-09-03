# set_mounting.py — Pin down this machine's physical mounting, once.
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python calibration/set_mounting.py              shows, then asks
#     python calibration/set_mounting.py tube_water   sets it directly
#     python calibration/set_mounting.py --show       shows only
#     python calibration/set_mounting.py --forget     drops the setting
#
# Run ONCE per computer, not once per session. The three mountings are
# bare_air, tube_air and tube_water. The answer is written to
# calibration/local_mounting.txt, which is deliberately not versioned so the
# pool PC and the office laptop can disagree without fighting over git.
# ===========================================================================
#
# WHY THIS SCRIPT EXISTS
# Two computers work on the same repository: the office laptop, where the
# camera sits bare on a table, and the poolside PC, where it is in the tube
# underwater. So the right mounting is not a property of the code, it is a
# property of the machine — and a machine does not change mounting between two
# git pulls.
#
# Before, the mounting's name was written in optics.py, a versioned file. The
# consequences: you had to warn each other by message before every run, one
# person's setting overwrote the other's at the next pull, and the day nobody
# warns anybody, the distances are a quarter out with no message at all.
#
# The setting now lives in calibration/local_mounting.txt, which is not
# versioned. Each person sets it ONCE on their machine and forgets about it.
import argparse
import os
import sys
from pathlib import Path

# This script asks the question itself, further down and with the full state
# in view. So optics.py is stopped from asking it at import time, otherwise it
# would be asked twice in a row.
os.environ.setdefault("UUV_MOUNTING_QUIET", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optics  # noqa: E402


def show():
    """The full state, changing nothing."""
    print("=" * 68)
    print("THIS MACHINE'S MOUNTING")
    print("=" * 68)
    print(f"  active          : {optics.ACTIVE_MOUNTING}")
    print(f"  decided by      : {optics.MOUNTING_SOURCE}")
    print(f"  local file      : {optics.LOCAL_MOUNTING_FILE}")

    local = optics._read_local_mounting()
    if local:
        print(f"                    contains '{local}'")
    elif optics.LOCAL_MOUNTING_FILE.exists():
        print("                    present but unreadable")
    else:
        print("                    ABSENT — nothing is set here yet")

    real = optics.source(optics.ACTIVE_MOUNTING)
    if real != optics.ACTIVE_MOUNTING:
        print(f"\n  WARNING: '{optics.ACTIVE_MOUNTING}' is not calibrated on")
        print(f"  this machine. The scripts will serve '{real}'s numbers.")
        print("  To calibrate it:")
        print(f"      python calibration/calibrate.py "
              f"--mounting {optics.ACTIVE_MOUNTING}")

    print("\n  calibrations present:")
    for name in optics.MOUNTINGS:
        if optics.source(name) == name:
            K, _ = optics.load(name, quiet=True)
            print(f"    {name:11s} yes   fx = {K[0, 0]:7.2f}   "
                  f"fy = {K[1, 1]:7.2f}")
        else:
            print(f"    {name:11s} no")
    print("=" * 68)


def ask():
    """Asks for the mounting at the terminal and writes it down."""
    suggested = optics.likely_mounting()
    print("\nWhat is this machine's mounting?\n")
    for index, name in enumerate(optics.MOUNTINGS, start=1):
        mark = "  <- suggested" if name == suggested else ""
        print(f"  {index}) {name:11s} {optics._DESCRIPTIONS[name]}{mark}")
    print(f"\n  Enter alone = {suggested}")
    try:
        answer = input("  Your choice: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAbandoned, nothing has changed.")
        return 1

    answer = optics.LEGACY_MOUNTING_NAMES.get(answer, answer)
    if not answer:
        chosen = suggested
    elif answer.isdigit() and 1 <= int(answer) <= len(optics.MOUNTINGS):
        chosen = optics.MOUNTINGS[int(answer) - 1]
    elif answer in optics.MOUNTINGS:
        chosen = answer
    else:
        print(f"'{answer}' is not a valid choice. Nothing has changed.")
        return 1
    return set_to(chosen)


def set_to(name):
    """Writes the setting down and says what has just changed."""
    name = optics.LEGACY_MOUNTING_NAMES.get(name, name)
    if name not in optics.MOUNTINGS:
        print(f"ERROR: '{name}' is unknown. "
              f"Possible: {', '.join(optics.MOUNTINGS)}")
        return 1

    path = optics.write_local_mounting(name)
    print(f"\nThis machine's mounting: {name}")
    print(f"  written to {path}")
    print("  that file is not versioned: the other PC keeps its own.")

    if optics.source(name) != name:
        print(f"\n  WARNING: '{name}' is not calibrated here yet.")
        print(f"  In the meantime the scripts will serve "
              f"'{optics.source(name)}'s numbers.")
        if optics.mounting_is_submerged(name):
            print("  Underwater that is NOT acceptable: the wall refracts, and")
            print("  distances will be about a quarter too short.")
        print(f"  To do:  python calibration/calibrate.py --mounting {name}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Sets this machine's physical mounting.")
    # The French spellings are accepted so that notes and habits from before
    # the handover keep working.
    parser.add_argument("mounting", nargs="?",
                        choices=(list(optics.MOUNTINGS)
                                 + list(optics.LEGACY_MOUNTING_NAMES)),
                        help="the mounting to remember on this machine")
    parser.add_argument("--show", "--montrer", dest="show",
                        action="store_true",
                        help="print the state without changing anything")
    parser.add_argument("--forget", "--effacer", dest="forget",
                        action="store_true",
                        help="drop this machine's setting")
    options = parser.parse_args()

    if options.forget:
        removed = False
        for path in (optics.LOCAL_MOUNTING_FILE,
                     optics.LEGACY_LOCAL_MOUNTING_FILE):
            if path.exists():
                path.unlink()
                print(f"Setting erased: {path}")
                removed = True
        if removed:
            print("The question will be asked again on the next run.")
        else:
            print("There was nothing to erase.")
        return 0

    show()
    if options.show:
        return 0
    if options.mounting:
        return set_to(options.mounting)
    return ask()


if __name__ == "__main__":
    sys.exit(main())
