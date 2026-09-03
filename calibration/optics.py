# optics.py — Everything the light crosses before reaching the sensor.
#
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python calibration/optics.py     full report on the current mounting
#     from optics import ...           in every other script
#
# This file is the SINGLE SOURCE OF TRUTH for every optical constant of the
# project: camera, tube, window, medium, tag sizes. Nothing else hard-codes a
# focal length or a tag size — they all read them from here.
#
# The mounting (bare camera in air / in the tube in air / in the tube
# underwater) is NOT written in any code file. It is a property of the
# MACHINE, not of the repository: the office laptop and the pool PC do not
# have the same camera in the same medium. It is set once per machine with
#
#     python calibration/set_mounting.py
#
# and stored in calibration/local_mounting.txt, which is deliberately not
# versioned so that a git pull can never change the other machine's setting.
#
# ===========================================================================
# WHY THIS FILE EXISTS
# ===========================================================================
# The optical constants used to be copied into a dozen scripts: the 604.1876
# focal length, water's 1.33 factor, the 43.4 deg field of view. The moment
# the camera goes into a tube, all of that changes at once. One place now
# holds the truth, and the scripts read it from there.
#
# ===========================================================================
# THE ACTUAL MOUNTING (photographs of 11/08)
# ===========================================================================
# Blue Robotics BR-100230-151 tube: cast acrylic, 150 mm long, 49.5 mm inner
# diameter. The D435i measures 90 x 25 x 25 mm: its 90 mm do not fit across
# the 49.5 mm diameter, but fit easily along the 150 mm of length. So it lies
# DOWN along the tube, its three lenses aligned WITH THE TUBE AXIS, and it
# looks out through the CYLINDRICAL WALL. A printed bracket wedges it against
# one side. That is ORIENTATION = "radial".
#
# ===========================================================================
# WHAT THAT CHANGES, AND IT IS COUNTER-INTUITIVE
# ===========================================================================
# A cylinder does not behave the same way in its two directions.
#
#   Along the TUBE AXIS — the image's HORIZONTAL axis, since the camera lies
#   down. In that plane the wall reduces to two parallel planes: a
#   plane-parallel slab. No effect in air, but underwater it is a flat
#   viewport, with its 1.33 factor.
#
#   In the CROSS-SECTION plane (circumferential) — the image's VERTICAL axis.
#   There the wall stays curved: it is a meniscus. A ray leaving exactly from
#   the TUBE AXIS strikes both surfaces perpendicularly and is not deflected
#   at all, in air as in water — that is the dome-port principle. Off the
#   axis, the meniscus acts.
#
# Underwater the system is therefore ANAMORPHIC: the two image axes are not
# magnified by the same factor. This is not a defect to be corrected, it is
# the geometry of the tube; but it does forbid summarising the optics with a
# single number.
#
# ===========================================================================
# BEING OFF-AXIS IS NOT FATAL
# ===========================================================================
# The D435i's pupil cannot sit on the axis by chance: the body is 25 mm deep
# for a 24.75 mm inner radius, and the pupil sits a few more millimetres
# behind the front face. Resting on the bottom of the tube, it ends up ~5 mm
# BEHIND the axis.
#
# The good news, computed below by `residu_section`: that offset translates
# almost entirely into a CHANGE OF FOCAL LENGTH, not into distortion. What
# the calibration does not absorb stays under 0.5 px, below the measured
# detection noise (0.215 px). In other words:
#
#   - the pupil does NOT need centring to a tenth of a millimetre;
#   - it is on the other hand IMPERATIVE to calibrate in the final
#     configuration, and for the camera not to move in its bracket
#     afterwards. One millimetre of slip after calibration is ~1 % of error
#     on every distance (see `sensibilite_slip`).
import os
import sys
from pathlib import Path

import numpy as np

RESOLUTION = (640, 480)

# --- the bare camera, measured on a checkerboard ---------------------------
K_BARE_AIR = np.array([
    [604.1876, 0.0000, 326.1973],
    [0.0000, 602.3668, 242.8850],
    [0.0000, 0.0000, 1.0000],
], dtype=np.float64)
DIST_BARE_AIR = np.array([0.013835, 0.733706, -0.002333, 0.001136, -2.707687],
                         dtype=np.float64)

# Detection noise on a tag corner, measured by measure_tag_noise.py. Used here
# as a yardstick: there is no point correcting an optical defect smaller than
# the noise that hides it.
CORNER_NOISE_PX = 0.215

# Overall size of the D435i. Its WIDTH carries the three aligned lenses; the
# optical axis leaves perpendicular to it, along the DEPTH.
CAMERA_WIDTH = 0.090
CAMERA_HEIGHT = 0.025
CAMERA_DEPTH = 0.025

# How far the entrance pupil sits behind the front face of the body.
# Value adjusted by the tube_air calibration of 11/08: the fy_tube / fy_bare
# ratio gives the real off-axis offset, which is what predicts the underwater
# focal lengths. See `off-axis offset_depuis_calibration`.
PUPIL_BEHIND_FACE = -0.0029

# --- the tube: Blue Robotics BR-100230-151, cast acrylic -------------------
TUBE_NAME = 'BR-100230-151, 2" cast acrylic, 150 mm'
TUBE_INNER_DIAMETER = 0.0495
TUBE_INNER_DIAMETER_TOLERANCE = 0.0015
TUBE_OUTER_DIAMETER = 0.0580
TUBE_OUTER_DIAMETER_TOLERANCE = 0.0010
TUBE_LENGTH = 0.150
TUBE_MASS = 0.115            # kg
TUBE_MAX_DEPTH = 130         # metres of water

# --- the two pool tags, side of the BLACK square, in metres ----------------
# Caliper-measured, not read off the print sheet: a printer does not reproduce
# the requested scale exactly. Distance comes from d = fx.S/s, where S is that
# side: 1 % of error in measuring the tag gives back 1 % of error at EVERY
# distance, without exception. So these two numbers are measured, never
# estimated.
#
# The two depart from nominal in OPPOSITE directions — the small one by
# -0.15 %, the large one by +0.40 %. So it is not a printer scale factor,
# which would have shifted both the same way: it is specific to each print
# run. Using the nominal values (0.223 / 0.115) instead of these amounts to a
# tag-measurement error we already knew about and chose not to correct.
LARGE_TAG_SIZE = 0.22389    # nominal 223.0 mm
SMALL_TAG_SIZE = 0.11732    # nominal 117.5 mm

# --- how the camera sits inside --------------------------------------------
# "radial": lying along the tube, looking out through the cylindrical wall.
#           The only mounting that fits a 49.5 mm tube, and the one in the
#           photographs.
# "axial":  looking out through a flat end cap. Requires the camera's width to
#           fit within the diameter — not the case here.
ORIENTATION = "radial"

# What the printed bracket leaves between the BACK of the camera and the wall,
# on the side opposite the view. It is the only thing the mechanics control,
# and it is measured with calipers. 0 = camera flat against the bottom.
BACK_CLEARANCE = 0.000

# Axial mounting only: how far the pupil sits back from the end cap.
PUPIL_SETBACK = 0.030

AIR_INDEX = 1.0
WATER_INDEX = 1.33
ACRYLIC_INDEX = 1.49

# Calibrations recorded by calibrate.py --mounting <name>
#
# The names are in English now. The three French names used until the handover
# are still accepted when read from local_mounting.txt or UUV_MOUNTING, so a
# machine already set up keeps working without anyone touching it.
MOUNTINGS = ("bare_air", "tube_air", "tube_water")
LEGACY_MOUNTING_NAMES = {"nue_air": "bare_air", "tube_air": "tube_air",
                         "tube_eau": "tube_water"}

# The calibration folder is looked for wherever it may be, depending on
# whether this file lives at the repository root or inside calibration/.
# Getting it wrong does not crash anything: `load` silently falls back to
# the bare camera, and one can measure for weeks with the wrong focal length
# without ever noticing. Both the English and the pre-handover French folder
# names are accepted.
_HERE = Path(__file__).resolve().parent
MOUNTINGS_FOLDER = next(
    (d for d in (_HERE / "mountings", _HERE / "montages",
                 _HERE / "calibration" / "mountings",
                 _HERE / "calibration" / "montages")
     if d.is_dir()),
    _HERE / "mountings")

# THE ACTIVE MOUNTING — found on its own, with nobody editing a Python file.
#
# The concrete problem: two computers work on the same repository. The office
# laptop has the bare camera on a table; the PC at the poolside has the camera
# in the tube, underwater. The right mounting is therefore not a property of
# the CODE, it is a property of the MACHINE. Writing the name into a versioned
# file forces the two to contradict each other at every git pull, and above
# all forces people to warn each other by message — and the day nobody warns
# anybody, distances are wrong by a third and nothing says so.
#
# The decision chain, strongest first:
#
#   1. the UUV_MOUNTING environment variable. It lasts only for one command:
#      the one-off override, to compare two mountings on the same run without
#      disturbing anything.
#          UUV_MOUNTING=bare_air python localization/world_frame_check.py
#
#   2. the local_mounting.txt file, written ONCE per machine. It is NOT
#      versioned (.gitignore): that is exactly what lets the two PCs disagree
#      without fighting. The pool PC puts "tube_water" in it once and for all,
#      and nobody has to say anything to anybody afterwards.
#
#   3. failing that, the question is asked at the terminal on first launch,
#      and the answer written to that file. Once per machine.
#
# What we deliberately do NOT do: guess in silence. No image reliably tells
# air from water — the camera's white balance removes the blue cast, and the
# RealSense depth is wrong by the same factor as the tags, so the two agree
# with each other even when both are wrong. A question at first launch costs
# ten seconds; one bad guess cost two weeks.
LOCAL_MOUNTING_FILE = _HERE / "local_mounting.txt"
LEGACY_LOCAL_MOUNTING_FILE = _HERE / "montage_local.txt"


def _normalise_mounting(name):
    """Accept the pre-handover French names, return the English one.

    A machine set up before the repository was translated still holds
    "tube_eau" in its file. Refusing it would silently send that machine back
    to the bare-camera calibration, which is exactly the kind of failure this
    module exists to prevent.
    """
    name = LEGACY_MOUNTING_NAMES.get(name, name)
    return name if name in MOUNTINGS else None


def _read_local_mounting():
    """The mounting kept on THIS machine, or None if there is none."""
    for path in (LOCAL_MOUNTING_FILE, LEGACY_LOCAL_MOUNTING_FILE):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for row in text.splitlines():
            row = row.split("#", 1)[0].strip()
            if not row:
                continue
            mounting = _normalise_mounting(row)
            if mounting:
                return mounting
            print(f"[optics] {path.name}: '{row}' is not a known mounting, "
                  f"line ignored.")
    return None


def write_local_mounting(name):
    """Pin down THIS machine's mounting, once and for all."""
    name = _normalise_mounting(name)
    if name is None:
        raise ValueError(f"unknown mounting: {name!r}")
    LOCAL_MOUNTING_FILE.write_text(
        "# The physical mounting of THIS machine.\n"
        "# One useful line: bare_air, tube_air or tube_water.\n"
        "# This file is not versioned: each computer keeps its own.\n"
        "# To change it:  python calibration/set_mounting.py\n"
        f"{name}\n", encoding="utf-8")
    return LOCAL_MOUNTING_FILE


def _quiet_requested():
    """Has the caller asked for no questions and no chatter?

    Both spellings are honoured: the English one and the French one used
    before the handover, so an existing script or CI job keeps working.

    Written as a helper rather than inline because the inline version was
    once `not A or B`, which parses as `(not A) or B` and silently stopped
    the legacy variable from working.
    """
    return bool(os.environ.get("UUV_MOUNTING_QUIET")
                or os.environ.get("UUV_MONTAGE_MUET"))


def likely_mounting():
    """The most plausible mounting, given what is calibrated on this machine.

    Only used as the default answer to the question asked at first launch: a
    machine on which somebody bothered to calibrate tube_water is very
    probably the pool one. It is a suggestion, never a decision.
    """
    for name in ("tube_water", "tube_air", "bare_air"):
        legacy = {v: k for k, v in LEGACY_MOUNTING_NAMES.items()}.get(name)
        for candidate in (name, legacy):
            if candidate and (MOUNTINGS_FOLDER / f"{candidate}.npz").exists():
                return name
    return "bare_air"


_DESCRIPTIONS = {
    "bare_air": "bare camera, in_air           (bench, desk, table)",
    "tube_air": "camera in the tube, in_air    (dry run)",
    "tube_water": "camera in the tube, IN WATER  (pool)",
}


def _ask_for_mounting():
    """Ask the question once, at the terminal. None if we cannot."""
    if _quiet_requested():
        return None
    try:
        if not (sys.stdin and sys.stdin.isatty()):
            return None
    except (AttributeError, ValueError):
        return None

    suggested = likely_mounting()
    print("\n" + "=" * 68)
    print("WHAT IS THIS MACHINE'S MOUNTING?")
    print("=" * 68)
    print("Asked ONCE per computer. The answer is kept in")
    print(f"{LOCAL_MOUNTING_FILE} and does not go into git:")
    print("every PC keeps its own.\n")
    for index, name in enumerate(MOUNTINGS, start=1):
        mark = " <- suggested" if name == suggested else ""
        print(f"  {index}) {name:11s} {_DESCRIPTIONS[name]}{mark}")
    print(f"\n  Enter alone = {suggested}")
    try:
        answer = input("  Your choice: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if not answer:
        chosen = suggested
    elif answer.isdigit() and 1 <= int(answer) <= len(MOUNTINGS):
        chosen = MOUNTINGS[int(answer) - 1]
    elif _normalise_mounting(answer):
        chosen = _normalise_mounting(answer)
    else:
        print(f"  '{answer}' is not a valid choice, keeping {suggested}.")
        chosen = suggested

    try:
        path = write_local_mounting(chosen)
        print(f"  -> mounting '{chosen}' kept, written to {path}")
        print("     To change it later:")
        print("       python calibration/set_mounting.py")
    except OSError as error:
        print(f"  -> mounting '{chosen}' kept (not saved: {error})")
    print("=" * 68 + "\n")
    return chosen


def _resolve_mounting():
    """Return (mounting, where it came from)."""
    forced = os.environ.get("UUV_MOUNTING") or os.environ.get("UUV_MONTAGE")
    if forced:
        mounting = _normalise_mounting(forced)
        if mounting is None:
            raise SystemExit(
                f"[optics] UUV_MOUNTING='{forced}' is unknown. "
                f"Possible mountings: {', '.join(MOUNTINGS)}")
        return mounting, "UUV_MOUNTING variable"

    local = _read_local_mounting()
    if local:
        return local, LOCAL_MOUNTING_FILE.name

    asked = _ask_for_mounting()
    if asked:
        return asked, "answer at first launch"

    # Neither a setting nor a terminal to ask on: take the most cautious
    # option — the bare camera — and SAY SO. Silence is the only real danger
    # here.
    default = "bare_air"
    if not _quiet_requested():
        print(f"[optics] no mounting set on this machine, using '{default}'.")
        print("[optics]   if the camera is in the tube or in the water, "
              "distances will be wrong.")
        print("[optics]   to set it: python calibration/set_mounting.py")
    return default, "default, nothing configured"


ACTIVE_MOUNTING, MOUNTING_SOURCE = _resolve_mounting()


def mounting_summary():
    """Une row lisible : quel mounting, et d'ou vient la decision."""
    reel = source(ACTIVE_MOUNTING)
    if reel != ACTIVE_MOUNTING:
        return (f"mounting {ACTIVE_MOUNTING} (via {MOUNTING_SOURCE}) "
                f"mais PAS CALIBRE -> chiffres de {reel}")
    return f"mounting {ACTIVE_MOUNTING} (via {MOUNTING_SOURCE})"


def _actif(mounting):
    """Resout le mounting demande. None = celui qui est actif."""
    return ACTIVE_MOUNTING if mounting is None else mounting


# --- chargement -------------------------------------------------------------
def source(mounting=None):
    """Le mounting dont les chiffres seront REELLEMENT servis.

    Tant qu'un mounting n'a pas ete calibre, `load` retombe sur la camera
    nue. Les conversions optiques ont besoin de savoir laquelle des deux elles
    ont sous la main, sinon elles corrigent deux fois.
    """
    mounting = _actif(mounting)
    return mounting if (MOUNTINGS_FOLDER / f"{mounting}.npz").exists() else "nue_air"


def load(mounting=None, quiet=False):
    """The matrix and distortion coefficients of a given mounting.

    Until a mounting has been calibrated we fall back to the bare camera, and
    say so. That is defensible IN AIR: the plane-parallel slab deflects
    nothing along the tube axis, and the meniscus costs just over 1 % along
    the other. It is NOT defensible underwater, where the wall becomes a real
    lens.
    """
    mounting = _actif(mounting)
    # A machine calibrated before the handover holds tube_eau.npz, not
    # tube_water.npz. Both names are tried, newest first.
    legacy = {v: k for k, v in LEGACY_MOUNTING_NAMES.items()}.get(mounting)
    for name in (mounting, legacy):
        if not name:
            continue
        path = MOUNTINGS_FOLDER / f"{name}.npz"
        if path.exists():
            data = np.load(path)
            return data["K"], data["dist"]
    if not quiet and mounting != "bare_air":
        verdict = ("acceptable in air" if mounting == "tube_air"
                   else "NOT VALID, the wall refracts")
        print(f"[optics] mounting '{mounting}' not calibrated yet, "
              f"falling back to the bare camera ({verdict}).")
        print(f"[optics]   to calibrate it: python calibration/calibrate.py "
              f"--mounting {mounting}")
    return K_BARE_AIR.copy(), DIST_BARE_AIR.copy()


def focal_length(mounting=None):
    """La focal_length horizontale du mounting, en pixels."""
    return float(load(mounting, quiet=True)[0][0, 0])


# --- garde-fou : ce que la camera voit contredit-il le mounting declare ? ----
# Ce test ne DECIDE rien, il alerte. L'water absorbe le rouge (~0.4 /m) et
# presque pas le bleu (~0.02 /m) : sur un aller-back de trois metres le canal
# rouge tombe a un tiers pendant que le bleu ne bouge pas. Une image de pool
# est donc franchement bleue, une image de bureau ne l'est pas.
#
# Pourquoi ce n'est qu'une alerte : la balance des blancs automatique de la
# D435i corrige une partie du bleu, un mur bleu en salle donne le meme signal,
# et un flux infrarouge est gris donc muet. Le test se tait des qu'il doute.
_SEUIL_EAU = 0.60          # rouge/bleu en dessous = tres probablement de l'water
_SEUIL_AIR = 0.85          # au dessus = tres probablement de l'air
_deja_alerte = False


def check_image_matches_mounting(image, mounting=None):
    """Compare la colour dominante au mounting declare.

    Renvoie un message d'alerte a afficher, ou None quand rien ne cloche ou
    que l'image ne permet pas de conclure.
    """
    global _deja_alerte
    if _deja_alerte or image is None or getattr(image, "ndim", 0) != 3:
        return None
    if image.shape[2] != 3:
        return None

    petite = np.asarray(image[::8, ::8], dtype=np.float64)
    bleu, vert, rouge = (float(np.median(petite[:, :, c])) for c in range(3))
    if max(bleu, vert, rouge) < 20.0:
        return None                                  # image trop sombre
    if max(abs(rouge - vert), abs(vert - bleu)) < 3.0:
        return None                                  # image grise : infrarouge
    if bleu < 1.0:
        return None

    report = rouge / bleu
    sous_leau = mounting_is_submerged(_actif(mounting))
    if report < _SEUIL_EAU and not sous_leau:
        _deja_alerte = True
        return (f"l'image est tres bleue (rouge/bleu = {report:.2f}) alors que "
                f"le mounting declare est '{_actif(mounting)}', qui est un "
                f"mounting in air.\n"
                f"    Si la camera est in the water, les distances seront "
                f"trop courtes d'environ 25 %.\n"
                f"    Pour correct : python calibration/set_mounting.py")
    if report > _SEUIL_AIR and sous_leau:
        _deja_alerte = True
        return (f"l'image n'a pas la teinte de l'water (rouge/bleu = "
                f"{report:.2f}) alors que le mounting declare est "
                f"'{_actif(mounting)}'.\n"
                f"    Si la camera est in air, les distances seront trop "
                f"longues d'environ 33 %.\n"
                f"    Pour correct : python calibration/set_mounting.py")
    return None


def mounting_is_submerged(mounting=None):
    """Le mounting donne assumed-t-il la camera in the water ?"""
    return _actif(mounting).endswith("_eau")


# --- le decalage du point de vue derriere le viewport -------------------------
# MESURE AU BASSIN, le 02/09, mounting tube_eau, calibration fx 791.34 :
#
#     0.50 m -> 0.4841 m   -3.18 %   (+/- 0.4 mm)
#     1.00 m -> 0.9839 m   -1.61 %   (+/- 2.3 mm)
#     1.50 m -> 1.4862 m   -0.92 %   (+/- 19 mm)
#     2.00 m -> 1.9944 m   -0.28 %   (+/- 26 mm)
#
# L'ajustement pondere donne une PENTE DE 1.0002 +/- 0.0044 et un DECALAGE de
# -15.9 mm a 7 sigma. Les deux chiffres comptent autant l'un que l'autre :
#
#   - la pente vaut 1 : la focal_length fx = 791.34 est juste, il n'y a plus rien a
#     correct de ce cote. Les %-d'error qui diminuent avec la distance ne
#     venaient pas d'une focal_length un peu fausse.
#   - le decalage est constant en METRES, pas en pourcentage. Aucune focal_length ne
#     peut produire cela : d = fx.S/s est une pure proportionnalite, elle
#     passe forcement par zero.
#
# Le model a un seul parametre (pente forcee a 1, decalage seul) donne un
# chi2 de 0.18 pour 3 degres de liberte, contre 48.7 pour le model en pure
# echelle. Ce n'est pas une preference, c'est un gap de deux ordres de
# grandeur.
#
# CE QUE C'EST PHYSIQUEMENT. Une camera derriere un viewport courbe n'a PAS de
# centre de projection unique : chaque radius est refracte par la wall, et les
# prolongements des rayons emergents ne se coupent pas tous au meme point. Le
# model stenope, lui, exige un point unique ; la calibration en choisit donc
# un, au mieux, et il tombe a cote. Tout se passe comme si l'oeil de la camera
# etait 16 mm plus loin qu'il ne l'est — le meme gap quelle que soit la
# distance visee, exactement ce qu'on measurement.
#
# 16 mm est du meme ordre que le tube lui-meme (radius interieur 24.75 mm,
# wall 4.25 mm), ce qui est le bon ordre de grandeur pour cet effet.
#
# Reference : Treibitz, Schechner, Kaplan, Negahdaripour, « Flat Refractive
# Geometry », IEEE TPAMI 34(1):51-65, 2012 — le viewport rend le systeme
# non-single-viewpoint, et le stenope n'en est qu'une approximation.
# WARNING : ce decalage a ete measurement avec fx = 791.34, et la focal_length installee
# vaut maintenant 838.45. Un decalage fixe et une focal_length ne sont pas
# independants — c'est tout le sujet du bloc ci-dessus — donc rien ne garantit
# que 16 mm soit encore la bonne value a cette focal_length-la.
#
# On le GARDE tel quel malgre tout, parce que c'est la seule value qui ait ete
# reellement measured (4 distances, 7 sigma). La correct au juge reviendrait a
# inventer un count : c'est exactement comme cela qu'un decalage de 77 mm,
# tire d'un ajustement sur des measurements qui ne venaient meme pas de cette
# calibration, s'est retrouve installe un moment.
#
# A REMESURER : trois distances ou plus avec la focal_length actuelle, dont 0.5 m,
# puis lire la section FORME DE L'ERREUR que check_distance.py affiche.
WINDOW_OFFSET = {
    "tube_eau": 0.0159,      # measurement au pool a fx 791.34, 4 distances, 7 sigma
    "tube_air": 0.0,         # jamais measurement
    "nue_air": 0.0,          # pas de viewport : rien a correct
}


def window_offset(mounting=None):
    """Metres a AJOUTER a une distance measured, pour ce mounting."""
    return WINDOW_OFFSET.get(_actif(mounting), 0.0)


def correct_window_offset(tvec, mounting=None):
    """Corrige un vector camera->objet du decalage du point de vue.

    La direction est juste — c'est un probleme de distance, pas d'angle — donc
    on allonge le vector sans le tourner. Sans correction, toutes les
    positions sont ramenees de 16 mm VERS la camera ; les tags d'une meme
    tag_map se retrouvent alors trop proches les uns des autres, et le filter
    voit un world qui retrecit.
    """
    decalage = window_offset(mounting)
    t = np.asarray(tvec, dtype=float)
    if decalage == 0.0:
        return t.copy()
    distance = float(np.linalg.norm(t))
    if distance < 1e-9:
        return t.copy()
    return t * ((distance + decalage) / distance)


def announce_mounting(prefixe="[optics]"):
    """Affiche le mounting kept. A appeler au demarrage de tout script qui
    measurement quelque chose : c'est la row qu'on relit six mois plus tard pour
    savoir avec quels chiffres la manip a tourne."""
    print(f"{prefixe} {mounting_summary()}")
    if MOUNTING_SOURCE.startswith("default"):
        print(f"{prefixe} regle-le une fois pour toutes : "
              f"python calibration/set_mounting.py")


# --- geometrie du champ -----------------------------------------------------
def half_fields_of_view(K=None):
    """Demi-angles du champ : horizontal, vertical, diagonal, en degres."""
    K = K_BARE_AIR if K is None else K
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return (float(np.degrees(np.arctan(cx / fx))),
            float(np.degrees(np.arctan(cy / fy))),
            float(np.degrees(np.arctan(np.hypot(cx / fx, cy / fy)))))


def rayon_tube(pire_cas=True):
    """Rayon interieur utile du tube."""
    return (TUBE_INNER_DIAMETER - (TUBE_INNER_DIAMETER_TOLERANCE if pire_cas else 0.0)) / 2


def rayon_exterieur(pire_cas=True):
    return (TUBE_OUTER_DIAMETER + (TUBE_OUTER_DIAMETER_TOLERANCE if pire_cas else 0.0)) / 2


# --- ou se trouve la pupil dans le tube -----------------------------------
# Convention : l'axis du tube est a 0, et le regard part vers les x positifs.
# Une pupil plaquee au fond est donc a un x NEGATIF, derriere l'axis.
def pupil_off_axis_offsetle(jeu_arriere=None):
    """Position de la pupil par report a l'axis du tube, en metres.

    Negatif = en retrait de l'axis (cas normal : le boitier bute au fond).
    Positif = en avant de l'axis, vers la wall regardee.
    """
    jeu = BACK_CLEARANCE if jeu_arriere is None else jeu_arriere
    return float(-rayon_tube(pire_cas=False) + jeu
                 + CAMERA_DEPTH - PUPIL_BEHIND_FACE)


def jeu_arriere_optimal():
    """Le jeu que le support doit menager pour poser la pupil sur l'axis.

    C'est le seul chiffre que la mecanique ait a respecter : de combien
    SURELEVER la camera au-dessus de la wall du fond.
    """
    return float(rayon_tube(pire_cas=False) - CAMERA_DEPTH
                 + PUPIL_BEHIND_FACE)


def encombrement_libre(jeu_arriere=None):
    """Marge restante entre la face avant de la camera et la wall regardee."""
    jeu = BACK_CLEARANCE if jeu_arriere is None else jeu_arriere
    return float(2 * rayon_tube() - jeu - CAMERA_DEPTH)


def demi_champ_tube(recul=None):
    """Demi-angle que le tube laisse passer en mounting AXIAL, en degres.

    Vu de la pupil, l'ouverture lointaine du tube est un disque de radius
    `rayon_tube` a la distance `recul`. Au-dela, la wall bouche la vue.
    En mounting radial, la wall est transparente sur toute sa length : rien
    ne vignette, et la fonction renvoie un champ non contraignant.
    """
    if ORIENTATION == "radiale":
        return 90.0
    recul = PUPIL_SETBACK if recul is None else recul
    return 90.0 if recul <= 0 else float(
        np.degrees(np.arctan(rayon_tube() / recul)))


def recul_maximal(K=None):
    """Le plus grand recul admissible avant que le tube ne rogne le champ."""
    _, _, diagonal = half_fields_of_view(K)
    return float(rayon_tube() / np.tan(np.radians(diagonal)))


def vignettage(recul=None, K=None):
    """Ce que le tube rogne du champ, s'il rogne quelque chose."""
    passe = demi_champ_tube(recul)
    h, v, d = half_fields_of_view(K)
    return {"demi_angle_tube": passe,
            "rogne_diagonale": passe < d,
            "rogne_horizontal": passe < h,
            "rogne_vertical": passe < v,
            "recul_maximal": recul_maximal(K)}


# --- trace de radius a travers la wall cylindrique --------------------------
def _refracter(direction, normale, eta):
    """Loi de Descartes sous forme vectorielle. None si reflexion totale."""
    normale = -normale if float(direction @ normale) > 0 else normale
    cosinus = -float(direction @ normale)
    sinus2 = eta * eta * (1.0 - cosinus * cosinus)
    if sinus2 > 1.0:
        return None
    return eta * direction + (eta * cosinus - np.sqrt(1.0 - sinus2)) * normale


def sortie_cylindre(angle_deg, off_axis_offset=None, indice_exterieur=WATER_INDEX):
    """Sous quel angle un radius ressort de la wall, dans le plan de section.

    Le radius part de la pupil, decalee de `off_axis_offset` par report a l'axis
    du tube, et traverse les deux surfaces cylindriques. Renvoie l'angle de
    output en degres, ou None en cas de reflexion totale.

    Pupille exactement sur l'axis : le radius est radial, donc perpendiculaire
    aux deux surfaces, et ressort sans avoir devie — quel que soit l'angle et
    quel que soit le milieu exterieur.
    """
    off_axis_offset = (pupil_off_axis_offsetle() if off_axis_offset is None
                    else off_axis_offset)
    point = np.array([off_axis_offset, 0.0])
    direction = np.array([np.cos(np.radians(angle_deg)),
                          np.sin(np.radians(angle_deg))])
    steps = ((rayon_tube(), AIR_INDEX / ACRYLIC_INDEX),
              (rayon_exterieur(), ACRYLIC_INDEX / indice_exterieur))
    for radius, eta in steps:
        b = float(point @ direction)
        c = float(point @ point) - radius * radius
        discriminant = b * b - c
        if discriminant < 0:
            return None
        point = point + (-b + np.sqrt(discriminant)) * direction
        direction = _refracter(direction, point / np.linalg.norm(point), eta)
        if direction is None:
            return None
    return float(np.degrees(np.arctan2(direction[1], direction[0])))


def _angles_de_sortie(off_axis_offset, indice_exterieur, demi_champ, points=40):
    """Couples (angle vise, angle reellement sorti), en radians."""
    vises = np.radians(np.linspace(demi_champ / points, demi_champ, points))
    sortis = []
    for angle in vises:
        output = sortie_cylindre(float(np.degrees(angle)), off_axis_offset,
                                 indice_exterieur)
        sortis.append(np.nan if output is None else np.radians(output))
    sortis = np.asarray(sortis, dtype=float)
    valides = ~np.isnan(sortis)
    return vises[valides], sortis[valides]


def erreur_off_axis_offset(off_axis_offset=None, indice_exterieur=WATER_INDEX,
                        K=None, demi_champ=None):
    """Deviation BRUTE due au off_axis_offset de la pupil, en pixels.

    C'est l'gap entre la direction visee et la direction reellement suivie,
    au bord du champ. Chiffre spectaculaire mais trompeur pris seul : une
    calibration faite dans cette configuration en absorbe la quasi-totalite
    sous forme de focal_length. Ce qui reste vraiment, c'est `residu_section`.
    """
    K = K_BARE_AIR if K is None else K
    if demi_champ is None:
        demi_champ = half_fields_of_view(K)[1]     # circonferentiel = vertical
    off_axis_offset = (pupil_off_axis_offsetle() if off_axis_offset is None
                    else off_axis_offset)
    vises, sortis = _angles_de_sortie(off_axis_offset, indice_exterieur, demi_champ)
    if len(vises) == 0:
        return 0.0
    return float(K[1, 1] * np.max(np.abs(sortis - vises)))


# --- ce que le meniscus fait vraiment a l'image -----------------------------
def grandissement_section(off_axis_offset=None, indice_exterieur=WATER_INDEX,
                          K=None):
    """Facteur par lequel le meniscus multiplie la focal_length VERTICALE.

    On ajuste au sens des moindres carres le seul parametre qu'une calibration
    puisse regler — la focal_length — sur le trace de radius exact, et on renvoie le
    report a la focal_length nue. 1.0 = pupil sur l'axis, le cylindre est
    optiquement absent.
    """
    K = K_BARE_AIR if K is None else K
    off_axis_offset = (pupil_off_axis_offsetle() if off_axis_offset is None
                    else off_axis_offset)
    vises, sortis = _angles_de_sortie(off_axis_offset, indice_exterieur,
                                      half_fields_of_view(K)[1])
    if len(vises) == 0:
        return 1.0
    # y_image = f * tan(angle_monde) ; on cherche f tel que f*tan(sortis)
    # colle a fy*tan(vises).
    return float(np.sum(np.tan(vises) * np.tan(sortis))
                 / np.sum(np.tan(sortis) ** 2))


def residu_section(off_axis_offset=None, indice_exterieur=WATER_INDEX, K=None):
    """Ce que le meniscus laisse APRES que la focal_length ait absorbe ce qu'elle peut.

    C'est la true error du mounting : la part de la deviation qu'aucune
    calibration ne peut ranger dans un parametre. A comparer a CORNER_NOISE_PX.
    """
    K = K_BARE_AIR if K is None else K
    fy = float(K[1, 1])
    off_axis_offset = (pupil_off_axis_offsetle() if off_axis_offset is None
                    else off_axis_offset)
    vises, sortis = _angles_de_sortie(off_axis_offset, indice_exterieur,
                                      half_fields_of_view(K)[1])
    if len(vises) == 0:
        return 0.0
    ajustee = fy * grandissement_section(off_axis_offset, indice_exterieur, K)
    return float(np.max(np.abs(fy * np.tan(vises) - ajustee * np.tan(sortis))))


def sensibilite_slip(indice_exterieur=WATER_INDEX, pas=0.001):
    """Combien coute un millimetre de slip APRES calibration, en %.

    La focal_length verticale est ce que la calibration a fige. Si la camera bouge
    dans son support, elle ne correspond plus, et l'error passe directement
    dans les distances : 1 % de focal_length = 1 % sur toutes les portees.
    """
    d = pupil_off_axis_offsetle()
    avant = grandissement_section(d - pas, indice_exterieur)
    apres = grandissement_section(d + pas, indice_exterieur)
    return float(100 * abs(apres - avant) / 2 / grandissement_section(d, indice_exterieur))


def off_axis_offset_from_calibration(K_mesure, K_reference=None,
                                    indice_exterieur=AIR_INDEX):
    """Retrouve le off_axis_offset reel a partir d'une calibration measured.

    C'est tout l'interet de calibrer D'ABORD DANS L'AIR. En air, la lame plane
    ne key pas a fx : si fx s'ecarte de la camera nue, c'est un probleme de
    mounting, pas d'optics. On the other hand fy passe par le meniscus, et le
    report fy_tube / fy_nue donne directement l'gap de la pupil a l'axis —
    sans demonter quoi que ce soit, et sans devoir croire la value supposee
    de PUPIL_BEHIND_FACE.
    """
    K_reference = K_BARE_AIR if K_reference is None else K_reference
    vise = float(K_mesure[1, 1]) / float(K_reference[1, 1])
    grille = np.arange(-0.015, 0.015, 0.0001)
    gaps = [abs(grandissement_section(float(d), indice_exterieur,
                                        K_reference) - vise) for d in grille]
    return float(grille[int(np.argmin(gaps))])


# --- refraction : ce que devient la focal_length ----------------------------------
def demi_champ_eau(demi_angle_air, direction="axis"):
    """Demi-champ seen in the water, pour l'une ou l'autre direction de l'image.

    `direction` vaut "axis" (le long du tube : la wall est une lame plane, et
    Descartes donne l'angle exact) ou "section" (circonferentiel : on suit le
    radius a travers les deux surfaces courbes).
    """
    if ORIENTATION == "radiale" and direction == "section":
        output = sortie_cylindre(demi_angle_air, indice_exterieur=WATER_INDEX)
        return demi_angle_air if output is None else output
    sine = np.sin(np.radians(demi_angle_air)) / WATER_INDEX
    return float(np.degrees(np.arcsin(np.clip(sine, -1.0, 1.0))))


def focales_eau(mounting=None):
    """Focales equivalentes underwater : (horizontale, verticale).

    Montage radial : la camera est couchee, sa width — donc l'axis HORIZONTAL
    de l'image — suit l'axis du tube et voit une lame plane, d'ou le facteur
    1.33. L'axis VERTICAL est circonferentiel et ne voit que le meniscus, dont
    l'effet depend du off_axis_offset de la pupil.

    On tient compte de ce que la calibration fournie contient DEJA : partir de
    `tube_air`, c'est partir d'un fy qui porte deja l'effet du meniscus en
    air ; il ne reste qu'a le convertir en water.

    Montage axial : les deux directions traversent le meme bouchon plat, et
    les deux focales sont multipliees.
    """
    mounting = _actif(mounting)
    K, _ = load(mounting, quiet=True)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    if mounting == "tube_eau" and source(mounting) == "tube_eau":
        return fx, fy                      # deja measurement underwater
    if ORIENTATION != "radiale":
        return fx * WATER_INDEX, fy * WATER_INDEX
    deja = (grandissement_section(indice_exterieur=AIR_INDEX)
            if source(mounting) == "tube_air" else 1.0)
    return fx * WATER_INDEX, fy * grandissement_section() / deja


def water_focal_length(mounting=None):
    """La focal_length underwater la plus DEFAVORABLE des deux.

    Un seul count ne peut pas decrire un systeme anamorphic. Pour tout ce
    qui est dimensionnement — size apparente d'un tag, uncertainty de pose —
    c'est la plus petite qui contraint, et c'est donc elle qu'on renvoie.
    """
    return float(min(focales_eau(mounting)))


def anamorphic_ratio(mounting=None):
    """Rapport entre les deux focales underwater. 1.0 = pas d'anamorphic_ratio."""
    fx, fy = focales_eau(mounting)
    return float(max(fx, fy) / min(fx, fy))


def portee_eau(portee_air, mounting=None):
    """Ce que devient, une fois immergee, une portee measured in air.

    Le raccourci current est « x 1.33 : underwater on voit plus loin ». Il ne
    vaut QUE pour un viewport plat, et ici seulement pour l'axis du tube. Un tag
    doit rester assez grand DANS LES DEUX directions pour etre decode, donc
    c'est la focal_length la plus petite qui decide — et en mounting radial avec une
    pupil en retrait, c'est la verticale, qui peut meme retrecir.
    """
    K, _ = load(mounting, quiet=True)
    limitante_air = min(float(K[0, 0]), float(K[1, 1]))
    return float(portee_air * water_focal_length(mounting) / limitante_air)


def rayon_image(angle_eau_deg, f=None):
    """Ou tombe vraiment un radius venu de l'water, et ou le model le croit.

    Vaut pour la direction ou la wall se comporte en lame plane : l'axis du
    tube en mounting radial, les deux directions en mounting axial.
    """
    # 'tube_air' est ecrit en dur A DESSEIN, et ne suit pas ACTIVE_MOUNTING :
    # cette fonction PART d'une focal_length in air pour lui apply la refraction.
    # Lui donner une focal_length deja measured underwater compterait l'water deux fois.
    f = focal_length("tube_air") if f is None else f
    angle_air = np.degrees(np.arcsin(np.clip(
        WATER_INDEX * np.sin(np.radians(angle_eau_deg)), -1.0, 1.0)))
    exact = f * np.tan(np.radians(angle_air))
    paraxial = WATER_INDEX * f * np.tan(np.radians(angle_eau_deg))
    return float(exact), float(paraxial)


def ecart_lame_plane(f=None, angles=(5, 10, 15, 20, 25)):
    """De combien le model paraxial se trompe, angle par angle."""
    return [(a, *rayon_image(a, f)) for a in angles]


def angle_modele_fiable(f=None, tolerance_px=1.0):
    """Jusqu'a quel angle le model « focal_length x 1.33 » reste sous la tolerance."""
    previous = 0.0
    for angle in np.arange(0.5, 45.0, 0.5):
        exact, paraxial = rayon_image(float(angle), f)
        if abs(exact - paraxial) > tolerance_px:
            return float(previous)
        previous = float(angle)
    return 45.0


# --- encombrement -----------------------------------------------------------
def budget_longueur():
    """Ce qu'il reste dans le tube une fois la camera dedans, en metres."""
    occupe = (CAMERA_WIDTH if ORIENTATION == "radiale"
              else CAMERA_DEPTH)
    return TUBE_LENGTH - occupe


def verifier_montage():
    """Les incompatibilites mecaniques et optiques du mounting decrit ici."""
    soucis = []
    libre = TUBE_INNER_DIAMETER - TUBE_INNER_DIAMETER_TOLERANCE
    section = np.hypot(CAMERA_HEIGHT, CAMERA_DEPTH)

    if ORIENTATION == "axiale":
        if CAMERA_WIDTH > libre:
            soucis.append(
                f"Montage axial : la camera fait {1000*CAMERA_WIDTH:.0f} mm de "
                f"large et le tube n'offre que {1000*libre:.1f} mm. One must la "
                "coucher (ORIENTATION = \"radiale\") ou passer en serie 4 pouces.")
        v = vignettage()
        if v["rogne_diagonale"]:
            soucis.append(
                f"Le tube ne laisse passer que {2*v['demi_angle_tube']:.0f} deg "
                f"quand la camera en couvre {2*half_fields_of_view()[2]:.0f} en diagonale : "
                f"corners noirs. Recul maximal {1000*v['recul_maximal']:.0f} mm, "
                f"contre {1000*PUPIL_SETBACK:.0f} prevus.")
        return soucis

    if section > libre:
        soucis.append(
            f"La section de la camera ({1000*section:.0f} mm en diagonale) ne "
            f"passe pas dans {1000*libre:.1f} mm.")
    if CAMERA_WIDTH > TUBE_LENGTH:
        soucis.append(
            f"La camera ({1000*CAMERA_WIDTH:.0f} mm) est plus longue que le "
            f"tube ({1000*TUBE_LENGTH:.0f} mm).")

    if encombrement_libre() < 0:
        soucis.append(
            f"Avec {1000*BACK_CLEARANCE:.1f} mm de jeu arriere, la camera depasse du "
            f"tube de {-1000*encombrement_libre():.1f} mm. Reduire le jeu.")

    soucis.append(
        f"La calibration doit etre faite CAMERA DEJA EN PLACE dans le tube, et "
        f"la camera ne doit plus bouger ensuite : {sensibilite_slip():.1f} % "
        "d'error sur toutes les distances par millimetre de slip. C'est "
        "le point faible du mounting, bien avant le centrage lui-meme.")

    soucis.append(
        f"PUPIL_BEHIND_FACE ({1000*PUPIL_BEHIND_FACE:.0f} mm) est une "
        "estimation, pas une measurement. La calibration in air la corrige : "
        "calibrate.py --mounting tube_air en deduit le off_axis_offset reel.")

    soucis.append(
        f"Anamorphose underwater : facteur {anamorphic_ratio():.2f} entre les deux "
        "axes de l'image. La distortion n'a plus de symetrie de revolution, "
        "et le model plumb_bob d'OpenCV la decrira mal — attendre des "
        "residus de calibration plus eleves qu'in air.")

    soucis.append(
        f"Le model « focal_length x {WATER_INDEX} » ne tient qu'a moins de "
        f"{angle_modele_fiable():.0f} deg de l'axis (a 1 px pres), et seulement "
        "selon l'axis du tube. Au-dela one must une calibration faite SOUS L'EAU.")
    return soucis


def report():
    """Un state des lieux lisible du mounting."""
    h, v, d = half_fields_of_view()
    f = focal_length("nue_air")
    fy_nue = float(K_BARE_AIR[1, 1])
    fx_eau, fy_eau = focales_eau()
    gap = pupil_off_axis_offsetle()
    servi = source(ACTIVE_MOUNTING)
    # En tete, et non en bas de page : c'est le first chiffre a check
    # apres une bascule. `servi` differe de `ACTIVE_MOUNTING` quand le mounting
    # demande n'est pas encore calibre — le seul cas ou l'on measurement avec une
    # optics qui n'est pas celle qu'on croit.
    rows = [
        "=" * 74, "OPTIQUE DU MONTAGE", "=" * 74,
        f"\nMONTAGE ACTIF  {ACTIVE_MOUNTING}  (via {MOUNTING_SOURCE})",
        (f"  source des chiffres : {servi}" if servi == ACTIVE_MOUNTING else
         f"  >>> WARNING : '{ACTIVE_MOUNTING}' n'est pas calibre, les chiffres "
         f"servis viennent de '{servi}'."),
        (f"  decalage du viewport : +{1000*window_offset():.1f} mm ajoutes a "
         f"chaque distance" if window_offset() else
         "  decalage du viewport : aucun (jamais measurement pour ce mounting)"),
        "\nCAMERA (nue, in air)",
        f"  focal_length {f:.1f} px, champ {2*h:.1f} x {2*v:.1f} deg (diagonale {2*d:.1f})",
        f"  encombrement {1000*CAMERA_WIDTH:.0f} x {1000*CAMERA_HEIGHT:.0f} x "
        f"{1000*CAMERA_DEPTH:.0f} mm",
        f"\nTUBE  {TUBE_NAME}",
        f"  interieur {1000*TUBE_INNER_DIAMETER:.1f} +/- {1000*TUBE_INNER_DIAMETER_TOLERANCE:.1f} mm "
        f"(pire cas {1000*(TUBE_INNER_DIAMETER-TUBE_INNER_DIAMETER_TOLERANCE):.1f}), "
        f"exterieur {1000*TUBE_OUTER_DIAMETER:.1f} +/- {1000*TUBE_OUTER_DIAMETER_TOLERANCE:.1f} mm",
        f"  wall {1000*(TUBE_OUTER_DIAMETER-TUBE_INNER_DIAMETER)/2:.2f} mm, length "
        f"{1000*TUBE_LENGTH:.0f} mm, {1000*TUBE_MASS:.0f} g, "
        f"tenue {TUBE_MAX_DEPTH} m",
        f"\nMONTAGE  {ORIENTATION}",
    ]
    if ORIENTATION == "radiale":
        rows += [
            "  camera couchee le long du tube, objectifs alignes selon l'axis,",
            "  regard a travers la wall cylindrique",
            f"  place occupee {1000*CAMERA_WIDTH:.0f} mm sur "
            f"{1000*TUBE_LENGTH:.0f}, reste {1000*budget_longueur():.0f} mm",
            "",
            "  OU EST LA PUPILLE  (l'axis du tube est l'origin, le regard va vers +)",
            f"    radius interieur          {1000*rayon_tube(False):+7.2f} mm",
            f"    jeu laisse par le support{1000*BACK_CLEARANCE:+7.2f} mm",
            f"    depth du boitier    {1000*CAMERA_DEPTH:+7.2f} mm",
            f"    retrait de la pupil    {-1000*PUPIL_BEHIND_FACE:+7.2f} mm",
            f"    ---------------------------------",
            f"    pupil / axis du tube    {1000*gap:+7.2f} mm"
            + ("   (en retrait de l'axis)" if gap < 0 else "   (en avant de l'axis)"),
            f"    pour la poser sur l'axis : surelever la camera de "
            f"{1000*jeu_arriere_optimal():.1f} mm",
        ]
    else:
        rows += [
            "  camera face au bouchon, regard par le bout du tube",
            f"  recul de la pupil {1000*PUPIL_SETBACK:.0f} mm, maximum sans "
            f"vignettage {1000*recul_maximal():.0f} mm",
        ]

    rows += ["", "CHAMP UTILE",
               f"  {'':22} {'horizontal':>12} {'vertical':>12}",
               f"  {'in air':22} {2*h:>10.1f} d {2*v:>10.1f} d"]
    if ORIENTATION == "radiale":
        rows.append(f"  {'sous l water':22} "
                      f"{2*demi_champ_eau(h, 'axis'):>10.1f} d "
                      f"{2*demi_champ_eau(v, 'section'):>10.1f} d")
        rows.append("  (horizontal = le long du tube, lame plane ;")
        rows.append("   vertical = circonferentiel, meniscus)")
    else:
        rows.append(f"  {'sous l water':22} {2*demi_champ_eau(h):>10.1f} d "
                      f"{2*demi_champ_eau(v):>10.1f} d")

    rows += ["", "FOCALES SOUS L'EAU",
               f"  horizontale {fx_eau:7.1f} px      verticale {fy_eau:7.1f} px",
               f"  anamorphic_ratio {anamorphic_ratio():.2f}"
               + ("  <- les deux axes ne grossissent pas pareil"
                  if anamorphic_ratio() > 1.01 else "")]

    if ORIENTATION == "radiale":
        rows += [
            "", "CE QUE COUTE LE DECENTREMENT DE LA PUPILLE",
            "  Sur l'axis, tout radius frappe les deux surfaces perpendiculairement",
            "  et ressort sans devier. Hors de l'axis le meniscus agit — mais",
            "  presque uniquement comme un CHANGEMENT DE FOCALE, que la",
            "  calibration absorbe. Seul le residu est une true error.",
            "",
            f"  {'gap a l axis':>14} {'deviation raw':>16} {'-> focal_length fy':>14} "
            f"{'residu reel':>13}",
        ]
        for millimetres in (0, 1, 2, 3, 5, 8):
            e = -millimetres / 1000        # en retrait, comme dans le tube
            rows.append(
                f"  {millimetres:>11} mm {erreur_off_axis_offset(e):>13.1f} px "
                f"{fy_nue*grandissement_section(e):>11.1f} px "
                f"{residu_section(e):>10.2f} px")
        rows += [
            f"\n  Le mounting actuel est a {1000*abs(gap):.1f} mm de l'axis : "
            f"residu {residu_section():.2f} px,",
            f"  a comparer au noise de detection measurement de {CORNER_NOISE_PX:.3f} px.",
            "  -> le centrage n'a pas besoin d'etre parfait ; la calibration suffit.",
            "",
            f"  EN REVANCHE la camera ne doit plus bouger apres calibration :",
            f"  {sensibilite_slip():.1f} % d'error sur toutes les distances "
            "par mm de slip",
            f"  ({sensibilite_slip()*30:.0f} mm d'error a 3 m pour 1 mm de "
            "jeu dans le support).",
            "",
            "  CE QUE LA CALIBRATION EN AIR VA DIRE",
            f"    fx doit retomber sur {K_BARE_AIR[0,0]:.1f} px : in air la lame "
            "plane ne devie rien,",
            "    donc tout gap la-dessus est un probleme de mounting, pas "
            "d'optics.",
            f"    fy doit valoir "
            f"{fy_nue*grandissement_section(indice_exterieur=AIR_INDEX):.1f} px "
            f"({100*(grandissement_section(indice_exterieur=AIR_INDEX)-1):+.2f} %) "
            "si la pupil est bien ou",
            "    on la croit. C'est ce report-la qui MESURE le off_axis_offset reel.",
        ]

    rows += ["", "CE QUE COUTE LA LAME PLANE (le long du tube)",
               "  Le model current multiplie la focal_length par l'index de l'water.",
               "  Voici ou tombe vraiment le radius, et ou ce model le croit :",
               f"\n  {'angle dans l water':>18} {'exact':>10} {'model':>10} {'gap':>9}"]
    for angle, exact, paraxial in ecart_lame_plane():
        rows.append(f"  {angle:>15} deg {exact:>8.1f} px {paraxial:>8.1f} px "
                      f"{exact-paraxial:>+7.1f} px")
    rows.append(f"\n  Le model reste a 1 px pres jusqu'a "
                  f"{angle_modele_fiable():.0f} deg de l'axis seulement, et le noise "
                  "de detection")
    rows.append(f"  measurement vaut {CORNER_NOISE_PX:.3f} px. "
                  "Seule une calibration en water corrige cela.")

    soucis = verifier_montage()
    if soucis:
        rows += ["", "A VERIFIER", "-" * 74]
        for numero, souci in enumerate(soucis, 1):
            rows.append(f"  {numero}. {souci}")

    rows += ["", "CALIBRATIONS ENREGISTREES"]
    for mounting in MOUNTINGS:
        path = MOUNTINGS_FOLDER / f"{mounting}.npz"
        if path.exists():
            K, _ = load(mounting, quiet=True)
            rows.append(f"  {mounting:10} fx = {K[0,0]:8.2f}  fy = {K[1,1]:8.2f}   "
                          f"{path.name}")
        elif mounting == "nue_air":
            rows.append(f"  {mounting:10} fx = {K_BARE_AIR[0,0]:8.2f}  "
                          f"fy = {K_BARE_AIR[1,1]:8.2f}   (en dur dans optics.py)")
        else:
            rows.append(f"  {mounting:10} {'—':>8}     pas encore measurement  "
                          f"(calibrate.py --mounting {mounting})")

    rows.append("=" * 74)
    return "\n".join(rows)


if __name__ == "__main__":
    print(report())
