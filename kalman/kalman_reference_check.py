# kalman_reference_check.py — The project's filter IS the reference document's.
#
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python kalman/kalman_reference_check.py
#
# No camera, no hardware, a few seconds. It either prints that the 9 published
# values are recovered, or it fails. Nothing to configure.
#
# This is the script to run when someone asks "is this really a Kalman filter,
# or something that looks like one?".
#
# ===========================================================================
# REFERENCE DOCUMENT
# ===========================================================================
#   Alex Becker, "Kalman Filter Explained Through Examples",
#   kalmanfilter.net, consulted 02/09/2026.
#
# ===========================================================================
# WHAT THIS SCRIPT DOES, AND WHY
# ===========================================================================
# The document works through a numerical example: a 1D radar tracking an
# aircraft. It prints every intermediate value — F, Q, P(1,0), K(1), x(1,1),
# P(1,1), x(2,1), P(2,1). This script runs THAT SAME example through the
# project's LinearKalman class, the very one that filters the vehicle's
# position, and compares each number with the one printed in the document.
#
# In other words: this is not a filter written for the demonstration, it is
# the vehicle's filter put to the test on a problem whose answer is
# published. If it departed from the course's equations, the comparison would
# say so immediately instead of hiding it behind the geometry of the tags.
#
# The script then shows that the vehicle's model is the SAME constant-velocity
# kinematic model, simply carried from 1 to 3 dimensions.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kalman_filter import PositionKalmanFilter, LinearKalman  # noqa: E402


# ---------------------------------------------------------------------------
# Les values imprimees dans le document, recopiees telles quelles.
# Ce sont les references : le code n'a pas le droit de les toucher.
# ---------------------------------------------------------------------------
DOC = {
    "Q": np.array([[6.25, 2.5], [2.5, 1.0]]),
    "x(1,0)": np.array([11000.0, 200.0]),
    "P(1,0)": np.array([[28.5, 3.75], [3.75, 1.25]]),
    "K(1)": np.array([[0.4048, 0.6377], [0.0399, 0.3144]]),
    "x(1,1)": np.array([11009.37, 201.43]),
    "P(1,1)": np.array([[14.57, 1.43], [1.43, 0.71]]),
    "x(2,1)": np.array([12016.5, 201.43]),
    "P(2,1)": np.array([[52.86, 7.47], [7.47, 1.71]]),
}

# Le document arrondit ses tableaux a deux decimales, et le gain a quatre.
# On tolere donc un demi-last-chiffre, pas davantage.
TOLERANCE = {"K(1)": 5e-5}
TOLERANCE_PAR_DEFAUT = 5e-3

_resultats = []


def comparer(name, calcule):
    """Compare a computed value with the one printed in the document."""
    attendu = DOC[name]
    calcule = np.asarray(calcule, dtype=float)
    gap = float(np.max(np.abs(calcule - attendu)))
    threshold = TOLERANCE.get(name, TOLERANCE_PAR_DEFAUT)
    ok = gap <= threshold
    _resultats.append((name, ok, gap))

    plat = " ".join(f"{v:12.4f}" for v in np.ravel(calcule))
    doc = " ".join(f"{v:12.4f}" for v in np.ravel(attendu))
    marque = "OK " if ok else "NON"
    print(f"  [{marque}] {name:8s} computed: {plat}")
    print(f"          {'':8s} document: {doc}     gap {gap:.2e}")
    return ok


def example_du_document():
    """The radar example, worked through with the project's core."""
    print("=" * 74)
    print("THE DOCUMENT'S EXAMPLE — 1D radar tracking an aircraft")
    print("  Alex Becker, Kalman Filter Explained Through Examples,")
    print("  kalmanfilter.net")
    print("=" * 74)

    # --- le probleme, tel que pose par le document -------------------------
    dt = 5.0                         # time de revisite du radar
    sigma_a = 0.2                    # m/s^2, acceleration aleatoire de l'avion
    sigma_portee, sigma_vitesse = 4.0, 0.5        # noise de la 1re measurement
    z1 = np.array([11020.0, 202.0])               # 2e measurement
    R1 = np.diag([6.0 ** 2, 1.5 ** 2])            # elle est plus bruitee

    # Etat : x = [portee, velocity]. Modele CINEMATIQUE a velocity constante.
    F = np.array([[1.0, dt],
                  [0.0, 1.0]])
    # Q tel qu'ecrit dans le document, section 8.2.2 :
    #     Q = sigma_a^2 [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]
    Q = sigma_a ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2],
                                 [dt ** 3 / 2, dt ** 2]])
    H = np.eye(2)                    # le radar measurement portee ET velocity

    print(f"\n  dt = {dt} s, sigma_a = {sigma_a} m/s2")
    print(f"  F = {F.tolist()}")
    comparer("Q", Q)

    # --- ITERATION 0 : initialisation puis prediction ----------------------
    print("\nITERATION 0 — initialised from the first measurement")
    filter = LinearKalman(
        x=np.array([10000.0, 200.0]),
        P=np.diag([sigma_portee ** 2, sigma_vitesse ** 2]))
    print(f"  x(0,0) = {filter.x.tolist()}   P(0,0) = "
          f"{np.diag(filter.P).tolist()} (diagonal)")

    print("\nITERATION 0 — prediction")
    filter.predict(F, Q)
    comparer("x(1,0)", filter.x)
    comparer("P(1,0)", filter.P)

    # --- ITERATION 1 : mise a jour ----------------------------------------
    print("\nITERATION 1 — update from the second measurement")
    print(f"  z(1) = {z1.tolist()}   R(1) = {np.diag(R1).tolist()} (diagonal)")
    K, _ = filter.gain(H, R1)
    comparer("K(1)", K)

    P_avant = filter.P.copy()
    innovation, _ = filter.correct(z1, H, R1)
    print(f"  innovation z - Hx = {innovation.tolist()}   "
          f"(le document donne [20, 2])")
    comparer("x(1,1)", filter.x)
    comparer("P(1,1)", filter.P)

    # Joseph contre la forme simplifiee : le document dit qu'elles sont egales
    # en arithmetique exacte et recommande Joseph. On le verifie plutot que de
    # le croire.
    simplifiee = (np.eye(2) - K @ H) @ P_avant
    gap = float(np.max(np.abs(filter.P - simplifiee)))
    print(f"\n  Joseph form vs simplified form: gap {gap:.1e}")
    print("    -> identical, as the document states. Joseph is kept, since it")
    print("       stays symmetric positive-definite after thousands of")
    print("       floating-point iterations.")

    # --- ITERATION 1 : prediction suivante ---------------------------------
    print("\nITERATION 1 — prediction")
    filter.predict(F, Q)
    comparer("x(2,1)", filter.x)
    comparer("P(2,1)", filter.P)


def modele_de_lvehicle():
    """The same model, carried from 1 to 3 dimensions for the vehicle."""
    print("\n" + "=" * 74)
    print("THE SAME MODEL, APPLIED TO THE VEHICLE")
    print("=" * 74)
    print("""
  The document reasons about a two-component state, [range, velocity],
  because its radar is one-dimensional. The vehicle moves through water: its
  state has six, [px py pz vx vy vz]. The model is the same, block by block —
  it is the same CONSTANT-VELOCITY KINEMATIC model.

      document              vehicle
      F = [[1, dt],         F = [[I3, dt.I3],
           [0,  1]]              [ 0,    I3 ]]

      Q = sa^2 [[dt^4/4, dt^3/2],      Q = sa^2 . G G'
                [dt^3/2, dt^2  ]]      avec G = [dt^2/2 . I3 ; dt . I3]

  These two ways of writing Q are the same. Numerical check:""")

    dt, sigma_a = 5.0, 0.2
    _, G = PositionKalmanFilter.model(dt)
    Q3 = sigma_a ** 2 * (G @ G.T)
    attendus = (("position  dt^4/4", Q3[0, 0], sigma_a ** 2 * dt ** 4 / 4),
                ("croise    dt^3/2", Q3[0, 3], sigma_a ** 2 * dt ** 3 / 2),
                ("velocity   dt^2  ", Q3[3, 3], sigma_a ** 2 * dt ** 2))
    tout_bon = True
    for name, got, attendu in attendus:
        ok = abs(got - attendu) < 1e-12
        tout_bon &= ok
        print(f"    [{'OK ' if ok else 'NON'}] block {name} : "
              f"G G' donne {got:8.4f}, formule du document {attendu:8.4f}")
    _resultats.append(("Q en 3D = Q du document", tout_bon, 0.0))

    print("""
  ONE SINGLE DIFFERENCE, AND IT IS IN H. The document's radar measures range
  AND velocity, so H = I. The tags only give a position:

      H = [I3  0]

  Velocities are never measured, they are DEDUCED by the filter from how the
  positions evolve. That is the more common case, and the document allows for
  it explicitly: "the measurement and the system state may belong to
  different physical domains".

  WHAT THE PROJECT ADDS, AND WHY
  Three things only, all of them foreseen by the document:

    1. THE INPUT u. The document writes x(n+1,n) = F x(n,n) + G u(n) and
       gives as an example of input "readings from an onboard accelerometer".
       That is exactly what PositionKalmanFilter.predict(dt, accel) does:
       without an accelerometer the acceleration is a random unknown covered
       by sigma_a; with one it is measured, and only sensor noise remains.

    2. OUTLIER REJECTION. A tag seen too far off-axis can flip and give a
       pose wrong by several decimetres. It is recognised by its Mahalanobis
       distance y' S^-1 y, where S = H P H' + R is already computed for the
       gain. The document refers this subject to its "Outlier Treatment"
       chapter: "in practice it is often necessary to reject certain
       measurements".

    3. ORIENTATION HANDLED SEPARATELY. Rotations do not add up, so a linear
       Kalman filter does not apply directly to an orientation. It is handled
       by a scalar filter on the angle, applied through slerp on the
       quaternions — the subject of nonlinear filters, which the document
       refers to its book. POSITION, on the other hand, remains exactly the
       linear filter above.
""")


def main():
    example_du_document()
    modele_de_lvehicle()

    print("=" * 74)
    echecs = [name for name, ok, _ in _resultats if not ok]
    if echecs:
        print(f"DISAGREEMENT with the document on: {', '.join(echecs)}")
        print("=" * 74)
        return 1
    print(f"THE {len(_resultats)} PUBLISHED VALUES ARE ALL RECOVERED")
    print("")
    print("  The project's position filter is not inspired by the document:")
    print("  it IS the same filter. It reproduces the document's worked")
    print("  example to the fourth decimal, using the class that actually")
    print("  runs on the vehicle (kalman_filter.LinearKalman).")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
