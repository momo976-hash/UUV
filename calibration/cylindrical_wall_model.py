# cylindrical_wall_model.py — Ray tracing through the tube's curved wall.
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python calibration/cylindrical_wall_model.py
#
# Prints the table and its own self-check. Nothing to install, nothing to
# plug in. You only need it if you doubt calibrate_tube.py's
# AXIAL_MAGNIFICATION table, or want to re-derive it after changing the
# tube's dimensions in optics.py.
# ===========================================================================
#
# A QUOI CA SERT
# To check, by calculation, what the tube's wall does to the focal length
# underwater. It is not an everyday tool: it produces calibrate_tube.py's
# AXIAL_MAGNIFICATION table, and stands as the justification for the numbers
# copied there.
#
#     python modele_wall_cylindrique.py     affiche la table
#
# CE QU'IL ETABLIT
# Along the tube axis, the cylindrical wall is locally FLAT: a plane
# containing the axis cuts it in two parallel straight lines. So it is a
# plane-parallel slab, which underwater multiplies the focal length by the
# index, 1.33. The ray trace confirms that to within 0.5 % against the
# analytical formula, and shows that the object's distance changes almost
# nothing (1.2 % at 0.30 m, 0.4 % at 1 m). Around the circumference, the
# meniscus gives x1.038: whence a predicted anamorphic ratio of 1.277, in
# agreement with the 1.268 computed another way in optics.py.
#
# WHAT IT DOES NOT EXPLAIN
# The focal lengths measured underwater come out 12.7 % (fx) and 6.6 % (fy)
# below these
# previsions. Cet gap reste ouvert.
#
# A TRAP FIXED HERE, NOT TO BE REINTRODUCED
# The normal of a cylinder crossed from the inside points the SAME way as the
# radius. Snell's vector formula assumes the opposite: without flipping the
# normal, cos(i) comes out negative and the deviation is computed backwards.
# That bug gave a magnification of 1.17 instead of 1.32, and had led to the
# wrong conclusion that the focal length depended strongly on distance. The
# check against the plane-slab formula is there to catch it: if the trace
# departs from it by more than ~1 %, something is wrong.
import numpy as np, cv2

R1, R2 = 0.02475, 0.02900          # rayons interieur / exterieur du tube
N_AIR, N_AC, N_EAU = 1.0, 1.49, 1.33
PUPIL = 0.00315                    # measured off-axis offset, towards the wall
FX_NUE, FY_NUE = 615.56, 615.09    # camera nue, measured aujourd'hui
CX, CY = 320.0, 240.0

def _inter_cylindre(o, d, R):
    """Intersection radius / cylindre d'axis X, en marchant vers +z."""
    a = d[1]**2 + d[2]**2
    b = 2*(o[1]*d[1] + o[2]*d[2])
    c = o[1]**2 + o[2]**2 - R*R
    disc = b*b - 4*a*c
    if disc < 0: return None
    t = (-b + np.sqrt(disc)) / (2*a)
    return o + t*d if t > 1e-12 else None

def _refracter(d, n, eta):
    """Snell vectoriel. eta = n1/n2.

    The classic formula assumes the normal faces INTO the incident ray. On a
    cylinder crossed from inside to outside, the radial normal points the
    same way as the ray: it has to be flipped, otherwise cos(i) comes out
    negative and the deviation is computed backwards.
    """
    cosi = -float(d @ n)
    if cosi < 0.0:                        # normal on the same side as the ray
        n = -n
        cosi = -cosi
    k = 1 - eta*eta*(1 - cosi*cosi)
    if k < 0: return None                     # reflexion totale
    return eta*d + (eta*cosi - np.sqrt(k))*n

def tracer(dx, dy):
    """Direction at the pupil -> the ray in the water."""
    d = np.array([dx, dy, 1.0]); d /= np.linalg.norm(d)
    o = np.array([0.0, 0.0, PUPIL])
    p1 = _inter_cylindre(o, d, R1)
    if p1 is None: return None
    n1 = np.array([0.0, p1[1], p1[2]]); n1 /= np.linalg.norm(n1)
    d1 = _refracter(d, n1, N_AIR/N_AC)
    if d1 is None: return None
    p2 = _inter_cylindre(p1, d1, R2)
    if p2 is None: return None
    n2 = np.array([0.0, p2[1], p2[2]]); n2 /= np.linalg.norm(n2)
    d2 = _refracter(d1, n2, N_AC/N_EAU)
    if d2 is None: return None
    return p2, d2 / np.linalg.norm(d2)

def _gap(dx, dy, W):
    """Distance from the point W to the ray emerging for direction (dx, dy)."""
    r = tracer(dx, dy)
    if r is None: return None
    p, d = r
    v = W - p
    return v - float(v @ d)*d          # composante perpendiculaire au radius

def project(W):
    """A 3D point in the water -> a pixel. 2D Newton on (dx, dy)."""
    # start: the plane-slab approximation, x direction reduced by 1/1.33
    dx, dy = W[0]/W[2]*N_EAU, W[1]/W[2]*N_EAU
    for _ in range(60):
        e = _gap(dx, dy, W)
        if e is None: return None
        if np.linalg.norm(e) < 1e-9: break
        h = 1e-6
        ex = _gap(dx+h, dy, W); ey = _gap(dx, dy+h, W)
        if ex is None or ey is None: return None
        J = np.column_stack([(ex-e)/h, (ey-e)/h])       # 3x2
        step, *_ = np.linalg.lstsq(J, -e, rcond=None)
        dx += float(step[0]); dy += float(step[1])
    else:
        return None
    return np.array([FX_NUE*dx + CX, FY_NUE*dy + CY])


if __name__ == "__main__":
    print("Magnification along the tube axis, by object distance")
    print(f"{'distance (m)':>13} {'magnification':>15} {'implied fx':>13}")
    print("-" * 44)
    eps = 0.002
    for Z in (0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0, 3.0, 10.0, 1000.0):
        p = project(np.array([eps * Z, 0.0, Z]))
        g = (p[0] - CX) / (FX_NUE * eps)
        print(f"{Z:13.2f} {g:15.4f} {FX_NUE * g:13.1f}")
    print("-" * 44)
    # Check: the axial direction must give the plane-parallel slab back. It
    # is this test that caught the sign error on the normal.
    n, d0 = N_EAU, R1 - PUPIL
    pires = []
    for Z in (0.3, 0.75, 2.0, 1000.0):
        p = project(np.array([eps * Z, 0.0, Z]))
        trace = (p[0] - CX) / (FX_NUE * eps)
        analytique = n * Z / (Z + (d0 + (R2 - R1) * (1 - n / N_AC)) * (n - 1))
        pires.append(abs(trace / analytique - 1))
    print(f"  Plane-slab check: max gap {100*max(pires):.2f} % "
          f"(must stay under ~1.5 %)")
    print("  So the axial magnification really is ~1.33, near enough")
    print("  independent of distance. And yet the focal lengths measured")
    print("  underwater come out 12.7 % (fx) and 6.6 % (fy) lower: that gap")
    print("  remains open.")
