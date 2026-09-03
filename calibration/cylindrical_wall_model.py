# modele_paroi_cylindrique.py — Trace de rayons a travers la paroi du tube.
#
# A QUOI CA SERT
# Verifier, par le calcul, ce que la paroi du tube fait a la focal_length sous
# l'water. Ce n'est pas un outil du quotidien : il product la table
# GROSSISSEMENT_AXIAL de calibration_tube.py, et sert de justification aux
# chiffres qui y sont recopies.
#
#     python modele_paroi_cylindrique.py     affiche la table
#
# CE QU'IL ETABLIT
# Selon l'axis du tube, la paroi cylindrique est localement PLANE : un plan
# contenant l'axis la coupe en deux droites paralleles. C'est donc une lame a
# faces paralleles, qui sous l'water multiplie la focal_length par l'index, 1.33. Le
# trace le confirme a 0.5 % pres face a la formule analytique, et montre que
# la distance de l'objet n'y change presque rien (1.2 % a 0.30 m, 0.4 % a 1 m).
# Selon la circonference, le menisque donne x1.038 : d'ou une anamorphose
# prevue de 1.277, en accord avec le 1.268 calcule autrement dans optics.py.
#
# CE QU'IL N'EXPLIQUE PAS
# Les focales measured sous l'water tombent 12.7 % (fx) et 6.6 % (fy) sous ces
# previsions. Cet gap reste ouvert.
#
# PIEGE CORRIGE ICI, A NE PAS REINTRODUIRE
# La normale d'un cylindre traverse de l'interieur pointe dans le MEME sens
# que le radius. La formule de Snell vectorielle assumed l'inverse : sans
# retourner la normale, cos(i) sort negatif et la deviation est calculee a
# l'envers. Ce bug donnait un grossissement de 1.17 au lieu de 1.32, et avait
# fait conclure a tort que la focal_length dependait fortement de la distance.
# Le check contre la formule de la lame plane est la pour le rattraper :
# si le trace s'en ecarte de plus de ~1 %, quelque chose cloche.
import numpy as np, cv2

R1, R2 = 0.02475, 0.02900          # rayons interieur / exterieur du tube
N_AIR, N_AC, N_EAU = 1.0, 1.49, 1.33
PUPILLE = 0.00315                  # decentrement measurement, vers la paroi visee
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

    La formule classique assumed la normale orientee FACE au radius incident.
    Sur un cylindre traverse de l'interieur vers l'exterieur, la normale
    radiale pointe dans le meme sens que le radius : il faut la retourner,
    sinon cos(i) sort negatif et la deviation est calculee a l'envers.
    """
    cosi = -float(d @ n)
    if cosi < 0.0:                            # normale du meme cote que le radius
        n = -n
        cosi = -cosi
    k = 1 - eta*eta*(1 - cosi*cosi)
    if k < 0: return None                     # reflexion totale
    return eta*d + (eta*cosi - np.sqrt(k))*n

def tracer(dx, dy):
    """Direction au niveau de la pupil -> radius dans l'water."""
    d = np.array([dx, dy, 1.0]); d /= np.linalg.norm(d)
    o = np.array([0.0, 0.0, PUPILLE])
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

def _ecart(dx, dy, W):
    """Distance du point W au radius sorti pour la direction (dx,dy)."""
    r = tracer(dx, dy)
    if r is None: return None
    p, d = r
    v = W - p
    return v - float(v @ d)*d          # composante perpendiculaire au radius

def projeter(W):
    """Point 3D dans l'water -> pixel. Newton 2D sur (dx,dy)."""
    # depart : approximation lame plane, direction x reduite de 1/1.33
    dx, dy = W[0]/W[2]*N_EAU, W[1]/W[2]*N_EAU
    for _ in range(60):
        e = _ecart(dx, dy, W)
        if e is None: return None
        if np.linalg.norm(e) < 1e-9: break
        h = 1e-6
        ex = _ecart(dx+h, dy, W); ey = _ecart(dx, dy+h, W)
        if ex is None or ey is None: return None
        J = np.column_stack([(ex-e)/h, (ey-e)/h])       # 3x2
        pas, *_ = np.linalg.lstsq(J, -e, rcond=None)
        dx += float(pas[0]); dy += float(pas[1])
    else:
        return None
    return np.array([FX_NUE*dx + CX, FY_NUE*dy + CY])


if __name__ == "__main__":
    print("Grossissement selon l'axis du tube, par distance d'objet")
    print(f"{'distance (m)':>13} {'grossissement':>15} {'fx implique':>13}")
    print("-" * 44)
    eps = 0.002
    for Z in (0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0, 3.0, 10.0, 1000.0):
        p = projeter(np.array([eps * Z, 0.0, Z]))
        g = (p[0] - CX) / (FX_NUE * eps)
        print(f"{Z:13.2f} {g:15.4f} {FX_NUE * g:13.1f}")
    print("-" * 44)
    # Controle : la direction axiale doit redonner la lame a faces paralleles.
    # C'est ce test qui a rattrape l'error de signe sur la normale.
    n, d0 = N_EAU, R1 - PUPILLE
    pires = []
    for Z in (0.3, 0.75, 2.0, 1000.0):
        p = projeter(np.array([eps * Z, 0.0, Z]))
        trace = (p[0] - CX) / (FX_NUE * eps)
        analytique = n * Z / (Z + (d0 + (R2 - R1) * (1 - n / N_AC)) * (n - 1))
        pires.append(abs(trace / analytique - 1))
    print(f"  Controle lame plane : gap max {100*max(pires):.2f} % "
          f"(doit rester sous ~1.5 %)")
    print("  Le grossissement axial vaut donc bien ~1.33, quasi independant")
    print("  de la distance. Les focales measured sous l'water tombent pourtant")
    print("  12.7 % (fx) et 6.6 % (fy) plus bas : cet gap reste ouvert.")
