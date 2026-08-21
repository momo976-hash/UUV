# modele_paroi_cylindrique.py — Trace de rayons a travers la paroi du tube.
#
# POURQUOI CE FICHIER EXISTE
# On a longtemps attendu que la focale sous l'eau vaille 1.33 x la focale en
# air, et rejete comme fausses des calibrations qui donnaient 1.17. Le 1.33
# est la limite d'un hublot PLAN pour un objet a l'INFINI. Notre paroi est
# CYLINDRIQUE et le damier est a moins d'un metre : aucune des deux
# hypotheses ne tient.
#
# Ce module trace les rayons a travers les deux dioptres courbes reels
# (air -> acrylique -> eau) et montre que le grossissement axial monte de
# 0.97 a 0.30 m jusqu'a 1.33 seulement a l'infini. Le systeme n'a donc PAS
# de point de vue unique sous l'eau : la focale qu'une calibration en retire
# depend des distances auxquelles le damier a ete tenu.
#
# La table GROSSISSEMENT_AXIAL de calibration_tube.py en est tiree.
#
#     python modele_paroi_cylindrique.py     affiche la table
#
# CE QUE CE MODELE NE FAIT PAS. Il reproduit fx (mesure 711 a 0.75 m, prevu
# 703) mais PAS fy : selon le signe du decentrement de pupille il donne 525
# ou 1040 la ou l'on mesure 596. La partie menisque reste donc a comprendre,
# et l'anamorphose predite (1.27) ne doit pas etre prise pour acquise.
import numpy as np, cv2

R1, R2 = 0.02475, 0.02900          # rayons interieur / exterieur du tube
N_AIR, N_AC, N_EAU = 1.0, 1.49, 1.33
PUPILLE = 0.00315                  # decentrement mesure, vers la paroi visee
FX_NUE, FY_NUE = 615.56, 615.09    # camera nue, mesuree aujourd'hui
CX, CY = 320.0, 240.0

def _inter_cylindre(o, d, R):
    """Intersection rayon / cylindre d'axe X, en marchant vers +z."""
    a = d[1]**2 + d[2]**2
    b = 2*(o[1]*d[1] + o[2]*d[2])
    c = o[1]**2 + o[2]**2 - R*R
    disc = b*b - 4*a*c
    if disc < 0: return None
    t = (-b + np.sqrt(disc)) / (2*a)
    return o + t*d if t > 1e-12 else None

def _refracter(d, n, eta):
    """Snell vectoriel. n : normale unitaire sortante. eta = n1/n2."""
    cosi = -float(d @ n)
    k = 1 - eta*eta*(1 - cosi*cosi)
    if k < 0: return None                     # reflexion totale
    return eta*d + (eta*cosi - np.sqrt(k))*n

def tracer(dx, dy):
    """Direction au niveau de la pupille -> rayon dans l'eau."""
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
    """Distance du point W au rayon sorti pour la direction (dx,dy)."""
    r = tracer(dx, dy)
    if r is None: return None
    p, d = r
    v = W - p
    return v - float(v @ d)*d          # composante perpendiculaire au rayon

def projeter(W):
    """Point 3D dans l'eau -> pixel. Newton 2D sur (dx,dy)."""
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
    print("Grossissement selon l'axe du tube, par distance d'objet")
    print(f"{'distance (m)':>13} {'grossissement':>15} {'fx implique':>13}")
    print("-" * 44)
    eps = 0.002
    for Z in (0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0, 3.0, 10.0, 1000.0):
        p = projeter(np.array([eps * Z, 0.0, Z]))
        g = (p[0] - CX) / (FX_NUE * eps)
        print(f"{Z:13.2f} {g:15.4f} {FX_NUE * g:13.1f}")
    print("-" * 44)
    print("  A l'infini on retrouve le 1.33 des manuels ; a 0.75 m, 1.16.")
    print("  C'est tout l'ecart qu'on a passe des jours a chercher.")
