import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid

#----------------------Densidad superficial de número de King-------------------
def num_surface_density(R, rc, rt, k):
    """Densidad superficial de número de King Sigma(R)."""
    R = np.asarray(R, dtype=float)
    
    term1 = 1.0 / np.sqrt(1.0 + (R / rc)**2)
    term2 = 1.0 / np.sqrt(1.0 + (rt / rc)**2)
    return k * (term1 - term2)**2

#-------------------Densidad espaciald de número de King------------------------
def num_density(r, rc, rt, k):
    """
    Densidad espacial de número de King n(r).
    Válida solo para r < rt; retorna 0 fuera.
    """
    r = np.asarray(r, dtype=float)
    values = np.zeros_like(r)
    valid = r < rt
 
    rv = r[valid]

    # w = sqrt[(1 + (r/rc)^2) / (1 + (rt/rc)^2)]
    w = np.clip(
        np.sqrt((1.0 + (rv / rc)**2) / (1.0 + (rt / rc)**2)),
        1e-12, 1.0 - 1e-12,
    )
 
    C = k / (np.pi * rc * (1.0 + (rt / rc)**2)**1.5)
    term1 = 1.0 / w**2
    term2 = (1.0 / w) * np.arccos(w) - np.sqrt(1.0 - w**2)
    values[valid] = C * term1 * term2
    return values

#--------------------------Proyección al plano tangente-------------------------
def tangent_plane_projection(ras, decs, ra0, dec0, d0):
    """
    Proyección gnomónica de (RA, Dec) al plano tangente centrado en
    (ra0, dec0). Devuelve coordenadas cartesianas (X, Y) en parsecs.
 
    Parámetros
    ----------
    ras, decs : array  — coordenadas de los miembros en grados
    ra0, dec0 : float  — coordenadas del centro del cúmulo en grados (escalares)
    d0        : float  — distancia al cúmulo en parsecs              (escalar)
    """

    ras_r = np.radians(ras)
    decs_r = np.radians(decs)
    ra0_r = np.radians(ra0)   # escalar
    dec0_r = np.radians(dec0)  # escalar
 
    delta_ra = ras_r - ra0_r
 
    den = (np.sin(dec0_r) * np.sin(decs_r) +
           np.cos(dec0_r) * np.cos(decs_r) * np.cos(delta_ra))
 
    xi = np.cos(decs_r) * np.sin(delta_ra) / den
    eta = (np.cos(dec0_r) * np.sin(decs_r) -
           np.sin(dec0_r) * np.cos(decs_r) * np.cos(delta_ra)) / den
 
    X = xi  * d0
    Y = eta * d0
    return X, Y

#-----------------------------Rejection sampling--------------------------------
def rejection_sampling(R, rc, rt, k, seed=23):
    """
    Para cada estrella con radio proyectado R < rt, muestrea Z tal que
    la distribución 3D siga n(r) de King.
 
    Las estrellas con R >= rt reciben Z = NaN.
    """
    rng = np.random.default_rng(seed)
    Z_samples = np.full_like(R, np.nan, dtype=float)
 
    mask = R < rt
    R_valid = R[mask]
    N = len(R_valid)
    if N == 0:
        return Z_samples
 
    # Límite superior del rango de Z para cada estrella
    Z_max = np.sqrt(np.clip(rt**2 - R_valid**2, 0.0, None))
 
    # Máximo de n(r) dado R fijo ocurre en Z=0 ya que r = R
    rho_max = num_density(R_valid, rc, rt, k)
 
    Z_valid = np.zeros(N)
    unaccepted = np.ones(N, dtype=bool)
 
    while np.any(unaccepted):
        idx = np.nonzero(unaccepted)[0]
        Z_cand = rng.uniform(-Z_max[idx], Z_max[idx])
        r_cand = np.sqrt(R_valid[idx]**2 + Z_cand**2)
        rho_c = num_density(r_cand, rc, rt, k)
        u = rng.uniform(0.0, rho_max[idx])
        accept = u <= rho_c
        Z_valid[idx[accept]]    = Z_cand[accept]
        unaccepted[idx[accept]] = False
 
    Z_samples[mask] = Z_valid
    return Z_samples