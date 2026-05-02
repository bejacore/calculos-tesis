import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

#===============================================================================
# MODELOS A AJUSTAR
#===============================================================================
def rho(r, rc, rt, k):
  values = np.zeros_like(r, dtype=float)

  mask = r < rt
  r_valid = r[mask]

  w = np.sqrt((1 + (r_valid/rc)**2) / (1 + (rt/rc)**2))

  C = k / (np.pi * rc * (1 + (rt/rc)**2)**1.5)

  term1 = 1.0 / (w**2)
  term2 = (1.0 / w) * np.arccos(w) - np.sqrt(1 - w**2)

  values[mask] = C * term1 * term2
  
  return values

def densidad_king_adimensional(r, rc, rt):
    """
    Calcula la forma de la densidad espacial de King (sin normalizar).
    Equivale a la ecuación 27 de King 1962, ignorando el factor k.
    """
    # Evitar evaluar radios mayores a rt
    r_val = np.clip(r, 0, rt * 0.99999)
    
    # Variable z del modelo de King
    z = np.sqrt((1 + (r_val/rc)**2) / (1 + (rt/rc)**2))
    
    # Ecuación de densidad
    n = (1.0 / z**2) * ((1.0 / z) * np.arccos(z) - np.sqrt(1 - z**2))
    
    # La densidad es estrictamente 0 más allá del radio de marea
    n[r >= rt] = 0.0
    return n

def modelo_sigma2(r_bins, A, rc, rt):
    """
    Evalúa la integral de Jeans para la dispersión de velocidades.
    r_bins: Radios donde tienes tus datos.
    A: Amplitud de la dispersión de velocidades (parámetro libre).
    rc: Radio del núcleo (parámetro libre).
    rt: Radio de marea (parámetro libre).
    """
    # Crear una grilla densa desde el centro hasta el radio de marea
    r_grid = np.linspace(1e-5, rt, 2000)
    
    # Densidad en la grilla
    n_grid = densidad_king_adimensional(r_grid, rc, rt)
    
    # Masa encerrada (acumulada) en la grilla
    # M(<r) ~ integral( r^2 * n(r) dr )
    integrando_M = r_grid**2 * n_grid
    M_grid = cumulative_trapezoid(integrando_M, r_grid, initial=0)
    
    # Integral desde r hasta rt de (n * M / r^2)
    integrando_jeans = n_grid * M_grid / r_grid**2
    
    # integral_r_rt = integral_0_rt - integral_0_r
    integral_0_r = cumulative_trapezoid(integrando_jeans, r_grid, initial=0)
    integral_r_rt = integral_0_r[-1] - integral_0_r
    
    # Dispersión de velocidades teórica en la grilla
    sigma2_grid = np.zeros_like(r_grid)
    mask = n_grid > 0
    sigma2_grid[mask] = A * integral_r_rt[mask] / n_grid[mask]
    
    # Interpolar los resultados teóricos a los r_bins
    interpolador = interp1d(r_grid, sigma2_grid, kind='cubic', 
                            fill_value=0.0, bounds_error=False)
    
    return interpolador(r_bins)