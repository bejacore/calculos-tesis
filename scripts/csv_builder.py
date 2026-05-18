import sys
from pathlib import Path

# Agrega la raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import utils
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================
def calculate_num_bins(data):
    '''
    Calcula el número de bins adecuado para cada cúmulo usando la regla N^1/2.
    '''

    N = len(data)
    num_bins = int(np.sqrt(N))
    return num_bins

def calculate_num_surface_density(Rs, num_bins=30):
    counts, bin_edges = np.histogram(Rs, bins=num_bins)
    areas = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)

    Rs_mid = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    Sigma_obs = counts / areas
    return Rs_mid, Sigma_obs

def calculate_num_density(rs, num_bins=30):
    counts, bin_edges = np.histogram(rs, bins=num_bins)
    vols = (4.0/3.0) * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
    
    rs_mid = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    n_obs = counts / vols
    return rs_mid, n_obs

def calculate_sig2_obs(rs, Ms, num_bins=30, min_stars=3):
    """
    Dispersión de velocidades observada por bins usando la ecuación de Jeans.
 
    Parámetros
    ----------
    rs: radios 3D de los miembros válidos (pc)
    Ms: masas de los miembros válidos (M_sun)
    rc: parámetros del ajuste 3D de King
    """

    G = 4.300e-3 # pc * M_sun^-1 * (km/s)^2
 
    rs_bins = np.linspace(np.min(rs), np.max(rs), num_bins + 1)
    rs_mid = 0.5 * (rs_bins[1:] + rs_bins[:-1])
 
    rs_mid_valid = []
    rhos_list = []
    M_acc_bin = []
 
    for i in range(num_bins):
        in_bin = (rs >= rs_bins[i]) & (rs < rs_bins[i + 1])
        num_stars = np.sum(in_bin)
 
        if num_stars < min_stars:
            continue
 
        vol = (4.0/3.0) * np.pi * (rs_bins[i+1]**3 - rs_bins[i]**3)
        mass_in_bin = np.sum(Ms[in_bin])
        rho = mass_in_bin / vol
 
        rs_mid_valid.append(rs_mid[i])
        rhos_list.append(rho)
        M_acc_bin.append(mass_in_bin)
 
    rs_mid_valid = np.array(rs_mid_valid)
    rhos = np.array(rhos_list)
    M_acc_bin = np.cumsum(M_acc_bin)
 
    # Integral de Jeans
    integrand = rhos * M_acc_bin / rs_mid_valid**2
    int_0_r = cumulative_trapezoid(integrand, rs_mid_valid, initial=0)
    int_r_rt = int_0_r[-1] - int_0_r
 
    sig2 = G / rhos * int_r_rt
    return rs_mid_valid, sig2

def sig2_model_for_fit(rs_obs, M_mean, rc, rt, k):
    """
    Envolvente de calculate_sig2 para curve_fit.
 
    Evalúa sig^2(r) sobre un grid denso y luego interpola en los puntos 
    observados rs_obs.
    """

    # Grid interno denso
    r_min = max(0.01 * rc, 1e-3)
    r_max = max(rt, np.max(rs_obs) * 1.05)
    rs_grid = np.linspace(r_min, r_max, 2000)
 
    sig2_grid = utils.calculate_sig2(rs_grid, M_mean, rc, rt, k)
 
    # Interpolación lineal en los puntos observados
    sig2_interp = np.interp(rs_obs, rs_grid, sig2_grid)
    return sig2_interp

# ==============================================================================
# PROCESAMIENTO DE DATOS
# ==============================================================================
def process_and_export_data(path_clusters, path_members):
    '''
    Procesa los datos de múltiples cúmulos y exporta sus perfiles radiales. 
    '''
    # Leer las tablas de cúmulos y miembros
    clusters = pd.read_csv(path_clusters)
    members = pd.read_csv(path_members)

    # Lista para almacenar los datos globales del cúmulo
    global_data = []

    clusters_groups = members.groupby('Name')

    id = 0
    for cluster, cluster_members in clusters_groups:
        # Extraer información del cúmulo
        num_members = len(cluster_members)

        ra0 = clusters[clusters['Name'] == cluster]['RA_ICRS'].values[0]
        dec0 = clusters[clusters['Name'] == cluster]['DE_ICRS'].values[0]
        d0 = clusters[clusters['Name'] == cluster]['dist50'].values[0]

        ras = cluster_members['RA_ICRS'].values
        decs = cluster_members['DE_ICRS'].values
        Ms = cluster_members['Mass50'].values

        # -------------------- Ajuste perfil superficial -----------------------
        # Realizar proyección gnomónica
        X, Y = utils.tangent_plane_projection(ras, decs, ra0, dec0, d0)

        # Calcular radios superficiales
        Rs = np.sqrt(X**2 + Y**2)

        # Calcular el número de bins
        num_bins_2d = calculate_num_bins(Rs)

        # Calcular densidad de número observada
        Rs_mid, Sigma_obs = calculate_num_surface_density(Rs, num_bins_2d)

        # Estimaciones iniciales para el ajuste
        k_guess = np.max(Sigma_obs)
        rc_guess = np.median(Rs) * 0.5
        rt_guess = np.max(Rs)

        # Realizar ajuste
        popt_2d, pcov_2d = curve_fit(
        utils.num_surface_density,
            Rs_mid, Sigma_obs,
            p0=[rc_guess, rt_guess, k_guess],
            bounds=(0, np.inf),
            maxfev=10000,
        )

        # Valores obtenidos
        rc_2d, rt_2d, k_2d = popt_2d

        # ----------------------- Ajuste perfil espacial -----------------------
        # Estimar coordenada Z
        Z = utils.rejection_sampling(Rs, rc_2d, rt_2d, k_2d, seed=23)

        # Quitar valores que no sean finitos
        valid_mask = np.isfinite(Z)
        X_v = X[valid_mask]
        Y_v = Y[valid_mask]
        Z_v = Z[valid_mask]
        Ms_v = Ms[valid_mask]

        # Calcular radios espaciales
        rs = np.sqrt(X_v**2 + Y_v**2 + Z_v**2)

        # Calcular el número de bins
        num_bins_3d = calculate_num_bins(rs)

        # Calcular densidad de número espacial
        rs_mid, n_obs = calculate_num_density(rs, num_bins_3d)

        # Realizar ajuste
        popt_3d, pcov_3d = curve_fit(
            utils.num_density,
            rs_mid, n_obs,
            p0=[rc_2d, rt_2d, k_2d], # se usa el ajuste 2D
            bounds=(0, np.inf),
            maxfev=10000,
        )

        # Valores obtenidos
        rc_3d, rt_3d, k_3d = popt_3d

        # ------------------ Ajuste dispersión de velocidades ------------------
        rs_mid_sig2, sig2_obs = calculate_sig2_obs(rs, Ms_v, num_bins_3d)

        # Parámetros de ajuste
        M_mean = np.mean(Ms_v)
        M_min = np.min(Ms_v)
        M_max = np.max(Ms_v)

        # Realizar ajuste
        popt_sig2, pcov_sig2 = curve_fit(
            sig2_model_for_fit,
            rs_mid_sig2, sig2_obs,
            p0=[M_mean, rc_3d, rt_3d, k_3d],
            bounds=(
                [M_min * 0.1, 0.01, 0.01, 1e-6], 
                [M_max * 10, np.inf, np.inf, np.inf]
                ),
            maxfev=10000,
        )

        M_mean_sig2, rc_sig2, rt_sig2, k_sig2 = popt_sig2

        global_data.append({
            'id': id,
            'nombre': cluster,
            'num_miembros': num_members,
            'rc_2d': rc_2d,
            'rt_2d': rt_2d,
            'k_2d': k_2d,
            'rc_3d': rc_3d,
            'rt_3d': rt_3d,
            'k_3d': k_3d,
            'rc_sig2': rc_sig2,
            'rt_sig2': rt_sig2,
            'k_sig2': k_sig2,
            'M_mean_sig2': M_mean_sig2
        })

        id += 1

# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    clusters_table = 'data/processed/largest_clusters.csv'
    members_table = 'data/processed/largest_clusters_members.csv'
    
    process_and_export_data(clusters_table, members_table)