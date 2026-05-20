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
# PEFILES RADIALES SUPERFICIALES
# ==============================================================================
def generate_surface_profile(Rs, rvs, num_bins=10, min_stars=3):
    # Definir los border de cada bin
    percentiles = np.linspace(0, 100, num_bins + 1)
    edges = np.percentile(Rs, percentiles)
    edges[0] -= 1e-6 # incluir el radio mínimo exacto
    edges[-1] += 1e-6

    Rs_mid = []
    num_stars = []
    num_density = []
    sig_obs = []
    for i in range(num_bins):
        R_lo, R_hi = edges[i], edges[i + 1]
        R_mid = 0.5 * (R_lo + R_hi)

        # Estrellas del bin
        in_bin = (Rs >= R_lo) & (Rs < R_hi)

        # Contar estrellas en el bin
        num_stars_in_bin = np.sum(in_bin)

        if num_stars_in_bin < min_stars:
            continue
        
        # Área del anillo
        area = np.pi * (R_hi**2 - R_lo**2)
        
        # Filtrar estrellas con VR válida dentro del bin
        rvs_in_bin = rvs[in_bin]
        valid_rv = rvs_in_bin[np.isfinite(rvs_in_bin)]

        # Calcular dispersión solo si hay suficientes estrellas con RV
        if len(valid_rv) >= min_stars:
            v_mean = np.mean(valid_rv)
            sig2 = np.sum((valid_rv - v_mean)**2) / (len(valid_rv) - 1)
            sig = np.sqrt(sig2)
        else:
            sig = np.nan  # Bin válido en densidad, pero sin dato de sigma

        # Guardar datos válidos
        Rs_mid.append(R_mid)
        num_stars.append(num_stars_in_bin)
        num_density.append(num_stars_in_bin / area)
        sig_obs.append(sig)

    # Convertir arrays a numpy arrays
    Rs_mid = np.array(Rs_mid)
    num_stars = np.array(num_stars)
    num_density = np.array(num_density)
    sig_obs = np.array(sig_obs)

    # Contruir dataframe con los resultados
    perfil_data = {
        'R_bin': Rs_mid,
        'num_estrellas': num_stars,
        'densidad_num': num_density,
        'sig_obs': sig_obs
    }

    return pd.DataFrame(perfil_data)

# ==============================================================================
# PERFILES RADIALES ESPACIALES
# ==============================================================================
def generate_spatial_profile(rs, Ms, num_bins=10, min_stars=3):
    # Constante gravitacional G (pc * M_sun^-1 * (km/s)^2)
    G = 4.3009e-3 

    # Definir bordes de cada bin
    percentiles = np.linspace(0, 100, num_bins + 1)
    edges = np.percentile(rs, percentiles)
    edges[0] -= -1e-6 # incluir el radio mínimo exacto
    edges[-1] += 1e-6

    rs_mid = []
    num_stars = []
    num_density = []
    rhos = []
    Ms_in_bin = []
    for i in range(num_bins):
        r_lo, r_hi = edges[i], edges[i + 1]
        r_mid = 0.5 * (r_lo + r_hi)

        # Estrellas en el bin
        in_bin = (rs >= r_lo) & (rs < r_hi)

        # Contar estrellas en el bin
        num_stars_in_bin = np.sum(in_bin)

        if num_stars_in_bin < min_stars:
            continue

        # Volumen del cascarón esférico
        vol = (4.0 / 3.0) * np.pi * (r_hi**3 - r_lo**3)

        # Masa en el bin
        mass_in_bin = np.sum(Ms[in_bin])

        # Guardar datos validos
        rs_mid.append(r_mid)
        num_stars.append(num_stars_in_bin)
        num_density.append(num_stars_in_bin / vol)
        rhos.append(mass_in_bin / vol)
        Ms_in_bin.append(mass_in_bin)

    # Convertir arrays a numpy arrays
    rs_mid = np.array(rs_mid)
    num_stars = np.array(num_stars)
    num_density = np.array(num_density)
    rhos = np.array(rhos)
    Ms_acc = np.cumsum(Ms_in_bin)

    # Calcular dispersión de velocidades
    # Se construye el integrando: rho(r) * G * M(<r) / r^2
    integrand = rhos * G * Ms_acc / rs_mid**2

    # Se calcula la integral acumulada de 0 a r
    # (Se usa initial=0 para mantener la misma longitud del array)
    integral_0_r = cumulative_trapezoid(integrand, x=rs_mid, initial=0)

    # La integral de r a rt es la integral total menos la integral hasta r
    integral_r_rt = integral_0_r[-1] - integral_0_r

    # Calcular integral de jeans
    sig2 = integral_r_rt / rhos
    sig = np.sqrt(sig2)

    # Contruir dataframe con los resultados
    perfil_data = {
        'r_bin': rs_mid,
        'num_estrellas': num_stars,
        'densidad_num': num_density,
        'densidad_vol': rhos,
        'sigma': sig,
        'masa_acumulada': Ms_acc
    }

    return pd.DataFrame(perfil_data)

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

    # Crear directorios para los perfiles si no existen
    spatial_output = 'data/processed/perfiles_radiales/espaciales'
    os.makedirs(spatial_output, exist_ok=True)

    surface_output = 'data/processed/perfiles_radiales/superficiales'
    os.makedirs(surface_output, exist_ok=True)

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
        rvs = cluster_members['RV'].values

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

        # Realizar ajuste
        popt_sig2, pcov_sig2 = curve_fit(
            lambda rs_obs, rc, rt, k: sig2_model_for_fit(
                rs_obs, M_mean, rc, rt, k
                ),
            rs_mid_sig2, sig2_obs,
            p0=[rc_3d, rt_3d, k_3d],
            bounds=(
                [0.01, 0.01, 1e-6], 
                [np.inf, np.inf, np.inf]
                ),
            maxfev=10000,
        )

        rc_sig2, rt_sig2, k_sig2 = popt_sig2

        # ----------------------- Datos globales -------------------------------
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
            'k_sig2': k_sig2
            # 'M_mean_sig2': M_mean_sig2
        })

        # ------------------------ Perfil espacial -----------------------------
        # Generar el perfil radial 3D del cúmulo
        spatial_profile = generate_spatial_profile(rs, Ms_v, num_bins_3d)

        # Exportar a un archivo CSV
        file_name = f"cluster_{id}.csv"
        spatial_path = os.path.join(spatial_output, file_name)
        spatial_profile.to_csv(spatial_path, index=False)

        # ----------------------- Perfil superficial ---------------------------
        # Generar el perfil radial 2D del cúmulo
        surface_profile = generate_surface_profile(Rs, rvs, num_bins_2d)

        # Exportar a un archivo CSV
        surface_path = os.path.join(surface_output, file_name)
        surface_profile.to_csv(surface_path, index=False)

        id += 1

    # Exportar los parámetros globales de todos los cúmulos a un archivo CSV
    global_df = pd.DataFrame(global_data)
    global_file = 'data/processed/parametros_globales.csv'
    global_df.to_csv(global_file, index=False)

# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    clusters_table = 'data/processed/largest_clusters.csv'
    members_table = 'data/processed/largest_clusters_members.csv'
    
    process_and_export_data(clusters_table, members_table)