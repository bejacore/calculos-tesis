import sys
from pathlib import Path

# Agrega la raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import utils
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

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

        rc_2d, rt_2d, k_2d = popt_2d

        global_data.append({
            'd': id,
            'nombre': cluster,
            'num_miembros': num_members,
            'rc_2d': rc_2d,
            'rt_2d': rt_2d,
            'k_2d': k_2d
        })

# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    clusters_table = 'data/processed/largest_clusters.csv'
    members_table = 'data/processed/largest_clusters_members.csv'
    
    process_and_export_data(clusters_table, members_table)