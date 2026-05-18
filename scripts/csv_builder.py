import sys
from pathlib import Path

# Agrega la raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import utils
import numpy as np
import pandas as pd


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

        # Realizar proyección gnomónica
        X, Y = utils.tangent_plane_projection(ras, decs, ra0, dec0, d0)

# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    clusters_table = 'data/processed/largest_clusters.csv'
    members_table = 'data/processed/largest_clusters_members.csv'
    
    process_and_export_data(clusters_table, members_table)