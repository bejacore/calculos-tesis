import sys
from pathlib import Path

# Agrega la raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import utils
import numpy as np
import pandas as pd

def process_and_export_data(path_clusters, path_members):
    '''
    Procesa los datos de múltiples cúmulos y exporta sus perfiles radiales. 
    '''
    # Leer las tablas de cúmulos y miembros
    clusters = pd.read_csv(path_clusters)
    members = pd.read_csv(path_members)

    print(clusters.head())

# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    clusters_table = 'data/processed/largest_clusters.csv'
    members_table = 'data/processed/largest_clusters_members.csv'
    
    process_and_export_data(clusters_table, members_table)