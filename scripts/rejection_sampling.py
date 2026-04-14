import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================
def angular_distances(ra0, dec0, ras, decs):
    """
    Calcula la distancia angular entre un punto central (ra0, dec0) y una lista 
    de puntos (ras, decs).
    """
    ra0 = np.radians(ra0)
    dec0 = np.radians(dec0)
    ras = np.radians(ras)
    decs = np.radians(decs)

    delta_ra = ras - ra0
    delta_dec = decs - dec0

    a = np.sin(delta_dec / 2)**2 + np.cos(dec0) * np.cos(decs) * np.sin(delta_ra / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return c

def bin_superficial_density(radii, num_bins=30):
    """Calcula la densidad superficial a partir de las distancias proyectadas."""
    counts, bin_edges = np.histogram(radii, bins=num_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    areas = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)
    sigma_obs = counts / areas

    return bin_centers, sigma_obs

def get_stars_labels(Rs, rc, rt):
    """
    Selecciona índices de estrellas representativas para las posiciones clave en 
    el cúmulo.
    """
    idx = [
        np.argmin(np.abs(Rs - 0.1)),
        np.argmin(np.abs(Rs - rc)),
        np.argmin(np.abs(Rs - rt/2)),
        np.argmin(np.abs(Rs - 0.95*rt))
    ]

    labels = [
        f'Estrella en Núcleo (R={Rs[idx[0]]:.2f} pc)',
        f'Estrella en $r_c$ (R={Rs[idx[1]]:.2f} pc)',
        f'Estrella en Intermedia (R={Rs[idx[2]]:.2f} pc)',
        f'Estrella en Borde (R={Rs[idx[3]]:.2f} pc)'
    ]

    return idx, labels

def make_plots(Z_matrix, idx, labels, rt):
    """
    Realiza los histogramas de la distribución marginal P(Z) y las 
    distribuciones condicionales P(Z|X,Y) para las estrellas seleccionadas.
    """
    plt.figure(figsize=(8, 5))

    Z_total = Z_matrix.flatten()

    plt.hist(Z_total, bins=100, density=True, color='gray', alpha=0.7, edgecolor='black')
    plt.title('Distribución de Probabilidad Marginal $P(Z)$ del cúmulo King 11')
    plt.xlabel('Z [pc]')
    plt.ylabel('Densidad de Probabilidad')
    plt.grid(True, alpha=0.3)
    plt.show()

    colors = ['blue', 'green', 'orange', 'red']
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    axs = axs.flatten()

    for i, idx_star in enumerate(idx):
        Z_star = Z_matrix[idx_star, :]
    
        axs[i].hist(Z_star, bins=40, density=True, color=colors[i], alpha=0.7, edgecolor='black')
        axs[i].set_title(labels[i])
        axs[i].set_xlabel('Z [pc]')
        axs[i].set_ylabel('$P(Z | X, Y)$')
        axs[i].grid(True, alpha=0.3)

        axs[i].set_xlim(-rt, rt)

    plt.tight_layout()
    plt.show()

# ==============================================================================
# MODELO DE KING
# ==============================================================================
def king_surface_density(r, k, rc, rt):
    """Perfil de densidad superficial de King."""
    term1 = 1.0 / np.sqrt(1.0 + (r / rc)**2)
    term2 = 1.0 / np.sqrt(1.0 + (rt / rc)**2)
    
    sigma = k * (term1 - term2)**2
    return np.where(r < rt, sigma, 0.0)

def king_spatial_density(r, rc, rt, k=1.0):
    """Perfil de densidad espacial de King."""
    # Se evita que se evalúen radios mayores al radio de marea (rt)
    # Fuera del cúmulo, la densidad debe ser 0.
    r_safe = np.clip(r, 0.0, rt)
    
    # Se calcula w
    w = np.sqrt((1 + (r_safe/rc)**2) / (1 + (rt/rc)**2))
    
    # Constante C (k / (pi * rc * [1+(rt/rc)^2]^1.5))
    C = k / (np.pi * rc * (1 + (rt/rc)**2)**1.5)
    
    # Se evalúa la densidad espacial
    termino_1 = (1.0 / w) * np.arccos(w)
    termino_2 = np.sqrt(1.0 - w**2)
    
    rho = C * (1.0 / w**2) * (termino_1 - termino_2)
    
    # Se impone que la densidad sea 0 para r >= rt
    rho[r >= rt] = 0.0
    
    return rho

# ==============================================================================
# FUNCIÓN DE AJUSTE
# ==============================================================================
def fit_king_profile(radii, densities):
    """
    Ajusta el perfil de King superficial a los datos observados.
    radii: distancias proyectadas (R) de los bines
    densities: densidad estelar observada en cada bin
    """
    p0 = [np.max(densities), np.median(radii), np.max(radii)]
    bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
    
    popt, pcov = curve_fit(king_surface_density, radii, densities, p0=p0, bounds=bounds)
    return popt[0], popt[1], popt[2] # k, rc, rt

# ==============================================================================
# REJECTION SAMPLING
# ==============================================================================
def rejection_sampling(R, rc, rt, k):
    """
    Realiza el muestreo de rechazo para obtener las coordenadas Z de las 
    estrellas dado R.
    """
    N = len(R)
    Z_accepted = np.zeros(N)
    
    # Máscara para saber cuáles estrellas faltan por calcular
    # Se descartan las estrellas con R >= rt (no pertenecen al modelo)
    pending = R < rt 
    
    # 1. Se calcula el rho máximo (ocurre en el centro del cúmulo, r = 0)
    # Se usa como techo para la variable uniforme 'u'
    rho_max = king_spatial_density(np.array([0.0]), rc, rt, k)[0]

    while np.any(pending):
        # Se obtienen las estrellas que aún no han sido aceptadas
        R_pending = R[pending]
        n_pending = len(R_pending)
        
        # 2. Límite de Z geométrico
        # Dado un R, la estrella no puede estar más allá del radio de marea.
        # r^2 = R^2 + Z^2  =>  Z_max = sqrt(rt^2 - R^2)
        Z_max = np.sqrt(rt**2 - R_pending**2)
        
        # 3. Se genera el Z candidato uniformemente entre -Z_max y +Z_max
        Z_cand = np.random.uniform(-Z_max, Z_max, size=n_pending)
        
        # 4. Se calcula el radio 3D (r) para los Z candidatos
        r_cand = np.sqrt(R_pending**2 + Z_cand**2)
        
        # 5. Se evalúa la densidad espacial (rho) en r_cand
        rho_cand = king_spatial_density(r_cand, rc, rt, k)
        
        # 6. Muestreo de rechazo: Se generan 'u' usando el rho máximo
        u = np.random.uniform(0.0, rho_max, size=n_pending)
        
        # 7. Condición de aceptación: si u < rho(cand), la muestra es válida
        accept = u < rho_cand
        
        # 8. Se actualizan los candidatos aceptados
        # Se obtiene los índices globales de las estrellas que estaban pendientes
        idx_pending = np.where(pending)[0]
        # Se obtiene los índices globales de las que sí fueron aceptadas en esta ronda
        idx_accepted = idx_pending[accept]
        
        # Se guardan los valores Z aceptados
        Z_accepted[idx_accepted] = Z_cand[accept]
        
        # Se actualiza la máscara para no volver a calcular estas estrellas
        pending[idx_accepted] = False
        
    return Z_accepted

def build_z_matrix(Rs, rc, rt, k, N_stars, M_realization=100):
    """
    Construye una matriz de dimensiones (N_stars, M_realization) con las 
    coordenadas Z obtenidas por muestreo de rechazo para cada estrella y cada 
    realización.
    """
    Z_matrix = np.zeros((N_stars, M_realization))
    
    for i in range(M_realization):
        Z_matrix[:, i] = rejection_sampling(Rs, rc, rt, k)
    
    return Z_matrix

# ==============================================================================
# PROCESAMIENTO DE DATOS
# ==============================================================================
def process_cluster_data(clusters_table, members_table):
    """
    Procesa los datos del cúmulo, ajusta el perfil de King y realiza el muestreo 
    de rechazo, finalmente generando los gráficos de las distribuciones 
    marginales y condicionales. 
    """
    clusters = pd.read_csv(clusters_table)
    members = pd.read_csv(members_table)

    cluster = clusters[clusters['Name'] == 'King_11']
    cluster_members = members[members['Name'] == 'King_11']

    ra0 = cluster['RA_ICRS'].values[0]
    dec0 = cluster['DE_ICRS'].values[0]
    d0 = cluster['dist50'].values[0]

    ras = cluster_members['RA_ICRS'].values
    decs = cluster_members['DE_ICRS'].values

    angular_dist = angular_distances(ra0, dec0, ras, decs)
    Rs = angular_dist * d0

    Rs_centers, sigma_obs = bin_superficial_density(Rs)
    k, rc, rt = fit_king_profile(Rs_centers, sigma_obs)

    Z_matrix = build_z_matrix(Rs, rc, rt, k, len(Rs), M_realization=1)
    idx, labels = get_stars_labels(Rs_centers, rc, rt)

    make_plots(Z_matrix, idx, labels, rt)

# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    clusters_table = 'data/processed/clusters.csv'
    members_table = 'data/processed/members.csv'
    
    process_cluster_data(clusters_table, members_table)
    