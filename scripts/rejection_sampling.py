import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.integrate import cumulative_trapezoid

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

def tangent_plane_projection(ras, decs, ra0, dec0, d0):
    """
    Proyecta las coordenadas (ras, decs) al plano tangente centrado en (ra0, dec0).
    """

    ras = np.radians(ras)
    decs = np.radians(decs)

    ra0 = np.radians(ra0)
    dec0 = np.radians(dec0)

    delta_ras = ras - ra0

    den = (np.sin(dec0) * np.sin(decs) + 
           np.cos(dec0) * np.cos(decs) * np.cos(delta_ras))
    
    xi = (np.cos(decs) * np.sin(delta_ras)) / den

    eta = (np.cos(dec0) * np.sin(decs) - 
           np.sin(dec0) * np.cos(decs) * np.cos(delta_ras)) / den
    
    X = xi * d0
    Y = eta * d0

    return X, Y

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
def prob_Z_given_R(Z, R, rc, rt, k):
    """Implementación de la probabilidad derivada P(Z | X, Y)."""
    r = np.sqrt(R**2 + Z**2)
    rho = king_spatial_density(r, rc, rt, k)
    
    # P(Z) proporcional a r^2 * rho(r) * (|Z| / r) = r * rho(r) * |Z|
    return r * rho * np.abs(Z)

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
    
    # El máximo de la función ocurre cuando R=0, lo que equivale a maximizar 
    # r^2 * rho(r).
    # Evaluamos un grid fino de r entre 0 y rt para encontrar este techo.
    r_grid = np.linspace(0, rt, 5000)
    P_grid = (r_grid**2) * king_spatial_density(r_grid, rc, rt, k)
    P_global_max = np.max(P_grid)

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
        P_cand = prob_Z_given_R(Z_cand, R_pending, rc, rt, k)
        
        # 6. Muestreo de rechazo: Se generan 'u' usando el máximo global de P(Z|R)
        u = np.random.uniform(0.0, P_global_max, size=n_pending)
        
        # 7. Condición de aceptación: si u < rho(cand), la muestra es válida
        accept = u < P_cand
        
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
# CALCULOS EN LAS TRES DIMESIONES
# ==============================================================================
def calculate_velocity_dispersion(X, Y, Z, M_star, num_bins=20):
    '''
    Calcula la dispersión de velocidades a partir de la densidad espacial de 
    masa y la masa acumulada.
    '''
    # Constante gravitacional G (pc * M_sun^-1 * (km/s)^2)
    G = 4.3009e-3 
    
    # Se calculan los radios y se ordenan
    r = np.sqrt(X**2 + Y**2 + Z**2)
    sort_idx = np.argsort(r)
    r_sorted = r[sort_idx]
    M_sorted = M_star[sort_idx]

    # Masa acumulada exacta por estrella
    M_cum_exact = np.cumsum(M_sorted)

    # Se definen cascarones esféricos (bins radiales)
    r_bins = np.linspace(np.min(r_sorted), np.max(r_sorted), num_bins + 1)
    r_centers = (r_bins[1:] + r_bins[:-1]) / 2

    # Se inicializan los arrays para los perfiles
    rho_bins = np.zeros(num_bins)
    M_cum_bins = np.zeros(num_bins)

    for i in range(num_bins):
        # Máscara para las estrellas que caen dentro del cascarón actual
        in_bin = (r_sorted >= r_bins[i]) & (r_sorted < r_bins[i+1])
        
        # Volumen del cascarón esférico
        volumen = (4/3) * np.pi * (r_bins[i+1]**3 - r_bins[i]**3)
        
        # Densidad = Masa en el bin / volumen
        rho_bins[i] = np.sum(M_sorted[in_bin]) / volumen
        
        # Masa acumulada hasta el centro del bin
        idx_center = np.searchsorted(r_sorted, r_centers[i])
        if idx_center < len(M_cum_exact):
            M_cum_bins[i] = M_cum_exact[idx_center]
        else:
            M_cum_bins[i] = M_cum_exact[-1]

    # Se evitan divisiones por cero en zonas vacías
    rho_bins[rho_bins == 0] = np.nan 

    # Se construye el integrando: rho(r) * G * M(<r) / r^2
    integrando = rho_bins * G * M_cum_bins / r_centers**2

    # Se calcula la integral acumulada de 0 a r
    # (Se usa initial=0 para mantener la misma longitud del array)
    integral_0_r = cumulative_trapezoid(integrando, x=r_centers, initial=0)
    
    # La integral de r a R_max es la integral total menos la integral hasta r
    integral_r_inf = integral_0_r[-1] - integral_0_r

    # Se calcula la dispersión de velocidades al cuadrado
    sigma_cuadrado = integral_r_inf / rho_bins

    return r_centers, rho_bins, np.sqrt(sigma_cuadrado)

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
    Ms = cluster_members['Mass50'].values

    angular_dist = angular_distances(ra0, dec0, ras, decs)
    Rs = angular_dist * d0

    Rs_centers, sigma_obs = bin_superficial_density(Rs)
    k, rc, rt = fit_king_profile(Rs_centers, sigma_obs)

    X, Y = tangent_plane_projection(ras, decs, ra0, dec0, d0)
    Z_samples = rejection_sampling(Rs, rc, rt, k)

    rs, rhos_m, sigma_v = calculate_velocity_dispersion(X, Y, Z_samples, Ms)

# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    clusters_table = 'data/processed/clusters.csv'
    members_table = 'data/processed/members_with_estimated_masses.csv'
    
    process_cluster_data(clusters_table, members_table)