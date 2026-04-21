import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
    
    try:
        popt, pcov = curve_fit(king_surface_density, radii, densities, p0=p0, bounds=bounds)
        return popt[0], popt[1], popt[2] # k, rc, rt
    except Exception as e:
        print(f"Error en el ajuste: {e}")
        return np.nan, np.nan, np.nan

# ==============================================================================
# REJECTION SAMPLING
# ==============================================================================
def rejection_sampling(R, rc, rt, k):
    """
    Genera muestras de Z para cada R usando el método de rechazo.
    """
    Z_samples = np.zeros_like(R, dtype=float)

    mask = R < rt
    R_valid = R[mask]

    N = len(R_valid)
    Z_valid = np.zeros(N)

    Z_max = np.sqrt(rt**2 - R_valid**2)

    rho_max = king_spatial_density(R_valid, rc, rt, k)

    unaccepted = np.ones(N, dtype=bool)

    while np.any(unaccepted):
        indices = np.nonzero(unaccepted)[0]

        Z_cand = np.random.uniform(-Z_max[indices], Z_max[indices])

        r_cand = np.sqrt(R_valid[indices]**2 + Z_cand**2)

        rho_cand = king_spatial_density(r_cand, rc, rt, k)

        u = np.random.uniform(0, rho_max[indices])

        accept = u <= rho_cand

        Z_valid[indices[accept]] = Z_cand[accept]
        unaccepted[indices[accept]] = False

    Z_samples[mask] = Z_valid

    return Z_samples

# ==============================================================================
# CALCULOS EN LAS TRES DIMESIONES
# ==============================================================================
def generate_radial_profile(X, Y, Z, M_star, num_bins=10):
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
    n_stars_in_bin = np.zeros(num_bins)
    n_density_bin = np.zeros(num_bins)
    rho_bins = np.zeros(num_bins)
    M_cum_bins = np.zeros(num_bins)

    for i in range(num_bins):
        # Máscara para las estrellas que caen dentro del cascarón actual
        in_bin = (r_sorted >= r_bins[i]) & (r_sorted < r_bins[i+1])

        # Número de estrellas en el bin
        n_stars_in_bin[i] = len(r[in_bin])
        
        # Volumen del cascarón esférico
        volumen = (4/3) * np.pi * (r_bins[i+1]**3 - r_bins[i]**3)

        # Densidad de número en el bin
        n_density_bin[i] = n_stars_in_bin[i] / volumen
        
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

    # Se construye el DataFrame con los resultados
    perfil_data = {
        'r_bin': r_centers,
        'n_estrellas_bin': n_stars_in_bin,
        'densidad_n': n_density_bin,
        'densidad_vol': rho_bins,
        'sigma_cuadrado': sigma_cuadrado,
        'mass_accum': M_cum_bins
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
    
    # Crear un directorio para los perfiles si no existe
    output_directory = 'data/processed/perfiles_radiales/'
    os.makedirs(output_directory, exist_ok=True)

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

        # Calcular las distancias proyectadas R para cada miembro
        angular_dist = angular_distances(ra0, dec0, ras, decs)
        Rs = angular_dist * d0

        # Calcular la densidad superficial y ajustar el perfil de King
        Rs_centers, sigma_obs = bin_superficial_density(Rs)
        k, rc, rt = fit_king_profile(Rs_centers, sigma_obs)

        # Guardar los parámetros globales del cúmulo
        global_data.append({
            'id': id,
            'nombre': cluster,
            'n_estrellas': num_members,
            'rc': rc,
            'rt': rt,
            'k': k
        })

        # Proyectar las coordenadas al plano tangente y generar muestras de Z 
        # usando rejection sampling
        X, Y = tangent_plane_projection(ras, decs, ra0, dec0, d0)
        Z_samples = rejection_sampling(Rs, rc, rt, k)

        # Se filtran las muestras de Z para quedarse solo con las que fueron 
        # aceptadas
        mask = Z_samples != 0

        X_clean = X[mask]
        Y_clean = Y[mask]
        Z_clean = Z_samples[mask]
        M_clean = Ms[mask]

        # Se genera el perfil radial 3D y se exporta a un archivo CSV
        perfil_df = generate_radial_profile(X_clean, Y_clean, Z_clean, M_clean)

        file_name = f"cluster_{id}.csv"
        path_file = os.path.join(output_directory, file_name)

        perfil_df.to_csv(path_file, index=False)

        id += 1
    
    # Exportar los parámetros globales de todos los cúmulos a un archivo CSV
    global_df = pd.DataFrame(global_data)
    global_file = 'parametros_globales.csv'
    global_df.to_csv(global_file, index=False)

# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    clusters_table = 'data/processed/largest_clusters.csv'
    members_table = 'data/processed/largest_clusters_members.csv'
    
    process_and_export_data(clusters_table, members_table)