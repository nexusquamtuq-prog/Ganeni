# Ganeni
Modelo Ganeni
# Marco Ganeni - Implementación de Referencia
# Observatorio de la Latencia Universal de Brandon N'Hain (OLUB)
# Autores: N. Quintana, J. Ruttan, I. Castillo, F. Cerecera, R. Arellano, J. Puc
# Licencia: MIT para uso académico

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.integrate import odeint
from sklearn.decomposition import PCA
from sklearn.manifold import UMAP
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional
import warnings

class GaneniField:
    """
    Clase principal para modelar un campo vectorial Ganeni.
    
    Parámetros:
    -----------
    dimension : int
        Dimensión del espacio configuracional
    metric : str o Callable
        Métrica de distancia ('euclidean', 'cosine', o función personalizada)
    alpha : float
        Parámetro de sensibilidad de repulsión configuracional (default: 1.0)
    beta : float
        Parámetro de alcance de interacción (default: 0.5)
    """
    
    def __init__(self, dimension: int, metric: str = 'euclidean', 
                 alpha: float = 1.0, beta: float = 0.5):
        self.dimension = dimension
        self.metric = metric
        self.alpha = alpha
        self.beta = beta
        self.configurations = None
        self.distance_matrix = None
        
    def add_configurations(self, configurations: np.ndarray):
        """
        Agrega conjunto de configuraciones al sistema.
        
        Parámetros:
        -----------
        configurations : np.ndarray
            Matriz de forma (N, d) donde N es número de configuraciones
            y d es la dimensión del espacio configuracional
        """
        if configurations.shape[1] != self.dimension:
            raise ValueError(f"Dimensión incorrecta: esperada {self.dimension}, "
                           f"recibida {configurations.shape[1]}")
        
        self.configurations = configurations
        self._compute_distance_matrix()
        
    def _compute_distance_matrix(self):
        """Calcula matriz de distancias entre todas las configuraciones."""
        distances = pdist(self.configurations, metric=self.metric)
        self.distance_matrix = squareform(distances)
        
    def compute_field(self, x: np.ndarray, t: float = 0.0) -> np.ndarray:
        """
        Calcula el campo vectorial Ganeni en un punto específico.
        
        Parámetros:
        -----------
        x : np.ndarray
            Configuración actual (vector de dimensión d)
        t : float
            Tiempo (para dependencia temporal, opcional)
            
        Retorna:
        --------
        v : np.ndarray
            Vector del campo Ganeni en x
        """
        if self.configurations is None:
            raise ValueError("Debe agregar configuraciones primero usando add_configurations()")
        
        N = len(self.configurations)
        v = np.zeros(self.dimension)
        
        # Calcular contribución de cada configuración
        for i in range(N):
            xi = self.configurations[i]
            diff = x - xi
            dist = np.linalg.norm(diff)
            
            if dist &lt; 1e-10:  # Evitar división por cero
                continue
                
            # Fuerza de repulsión configuracional
            weight = self.alpha * np.exp(-dist / self.beta)
            v += weight * (diff / dist)
            
        return v
    
    def compute_redundancy_index(self) -> float:
        """
        Calcula el Índice de Redundancia Configuracional (R).
        
        Retorna:
        --------
        R : float
            Valor entre 0 y 1, donde valores cercanos a 1 indican
            alta redundancia (colapso configuracional)
        """
        if self.distance_matrix is None:
            raise ValueError("Debe agregar configuraciones primero")
        
        N = len(self.configurations)
        
        # Distancia promedio entre configuraciones
        avg_distance = np.mean(self.distance_matrix[np.triu_indices(N, k=1)])
        
        # Distancia característica del espacio (escala de referencia)
        characteristic_distance = np.sqrt(self.dimension)
        
        # R normalizado
        R = 1.0 - np.tanh(avg_distance / characteristic_distance)
        
        return R
    
    def compute_rigidity_index(self) -> float:
        """
        Calcula el Índice de Rigidez Configuracional (IRC).
        
        Retorna:
        --------
        IRC : float
            Medida de resistencia a perturbaciones
        """
        if self.distance_matrix is None:
            raise ValueError("Debe agregar configuraciones primero")
        
        N = len(self.configurations)
        
        # Varianza de distancias (mide heterogeneidad)
        distances = self.distance_matrix[np.triu_indices(N, k=1)]
        variance = np.var(distances)
        
        # IRC inversamente proporcional a varianza
        IRC = 1.0 / (1.0 + variance)
        
        return IRC
    
    def compute_adjustment_capacity(self) -> float:
        """
        Calcula el Índice de Capacidad de Ajuste (ICA).
        
        Retorna:
        --------
        ICA : float
            Medida de rango de respuestas disponibles
        """
        if self.configurations is None:
            raise ValueError("Debe agregar configuraciones primero")
        
        # Volumen efectivo ocupado en espacio configuracional
        # Aproximado mediante determinante de matriz de covarianza
        cov_matrix = np.cov(self.configurations.T)
        
        # Evitar problemas numéricos
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        
        # ICA proporcional a logaritmo del volumen
        log_volume = np.sum(np.log(eigenvalues))
        ICA = np.tanh(log_volume / self.dimension)
        
        return ICA
    
    def integrate_trajectory(self, x0: np.ndarray, t_span: Tuple[float, float], 
                           n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integra trayectoria configuracional desde condición inicial.
        
        Parámetros:
        -----------
        x0 : np.ndarray
            Configuración inicial
        t_span : tuple
            (t_inicial, t_final)
        n_points : int
            Número de puntos temporales
            
        Retorna:
        --------
        t : np.ndarray
            Vector de tiempos
        trajectory : np.ndarray
            Matriz (n_points, dimension) con trayectoria
        """
        def dynamics(x, t):
            return self.compute_field(x, t)
        
        t = np.linspace(t_span[0], t_span[1], n_points)
        trajectory = odeint(dynamics, x0, t)
        
        return t, trajectory
    
    def detect_critical_transition(self, R_threshold: float = 0.75,
                                  window_size: int = 10) -> Optional[int]:
        """
        Detecta transición crítica basada en umbral de R.
        
        Parámetros:
        -----------
        R_threshold : float
            Umbral crítico de redundancia
        window_size : int
            Tamaño de ventana para suavizado temporal
            
        Retorna:
        --------
        critical_point : int o None
            Índice temporal de transición crítica, o None si no se detecta
        """
        R = self.compute_redundancy_index()
        
        if R &gt;= R_threshold:
            warnings.warn(f"Sistema en estado crítico: R = {R:.3f} &gt;= {R_threshold}")
            return 0
        
        return None
    
    def visualize_configuration_space(self, method: str = 'pca', 
                                     highlight_points: Optional[np.ndarray] = None):
        """
        Visualiza espacio configuracional en 2D.
        
        Parámetros:
        -----------
        method : str
            Método de reducción dimensional ('pca' o 'umap')
        highlight_points : np.ndarray
            Puntos específicos a resaltar (opcional)
        """
        if self.configurations is None:
            raise ValueError("Debe agregar configuraciones primero")
        
        if method == 'pca':
            reducer = PCA(n_components=2)
            coords_2d = reducer.fit_transform(self.configurations)
        elif method == 'umap':
            reducer = UMAP(n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(self.configurations)
        else:
            raise ValueError("Método debe ser 'pca' o 'umap'")
        
        plt.figure(figsize=(10, 8))
        plt.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                   c='blue', alpha=0.6, s=50, label='Configuraciones')
        
        if highlight_points is not None:
            highlight_2d = reducer.transform(highlight_points)
            plt.scatter(highlight_2d[:, 0], highlight_2d[:, 1],
                       c='red', alpha=0.8, s=100, marker='*',
                       label='Puntos destacados')
        
        R = self.compute_redundancy_index()
        IRC = self.compute_rigidity_index()
        ICA = self.compute_adjustment_capacity()
        
        plt.title(f'Espacio Configuracional (R={R:.3f}, IRC={IRC:.3f}, ICA={ICA:.3f})')
        plt.xlabel(f'{method.upper()} Componente 1')
        plt.ylabel(f'{method.upper()} Componente 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def generate_test_configurations(n_configs: int, dimension: int, 
                                distribution: str = 'uniform') -&gt; np.ndarray:
    """
    Genera configuraciones de prueba.
    
    Parámetros:
    -----------
    n_configs : int
        Número de configuraciones
    dimension : int
        Dimensión del espacio
    distribution : str
        Tipo de distribución ('uniform', 'gaussian', 'clustered')
        
    Retorna:
    --------
    configs : np.ndarray
        Matriz de configuraciones
    """
    if distribution == 'uniform':
        configs = np.random.uniform(-1, 1, size=(n_configs, dimension))
    elif distribution == 'gaussian':
        configs = np.random.randn(n_configs, dimension)
    elif distribution == 'clustered':
        # Genera k clusters
        k = max(3, n_configs // 20)
        centers = np.random.randn(k, dimension) * 2
        configs = []
        for i in range(n_configs):
            center = centers[i % k]
            config = center + np.random.randn(dimension) * 0.3
            configs.append(config)
        configs = np.array(configs)
    else:
        raise ValueError("Distribución no reconocida")
    
    return configs


def simulate_collapse_dynamics(field: GaneniField, x0: np.ndarray,
                               t_max: float = 100.0, n_steps: int = 1000) -&gt; dict:
    """
    Simula dinámica de colapso configuracional.
    
    Parámetros:
    -----------
    field : GaneniField
        Campo Ganeni configurado
    x0 : np.ndarray
        Configuración inicial
    t_max : float
        Tiempo máximo de simulación
    n_steps : int
        Número de pasos temporales
        
    Retorna:
    --------
    results : dict
        Diccionario con resultados de simulación
    """
    t, trajectory = field.integrate_trajectory(x0, (0, t_max), n_steps)
    
    # Calcular métricas a lo largo de trayectoria
    R_history = []
    for i in range(len(t)):
        # Actualizar configuraciones con punto actual
        current_configs = np.vstack([field.configurations, trajectory[i]])
        field_temp = GaneniField(field.dimension, field.metric)
        field_temp.add_configurations(current_configs)
        R_history.append(field_temp.compute_redundancy_index())
    
    results = {
        'time': t,
        'trajectory': trajectory,
        'redundancy': np.array(R_history),
        'collapse_detected': np.any(np.array(R_history) &gt; 0.75)
    }
    
    return results


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    print("Marco Ganeni - Implementación de Referencia")
    print("=" * 60)
    
    # Configuración del sistema
    dimension = 10
    n_configurations = 50
    
    # Generar configuraciones de prueba
    print(f"\nGenerando {n_configurations} configuraciones en {dimension}D...")
    configs = generate_test_configurations(n_configurations, dimension, 'gaussian')
    
    # Crear campo Ganeni
    field = GaneniField(dimension=dimension, metric='euclidean', alpha=1.0, beta=0.5)
    field.add_configurations(configs)
    
    # Calcular métricas
    print("\nMétricas del sistema:")
    print(f"  Índice de Redundancia (R): {field.compute_redundancy_index():.4f}")
    print(f"  Índice de Rigidez (IRC): {field.compute_rigidity_index():.4f}")
    print(f"  Índice de Capacidad de Ajuste (ICA): {field.compute_adjustment_capacity():.4f}")
    
    # Integrar trayectoria desde configuración inicial
    print("\nIntegrando trayectoria configuracional...")
    x0 = np.random.randn(dimension)
    t, trajectory = field.integrate_trajectory(x0, (0, 50), n_points=200)
    
    print(f"  Configuración inicial: {x0[:3]}... (primeras 3 componentes)")
    print(f"  Configuración final: {trajectory[-1, :3]}... (primeras 3 componentes)")
    
    # Simular dinámica de colapso
    print("\nSimulando dinámica de colapso...")
    results = simulate_collapse_dynamics(field, x0, t_max=50.0, n_steps=200)
    
    if results['collapse_detected']:
        print("  ⚠ ADVERTENCIA: Colapso configuracional detectado")
        collapse_time = results['time'][np.argmax(results['redundancy'] &gt; 0.75)]
        print(f"  Tiempo estimado de colapso: {collapse_time:.2f}")
    else:
        print("  ✓ Sistema mantiene heterogeneidad configuracional")
    
    # Visualización
    print("\nGenerando visualización...")
    field.visualize_configuration_space(method='pca', 
                                       highlight_points=trajectory[::20])
    
    print("\n" + "=" * 60)
    print("Simulación completada. Consulte el repositorio oficial para más ejemplos.")
    print("Repositorio: [URL del repositorio oficial]")
