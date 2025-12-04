"""
Módulo de clustering para segmentación de estudiantes.
Implementa K-Means con optimización de hiperparámetros.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns


class StudentClustering:
    """
    Segmentación de estudiantes usando K-Means con selección óptima de k.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.optimal_k = None
        self.cluster_labels = None
        self.cluster_centers = None
        self.metrics = {}

    def find_optimal_k(self, X: np.ndarray, k_range: range = range(2, 11),
                      plot: bool = True) -> int:
        """
        Determina el número óptimo de clusters usando Elbow y Silhouette.

        Args:
            X: Matriz de características
            k_range: Rango de valores k a evaluar
            plot: Si True, genera gráficos de métricas

        Returns:
            Número óptimo de clusters
        """
        print("\nDeterminando número óptimo de clusters...")

        wcss = []  # Within-Cluster Sum of Squares
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=50,
                          max_iter=300, random_state=self.random_state)
            kmeans.fit(X)

            wcss.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
            calinski_scores.append(calinski_harabasz_score(X, kmeans.labels_))
            davies_bouldin_scores.append(davies_bouldin_score(X, kmeans.labels_))

            print(f"k={k} | Silhouette={silhouette_scores[-1]:.3f} | "
                  f"Calinski-Harabasz={calinski_scores[-1]:.2f}")

        # Método del codo: encontrar punto de inflexión
        optimal_k_elbow = self._find_elbow_point(k_range, wcss)

        # Método Silhouette: máximo score
        optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]

        print(f"\nK óptimo por método del codo: {optimal_k_elbow}")
        print(f"K óptimo por Silhouette: {optimal_k_silhouette}")

        # Usar Silhouette como criterio principal
        self.optimal_k = optimal_k_silhouette

        # Guardar métricas
        self.metrics = {
            'k_range': list(k_range),
            'wcss': wcss,
            'silhouette': silhouette_scores,
            'calinski_harabasz': calinski_scores,
            'davies_bouldin': davies_bouldin_scores
        }

        if plot:
            self._plot_optimization_metrics(k_range, wcss, silhouette_scores)

        return self.optimal_k

    def _find_elbow_point(self, k_range: range, wcss: list) -> int:
        """
        Encuentra el punto de inflexión (codo) en la curva WCSS.

        Args:
            k_range: Rango de valores k
            wcss: Lista de WCSS para cada k

        Returns:
            Valor k óptimo
        """
        # Método de la segunda derivada
        k_values = list(k_range)
        wcss_diff = np.diff(wcss)
        wcss_diff2 = np.diff(wcss_diff)

        # El codo es donde la segunda derivada es máxima
        elbow_idx = np.argmax(wcss_diff2) + 2  # +2 por doble diff
        elbow_k = k_values[elbow_idx] if elbow_idx < len(k_values) else k_values[-1]

        return elbow_k

    def _plot_optimization_metrics(self, k_range: range, wcss: list,
                                   silhouette_scores: list):
        """
        Genera gráficos de métricas de optimización.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Método del codo
        ax1.plot(list(k_range), wcss, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Número de Clusters (k)', fontsize=12)
        ax1.set_ylabel('WCSS (Inertia)', fontsize=12)
        ax1.set_title('Método del Codo', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Silhouette Score
        ax2.plot(list(k_range), silhouette_scores, 'go-', linewidth=2, markersize=8)
        ax2.axvline(x=self.optimal_k, color='r', linestyle='--', label=f'k óptimo = {self.optimal_k}')
        ax2.set_xlabel('Número de Clusters (k)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Índice de Silhouette', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('d:/Elias/Universidad/Mineria de datos/clasificacion_estudiantes/results/cluster_optimization.png',
                   dpi=300, bbox_inches='tight')
        print("\nGráfico guardado: results/cluster_optimization.png")

    def fit(self, X: np.ndarray, n_clusters: int = None) -> np.ndarray:
        """
        Entrena el modelo K-Means.

        Args:
            X: Matriz de características
            n_clusters: Número de clusters (si None, usa optimal_k)

        Returns:
            Array de etiquetas de cluster
        """
        if n_clusters is None:
            if self.optimal_k is None:
                raise ValueError("Debe ejecutar find_optimal_k() primero o especificar n_clusters")
            n_clusters = self.optimal_k

        print(f"\nEntrenando K-Means con k={n_clusters}...")

        self.model = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=50,
            max_iter=300,
            tol=1e-4,
            random_state=self.random_state,
            algorithm='lloyd'
        )

        self.cluster_labels = self.model.fit_predict(X)
        self.cluster_centers = self.model.cluster_centers_

        # Calcular métricas finales
        sil_score = silhouette_score(X, self.cluster_labels)
        cal_score = calinski_harabasz_score(X, self.cluster_labels)
        dav_score = davies_bouldin_score(X, self.cluster_labels)

        print(f"\nMétricas de Clustering:")
        print(f"  Silhouette Score: {sil_score:.4f}")
        print(f"  Calinski-Harabasz: {cal_score:.2f}")
        print(f"  Davies-Bouldin: {dav_score:.4f} (menor es mejor)")

        # Distribución de clusters
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        print(f"\nDistribución de estudiantes por cluster:")
        for cluster_id, count in zip(unique, counts):
            percentage = (count / len(self.cluster_labels)) * 100
            print(f"  Cluster {cluster_id}: {count:,} estudiantes ({percentage:.1f}%)")

        return self.cluster_labels

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice clusters para nuevos datos.

        Args:
            X: Matriz de características

        Returns:
            Array de etiquetas de cluster
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado. Ejecute fit() primero.")

        return self.model.predict(X)

    def analyze_clusters(self, X: np.ndarray, feature_names: list,
                        original_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Analiza características de cada cluster.

        Args:
            X: Matriz de características normalizadas
            feature_names: Nombres de características
            original_df: DataFrame original con datos sin normalizar

        Returns:
            DataFrame con estadísticas por cluster
        """
        if self.cluster_labels is None:
            raise ValueError("Modelo no entrenado. Ejecute fit() primero.")

        print("\n" + "="*60)
        print("ANÁLISIS DE CLUSTERS")
        print("="*60)

        # Crear DataFrame con datos y clusters
        df_clustered = pd.DataFrame(X, columns=feature_names)
        df_clustered['cluster'] = self.cluster_labels

        # Calcular estadísticas por cluster
        cluster_stats = []

        for cluster_id in sorted(df_clustered['cluster'].unique()):
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]

            stats = {
                'Cluster': cluster_id,
                'n_estudiantes': len(cluster_data),
                'porcentaje': f"{(len(cluster_data)/len(df_clustered))*100:.1f}%"
            }

            # Estadísticas de variables principales
            if 'creditos_aprobados' in feature_names and original_df is not None:
                creditos = original_df.iloc[cluster_data.index]['creditos_aprobados']
                stats['creditos_mean'] = f"{creditos.mean():.0f}"
                stats['creditos_std'] = f"{creditos.std():.0f}"

            if 'ranking_facultad' in feature_names and original_df is not None:
                ranking = original_df.iloc[cluster_data.index]['ranking_facultad']
                stats['ranking_mean'] = f"{ranking.mean():.0f}"

            if 'tercio_superior' in feature_names:
                tercio = cluster_data['tercio_superior'].mean()
                stats['tercio_superior_%'] = f"{tercio*100:.1f}%"

            cluster_stats.append(stats)

        df_stats = pd.DataFrame(cluster_stats)
        print(df_stats.to_string(index=False))

        return df_stats

    def save_model(self, filepath: str):
        """Guarda el modelo entrenado."""
        import joblib
        if self.model is None:
            raise ValueError("No hay modelo entrenado para guardar")

        joblib.dump(self.model, filepath)
        print(f"\nModelo guardado en: {filepath}")

    def load_model(self, filepath: str):
        """Carga un modelo previamente entrenado."""
        import joblib
        self.model = joblib.load(filepath)
        print(f"Modelo cargado desde: {filepath}")
