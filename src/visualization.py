"""
Módulo de visualización y generación de reportes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
sns.set_style('whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


class Visualizer:
    """
    Generador de visualizaciones para análisis de estudiantes.
    """

    def __init__(self, output_dir: str = 'results'):
        self.output_dir = output_dir

    def plot_cluster_distribution(self, cluster_labels: np.ndarray,
                                  cluster_names: List[str] = None,
                                  save_path: str = None):
        """
        Visualiza la distribución de estudiantes por cluster.

        Args:
            cluster_labels: Array de etiquetas de cluster
            cluster_names: Nombres personalizados para clusters
            save_path: Ruta para guardar la figura
        """
        unique, counts = np.unique(cluster_labels, return_counts=True)
        percentages = (counts / len(cluster_labels)) * 100

        if cluster_names is None:
            cluster_names = [f'Cluster {i}' for i in unique]

        # Colores personalizados (ajustar según número de clusters)
        all_colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db', '#9b59b6']
        colors = all_colors[:len(unique)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Gráfico de barras
        bars = ax1.bar(range(len(unique)), counts, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_xlabel('Segmento de Estudiantes', fontsize=12)
        ax1.set_ylabel('Cantidad de Estudiantes', fontsize=12)
        ax1.set_title('Distribución de Estudiantes por Segmento', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(unique)))
        ax1.set_xticklabels(cluster_names, rotation=15, ha='right')

        # Añadir valores sobre las barras
        for bar, count, pct in zip(bars, counts, percentages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Gráfico de pastel
        wedges, texts, autotexts = ax2.pie(counts, labels=cluster_names, autopct='%1.1f%%',
                                           colors=colors, startangle=90,
                                           explode=[0.05]*len(unique))
        for text in texts:
            text.set_fontsize(11)
            text.set_fontweight('bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')

        ax2.set_title('Proporción de Estudiantes', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualización guardada en: {save_path}")

    def plot_cluster_characteristics(self, df: pd.DataFrame, cluster_col: str,
                                    features: List[str], save_path: str = None):
        """
        Visualiza características promedio de cada cluster.

        Args:
            df: DataFrame con datos y clusters
            cluster_col: Nombre de la columna de clusters
            features: Lista de características a visualizar
            save_path: Ruta para guardar la figura
        """
        n_features = len(features)
        n_cols = 2
        n_rows = (n_features + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
        axes = axes.flatten() if n_features > 1 else [axes]

        # Detectar número de clusters automáticamente
        n_clusters = df[cluster_col].nunique()
        all_colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db', '#9b59b6']
        colors = all_colors[:n_clusters]

        # Asignar nombres según número de clusters
        if n_clusters == 2:
            cluster_names = ['Bajo Rendimiento', 'Alto Rendimiento']
        elif n_clusters == 3:
            cluster_names = ['Riesgo Alto', 'Rendimiento Medio', 'Alto Rendimiento']
        else:
            cluster_names = [f'Cluster {i}' for i in range(n_clusters)]

        for idx, feature in enumerate(features):
            ax = axes[idx]

            # Calcular estadísticas por cluster
            cluster_means = df.groupby(cluster_col)[feature].mean().values
            cluster_stds = df.groupby(cluster_col)[feature].std().values

            x_pos = np.arange(len(cluster_means))
            bars = ax.bar(x_pos, cluster_means, yerr=cluster_stds, capsize=5,
                         color=colors[:len(cluster_means)], alpha=0.8, edgecolor='black')

            ax.set_xlabel('Segmento', fontsize=11)
            ax.set_ylabel(f'{feature}', fontsize=11)
            ax.set_title(f'Promedio de {feature} por Segmento', fontsize=12, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(cluster_names, rotation=15, ha='right')
            ax.grid(axis='y', alpha=0.3)

            # Añadir valores
            for bar, mean in zip(bars, cluster_means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.0f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Ocultar subplots vacíos
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Características de clusters guardadas en: {save_path}")

    def plot_feature_importance(self, feature_names: List[str],
                               importances: np.ndarray,
                               top_n: int = 15, save_path: str = None):
        """
        Visualiza importancia de características (para Random Forest, etc.).

        Args:
            feature_names: Nombres de características
            importances: Valores de importancia
            top_n: Número de características más importantes a mostrar
            save_path: Ruta para guardar la figura
        """
        # Ordenar por importancia
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices], color='steelblue', alpha=0.8)
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importancia', fontsize=12)
        plt.title(f'Top {top_n} Características Más Importantes', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Importancia de características guardada en: {save_path}")

    def plot_performance_comparison(self, metrics: Dict, save_path: str = None):
        """
        Compara métricas de rendimiento del modelo.

        Args:
            metrics: Diccionario con métricas (accuracy, f1, etc.)
            save_path: Ruta para guardar la figura
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(metric_names)))
        bars = ax.barh(metric_names, metric_values, color=colors, alpha=0.8, edgecolor='black')

        ax.set_xlabel('Score', fontsize=12)
        ax.set_title('Métricas de Rendimiento del Modelo', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)

        # Añadir valores
        for bar, value in zip(bars, metric_values):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{value:.4f}',
                   ha='left', va='center', fontsize=10, fontweight='bold', color='black')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparación de métricas guardada en: {save_path}")

    def generate_summary_report(self, clustering_stats: pd.DataFrame,
                               svm_metrics: Dict, save_path: str = None):
        """
        Genera un reporte resumen en formato de texto.

        Args:
            clustering_stats: DataFrame con estadísticas de clusters
            svm_metrics: Diccionario con métricas de SVM
            save_path: Ruta para guardar el reporte
        """
        report = []
        report.append("="*80)
        report.append("REPORTE DE CLASIFICACIÓN DE ESTUDIANTES - UNMSM")
        report.append("="*80)
        report.append("")

        # Sección de Clustering
        report.append("1. SEGMENTACIÓN DE ESTUDIANTES (K-Means)")
        report.append("-"*80)
        report.append(clustering_stats.to_string(index=False))
        report.append("")

        # Sección de Clasificación SVM
        report.append("2. MODELO DE CLASIFICACIÓN (SVM)")
        report.append("-"*80)
        report.append(f"Accuracy:        {svm_metrics.get('accuracy', 0):.4f} ({svm_metrics.get('accuracy', 0)*100:.2f}%)")
        report.append(f"F1-Score (weighted): {svm_metrics.get('f1_weighted', 0):.4f}")
        report.append(f"F1-Score (macro):    {svm_metrics.get('f1_macro', 0):.4f}")
        if 'auc_roc' in svm_metrics and svm_metrics['auc_roc']:
            report.append(f"AUC-ROC:         {svm_metrics['auc_roc']:.4f}")
        report.append("")

        # Conclusiones
        report.append("3. INTERPRETACIÓN")
        report.append("-"*80)

        # Identificar cluster de riesgo
        if not clustering_stats.empty:
            risk_cluster = clustering_stats.iloc[0]
            report.append(f"• Estudiantes en Riesgo Alto: {risk_cluster['n_estudiantes']} ({risk_cluster['porcentaje']})")
            report.append(f"  - Requieren intervención prioritaria")
            report.append(f"  - Créditos promedio: {risk_cluster.get('creditos_mean', 'N/A')}")

        # Evaluar rendimiento del modelo
        accuracy = svm_metrics.get('accuracy', 0)
        if accuracy >= 0.85:
            report.append(f"• El modelo SVM alcanza un rendimiento EXCELENTE (accuracy={accuracy*100:.1f}%)")
        elif accuracy >= 0.75:
            report.append(f"• El modelo SVM alcanza un rendimiento BUENO (accuracy={accuracy*100:.1f}%)")
        else:
            report.append(f"• El modelo SVM requiere optimización (accuracy={accuracy*100:.1f}%)")

        report.append("")
        report.append("="*80)

        report_text = "\n".join(report)
        print("\n" + report_text)

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"\nReporte guardado en: {save_path}")

        return report_text
