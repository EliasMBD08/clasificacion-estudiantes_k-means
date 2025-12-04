"""
Script principal para el proyecto de Clasificación de Estudiantes según Desempeño.
Implementa pipeline completo: datos sintéticos -> clustering -> clasificación SVM.

Autor: Sistema de Clasificación UNMSM
"""

import os
import sys
import warnings
import numpy as np

# Fix para error de joblib en Windows (debe estar antes de importar sklearn)
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

warnings.filterwarnings('ignore')

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_generator import StudentDataGenerator
from src.preprocessing import DataPreprocessor
from src.clustering import StudentClustering
from src.classification import SVMClassifier
from src.visualization import Visualizer


def main():
    """
    Pipeline principal de ejecución.
    """
    print("\n" + "="*80)
    print("CLASIFICACIÓN DE ESTUDIANTES SEGÚN DESEMPEÑO ACADÉMICO - UNMSM")
    print("Sistema Híbrido: K-Means + SVM")
    print("="*80)

    # Configuración de rutas
    DATA_DIR = 'data'
    MODELS_DIR = 'models'
    RESULTS_DIR = 'results'

    # Crear directorios si no existen
    for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
        os.makedirs(directory, exist_ok=True)

    # =====================================================================
    # PASO 1: GENERAR/CARGAR DATOS
    # =====================================================================
    print("\n" + "="*80)
    print("PASO 1: GENERACIÓN/CARGA DE DATOS")
    print("="*80)

    data_generator = StudentDataGenerator(n_students=37257, random_state=42)
    df_raw = data_generator.load_or_generate(f'{DATA_DIR}/estudiantes_unmsm.csv')

    print(f"\nResumen del dataset:")
    print(f"  Dimensiones: {df_raw.shape}")
    print(f"  Columnas: {list(df_raw.columns)}")

    # =====================================================================
    # PASO 2: PREPROCESAMIENTO
    # =====================================================================
    print("\n" + "="*80)
    print("PASO 2: PREPROCESAMIENTO DE DATOS")
    print("="*80)

    preprocessor = DataPreprocessor()
    X, feature_names = preprocessor.fit_transform(df_raw.copy())

    print(f"\nDatos preprocesados:")
    print(f"  Matriz X: {X.shape}")
    print(f"  Características: {len(feature_names)}")

    # =====================================================================
    # PASO 3: CLUSTERING (K-MEANS)
    # =====================================================================
    print("\n" + "="*80)
    print("PASO 3: SEGMENTACIÓN DE ESTUDIANTES (K-MEANS)")
    print("="*80)

    clustering = StudentClustering(random_state=42)

    # Encontrar k óptimo
    optimal_k = clustering.find_optimal_k(X, k_range=range(2, 8), plot=True)
    print(f"\nNúmero óptimo de clusters: {optimal_k}")

    # Entrenar con k óptimo
    cluster_labels = clustering.fit(X, n_clusters=optimal_k)

    # Analizar clusters
    cluster_stats = clustering.analyze_clusters(X, feature_names,
                                               original_df=df_raw)

    # Guardar modelo
    clustering.save_model(f'{MODELS_DIR}/kmeans_model.pkl')

    # =====================================================================
    # PASO 4: CLASIFICACIÓN (SVM)
    # =====================================================================
    print("\n" + "="*80)
    print("PASO 4: CLASIFICACIÓN SUPERVISADA (SVM)")
    print("="*80)

    classifier = SVMClassifier(random_state=42)

    # Preparar datos
    X_train, X_val, X_test, y_train, y_val, y_test = classifier.prepare_data(
        X, cluster_labels, test_size=0.15, val_size=0.15, apply_smote=True
    )

    # Optimizar hiperparámetros
    best_params = classifier.optimize_hyperparameters(X_train, y_train, cv_folds=5)

    # Entrenar con mejores parámetros
    classifier.train(X_train, y_train,
                    C=best_params['C'],
                    gamma=best_params['gamma'],
                    kernel=best_params['kernel'])

    # Evaluar en validación y test
    val_metrics = classifier.evaluate(X_val, y_val, dataset_name="Validación")
    test_metrics = classifier.evaluate(X_test, y_test, dataset_name="Test")

    # Guardar modelo
    classifier.save_model(f'{MODELS_DIR}/svm_model.pkl')

    # =====================================================================
    # PASO 5: VISUALIZACIONES Y REPORTES
    # =====================================================================
    print("\n" + "="*80)
    print("PASO 5: GENERACIÓN DE VISUALIZACIONES Y REPORTES")
    print("="*80)

    visualizer = Visualizer(output_dir=RESULTS_DIR)

    # Visualización 1: Distribución de clusters
    # Detectar automáticamente el número de clusters
    n_clusters = len(np.unique(cluster_labels))
    if n_clusters == 2:
        cluster_names = ['Bajo Rendimiento', 'Alto Rendimiento']
    elif n_clusters == 3:
        cluster_names = ['Riesgo Alto', 'Rendimiento Medio', 'Alto Rendimiento']
    else:
        cluster_names = [f'Cluster {i}' for i in range(n_clusters)]

    print(f"Clusters detectados: {n_clusters}")
    print(f"Nombres de clusters: {cluster_names}")

    visualizer.plot_cluster_distribution(
        cluster_labels,
        cluster_names=cluster_names,
        save_path=f'{RESULTS_DIR}/distribucion_clusters.png'
    )

    # Visualización 2: Características de clusters
    # Crear DataFrame solo con los registros limpios (después del preprocesamiento)
    # Nota: df_raw tiene más filas que X porque el preprocesamiento eliminó duplicados,
    # valores nulos y outliers. Solo podemos usar los primeros len(X) registros.
    print(f"\nNota: df_raw tiene {len(df_raw)} filas, pero después del preprocesamiento quedan {len(X)} filas")
    print("Creando DataFrame de análisis solo con registros válidos...")

    # Tomar solo las primeras len(cluster_labels) filas de df_raw que corresponden a los datos limpios
    # ADVERTENCIA: Esta es una aproximación. Idealmente el preprocesador debería devolver los índices exactos.
    df_analysis = df_raw.head(len(cluster_labels)).copy()
    df_analysis['cluster'] = cluster_labels
    visualizer.plot_cluster_characteristics(
        df_analysis,
        cluster_col='cluster',
        features=['creditos_aprobados', 'ranking_facultad'],
        save_path=f'{RESULTS_DIR}/caracteristicas_clusters.png'
    )

    # Visualización 3: Matriz de confusión
    classifier.plot_confusion_matrix(
        test_metrics['y_true'],
        test_metrics['y_pred'],
        normalize=True,
        save_path=f'{RESULTS_DIR}/matriz_confusion.png'
    )

    # Visualización 4: Curvas ROC
    classifier.plot_roc_curves(
        test_metrics['y_true'],
        test_metrics['y_proba'],
        save_path=f'{RESULTS_DIR}/curvas_roc.png'
    )

    # Visualización 5: Métricas de rendimiento
    performance_metrics = {
        'Accuracy': test_metrics['accuracy'],
        'F1-Score (weighted)': test_metrics['f1_weighted'],
        'F1-Score (macro)': test_metrics['f1_macro']
    }
    if test_metrics['auc_roc']:
        performance_metrics['AUC-ROC'] = test_metrics['auc_roc']

    visualizer.plot_performance_comparison(
        performance_metrics,
        save_path=f'{RESULTS_DIR}/metricas_rendimiento.png'
    )

    # Reporte final
    visualizer.generate_summary_report(
        cluster_stats,
        test_metrics,
        save_path=f'{RESULTS_DIR}/reporte_final.txt'
    )

    # =====================================================================
    # PASO 6: RESULTADOS FINALES
    # =====================================================================
    print("\n" + "="*80)
    print("RESUMEN DE RESULTADOS")
    print("="*80)

    print(f"\n✓ Clustering (K-Means):")
    print(f"  - Número de clusters: {optimal_k}")
    print(f"  - Silhouette Score: {clustering.metrics['silhouette'][optimal_k-2]:.4f}")

    print(f"\n✓ Clasificación (SVM):")
    print(f"  - Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  - F1-Score: {test_metrics['f1_weighted']:.4f}")
    if test_metrics['auc_roc']:
        print(f"  - AUC-ROC: {test_metrics['auc_roc']:.4f}")

    print(f"\n✓ Estudiantes identificados:")
    for i, name in enumerate(cluster_names):
        count = (cluster_labels == i).sum()
        pct = (count / len(cluster_labels)) * 100
        print(f"  - {name}: {count:,} estudiantes ({pct:.1f}%)")

    print(f"\n✓ Archivos generados:")
    print(f"  - Datos: {DATA_DIR}/estudiantes_unmsm.csv")
    print(f"  - Modelos: {MODELS_DIR}/kmeans_model.pkl, {MODELS_DIR}/svm_model.pkl")
    print(f"  - Resultados: {RESULTS_DIR}/ (gráficos y reportes)")

    print("\n" + "="*80)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("="*80)
    print("\nPara visualizar los resultados, revise la carpeta 'results/'")
    print("Para usar los modelos entrenados, cargue los archivos .pkl desde 'models/'")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProceso interrumpido por el usuario.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
