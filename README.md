# Clasificación de Estudiantes según Desempeño Académico - UNMSM

Sistema híbrido de Machine Learning para segmentación y clasificación de estudiantes universitarios usando K-Means y SVM (Support Vector Machines).

## Descripción del Proyecto

Este proyecto implementa un sistema completo de análisis y clasificación de estudiantes de la Universidad Nacional Mayor de San Marcos (UNMSM) basado en su desempeño académico. Utiliza una metodología híbrida que combina:

1. **Clustering no supervisado (K-Means)**: Para identificar grupos naturales de estudiantes según sus características académicas
2. **Clasificación supervisada (SVM)**: Para automatizar la categorización de nuevos estudiantes en los segmentos identificados

### Objetivos

- Segmentar estudiantes en grupos homogéneos según su perfil de desempeño
- Identificar estudiantes en riesgo académico de manera temprana
- Automatizar la clasificación de nuevos estudiantes
- Proporcionar insights para intervenciones pedagógicas personalizadas

## Estructura del Proyecto

```
clasificacion_estudiantes/
│
├── main.py                      # Script principal de ejecución
├── requirements.txt             # Dependencias del proyecto
├── README.md                    # Este archivo
│
├── src/                         # Código fuente
│   ├── data_generator.py        # Generador de datos sintéticos
│   ├── preprocessing.py         # Preprocesamiento y limpieza de datos
│   ├── clustering.py            # Implementación de K-Means
│   ├── classification.py        # Implementación de SVM
│   └── visualization.py         # Visualizaciones y reportes
│
├── data/                        # Datos (generado automáticamente)
│   └── estudiantes_unmsm.csv    # Dataset de estudiantes
│
├── models/                      # Modelos entrenados
│   ├── kmeans_model.pkl         # Modelo K-Means
│   └── svm_model.pkl            # Modelo SVM
│
├── results/                     # Resultados y visualizaciones
│   ├── distribucion_clusters.png
│   ├── caracteristicas_clusters.png
│   ├── matriz_confusion.png
│   ├── curvas_roc.png
│   ├── cluster_optimization.png
│   ├── metricas_rendimiento.png
│   └── reporte_final.txt
│
└── notebooks/                   # Notebooks de análisis (opcional)
```

## Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalar Dependencias

Ejecuta el siguiente comando en la terminal:

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn joblib
```

O instala desde el archivo requirements.txt:

```bash
pip install -r requirements.txt
```

## Uso

### Ejecución Completa del Pipeline

Para ejecutar todo el proceso (generación de datos, preprocesamiento, clustering, clasificación y visualizaciones):

```bash
cd clasificacion_estudiantes
python main.py
```

### Salida Esperada

El script generará:

1. **Dataset sintético** de 37,257 estudiantes en `data/estudiantes_unmsm.csv`
2. **Modelos entrenados** en `models/`:
   - `kmeans_model.pkl`: Modelo de clustering
   - `svm_model.pkl`: Modelo de clasificación
3. **Visualizaciones** en `results/`:
   - Distribución de clusters
   - Características por segmento
   - Matriz de confusión
   - Curvas ROC
   - Métricas de rendimiento
4. **Reporte final** en `results/reporte_final.txt`

## Metodología

### 1. Preprocesamiento

- Limpieza de datos (duplicados, valores faltantes)
- Detección y eliminación de outliers usando IQR por facultad
- Codificación de variables categóricas (One-Hot Encoding)
- Normalización Min-Max [0, 1] de variables numéricas

### 2. Clustering (K-Means)

- Optimización del número de clusters usando:
  - Método del codo (Elbow Method)
  - Índice de Silhouette
  - Calinski-Harabasz Score
- Entrenamiento con k-means++ y 50 inicializaciones
- Identificación de 3 segmentos:
  - **Riesgo Alto**: Estudiantes con bajo rendimiento que requieren intervención
  - **Rendimiento Medio**: Estudiantes con potencial de mejora
  - **Alto Rendimiento**: Estudiantes destacados

### 3. Clasificación (SVM)

- División estratificada: 70% entrenamiento, 15% validación, 15% prueba
- Aplicación de SMOTE para balanceo de clases
- Grid Search para optimización de hiperparámetros (C, gamma)
- Kernel RBF (Radial Basis Function)
- Métricas de evaluación: Accuracy, Precision, Recall, F1-Score, AUC-ROC

## Variables del Dataset

### Variables de Identificación
- `apellidos_nombres`: Nombre completo del estudiante
- `codigo_alumno`: Código institucional
- `tipo_documento`: DNI, CE o Pasaporte
- `numero_documento`: Número de identificación
- `correo_institucional`: Email @unmsm.edu.pe

### Variables Académicas (Principales)
- `creditos_aprobados`: Número de créditos cursados y aprobados
- `ranking_facultad`: Posición en el ranking de su facultad
- `tercio_superior`: Pertenencia al tercio superior (SI/NO)

### Variables de Contexto
- `facultad`: Facultad de adscripción (20 facultades)
- `pais_nacimiento`: País de origen del estudiante

## Resultados Esperados

Basado en el documento técnico, el sistema debe lograr:

- **Clustering**: Silhouette Score ≈ 0.42
- **Clasificación SVM**:
  - Accuracy ≈ 87.3%
  - F1-Score weighted ≈ 0.86
  - AUC-ROC ≈ 0.93

### Distribución de Segmentos
- Riesgo Alto: ~25% (9,314 estudiantes)
- Rendimiento Medio: ~40% (14,903 estudiantes)
- Alto Rendimiento: ~35% (13,040 estudiantes)

## Aplicaciones Prácticas

1. **Sistema de Alerta Temprana**: Identificación proactiva de estudiantes en riesgo
2. **Asignación de Recursos**: Focalización de tutorías y apoyo académico
3. **Intervenciones Personalizadas**: Diseño de programas según perfil de estudiante
4. **Monitoreo Continuo**: Actualización automática de clasificaciones cada semestre

## Personalización

### Usar Datos Propios

Para usar tus propios datos, reemplaza el contenido de `data/estudiantes_unmsm.csv` con tu dataset. Asegúrate de que contenga las columnas necesarias:

```python
# Columnas requeridas
required_columns = [
    'creditos_aprobados',
    'ranking_facultad',
    'tercio_superior',
    'facultad',
    'pais_nacimiento',
    'tipo_documento'
]
```

### Modificar Parámetros

Edita `main.py` para ajustar:

```python
# Número de estudiantes
data_generator = StudentDataGenerator(n_students=50000)

# Rango de clusters a evaluar
optimal_k = clustering.find_optimal_k(X, k_range=range(2, 10))

# División train/val/test
X_train, X_val, X_test, ... = classifier.prepare_data(
    X, y, test_size=0.20, val_size=0.10
)
```

## Autor

Sistema desarrollado para la Universidad Nacional Mayor de San Marcos (UNMSM) como parte del miniproyecto de Educational Data Mining.

## Licencia

Este proyecto es de uso académico y de investigación.

## Referencias

- Baker, R. S., & Inventado, P. S. (2014). Educational data mining and learning analytics.
- Romero, C., & Ventura, S. (2020). Educational data mining and learning analytics: An updated review.
- Martínez-Abad, F., & Chaparro-Peláez, J. (2022). Comparison of machine learning algorithms for predicting student academic success.

## Contacto

Para consultas sobre el proyecto, contactar a la Dirección Académica de la UNMSM.

---

**Nota**: Este proyecto utiliza datos sintéticos generados automáticamente para propósitos de demostración. Para uso en producción, debe ser alimentado con datos reales del sistema de información académica institucional.
