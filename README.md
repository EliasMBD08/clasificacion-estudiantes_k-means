# Clasificación de Estudiantes según Desempeño Académico - UNMSM

Sistema híbrido de Machine Learning para segmentación y clasificación de estudiantes universitarios usando K-Means y SVM (Support Vector Machines).

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-OpenSource-green.svg)](LICENSE)

## Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Instalación](#instalación)
- [Uso](#uso)
- [Metodología Completa](#metodología)
  - [1. Preprocesamiento](#1-preprocesamiento-preprocessingpy)
  - [2. Clustering K-Means](#2-clustering-k-means-clusteringpy)
  - [3. Clasificación SVM](#3-clasificación-svm-classificationpy)
  - [4. Visualización](#4-visualización-y-reportes-visualizationpy)
- [Flujo de Trabajo del Sistema](#flujo-de-trabajo-del-sistema)
- [Criterios de Selección del Modelo](#criterios-de-selección-del-modelo)
- [Uso en Producción](#uso-en-producción)
- [Personalización](#personalización)
- [FAQ](#preguntas-frecuentes-faq)
- [Troubleshooting](#troubleshooting)
- [Referencias](#referencias)

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

### 1. Preprocesamiento ([preprocessing.py](src/preprocessing.py))

El preprocesamiento transforma los datos crudos en características numéricas normalizadas listas para el modelado.

**Pipeline completo:**

1. **Limpieza de datos:**
   - Eliminación de duplicados basados en `numero_documento` (mantiene primer registro)
   - Eliminación de registros con valores faltantes en variables críticas (`creditos_aprobados`, `ranking_facultad`)
   - Relleno de valores faltantes en categóricas con 'DESCONOCIDO'

2. **Detección de outliers:**
   - Método IQR (Interquartile Range) agrupado por facultad
   - Límites: `Q1 - 3*IQR` y `Q3 + 3*IQR` (outliers extremos)
   - Aplicado a `creditos_aprobados` para remover valores anómalos respetando diferencias entre facultades

3. **Codificación de variables categóricas:**
   - **One-Hot Encoding** para `facultad`, `pais_nacimiento`, `tipo_documento`
   - **Label Encoding** para `tercio_superior` (SI=1, NO=0)
   - Se descarta la primera categoría (drop_first=True) para evitar multicolinealidad

4. **Normalización Min-Max [0, 1]:**
   - Aplicada a `creditos_aprobados` y `ranking_facultad`
   - Usa `MinMaxScaler` de scikit-learn para escalar al rango [0, 1]
   - Importante: el scaler se ajusta (fit) solo con datos de entrenamiento

5. **Selección de características:**
   - Excluye columnas identificatorias: `apellidos_nombres`, `codigo_alumno`, `numero_documento`, `correo_institucional`
   - Retiene todas las variables académicas y categóricas codificadas

**Características finales utilizadas:**
- Variables numéricas normalizadas: `creditos_aprobados`, `ranking_facultad`
- Variable binaria: `tercio_superior`
- Variables categóricas codificadas: ~60 columnas dummy para facultades, países y tipos de documento

### 2. Clustering K-Means ([clustering.py](src/clustering.py))

El clustering identifica segmentos naturales de estudiantes sin etiquetas previas.

**Optimización del número de clusters:**

1. **Métricas evaluadas** (rango: k=2 a k=7):
   - **WCSS (Within-Cluster Sum of Squares)**: Suma de distancias al cuadrado dentro de cada cluster
   - **Silhouette Score**: Mide cohesión y separación (rango [-1, 1], >0.5 es bueno)
   - **Calinski-Harabasz**: Ratio de dispersión entre/dentro clusters (mayor es mejor)
   - **Davies-Bouldin**: Promedio de similitud entre clusters (menor es mejor)

2. **Selección de k óptimo:**
   - **Método del codo**: Detecta punto de inflexión en WCSS usando segunda derivada
   - **Criterio Silhouette**: Selecciona k con máximo Silhouette Score (criterio principal)
   - El sistema genera gráficos comparativos en [cluster_optimization.png](results/cluster_optimization.png)

3. **Entrenamiento K-Means:**
   - **Inicialización**: k-means++ (ubica centroides iniciales de forma inteligente)
   - **n_init=50**: Ejecuta 50 veces con diferentes inicializaciones y selecciona la mejor
   - **max_iter=300**: Máximo de iteraciones por ejecución
   - **Algoritmo Lloyd**: Implementación clásica de k-means

4. **Interpretación de segmentos** (k=3 típicamente):
   - **Cluster 0 - Riesgo Alto**: Pocos créditos aprobados, ranking bajo, fuera del tercio superior
   - **Cluster 1 - Rendimiento Medio**: Créditos moderados, ranking medio, algunos en tercio superior
   - **Cluster 2 - Alto Rendimiento**: Muchos créditos, ranking alto, mayoría en tercio superior

**Análisis automático:**
- Calcula estadísticas descriptivas por cluster
- Identifica características distintivas de cada segmento
- Genera distribución de estudiantes por facultad en cada cluster

### 3. Clasificación SVM ([classification.py](src/classification.py))

El clasificador SVM automatiza la asignación de nuevos estudiantes a los segmentos identificados.

**Preparación de datos:**

1. **División estratificada:**
   - **Train: 70%** - Para entrenar el modelo
   - **Validation: 15%** - Para ajustar hiperparámetros
   - **Test: 15%** - Para evaluación final (nunca visto por el modelo)
   - `stratify=y` asegura proporciones iguales de cada clase en cada conjunto

2. **Balanceo con SMOTE** (Synthetic Minority Over-sampling Technique):
   - **Problema**: Los clusters pueden estar desbalanceados (ej. 25%-40%-35%)
   - **Solución**: SMOTE genera ejemplos sintéticos de clases minoritarias
   - Se aplica **solo al conjunto de entrenamiento** para evitar data leakage
   - **k_neighbors=5**: Crea nuevos ejemplos interpolando entre 5 vecinos más cercanos
   - **Resultado**: Clases balanceadas (~33%-33%-33% en train) mejoran el aprendizaje

**Optimización de hiperparámetros:**

Grid Search con validación cruzada estratificada (5 folds):

- **Parámetros evaluados:**
  - **C** (regularización): [1, 10, 100]
    - C pequeño: Margen más amplio, más regularización, menos overfitting
    - C grande: Margen más estrecho, menos regularización, más ajuste a datos
  - **gamma** (kernel RBF): ['scale', 0.001, 0.01]
    - gamma pequeño: Influencia amplia, decisión más suave
    - gamma grande: Influencia local, decisión más compleja
  - **kernel**: ['rbf'] (Radial Basis Function)
    - RBF permite decisiones no lineales (mejor para datos complejos)

- **Métrica de optimización**: F1-score weighted (balance entre precisión y recall)
- **Validación cruzada**: StratifiedKFold con 5 pliegues
- **Paralelización**: 6 núcleos con backend 'threading' (optimizado para Windows)

**Entrenamiento del modelo:**

- Usa mejores hiperparámetros del Grid Search
- **probability=True**: Habilita cálculo de probabilidades por clase
- **class_weight='balanced'**: Ajusta pesos inversamente proporcionales a frecuencia de clase
- **random_state=42**: Reproducibilidad de resultados

**Evaluación exhaustiva:**

1. **Métricas globales:**
   - **Accuracy**: Porcentaje de predicciones correctas
   - **F1-Score weighted**: Media armónica de precision/recall ponderada por clase
   - **F1-Score macro**: F1 promedio sin ponderar (trata clases por igual)
   - **AUC-ROC weighted**: Área bajo curva ROC (one-vs-rest para multiclase)

2. **Métricas por clase:**
   - **Precision**: De los predichos como clase X, cuántos realmente lo son
   - **Recall**: De los que realmente son clase X, cuántos se detectaron
   - **F1-Score**: Balance entre precision y recall
   - **Support**: Número de ejemplos de cada clase

3. **Visualizaciones:**
   - **Matriz de confusión normalizada**: Muestra patrones de error entre clases
   - **Curvas ROC por clase**: Evaluación de discriminación (one-vs-rest)
   - **Gráfico de métricas**: Comparación visual de accuracy, F1, AUC

**Consideraciones técnicas importantes:**

- **No hay fuga de datos**: Test set nunca se usa en entrenamiento ni optimización
- **SMOTE solo en train**: Evita generar sintéticos que luego se evalúen
- **Validación cruzada**: Reduce varianza en selección de hiperparámetros
- **Normalización consistente**: Mismo scaler para train/val/test

### 4. Visualización y Reportes ([visualization.py](src/visualization.py))

Genera visualizaciones profesionales y reportes para interpretar resultados:

**Gráficos generados:**

1. **Distribución de clusters** ([distribucion_clusters.png](results/distribucion_clusters.png)):
   - Gráfico de barras con conteos absolutos y porcentajes
   - Gráfico de pastel para proporciones
   - Colores diferenciados por segmento

2. **Características por cluster** ([caracteristicas_clusters.png](results/caracteristicas_clusters.png)):
   - Barras comparativas de promedios por segmento
   - Barras de error (desviación estándar)
   - Visualiza `creditos_aprobados` y `ranking_facultad`

3. **Matriz de confusión normalizada** ([matriz_confusion.png](results/matriz_confusion.png)):
   - Heatmap con valores normalizados por fila
   - Identifica confusiones entre clases
   - Diagonal principal = predicciones correctas

4. **Curvas ROC multiclase** ([curvas_roc.png](results/curvas_roc.png)):
   - One-vs-Rest para cada clase
   - AUC por clase individual
   - Línea diagonal de referencia (clasificador aleatorio)

5. **Métricas de rendimiento** ([metricas_rendimiento.png](results/metricas_rendimiento.png)):
   - Comparación visual de Accuracy, F1-Score, AUC-ROC
   - Barras horizontales con valores exactos

6. **Reporte textual** ([reporte_final.txt](results/reporte_final.txt)):
   - Estadísticas completas de clustering
   - Métricas de clasificación
   - Interpretaciones y recomendaciones

## Flujo de Trabajo del Sistema

### Pipeline de Ejecución ([main.py](main.py))

El script principal orquesta todo el proceso en 6 pasos:

```python
# PASO 1: Generación/Carga de Datos
- Genera dataset sintético de 37,257 estudiantes (o carga existente)
- 20 facultades de la UNMSM
- Distribuciones realistas de variables académicas

# PASO 2: Preprocesamiento
- Limpieza: duplicados, outliers, valores faltantes
- Codificación: One-Hot (categóricas) + Label (binarias)
- Normalización: Min-Max [0,1] para numéricas
- Output: Matriz X (n_samples x ~65 features)

# PASO 3: Clustering K-Means
- Optimización de k usando múltiples métricas
- Entrenamiento con k óptimo (típicamente 3)
- Análisis de características por cluster
- Output: Labels de cluster para cada estudiante

# PASO 4: Clasificación SVM
- División train (70%) / val (15%) / test (15%)
- Balanceo con SMOTE en train
- Grid Search de hiperparámetros con CV
- Entrenamiento con mejores parámetros
- Evaluación exhaustiva en val y test
- Output: Modelo entrenado + métricas

# PASO 5: Visualizaciones y Reportes
- Genera 6 visualizaciones en results/
- Crea reporte textual interpretable
- Output: PNG de alta resolución (300 dpi)

# PASO 6: Guardado de Modelos
- Serializa K-Means en models/kmeans_model.pkl
- Serializa SVM en models/svm_model.pkl
- Modelos reutilizables para predicción en producción
```

### Arquitectura Modular

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py                              │
│              (Orquestador del pipeline)                      │
└─────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│data_generator │  │preprocessing  │  │ clustering    │
└───────────────┘  └───────────────┘  └───────────────┘
        │                  │                  │
        │ CSV              │ X (normalized)   │ cluster_labels
        ▼                  ▼                  ▼
┌──────────────────────────────────────────────────────┐
│                  classification.py                    │
│         (SVM con Grid Search y SMOTE)                 │
└──────────────────────────────────────────────────────┘
                           │
                           │ metrics + predictions
                           ▼
┌──────────────────────────────────────────────────────┐
│                  visualization.py                     │
│          (Gráficos + Reportes interpretativos)        │
└──────────────────────────────────────────────────────┘
```

### Consideraciones de Diseño

**Reproducibilidad:**
- `random_state=42` en todos los componentes estocásticos
- Semillas fijas para: generación de datos, train/test split, K-Means, SVM, SMOTE
- Resultados consistentes entre ejecuciones

**Modularidad:**
- Cada módulo (`preprocessing`, `clustering`, `classification`, `visualization`) es independiente
- Interfaces claras: `fit()`, `transform()`, `predict()`
- Reutilizable en otros proyectos de clasificación

**Escalabilidad:**
- Min-Max normalización: preserva relaciones entre variables
- K-Means++: Convergencia más rápida que inicialización aleatoria
- SVM con kernel RBF: Maneja relaciones no lineales complejas
- Grid Search paralelo: Usa 6 núcleos para acelerar búsqueda

**Prevención de errores comunes:**
- No hay data leakage (scaler se ajusta solo con train)
- SMOTE solo en train (evita sintéticos en evaluación)
- Validación cruzada estratificada (mantiene proporción de clases)
- Normalización antes de clustering/clasificación (distancias euclidianas correctas)

**Optimizaciones Windows:**
- `LOKY_MAX_CPU_COUNT=6` para evitar errores de paralelización
- Backend 'threading' en Grid Search (más estable que 'multiprocessing' en Windows)
- `cache_size=500` en SVM para acelerar entrenamiento

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

## Criterios de Selección del Modelo

### ¿Por qué K-Means para Clustering?

**Ventajas que justifican su elección:**
1. **Eficiencia computacional**: O(n*k*i*d) - escala bien con 37K estudiantes
2. **Interpretabilidad**: Centroides representan estudiante "promedio" de cada segmento
3. **Convergencia garantizada**: Con k-means++ la inicialización es robusta
4. **Segmentación clara**: Asignación hard (cada estudiante a un único cluster)
5. **Funciona bien con datos normalizados**: Variables en misma escala [0,1]

**Alternativas consideradas:**
- **DBSCAN**: Descartado por sensibilidad a parámetros eps/min_samples
- **Hierarchical**: Computacionalmente costoso para 37K muestras
- **GMM (Gaussian Mixture)**: Mayor complejidad sin mejora significativa en datos académicos

### ¿Por qué SVM con Kernel RBF?

**Ventajas que justifican su elección:**
1. **Clasificación no lineal**: RBF mapea a espacio de alta dimensión sin costo computacional explícito
2. **Robustez con clases desbalanceadas**: `class_weight='balanced'` ajusta pesos automáticamente
3. **Generalización**: Maximización del margen reduce overfitting
4. **Probabilidades calibradas**: `probability=True` permite cuantificar certeza de predicción
5. **Rendimiento superior en datasets medianos**: <100K muestras con ~65 features

**Alternativas consideradas:**
- **Random Forest**: Menos interpretable, no aprovecha estructura de margen
- **XGBoost**: Mayor complejidad, requiere más tuning
- **Logistic Regression**: Asume linealidad (insuficiente para datos complejos)
- **Neural Networks**: Overkill para este tamaño de datos, requiere más datos para entrenar

### ¿Por qué SMOTE para Balanceo?

**Ventajas que justifican su elección:**
1. **Genera ejemplos sintéticos realistas**: Interpolación entre vecinos cercanos
2. **Preserva distribución**: No duplica exactamente, crea variabilidad
3. **Mejora recall de clases minoritarias**: Modelo aprende mejor sus patrones
4. **No altera test set**: Solo se aplica a train, evaluación justa

**Alternativas consideradas:**
- **Oversampling simple**: Duplicación exacta causa overfitting
- **Undersampling**: Pérdida de información de clases mayoritarias
- **Class weights solamente**: Insuficiente cuando desbalance es severo (>2:1)

### Selección de Hiperparámetros Clave

**K-Means:**
- `n_init=50`: Múltiples inicializaciones garantizan encontrar buen mínimo local
- `max_iter=300`: Suficiente para convergencia (típicamente <100 iteraciones)
- `init='k-means++'`: Inicialización inteligente reduce iteraciones y mejora resultado

**SVM:**
- `C=[1, 10, 100]`: Rango cubre regularización suave a estricta
- `gamma=['scale', 0.001, 0.01]`: Rango cubre influencia global a local
- `kernel='rbf'`: RBF universal approximator, mejor que lineal/polinomial para estos datos

**Grid Search:**
- `cv=5`: Balance entre varianza y costo computacional
- `scoring='f1_weighted'`: Prioriza balance precision/recall ponderado por clase
- `n_jobs=6`: Paralelización maximiza velocidad en CPU de 6+ núcleos

## Aplicaciones Prácticas

1. **Sistema de Alerta Temprana**: Identificación proactiva de estudiantes en riesgo
2. **Asignación de Recursos**: Focalización de tutorías y apoyo académico
3. **Intervenciones Personalizadas**: Diseño de programas según perfil de estudiante
4. **Monitoreo Continuo**: Actualización automática de clasificaciones cada semestre
5. **Predicción para nuevos estudiantes**: Usar modelos guardados (`.pkl`) para clasificar estudiantes recién matriculados
6. **Dashboard institucional**: Integrar métricas en sistema de información académica

## Uso en Producción

### Cargar Modelos Entrenados

Los modelos guardados en `models/` pueden reutilizarse para clasificar nuevos estudiantes sin reentrenar:

```python
import joblib
import pandas as pd
from src.preprocessing import DataPreprocessor

# 1. Cargar modelos entrenados
kmeans = joblib.load('models/kmeans_model.pkl')
svm = joblib.load('models/svm_model.pkl')

# 2. Cargar nuevos datos
df_new = pd.read_csv('nuevos_estudiantes.csv')

# 3. Preprocesar (importante: usar el mismo preprocesador)
preprocessor = DataPreprocessor()
X_new = preprocessor.transform(df_new)

# 4. Predecir segmento con SVM
predictions = svm.predict(X_new)
probabilities = svm.predict_proba(X_new)

# 5. Interpretar resultados
class_names = ['Riesgo Alto', 'Rendimiento Medio', 'Alto Rendimiento']
for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
    student_name = df_new.iloc[i]['apellidos_nombres']
    confidence = proba[pred] * 100
    print(f"{student_name} -> {class_names[pred]} (Confianza: {confidence:.1f}%)")
```

### Ejemplo de Integración con API

```python
from flask import Flask, request, jsonify
import joblib
from src.preprocessing import DataPreprocessor

app = Flask(__name__)

# Cargar modelos al iniciar
svm_model = joblib.load('models/svm_model.pkl')
preprocessor = DataPreprocessor()

@app.route('/clasificar', methods=['POST'])
def clasificar_estudiante():
    """
    Endpoint para clasificar un estudiante.

    Input JSON:
    {
        "creditos_aprobados": 120,
        "ranking_facultad": 15,
        "tercio_superior": "SI",
        "facultad": "Ingeniería de Sistemas",
        "pais_nacimiento": "Perú",
        "tipo_documento": "DNI"
    }

    Output JSON:
    {
        "segmento": "Alto Rendimiento",
        "confianza": 92.5,
        "probabilidades": {
            "Riesgo Alto": 2.1,
            "Rendimiento Medio": 5.4,
            "Alto Rendimiento": 92.5
        }
    }
    """
    data = request.json
    df = pd.DataFrame([data])

    # Preprocesar
    X = preprocessor.transform(df)

    # Predecir
    pred = svm_model.predict(X)[0]
    proba = svm_model.predict_proba(X)[0]

    class_names = ['Riesgo Alto', 'Rendimiento Medio', 'Alto Rendimiento']

    return jsonify({
        'segmento': class_names[pred],
        'confianza': float(proba[pred] * 100),
        'probabilidades': {
            name: float(prob * 100)
            for name, prob in zip(class_names, proba)
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
```

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

**Importante**: Si tus datos tienen facultades, países o tipos de documento diferentes a los del entrenamiento, deberás reentrenar el modelo completamente.

### Modificar Parámetros

Edita [main.py](main.py) para ajustar:

```python
# Número de estudiantes (si generas sintéticos)
data_generator = StudentDataGenerator(n_students=50000)

# Rango de clusters a evaluar
optimal_k = clustering.find_optimal_k(X, k_range=range(2, 10))

# División train/val/test
X_train, X_val, X_test, ... = classifier.prepare_data(
    X, y, test_size=0.20, val_size=0.10
)

# Grid de hiperparámetros SVM (en src/classification.py)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'poly']
}
```

### Experimentar con Otros Modelos

Puedes reemplazar SVM por otros clasificadores manteniendo la misma interfaz:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# En lugar de SVM
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# O usar XGBoost
import xgboost as xgb
model = xgb.XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

## Preguntas Frecuentes (FAQ)

### ¿Cómo funcionan las predicciones del modelo?

El sistema utiliza un enfoque híbrido en dos etapas:

1. **K-Means identifica patrones** en datos históricos sin etiquetas previas
2. **SVM aprende a clasificar** nuevos estudiantes en esos patrones identificados
3. Para un nuevo estudiante, SVM predice su segmento basándose en características similares a estudiantes históricos

### ¿Qué significa el "Silhouette Score"?

- Mide qué tan bien están separados los clusters
- Rango: -1 (mal agrupado) a +1 (perfectamente agrupado)
- **Interpretación**:
  - > 0.7: Estructura fuerte y clara
  - 0.5-0.7: Estructura razonable (típico en datos reales)
  - 0.25-0.5: Estructura débil pero detectable
  - < 0.25: No hay estructura clara

### ¿Por qué usar SMOTE? ¿No basta con class_weight='balanced'?

`class_weight='balanced'` ajusta la función de pérdida pero no aumenta datos. SMOTE:
- Genera nuevos ejemplos sintéticos de clases minoritarias
- Ayuda al modelo a aprender patrones de clases subrepresentadas
- Mejora recall de clases minoritarias sin sacrificar precision de mayoritarias
- Es especialmente útil cuando el desbalance es severo (>2:1)

### ¿Cómo interpretar la matriz de confusión?

Ejemplo para 3 clases:
```
                 Predicho
              RA    RM    AR
Real    RA   [85%   12%   3%]
        RM   [8%    80%   12%]
        AR   [2%    10%   88%]
```

- **Diagonal** (85%, 80%, 88%): Predicciones correctas
- **Fuera de diagonal**: Confusiones entre clases
- Ejemplo: 12% de "Rendimiento Medio" se clasifican erróneamente como "Alto Rendimiento"

### ¿Puedo usar este código para otros problemas de clasificación?

¡Sí! El código es modular y adaptable. Solo necesitas:
1. Reemplazar el dataset con tus datos
2. Ajustar variables en `preprocessing.py` (líneas 24-25)
3. Modificar nombres de clases según tu problema
4. Reentrenar los modelos ejecutando `python main.py`

### ¿Qué hacer si el accuracy es bajo (<70%)?

1. **Revisar preprocesamiento**: ¿Hay outliers extremos? ¿Variables categóricas mal codificadas?
2. **Aumentar Grid Search**: Probar más valores de C y gamma
3. **Feature engineering**: Crear nuevas variables (ej. ratio créditos/ranking)
4. **Recolectar más datos**: <1000 muestras suelen ser insuficientes para SVM complejo
5. **Probar otros modelos**: Random Forest, XGBoost pueden funcionar mejor

## Troubleshooting

### Error: `LOKY_MAX_CPU_COUNT`

**Síntoma**: `BrokenProcessPool` o errores de paralelización en Windows

**Solución**: Ya está configurado en el código (`main.py` línea 14 y `classification.py` línea 8). Si persiste:
```python
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '1'  # Deshabilita paralelización
```

### Error: `MemoryError` durante Grid Search

**Síntoma**: Sistema se queda sin RAM durante optimización

**Solución**: Reducir Grid Search:
```python
param_grid = {
    'C': [10],  # Solo un valor
    'gamma': ['scale'],
    'kernel': ['rbf']
}
```

O reducir `n_jobs`:
```python
grid_search = GridSearchCV(..., n_jobs=2)  # Menos paralelización
```

### Error: `ValueError: could not convert string to float`

**Síntoma**: Error durante preprocesamiento

**Solución**: Revisar que variables categóricas se codifiquen correctamente:
```python
# En preprocessing.py, verificar columnas categóricas
self.categorical_features = ['facultad', 'pais_nacimiento', 'tipo_documento']
```

### Warning: `Convergence not reached`

**Síntoma**: K-Means no converge en 300 iteraciones

**Solución**: No es crítico pero puedes aumentar `max_iter`:
```python
kmeans = KMeans(n_clusters=k, max_iter=500)  # Más iteraciones
```

### Resultados varían entre ejecuciones

**Síntoma**: Métricas cambian cada vez que ejecutas

**Solución**: Verificar que `random_state=42` esté en todos los componentes:
- `StudentDataGenerator(random_state=42)`
- `train_test_split(..., random_state=42)`
- `KMeans(random_state=42)`
- `SVC(random_state=42)`
- `SMOTE(random_state=42)`

### Gráficos no se guardan correctamente

**Síntoma**: Archivos PNG vacíos o corruptos

**Solución**: Verificar que la carpeta `results/` existe:
```python
import os
os.makedirs('results', exist_ok=True)
```

## Autor

Sistema desarrollado por el estudiante MARCOS BERNARDO, Elias Daniel en colaboración con el equipo de trabajo del curso de Minería de Datos:
- Jean Silva López
- Luis Fernando Ruiz Palacios
- Renzo Luis Campos Vergara
- Marcos Bernardo Elias
- George Huayhuas Galvan
- Kevin Anderson Gonzalez Cabezas

Los estudiantes pertenecen a la Universidad Nacional Mayor de San Marcos (UNMSM) como parte del miniproyecto de Educational Data Mining.

## Licencia

Este proyecto es de código abierto (OperSource) para uso académico y de investigación.

## Referencias

- Baker, R. S., & Inventado, P. S. (2014). Educational data mining and learning analytics.
- Romero, C., & Ventura, S. (2020). Educational data mining and learning analytics: An updated review.
- Martínez-Abad, F., & Chaparro-Peláez, J. (2022). Comparison of machine learning algorithms for predicting student academic success.

---

**Nota**: Este proyecto utiliza datos sintéticos generados automáticamente para propósitos de demostración. Para uso en producción, debe ser alimentado con datos reales del sistema de información académica institucional.
