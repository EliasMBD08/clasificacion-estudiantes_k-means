"""
Módulo de clasificación supervisada usando SVM.
Implementa entrenamiento, evaluación y predicción con optimización de hiperparámetros.
"""

import os
# Fix para error de joblib en Windows
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                            f1_score, roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns


class SVMClassifier:
    """
    Clasificador SVM para categorización de estudiantes en segmentos de desempeño.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.class_names = ['Riesgo Alto', 'Rendimiento Medio', 'Alto Rendimiento']
        self.metrics = {}

    def prepare_data(self, X: np.ndarray, y: np.ndarray,
                    test_size: float = 0.15, val_size: float = 0.15,
                    apply_smote: bool = True) -> Tuple:
        """
        Prepara datos: split train/val/test y aplica SMOTE para balanceo.

        Args:
            X: Matriz de características
            y: Vector de etiquetas (clusters)
            test_size: Proporción para conjunto de prueba
            val_size: Proporción para conjunto de validación
            apply_smote: Si True, aplica SMOTE al conjunto de entrenamiento

        Returns:
            Tupla (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("\nPreparando datos para clasificación...")

        # Split inicial: train+val / test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )

        # Split: train / val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp,
            random_state=self.random_state
        )

        print(f"  Train: {len(X_train)} muestras")
        print(f"  Val:   {len(X_val)} muestras")
        print(f"  Test:  {len(X_test)} muestras")

        # Distribución de clases antes de SMOTE
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\nDistribución de clases (train antes de SMOTE):")
        for cls, count in zip(unique, counts):
            print(f"  Clase {cls}: {count} ({count/len(y_train)*100:.1f}%)")

        # Aplicar SMOTE solo al conjunto de entrenamiento
        if apply_smote:
            smote = SMOTE(random_state=self.random_state, k_neighbors=5)
            X_train, y_train = smote.fit_resample(X_train, y_train)

            print(f"\nDistribución de clases (train después de SMOTE):")
            unique, counts = np.unique(y_train, return_counts=True)
            for cls, count in zip(unique, counts):
                print(f"  Clase {cls}: {count} ({count/len(y_train)*100:.1f}%)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                cv_folds: int = 5) -> Dict:
        """
        Optimiza hiperparámetros usando Grid Search con validación cruzada.

        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento
            cv_folds: Número de pliegues para CV

        Returns:
            Diccionario con mejores parámetros
        """
        print("\nOptimizando hiperparámetros de SVM...")

        # Grid de hiperparámetros
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf']
        }

        # Configurar Grid Search
        svm = SVC(random_state=self.random_state, probability=True)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        grid_search = GridSearchCV(
            svm, param_grid, cv=cv, scoring='f1_weighted',
            n_jobs=1, verbose=1, return_train_score=True
        )

        # Entrenar
        grid_search.fit(X_train, y_train)

        self.best_params = grid_search.best_params_
        print(f"\nMejores parámetros encontrados:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")

        print(f"\nMejor F1-score (CV): {grid_search.best_score_:.4f}")

        return self.best_params

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             C: float = 10, gamma: float = 0.01, kernel: str = 'rbf'):
        """
        Entrena el modelo SVM con parámetros especificados.

        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento
            C: Parámetro de regularización
            gamma: Coeficiente del kernel RBF
            kernel: Tipo de kernel
        """
        print(f"\nEntrenando SVM con kernel={kernel}, C={C}, gamma={gamma}...")

        self.model = SVC(
            C=C,
            gamma=gamma,
            kernel=kernel,
            random_state=self.random_state,
            probability=True,
            class_weight='balanced'
        )

        self.model.fit(X_train, y_train)
        print("Entrenamiento completado.")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                dataset_name: str = "Test") -> Dict:
        """
        Evalúa el modelo en un conjunto de datos.

        Args:
            X_test: Datos de evaluación
            y_test: Etiquetas verdaderas
            dataset_name: Nombre del conjunto (para reporte)

        Returns:
            Diccionario con métricas de evaluación
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado. Ejecute train() primero.")

        print(f"\n{'='*60}")
        print(f"EVALUACIÓN EN CONJUNTO {dataset_name.upper()}")
        print('='*60)

        # Predicciones
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        # Métricas globales
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')

        print(f"\nMétricas Globales:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  F1-Score (weighted): {f1_weighted:.4f}")
        print(f"  F1-Score (macro): {f1_macro:.4f}")

        # ROC-AUC (one-vs-rest para multiclase)
        try:
            auc_ovr = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            print(f"  AUC-ROC (weighted): {auc_ovr:.4f}")
        except:
            auc_ovr = None

        # Reporte de clasificación
        print(f"\nReporte de Clasificación:")
        print(classification_report(y_test, y_pred, target_names=self.class_names,
                                   digits=4))

        # Guardar métricas
        metrics = {
            'dataset': dataset_name,
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'auc_roc': auc_ovr,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba
        }

        self.metrics[dataset_name.lower()] = metrics

        return metrics

    def plot_confusion_matrix(self, y_test: np.ndarray, y_pred: np.ndarray,
                             normalize: bool = True, save_path: str = None):
        """
        Genera y visualiza la matriz de confusión.

        Args:
            y_test: Etiquetas verdaderas
            y_pred: Predicciones
            normalize: Si True, normaliza por filas
            save_path: Ruta para guardar la figura
        """
        cm = confusion_matrix(y_test, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Matriz de Confusión Normalizada'
        else:
            fmt = 'd'
            title = 'Matriz de Confusión'

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Proporción' if normalize else 'Cantidad'})

        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Clase Real', fontsize=12)
        plt.xlabel('Clase Predicha', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nMatriz de confusión guardada en: {save_path}")

    def plot_roc_curves(self, y_test: np.ndarray, y_proba: np.ndarray,
                       save_path: str = None):
        """
        Genera curvas ROC para cada clase (one-vs-rest).

        Args:
            y_test: Etiquetas verdaderas
            y_proba: Probabilidades predichas
            save_path: Ruta para guardar la figura
        """
        from sklearn.preprocessing import label_binarize

        # Binarizar etiquetas
        n_classes = len(np.unique(y_test))
        y_test_bin = label_binarize(y_test, classes=range(n_classes))

        plt.figure(figsize=(10, 8))

        # Calcular ROC para cada clase
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            auc = roc_auc_score(y_test_bin[:, i], y_proba[:, i])

            plt.plot(fpr, tpr, linewidth=2, label=f'{self.class_names[i]} (AUC={auc:.3f})')

        # Línea diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Azar')

        plt.xlabel('Tasa de Falsos Positivos', fontsize=12)
        plt.ylabel('Tasa de Verdaderos Positivos', fontsize=12)
        plt.title('Curvas ROC por Clase (One-vs-Rest)', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Curvas ROC guardadas en: {save_path}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones en nuevos datos.

        Args:
            X: Matriz de características

        Returns:
            Array de predicciones
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado.")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula probabilidades de pertenencia a cada clase.

        Args:
            X: Matriz de características

        Returns:
            Matriz de probabilidades (n_samples, n_classes)
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado.")

        return self.model.predict_proba(X)

    def save_model(self, filepath: str):
        """Guarda el modelo entrenado."""
        import joblib
        if self.model is None:
            raise ValueError("No hay modelo entrenado para guardar")

        joblib.dump(self.model, filepath)
        print(f"\nModelo SVM guardado en: {filepath}")

    def load_model(self, filepath: str):
        """Carga un modelo previamente entrenado."""
        import joblib
        self.model = joblib.load(filepath)
        print(f"Modelo SVM cargado desde: {filepath}")
