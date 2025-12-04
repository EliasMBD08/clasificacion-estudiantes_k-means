"""
Módulo de preprocesamiento de datos para clasificación de estudiantes.
Implementa limpieza, transformación y normalización de datos académicos.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Preprocesador de datos académicos con pipeline completo de limpieza,
    transformación y normalización.
    """

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.numeric_features = ['creditos_aprobados', 'ranking_facultad']
        self.categorical_features = ['facultad', 'pais_nacimiento', 'tipo_documento']

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia datos: elimina duplicados, maneja valores faltantes y outliers.

        Args:
            df: DataFrame con datos crudos

        Returns:
            DataFrame limpio
        """
        print(f"Registros iniciales: {len(df)}")

        # Eliminar duplicados basados en documento de identidad
        df = df.drop_duplicates(subset=['numero_documento'], keep='first')
        print(f"Después de eliminar duplicados: {len(df)}")

        # Eliminar registros sin variables académicas críticas
        df = df.dropna(subset=['creditos_aprobados', 'ranking_facultad'])
        print(f"Después de eliminar valores faltantes críticos: {len(df)}")

        # Rellenar valores faltantes en variables categóricas
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].fillna('DESCONOCIDO')

        # Detección de outliers usando IQR por facultad
        df = self._remove_outliers_by_group(df, 'creditos_aprobados', 'facultad')

        print(f"Registros finales después de limpieza: {len(df)}")
        return df

    def _remove_outliers_by_group(self, df: pd.DataFrame, column: str,
                                   group_by: str, k: float = 3.0) -> pd.DataFrame:
        """
        Elimina outliers usando método IQR agrupado por categoría.

        Args:
            df: DataFrame
            column: Columna numérica para detectar outliers
            group_by: Columna para agrupar
            k: Multiplicador del IQR (default: 3.0 para outliers extremos)

        Returns:
            DataFrame sin outliers
        """
        def filter_outliers(group):
            Q1 = group[column].quantile(0.25)
            Q3 = group[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - k * IQR
            upper_bound = Q3 + k * IQR
            return group[(group[column] >= lower_bound) & (group[column] <= upper_bound)]

        if group_by in df.columns:
            df_clean = df.groupby(group_by, group_keys=False).apply(filter_outliers)
            outliers_removed = len(df) - len(df_clean)
            print(f"Outliers eliminados en {column}: {outliers_removed}")
            return df_clean.reset_index(drop=True)

        return df

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Codifica variables categóricas usando One-Hot Encoding y Label Encoding.

        Args:
            df: DataFrame con variables categóricas

        Returns:
            DataFrame con variables codificadas
        """
        df = df.copy()

        # One-Hot Encoding para variables nominales
        for col in ['facultad', 'pais_nacimiento', 'tipo_documento']:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df.drop(col, axis=1, inplace=True)

        # Label Encoding para variable binaria tercio_superior
        if 'tercio_superior' in df.columns:
            df['tercio_superior'] = df['tercio_superior'].map({'SI': 1, 'NO': 0})
            df['tercio_superior'] = df['tercio_superior'].fillna(0).astype(int)

        return df

    def normalize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normaliza variables numéricas usando Min-Max Scaling [0, 1].

        Args:
            df: DataFrame con variables numéricas
            fit: Si True, ajusta el scaler. Si False, usa scaler existente.

        Returns:
            DataFrame con variables normalizadas
        """
        df = df.copy()

        numeric_cols = [col for col in self.numeric_features if col in df.columns]

        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        return df

    def select_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Selecciona características relevantes para el modelado.

        Args:
            df: DataFrame preprocesado

        Returns:
            Tupla (matriz de características, nombres de características)
        """
        # Excluir columnas identificatorias
        exclude_cols = ['apellidos_nombres', 'codigo_alumno', 'numero_documento',
                       'correo_institucional']

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_names = feature_cols

        X = df[feature_cols].values

        print(f"\nCaracterísticas seleccionadas: {len(feature_cols)}")
        print(f"Dimensiones de la matriz: {X.shape}")

        return X, feature_cols

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Pipeline completo de preprocesamiento (fit + transform).

        Args:
            df: DataFrame crudo

        Returns:
            Tupla (matriz de características, nombres de características)
        """
        print("="*60)
        print("INICIANDO PREPROCESAMIENTO DE DATOS")
        print("="*60)

        # 1. Limpieza
        df_clean = self.clean_data(df)

        # 2. Codificación
        df_encoded = self.encode_categorical(df_clean)

        # 3. Normalización
        df_normalized = self.normalize_features(df_encoded, fit=True)

        # 4. Selección de características
        X, feature_names = self.select_features(df_normalized)

        print("="*60)
        print("PREPROCESAMIENTO COMPLETADO")
        print("="*60)

        return X, feature_names

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Aplica transformaciones a nuevos datos usando parámetros ajustados.

        Args:
            df: DataFrame nuevo

        Returns:
            Matriz de características transformadas
        """
        df_clean = self.clean_data(df)
        df_encoded = self.encode_categorical(df_clean)
        df_normalized = self.normalize_features(df_encoded, fit=False)
        X, _ = self.select_features(df_normalized)

        return X
