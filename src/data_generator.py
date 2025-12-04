"""
Generador de datos sintéticos para simulación del dataset de estudiantes UNMSM.
"""

import numpy as np
import pandas as pd
from typing import Tuple


class StudentDataGenerator:
    """
    Generador de datos sintéticos realistas para estudiantes universitarios.
    """

    def __init__(self, n_students: int = 37257, random_state: int = 42):
        self.n_students = n_students
        self.random_state = random_state
        np.random.seed(random_state)

        # Datos de referencia
        self.facultades = [
            'Medicina', 'Ingeniería de Sistemas', 'Derecho', 'Economía',
            'Administración', 'Contabilidad', 'Psicología', 'Educación',
            'Ciencias Biológicas', 'Química', 'Física', 'Matemática',
            'Ciencias Sociales', 'Letras', 'Ingeniería Industrial',
            'Ingeniería Eléctrica', 'Medicina Veterinaria', 'Odontología',
            'Farmacia', 'Ingeniería Geológica'
        ]

        self.paises = ['Perú', 'Venezuela', 'Colombia', 'Ecuador', 'Bolivia', 'Chile']
        self.tipo_docs = ['DNI', 'CE', 'Pasaporte']

    def generate_dataset(self) -> pd.DataFrame:
        """
        Genera dataset completo de estudiantes con características realistas.

        Returns:
            DataFrame con datos sintéticos
        """
        print(f"\nGenerando dataset sintético con {self.n_students:,} estudiantes...")

        # 1. Identificación
        nombres = [f"Estudiante {i+1:05d}" for i in range(self.n_students)]
        codigos = [f"20{np.random.randint(15, 24):02d}{i:05d}" for i in range(self.n_students)]
        tipo_doc = np.random.choice(self.tipo_docs, self.n_students, p=[0.90, 0.07, 0.03])
        num_doc = [f"{np.random.randint(10000000, 99999999)}" for _ in range(self.n_students)]

        # 2. Variables académicas - crear 3 perfiles distintos

        # Perfil 1: Alto Rendimiento (35%)
        n_alto = int(self.n_students * 0.35)
        creditos_alto = np.random.normal(165, 42, n_alto).clip(120, 220)
        ranking_alto = np.random.normal(450, 280, n_alto).clip(1, 800)
        tercio_alto = np.random.choice(['SI', 'NO'], n_alto, p=[0.887, 0.113])

        # Perfil 2: Rendimiento Medio (40%)
        n_medio = int(self.n_students * 0.40)
        creditos_medio = np.random.normal(85, 31, n_medio).clip(50, 120)
        ranking_medio = np.random.normal(1200, 420, n_medio).clip(800, 1500)
        tercio_medio = np.random.choice(['SI', 'NO'], n_medio, p=[0.428, 0.572])

        # Perfil 3: Riesgo Alto (25%)
        n_riesgo = self.n_students - n_alto - n_medio
        creditos_riesgo = np.random.normal(48, 22, n_riesgo).clip(10, 70)
        ranking_riesgo = np.random.normal(2100, 580, n_riesgo).clip(1500, 3000)
        tercio_riesgo = np.random.choice(['SI', 'NO'], n_riesgo, p=[0.052, 0.948])

        # Combinar y mezclar
        creditos = np.concatenate([creditos_alto, creditos_medio, creditos_riesgo])
        ranking = np.concatenate([ranking_alto, ranking_medio, ranking_riesgo])
        tercio = np.concatenate([tercio_alto, tercio_medio, tercio_riesgo])

        # Shuffle para mezclar los perfiles
        indices = np.arange(self.n_students)
        np.random.shuffle(indices)
        creditos = creditos[indices]
        ranking = ranking[indices]
        tercio = tercio[indices]

        # 3. Variables de contexto
        facultad = np.random.choice(self.facultades, self.n_students)
        pais = np.random.choice(self.paises, self.n_students, p=[0.91, 0.04, 0.02, 0.015, 0.01, 0.005])

        # 4. Correo institucional (algunos faltantes)
        correos = []
        for i in range(self.n_students):
            if np.random.random() > 0.05:  # 95% tienen correo
                correos.append(f"estudiante{i+1:05d}@unmsm.edu.pe")
            else:
                correos.append(np.nan)

        # Crear DataFrame
        df = pd.DataFrame({
            'apellidos_nombres': nombres,
            'codigo_alumno': codigos,
            'tipo_documento': tipo_doc,
            'numero_documento': num_doc,
            'creditos_aprobados': creditos.astype(int),
            'ranking_facultad': ranking.astype(int),
            'tercio_superior': tercio,
            'facultad': facultad,
            'pais_nacimiento': pais,
            'correo_institucional': correos
        })

        # Introducir algunos valores faltantes realistas
        # 2% sin código de alumno
        missing_codigo = np.random.choice(df.index, size=int(self.n_students * 0.02), replace=False)
        df.loc[missing_codigo, 'codigo_alumno'] = np.nan

        print(f"Dataset generado exitosamente:")
        print(f"  - Total estudiantes: {len(df):,}")
        print(f"  - Facultades: {df['facultad'].nunique()}")
        print(f"  - Créditos rango: [{df['creditos_aprobados'].min()}, {df['creditos_aprobados'].max()}]")
        print(f"  - Ranking rango: [{df['ranking_facultad'].min()}, {df['ranking_facultad'].max()}]")
        print(f"  - Tercio superior: {(df['tercio_superior']=='SI').sum()} estudiantes ({(df['tercio_superior']=='SI').sum()/len(df)*100:.1f}%)")

        return df

    def save_dataset(self, df: pd.DataFrame, filepath: str):
        """
        Guarda el dataset generado en CSV.

        Args:
            df: DataFrame a guardar
            filepath: Ruta del archivo
        """
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\nDataset guardado en: {filepath}")

    def load_or_generate(self, filepath: str) -> pd.DataFrame:
        """
        Carga dataset existente o genera uno nuevo si no existe.

        Args:
            filepath: Ruta del archivo

        Returns:
            DataFrame con datos
        """
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
            print(f"Dataset cargado desde: {filepath}")
            print(f"Total de registros: {len(df):,}")
            return df
        except FileNotFoundError:
            print(f"Archivo no encontrado. Generando nuevo dataset...")
            df = self.generate_dataset()
            self.save_dataset(df, filepath)
            return df
