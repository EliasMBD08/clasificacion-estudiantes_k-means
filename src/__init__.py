"""
Paquete de clasificación de estudiantes según desempeño académico.
Sistema híbrido K-Means + SVM para UNMSM.
"""

__version__ = "1.0.0"
__author__ = "UNMSM - Sistema de Clasificación de Estudiantes"

from .preprocessing import DataPreprocessor
from .clustering import StudentClustering
from .classification import SVMClassifier
from .visualization import Visualizer
from .data_generator import StudentDataGenerator

__all__ = [
    'DataPreprocessor',
    'StudentClustering',
    'SVMClassifier',
    'Visualizer',
    'StudentDataGenerator'
]
