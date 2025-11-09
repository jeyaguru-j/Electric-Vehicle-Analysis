"""
EV Range Prediction Package
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

from .data_preprocessing import EVDataPreprocessor
from .model_training import EVModelTrainer
from .predict import EVRangePredictor

__all__ = ['EVDataPreprocessor', 'EVModelTrainer', 'EVRangePredictor']