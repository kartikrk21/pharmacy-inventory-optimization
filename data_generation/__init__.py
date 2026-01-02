"""
Data generation module for synthetic prescription data
Includes privacy evaluation and CTGAN support
"""
from .prescription_generator import PrescriptionDataGenerator

try:
    from .synthetic_data import SyntheticDataGenerator, CTGANWrapper
    from .privacy_evaluator import PrivacyEvaluator
    
    __all__ = [
        'PrescriptionDataGenerator',
        'SyntheticDataGenerator',
        'CTGANWrapper',
        'PrivacyEvaluator'
    ]
except ImportError:
    # Optional dependencies not installed
    __all__ = ['PrescriptionDataGenerator']