"""
Kafka streaming module for real-time data processing
Includes producer, consumer, and advanced stream processing
"""
from .producer import PrescriptionKafkaProducer
from .consumer import PrescriptionStreamProcessor

try:
    from .stream_processor import (
        WindowedStreamProcessor,
        FeatureExtractor,
        CoprescriptionGraphBuilder
    )
    
    __all__ = [
        'PrescriptionKafkaProducer',
        'InventoryUpdateProducer',
        'PrescriptionStreamProcessor',
        'WindowedStreamProcessor',
        'FeatureExtractor',
        'CoprescriptionGraphBuilder'
    ]
except ImportError:
    __all__ = [
        'PrescriptionKafkaProducer',
        'InventoryUpdateProducer',
        'PrescriptionStreamProcessor'
    ]
