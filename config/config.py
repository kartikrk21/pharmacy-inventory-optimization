# PYTHON FILE - Replace content:
import os
from datetime import timedelta

class Config:
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'pharmacy-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', '1') == '1'
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///pharmacy.db')
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Kafka Configuration
    KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:29093')
    KAFKA_TOPIC_PRESCRIPTIONS = 'prescriptions'
    KAFKA_TOPIC_INVENTORY = 'inventory_updates'
    KAFKA_TOPIC_ORDERS = 'orders'
    KAFKA_CONSUMER_GROUP = 'pharmacy_consumer_group'
    
    # Stream Processing Configuration
    WINDOW_SIZE = 300
    SLIDE_INTERVAL = 60
    BATCH_SIZE = 100
    NUM_MEDICINES = 500
    # ML Model Configuration
    MODEL_PATH = 'ml_models/trained_models/'
    FORECAST_HORIZON = 30
    CONFIDENCE_LEVEL = 0.95
    
    # Optimization Configuration
    OPTIMIZATION_METHOD = 'ROBUST_OR'
    LEAD_TIME = 7
    SERVICE_LEVEL = 0.95
    STORAGE_CAPACITY = 10000
    
    # Performance Targets
    TARGET_FILL_RATE = 0.9995
    TARGET_WASTE = 0.10
    TARGET_COST_REDUCTION = 0.30
    MAX_LATENCY_MS = 305
    MIN_THROUGHPUT = 10000
    MIN_AVAILABILITY = 0.9995
    
    # SocketIO
    SOCKETIO_ASYNC_MODE = 'threading'

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

class TestingConfig(Config):
    TESTING = True
    DATABASE_URL = 'sqlite:///test_pharmacy.db'

config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}