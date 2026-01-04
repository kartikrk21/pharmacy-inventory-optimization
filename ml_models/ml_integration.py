"""
ML Model Integration Layer - FIXED VERSION
Handles all ML operations with proper error handling
"""
import pandas as pd
import numpy as np
import pickle
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import models, handle if not available
try:
    from ml_models.demand_forecasting import DemandForecaster
    FORECASTER_AVAILABLE = True
except ImportError:
    logger.warning("DemandForecaster not available")
    FORECASTER_AVAILABLE = False

class MLModelManager:
    """Manages all ML models with fallback support"""
    
    def __init__(self, csv_path: str = 'historical_prescriptions.csv'):
        self.csv_path = csv_path
        self.models_dir = 'trained_models'
        self.cache = {}
        self.forecaster = None
        self.df = None
        
        os.makedirs(self.models_dir, exist_ok=True)
        self.load_data()
    
    def load_data(self):
        """Load prescription data"""
        try:
            if not os.path.exists(self.csv_path):
                logger.error(f"CSV not found: {self.csv_path}")
                return False
            
            self.df = pd.read_csv(self.csv_path)
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            logger.info(f"Loaded {len(self.df)} prescriptions")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def train_models_for_medicine(self, medicine_id: str, force_retrain: bool = False) -> bool:
        """Train models for specific medicine"""
        try:
            if not FORECASTER_AVAILABLE:
                logger.warning("Forecaster not available, skipping training")
                return False
            
            model_path = os.path.join(self.models_dir, f'{medicine_id}_models.pkl')
            
            if os.path.exists(model_path) and not force_retrain:
                logger.info(f"Models exist for {medicine_id}")
                return True
            
            logger.info(f"Training {medicine_id}...")
            
            if self.forecaster is None:
                self.forecaster = DemandForecaster(forecast_horizon=30)
            
            try:
                result = self.forecaster.train_all_models(self.df, medicine_id)
            except Exception as train_error:
                logger.error(f"Training failed for {medicine_id}: {train_error}")
                return False
            
            if result is None:
                logger.warning(f"Training returned None for {medicine_id}")
                return False
            
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(self.forecaster.models[medicine_id], f)
                logger.info(f"✅ Models saved for {medicine_id}")
            except Exception as save_error:
                logger.error(f"Save failed for {medicine_id}: {save_error}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error training {medicine_id}: {e}")
            return False
    
    def load_trained_models(self, medicine_id: str) -> Optional[Dict]:
        """Load pre-trained models"""
        try:
            model_path = os.path.join(self.models_dir, f'{medicine_id}_models.pkl')
            
            if not os.path.exists(model_path):
                return None
            
            with open(model_path, 'rb') as f:
                models = pickle.load(f)
            
            logger.info(f"✅ Loaded models for {medicine_id}")
            return models
        
        except Exception as e:
            logger.error(f"Load error for {medicine_id}: {e}")
            return None
    
    def get_forecast(self, medicine_id: str, horizon: int = 30) -> Optional[Dict]:
        """Get forecast with fallback"""
        try:
            cache_key = f"{medicine_id}_{horizon}"
            if cache_key in self.cache:
                logger.info(f"Using cached forecast for {medicine_id}")
                return self.cache[cache_key]
            
            # Try trained models
            models = self.load_trained_models(medicine_id)
            
            if models is None and FORECASTER_AVAILABLE:
                logger.info(f"Training models for {medicine_id}...")
                success = self.train_models_for_medicine(medicine_id)
                if success:
                    models = self.load_trained_models(medicine_id)
            
            # Extract forecast
            if models and 'ensemble' in models and models['ensemble'] is not None:
                ensemble = models['ensemble']
                result = {
                    'medicine_id': medicine_id,
                    'horizon_days': horizon,
                    'forecast': ensemble['forecast'][:horizon].tolist(),
                    'lower_bound': ensemble['lower_bound'][:horizon].tolist(),
                    'upper_bound': ensemble['upper_bound'][:horizon].tolist(),
                    'uncertainty': ensemble['uncertainty'][:horizon].tolist(),
                    'model': 'ensemble_arima_lstm_prophet',
                    'confidence': 0.95
                }
            elif models:
                result = self._extract_best_forecast(models, medicine_id, horizon)
            else:
                result = self._generate_fallback_forecast(medicine_id, horizon)
            
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Forecast error for {medicine_id}: {e}")
            return self._generate_fallback_forecast(medicine_id, horizon)
    
    def _extract_best_forecast(self, models: Dict, medicine_id: str, horizon: int) -> Dict:
        """Extract from best available model"""
        try:
            if models.get('arima') and models['arima'] is not None:
                arima = models['arima']
                return {
                    'medicine_id': medicine_id,
                    'horizon_days': horizon,
                    'forecast': arima['forecast_mean'][:horizon].tolist(),
                    'lower_bound': arima['forecast_lower'][:horizon].tolist(),
                    'upper_bound': arima['forecast_upper'][:horizon].tolist(),
                    'uncertainty': (arima['forecast_upper'][:horizon] - arima['forecast_lower'][:horizon]).tolist(),
                    'model': 'arima',
                    'confidence': 0.95
                }
            
            if models.get('prophet') and models['prophet'] is not None:
                prophet = models['prophet']
                return {
                    'medicine_id': medicine_id,
                    'horizon_days': horizon,
                    'forecast': prophet['forecast_mean'][:horizon].tolist(),
                    'lower_bound': prophet['forecast_lower'][:horizon].tolist(),
                    'upper_bound': prophet['forecast_upper'][:horizon].tolist(),
                    'uncertainty': (prophet['forecast_upper'][:horizon] - prophet['forecast_lower'][:horizon]).tolist(),
                    'model': 'prophet',
                    'confidence': 0.95
                }
            
            if models.get('lstm') and models['lstm'] is not None:
                lstm = models['lstm']
                forecast = lstm['forecast'][:horizon]
                uncertainty = np.std(forecast) * np.ones(len(forecast))
                return {
                    'medicine_id': medicine_id,
                    'horizon_days': horizon,
                    'forecast': forecast.tolist(),
                    'lower_bound': (forecast - 1.96 * uncertainty).tolist(),
                    'upper_bound': (forecast + 1.96 * uncertainty).tolist(),
                    'uncertainty': uncertainty.tolist(),
                    'model': 'lstm',
                    'confidence': 0.95
                }
            
            return self._generate_fallback_forecast(medicine_id, horizon)
            
        except Exception as e:
            logger.error(f"Extract error: {e}")
            return self._generate_fallback_forecast(medicine_id, horizon)
    
    def _generate_fallback_forecast(self, medicine_id: str, horizon: int) -> Dict:
        """Simple fallback forecast"""
        try:
            if self.df is None:
                base_demand = 10.0
                std_demand = 2.0
            else:
                med_data = self.df[self.df['medicine_id'] == medicine_id].copy()
                
                if len(med_data) == 0:
                    base_demand = 10.0
                    std_demand = 2.0
                else:
                    med_data['date'] = med_data['timestamp'].dt.date
                    daily_demand = med_data.groupby('date')['quantity'].sum()
                    base_demand = daily_demand.mean()
                    std_demand = daily_demand.std()
            
            forecast = []
            lower_bound = []
            upper_bound = []
            uncertainty = []
            
            for i in range(horizon):
                seasonal = 0.2 * base_demand * np.sin(2 * np.pi * i / 7)
                value = base_demand + seasonal
                unc = std_demand
                
                forecast.append(float(max(0, value)))
                uncertainty.append(float(unc))
                lower_bound.append(float(max(0, value - 1.96 * unc)))
                upper_bound.append(float(value + 1.96 * unc))
            
            return {
                'medicine_id': medicine_id,
                'horizon_days': horizon,
                'forecast': forecast,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'uncertainty': uncertainty,
                'model': 'historical_baseline',
                'confidence': 0.80
            }
            
        except Exception as e:
            logger.error(f"Fallback error: {e}")
            forecast = [10.0] * horizon
            return {
                'medicine_id': medicine_id,
                'horizon_days': horizon,
                'forecast': forecast,
                'lower_bound': [5.0] * horizon,
                'upper_bound': [15.0] * horizon,
                'uncertainty': [2.5] * horizon,
                'model': 'simple_baseline',
                'confidence': 0.50
            }
    
    def train_top_medicines(self, top_n: int = 50):
        """Train top N medicines"""
        try:
            if self.df is None:
                logger.error("No data loaded")
                return
            
            if not FORECASTER_AVAILABLE:
                logger.warning("Forecaster not available")
                return
            
            top_medicines = (
                self.df.groupby('medicine_id')['quantity']
                .sum()
                .nlargest(top_n)
                .index.tolist()
            )
            
            logger.info(f"Training top {len(top_medicines)} medicines...")
            
            trained = 0
            failed = 0
            
            for i, med_id in enumerate(top_medicines, 1):
                logger.info(f"[{i}/{len(top_medicines)}] {med_id}...")
                
                try:
                    if self.train_models_for_medicine(med_id):
                        trained += 1
                    else:
                        failed += 1
                except KeyboardInterrupt:
                    logger.warning("Training interrupted")
                    break
                except Exception as e:
                    logger.error(f"Error training {med_id}: {e}")
                    failed += 1
            
            logger.info(f"✅ Training complete: {trained} ok, {failed} failed")
            
        except Exception as e:
            logger.error(f"Train top error: {e}")
    
    def get_model_info(self, medicine_id: str) -> Dict:
        """Get model info"""
        try:
            models = self.load_trained_models(medicine_id)
            
            if models is None:
                return {'trained': False, 'medicine_id': medicine_id}
            
            info = {
                'trained': True,
                'medicine_id': medicine_id,
                'models_available': []
            }
            
            if models.get('arima'):
                info['models_available'].append('ARIMA')
            if models.get('lstm'):
                info['models_available'].append('LSTM')
            if models.get('prophet'):
                info['models_available'].append('Prophet')
            if models.get('ensemble'):
                info['models_available'].append('Ensemble')
            
            return info
            
        except Exception as e:
            logger.error(f"Model info error: {e}")
            return {'trained': False, 'medicine_id': medicine_id, 'error': str(e)}
    
    def clear_cache(self):
        """Clear cache"""
        self.cache = {}
        logger.info("Cache cleared")

# Global instance
_ml_manager = None

def get_ml_manager() -> MLModelManager:
    """Get singleton instance"""
    global _ml_manager
    if _ml_manager is None:
        _ml_manager = MLModelManager()
    return _ml_manager

def initialize_ml_models(csv_path: str = 'historical_prescriptions.csv', train_top_n: int = 20):
    """Initialize ML models"""
    logger.info("Initializing ML models...")
    
    try:
        manager = get_ml_manager()
        
        if manager.df is None:
            logger.error("Failed to load data")
            return False
        
        manager.train_top_medicines(top_n=train_top_n)
        
        logger.info("✅ ML models initialized")
        return True
        
    except Exception as e:
        logger.error(f"ML init error: {e}")
        return False

if __name__ == "__main__":
    manager = MLModelManager()
    manager.train_top_medicines(top_n=10)
    
    if manager.df is not None:
        top_med = manager.df['medicine_id'].value_counts().index[0]
        forecast = manager.get_forecast(top_med)
        print(f"\nForecast for {top_med}:")
        print(f"  Model: {forecast['model']}")
        print(f"  Mean: {np.mean(forecast['forecast']):.2f}")