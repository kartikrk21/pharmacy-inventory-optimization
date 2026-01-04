"""
ML Model Integration Layer
Connects demand forecasting models to the Flask application
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

# Import our model classes
from demand_forecasting import DemandForecaster
from arima_lstm_model import ARIMAModel, LSTMModel
from uncertainty_quantification import UncertaintyQuantifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLModelManager:
    """
    Manages all ML models for the application
    Handles training, prediction, and caching
    """
    
    def __init__(self, csv_path: str = 'historical_prescriptions.csv'):
        self.csv_path = csv_path
        self.models_dir = 'trained_models'
        self.cache = {}
        self.forecaster = None
        self.df = None
        
        # Create models directory
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load prescription data"""
        try:
            if not os.path.exists(self.csv_path):
                logger.error(f"CSV not found: {self.csv_path}")
                return False
            
            self.df = pd.read_csv(self.csv_path)
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            logger.info(f"Loaded {len(self.df)} prescriptions from {self.csv_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def train_models_for_medicine(self, medicine_id: str, force_retrain: bool = False) -> bool:
        """
        Train all models for a specific medicine
        
        Args:
            medicine_id: Medicine to train models for
            force_retrain: Force retraining even if cached models exist
        """
        try:
            model_path = os.path.join(self.models_dir, f'{medicine_id}_models.pkl')
            
            # Check if already trained
            if os.path.exists(model_path) and not force_retrain:
                logger.info(f"Models already trained for {medicine_id}")
                return True
            
            logger.info(f"Training models for {medicine_id}...")
            
            # Initialize forecaster if needed
            if self.forecaster is None:
                self.forecaster = DemandForecaster(forecast_horizon=30)
            
            # Train ensemble models
            result = self.forecaster.train_all_models(self.df, medicine_id)
            
            if result is None:
                logger.warning(f"Insufficient data for {medicine_id}")
                return False
            
            # Save models
            with open(model_path, 'wb') as f:
                pickle.dump(self.forecaster.models[medicine_id], f)
            
            logger.info(f"✓ Models trained and saved for {medicine_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error training models for {medicine_id}: {e}")
            return False
    
    def load_trained_models(self, medicine_id: str) -> Optional[Dict]:
        """Load pre-trained models for a medicine"""
        try:
            model_path = os.path.join(self.models_dir, f'{medicine_id}_models.pkl')
            
            if not os.path.exists(model_path):
                return None
            
            with open(model_path, 'rb') as f:
                models = pickle.load(f)
            
            logger.info(f"✓ Loaded trained models for {medicine_id}")
            return models
        
        except Exception as e:
            logger.error(f"Error loading models for {medicine_id}: {e}")
            return None
    
    def get_forecast(self, medicine_id: str, horizon: int = 30) -> Optional[Dict]:
        """
        Get forecast for a medicine
        Uses cached models or trains new ones if needed
        
        Returns:
            Dict with forecast, uncertainty bounds, etc.
        """
        try:
            # Check cache first
            cache_key = f"{medicine_id}_{horizon}"
            if cache_key in self.cache:
                logger.info(f"Using cached forecast for {medicine_id}")
                return self.cache[cache_key]
            
            # Try to load existing models
            models = self.load_trained_models(medicine_id)
            
            # If no models, train them
            if models is None:
                logger.info(f"No trained models found, training for {medicine_id}...")
                success = self.train_models_for_medicine(medicine_id)
                if not success:
                    return self._generate_fallback_forecast(medicine_id, horizon)
                models = self.load_trained_models(medicine_id)
            
            # Extract ensemble forecast
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
            else:
                # Fallback to individual models
                result = self._extract_best_available_forecast(models, medicine_id, horizon)
            
            # Cache result
            self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting forecast for {medicine_id}: {e}")
            return self._generate_fallback_forecast(medicine_id, horizon)
    
    def _extract_best_available_forecast(self, models: Dict, medicine_id: str, horizon: int) -> Dict:
        """Extract forecast from best available model"""
        try:
            # Try ARIMA
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
            
            # Try Prophet
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
            
            # Try LSTM
            if models.get('lstm') and models['lstm'] is not None:
                lstm = models['lstm']
                forecast = lstm['forecast'][:horizon]
                # Estimate uncertainty for LSTM
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
            
            # No models available
            return self._generate_fallback_forecast(medicine_id, horizon)
            
        except Exception as e:
            logger.error(f"Error extracting forecast: {e}")
            return self._generate_fallback_forecast(medicine_id, horizon)
    
    def _generate_fallback_forecast(self, medicine_id: str, horizon: int) -> Dict:
        """Generate simple forecast based on historical patterns"""
        try:
            if self.df is None:
                raise Exception("No data loaded")
            
            # Get historical data for this medicine
            med_data = self.df[self.df['medicine_id'] == medicine_id].copy()
            
            if len(med_data) == 0:
                # No historical data, use simple baseline
                base_demand = 10.0
                std_demand = 2.0
            else:
                # Calculate statistics
                med_data['date'] = med_data['timestamp'].dt.date
                daily_demand = med_data.groupby('date')['quantity'].sum()
                
                base_demand = daily_demand.mean()
                std_demand = daily_demand.std()
            
            # Generate simple forecast with seasonality
            forecast = []
            lower_bound = []
            upper_bound = []
            uncertainty = []
            
            for i in range(horizon):
                # Add weekly seasonality
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
            logger.error(f"Error in fallback forecast: {e}")
            # Return simple constant forecast
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
        """Train models for top N medicines by demand"""
        try:
            if self.df is None:
                logger.error("No data loaded")
                return
            
            # Get top medicines
            top_medicines = (
                self.df.groupby('medicine_id')['quantity']
                .sum()
                .nlargest(top_n)
                .index.tolist()
            )
            
            logger.info(f"Training models for top {len(top_medicines)} medicines...")
            
            trained = 0
            failed = 0
            
            for i, med_id in enumerate(top_medicines, 1):
                logger.info(f"[{i}/{len(top_medicines)}] Training {med_id}...")
                
                if self.train_models_for_medicine(med_id):
                    trained += 1
                else:
                    failed += 1
            
            logger.info(f"✓ Training complete: {trained} succeeded, {failed} failed")
            
        except Exception as e:
            logger.error(f"Error training top medicines: {e}")
    
    def get_model_info(self, medicine_id: str) -> Dict:
        """Get information about trained models for a medicine"""
        try:
            models = self.load_trained_models(medicine_id)
            
            if models is None:
                return {
                    'trained': False,
                    'medicine_id': medicine_id
                }
            
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
            logger.error(f"Error getting model info: {e}")
            return {'trained': False, 'medicine_id': medicine_id, 'error': str(e)}
    
    def clear_cache(self):
        """Clear forecast cache"""
        self.cache = {}
        logger.info("Cache cleared")

# Global instance
_ml_manager = None

def get_ml_manager() -> MLModelManager:
    """Get singleton ML manager instance"""
    global _ml_manager
    if _ml_manager is None:
        _ml_manager = MLModelManager()
    return _ml_manager

def initialize_ml_models(csv_path: str = 'historical_prescriptions.csv', train_top_n: int = 20):
    """
    Initialize ML models on startup
    
    Args:
        csv_path: Path to historical data
        train_top_n: Number of top medicines to pre-train
    """
    logger.info("Initializing ML models...")
    
    try:
        manager = get_ml_manager()
        
        # Check if data loaded successfully
        if manager.df is None:
            logger.error("Failed to load data")
            return False
        
        # Train models for top medicines
        manager.train_top_medicines(top_n=train_top_n)
        
        logger.info("✓ ML models initialized")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing ML models: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Initialize
    manager = MLModelManager()
    
    # Train top 10 medicines
    manager.train_top_medicines(top_n=10)
    
    # Get forecast
    top_med = manager.df['medicine_id'].value_counts().index[0]
    forecast = manager.get_forecast(top_med)
    
    print(f"\nForecast for {top_med}:")
    print(f"  Model: {forecast['model']}")
    print(f"  Mean demand (next 30 days): {np.mean(forecast['forecast']):.2f}")
    print(f"  Confidence: {forecast['confidence']*100}%")