import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import logging
from typing import Tuple, Dict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemandForecaster:
    """
    Ensemble demand forecasting using ARIMA/LSTM/Prophet
    Implements uncertainty quantification with confidence intervals
    """
    
    def __init__(self, forecast_horizon: int = 30):
        self.forecast_horizon = forecast_horizon
        self.models = {}
        self.scalers = {}
        self.history = {}
        
    def prepare_time_series(self, prescriptions_df: pd.DataFrame, 
                          medicine_id: str) -> pd.DataFrame:
        """Prepare time series data for a specific medicine"""
        try:
            # Filter for specific medicine
            med_data = prescriptions_df[
                prescriptions_df['medicine_id'] == medicine_id
            ].copy()
            
            if len(med_data) == 0:
                logger.warning(f"No data found for {medicine_id}")
                return None
            
            # Convert timestamp to datetime
            med_data['timestamp'] = pd.to_datetime(med_data['timestamp'])
            
            # Aggregate by day
            daily_demand = med_data.groupby(
                med_data['timestamp'].dt.date
            ).agg({
                'quantity': 'sum'
            }).reset_index()
            
            daily_demand.columns = ['date', 'demand']
            daily_demand['date'] = pd.to_datetime(daily_demand['date'])
            daily_demand = daily_demand.sort_values('date')
            
            # Fill missing dates with 0
            date_range = pd.date_range(
                start=daily_demand['date'].min(),
                end=daily_demand['date'].max(),
                freq='D'
            )
            
            daily_demand = daily_demand.set_index('date').reindex(date_range).reset_index()
            daily_demand.columns = ['date', 'demand']
            daily_demand['demand'] = daily_demand['demand'].fillna(0)
            
            # Add time features
            daily_demand['day_of_week'] = daily_demand['date'].dt.dayofweek
            daily_demand['month'] = daily_demand['date'].dt.month
            daily_demand['quarter'] = daily_demand['date'].dt.quarter
            daily_demand['is_weekend'] = daily_demand['day_of_week'].isin([5, 6]).astype(int)
            
            return daily_demand
        
        except Exception as e:
            logger.error(f"Error preparing time series for {medicine_id}: {e}")
            return None
    
    def train_arima_model(self, time_series: pd.DataFrame, 
                          medicine_id: str) -> Dict:
        """Train ARIMA model with seasonal components"""
        try:
            # Prepare data - FIXED: Convert to numpy array properly
            y = time_series['demand'].values
            
            # Check if we have enough data
            if len(y) < 14:
                logger.warning(f"Insufficient data for ARIMA: {len(y)} days")
                return None
            
            # Remove any NaN or infinite values
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            
            # If all zeros or constant, return None
            if np.std(y) < 0.1:
                logger.warning(f"Constant or near-zero demand for {medicine_id}")
                return None
            
            # Try simple ARIMA first (no seasonality for short series)
            if len(y) < 28:
                order = (1, 0, 1)
                seasonal_order = None
            else:
                order = (1, 1, 1)
                seasonal_order = (1, 0, 1, 7)  # Weekly seasonality
            
            # Fit SARIMA model
            if seasonal_order:
                model = SARIMAX(
                    y,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                from statsmodels.tsa.arima.model import ARIMA as SimpleARIMA
                model = SimpleARIMA(y, order=order)
            
            results = model.fit(disp=False, maxiter=100)
            
            # Generate forecast
            forecast_steps = min(self.forecast_horizon, len(y) // 2)  # Don't forecast too far
            forecast = results.get_forecast(steps=forecast_steps)
            forecast_mean = forecast.predicted_mean
            forecast_ci = forecast.conf_int()
            
            # Ensure positive forecasts
            forecast_mean = np.maximum(forecast_mean, 0)
            
            logger.info(f"✓ ARIMA model trained for {medicine_id}")
            
            return {
                'model': results,
                'forecast_mean': forecast_mean,
                'forecast_lower': np.maximum(forecast_ci.iloc[:, 0].values, 0),
                'forecast_upper': np.maximum(forecast_ci.iloc[:, 1].values, 0),
                'aic': results.aic,
                'bic': results.bic
            }
            
        except Exception as e:
            logger.error(f"ARIMA training failed for {medicine_id}: {e}")
            return None
    
    def build_lstm_model(self, input_shape: Tuple) -> keras.Model:
        """Build LSTM neural network for time series forecasting"""
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_sequences(self, data: np.ndarray, 
                        lookback: int = 14) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def train_lstm_model(self, time_series: pd.DataFrame, 
                        medicine_id: str,
                        lookback: int = 14) -> Dict:
        """Train LSTM model"""
        try:
            # Prepare data
            demand_data = time_series['demand'].values.reshape(-1, 1)
            
            # Need at least 3x lookback for meaningful training
            if len(demand_data) < lookback * 3:
                logger.warning(f"Insufficient data for LSTM: {len(demand_data)} days")
                return None
            
            # Check for constant values
            if np.std(demand_data) < 0.1:
                logger.warning(f"Constant demand for {medicine_id}, skipping LSTM")
                return None
            
            # Scale data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(demand_data)
            
            # Create sequences
            X, y = self.create_sequences(scaled_data, lookback)
            
            if len(X) < 50:
                logger.warning(f"Insufficient sequences for LSTM: {len(X)} samples")
                return None
            
            # Split train/val
            split = int(0.8 * len(X))
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
            
            # Build and train model
            model = self.build_lstm_model((lookback, 1))
            
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=16,
                callbacks=[early_stop],
                verbose=0
            )
            
            # Generate forecast
            last_sequence = scaled_data[-lookback:]
            forecasts = []
            current_seq = last_sequence.copy()
            
            for _ in range(self.forecast_horizon):
                pred = model.predict(current_seq.reshape(1, lookback, 1), verbose=0)
                forecasts.append(pred[0, 0])
                current_seq = np.append(current_seq[1:], pred)
            
            # Inverse transform
            forecasts = scaler.inverse_transform(
                np.array(forecasts).reshape(-1, 1)
            ).flatten()
            
            # Ensure positive forecasts
            forecasts = np.maximum(forecasts, 0)
            
            logger.info(f"✓ LSTM model trained for {medicine_id}")
            
            # Store scaler
            self.scalers[medicine_id] = scaler
            
            return {
                'model': model,
                'forecast': forecasts,
                'history': history.history,
                'val_loss': history.history['val_loss'][-1]
            }
            
        except Exception as e:
            logger.error(f"LSTM training failed for {medicine_id}: {e}")
            return None
    
    def train_prophet_model(self, time_series: pd.DataFrame,
                           medicine_id: str) -> Dict:
        """Train Prophet model for forecasting"""
        try:
            # Prepare data for Prophet
            prophet_df = time_series[['date', 'demand']].copy()
            prophet_df.columns = ['ds', 'y']
            
            # Need at least 2 non-zero values
            if (prophet_df['y'] > 0).sum() < 2:
                logger.warning(f"Insufficient non-zero data for Prophet: {medicine_id}")
                return None
            
            # Initialize model with simpler settings for small datasets
            model = Prophet(
                yearly_seasonality=False if len(prophet_df) < 365 else True,
                weekly_seasonality=True if len(prophet_df) >= 14 else False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                interval_width=0.95,
                uncertainty_samples=0  # Faster training
            )
            
            # Fit model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(prophet_df)
            
            # Generate forecast
            future = model.make_future_dataframe(
                periods=self.forecast_horizon,
                freq='D'
            )
            forecast = model.predict(future)
            
            # Extract forecast for future dates only
            future_forecast = forecast.tail(self.forecast_horizon)
            
            # Ensure positive forecasts
            forecast_mean = np.maximum(future_forecast['yhat'].values, 0)
            forecast_lower = np.maximum(future_forecast['yhat_lower'].values, 0)
            forecast_upper = np.maximum(future_forecast['yhat_upper'].values, 0)
            
            logger.info(f"✓ Prophet model trained for {medicine_id}")
            
            return {
                'model': model,
                'forecast': future_forecast,
                'forecast_mean': forecast_mean,
                'forecast_lower': forecast_lower,
                'forecast_upper': forecast_upper
            }
            
        except Exception as e:
            logger.error(f"Prophet training failed for {medicine_id}: {e}")
            return None
    
    def ensemble_forecast(self, arima_result: Dict, 
                         lstm_result: Dict,
                         prophet_result: Dict) -> Dict:
        """
        Create ensemble forecast with uncertainty quantification
        Combines ARIMA, LSTM, and Prophet predictions
        """
        forecasts = []
        weights = []
        
        if arima_result:
            forecasts.append(arima_result['forecast_mean'])
            weights.append(0.3)
        if lstm_result:
            forecasts.append(lstm_result['forecast'])
            weights.append(0.4)
        if prophet_result:
            forecasts.append(prophet_result['forecast_mean'])
            weights.append(0.3)
        
        if not forecasts:
            return None
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Pad shorter forecasts to match longest
        max_len = max(len(f) for f in forecasts)
        padded_forecasts = []
        for f in forecasts:
            if len(f) < max_len:
                # Pad with last value
                padding = np.full(max_len - len(f), f[-1])
                f = np.concatenate([f, padding])
            padded_forecasts.append(f[:max_len])
        
        # Weighted average
        ensemble_forecast = np.average(padded_forecasts, axis=0, weights=weights)
        
        # Confidence intervals
        if arima_result and prophet_result:
            # Average confidence intervals from ARIMA and Prophet
            lower_a = arima_result['forecast_lower']
            upper_a = arima_result['forecast_upper']
            lower_p = prophet_result['forecast_lower']
            upper_p = prophet_result['forecast_upper']
            
            # Pad if needed
            if len(lower_a) < max_len:
                lower_a = np.pad(lower_a, (0, max_len - len(lower_a)), 'edge')
                upper_a = np.pad(upper_a, (0, max_len - len(upper_a)), 'edge')
            if len(lower_p) < max_len:
                lower_p = np.pad(lower_p, (0, max_len - len(lower_p)), 'edge')
                upper_p = np.pad(upper_p, (0, max_len - len(upper_p)), 'edge')
            
            lower_bound = (lower_a + lower_p) / 2
            upper_bound = (upper_a + upper_p) / 2
        elif arima_result:
            lower_bound = arima_result['forecast_lower']
            upper_bound = arima_result['forecast_upper']
            if len(lower_bound) < max_len:
                lower_bound = np.pad(lower_bound, (0, max_len - len(lower_bound)), 'edge')
                upper_bound = np.pad(upper_bound, (0, max_len - len(upper_bound)), 'edge')
        elif prophet_result:
            lower_bound = prophet_result['forecast_lower']
            upper_bound = prophet_result['forecast_upper']
            if len(lower_bound) < max_len:
                lower_bound = np.pad(lower_bound, (0, max_len - len(lower_bound)), 'edge')
                upper_bound = np.pad(upper_bound, (0, max_len - len(upper_bound)), 'edge')
        else:
            # Estimate confidence interval from LSTM
            std_dev = np.std(lstm_result['forecast'])
            lower_bound = ensemble_forecast - 1.96 * std_dev
            upper_bound = ensemble_forecast + 1.96 * std_dev
        
        # Ensure non-negative bounds
        lower_bound = np.maximum(lower_bound, 0)
        
        return {
            'forecast': ensemble_forecast,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'uncertainty': upper_bound - lower_bound
        }
    
    def train_all_models(self, prescriptions_df: pd.DataFrame, 
                        medicine_id: str) -> Dict:
        """Train all models and create ensemble"""
        logger.info(f"Training models for {medicine_id}")
        
        try:
            # Prepare time series
            time_series = self.prepare_time_series(prescriptions_df, medicine_id)
            
            if time_series is None or len(time_series) < 14:
                logger.warning(f"Insufficient historical data for {medicine_id}")
                return None
            
            # Train individual models (each may return None if fails)
            arima_result = self.train_arima_model(time_series, medicine_id)
            lstm_result = self.train_lstm_model(time_series, medicine_id)
            prophet_result = self.train_prophet_model(time_series, medicine_id)
            
            # Check if at least one model succeeded
            if not any([arima_result, lstm_result, prophet_result]):
                logger.error(f"All models failed for {medicine_id}")
                return None
            
            # Create ensemble
            ensemble_result = self.ensemble_forecast(
                arima_result, lstm_result, prophet_result
            )
            
            # Store results
            self.models[medicine_id] = {
                'arima': arima_result,
                'lstm': lstm_result,
                'prophet': prophet_result,
                'ensemble': ensemble_result,
                'time_series': time_series
            }
            
            logger.info(f"✅ Successfully trained models for {medicine_id}")
            return ensemble_result
        
        except Exception as e:
            logger.error(f"Error training models for {medicine_id}: {e}")
            return None
    
    def save_models(self, filepath: str):
        """Save trained models"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers
            }, f)
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.models = data['models']
            self.scalers = data['scalers']
        logger.info(f"Models loaded from {filepath}")

# Usage example
if __name__ == "__main__":
    # Load historical data
    df = pd.read_csv('historical_prescriptions.csv')
    
    # Initialize forecaster
    forecaster = DemandForecaster(forecast_horizon=30)
    
    # Get top medicines by demand
    top_medicines = df.groupby('medicine_id')['quantity'].sum().nlargest(10).index
    
    # Train models for top medicines
    for medicine_id in top_medicines:
        result = forecaster.train_all_models(df, medicine_id)
        if result:
            print(f"\n✅ Forecast for {medicine_id}:")
            print(f"  Mean demand (next 30 days): {result['forecast'].mean():.2f}")
            print(f"  Uncertainty range: {result['uncertainty'].mean():.2f}")
    
    # Save models
    forecaster.save_models('trained_models/demand_models.pkl')