"""
Individual ARIMA and LSTM model implementations
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ARIMAModel:
    """ARIMA time series model"""
    
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.results = None
    
    def fit(self, time_series: np.ndarray):
        """Fit ARIMA model"""
        try:
            if self.seasonal_order:
                self.model = SARIMAX(
                    time_series,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                self.model = ARIMA(time_series, order=self.order)
            
            self.results = self.model.fit(disp=False)
            logger.info(f"ARIMA model fitted. AIC: {self.results.aic:.2f}")
            
        except Exception as e:
            logger.error(f"ARIMA fit error: {e}")
            raise
    
    def forecast(self, steps: int = 30):
        """Generate forecast"""
        if self.results is None:
            raise ValueError("Model not fitted")
        
        forecast = self.results.get_forecast(steps=steps)
        
        return {
            'mean': forecast.predicted_mean,
            'confidence_intervals': forecast.conf_int()
        }
    
    def get_residuals(self):
        """Get model residuals"""
        if self.results is None:
            return None
        return self.results.resid

class LSTMModel:
    """LSTM neural network for time series"""
    
    def __init__(self, lookback=14, units=[128, 64], dropout=0.2):
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.model = None
        self.scaler = MinMaxScaler()
    
    def build_model(self, input_shape):
        """Build LSTM architecture"""
        model = keras.Sequential()
        
        # First LSTM layer
        model.add(layers.LSTM(
            self.units[0],
            return_sequences=True if len(self.units) > 1 else False,
            input_shape=input_shape
        ))
        model.add(layers.Dropout(self.dropout))
        
        # Additional LSTM layers
        for i, units in enumerate(self.units[1:]):
            return_seq = i < len(self.units) - 2
            model.add(layers.LSTM(units, return_sequences=return_seq))
            model.add(layers.Dropout(self.dropout))
        
        # Dense layers
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(1))
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_data(self, time_series: np.ndarray):
        """Prepare data for LSTM"""
        # Scale data
        scaled = self.scaler.fit_transform(time_series.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(self.lookback, len(scaled)):
            X.append(scaled[i-self.lookback:i, 0])
            y.append(scaled[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for LSTM [samples, timesteps, features]
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        return X, y
    
    def fit(self, time_series: np.ndarray, epochs=50, batch_size=32,
            validation_split=0.2):
        """Train LSTM model"""
        X, y = self.prepare_data(time_series)
        
        if len(X) < 100:
            raise ValueError("Insufficient data for LSTM training")
        
        # Build model
        self.model = self.build_model((self.lookback, 1))
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Train
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        logger.info(f"LSTM trained. Final loss: {history.history['loss'][-1]:.4f}")
        
        return history
    
    def forecast(self, time_series: np.ndarray, steps: int = 30):
        """Generate multi-step forecast"""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        # Scale input
        scaled = self.scaler.transform(time_series.reshape(-1, 1))
        
        # Get last sequence
        last_sequence = scaled[-self.lookback:].flatten()
        
        # Generate forecasts
        forecasts = []
        current_seq = last_sequence.copy()
        
        for _ in range(steps):
            # Reshape for prediction
            X = current_seq.reshape(1, self.lookback, 1)
            
            # Predict
            pred = self.model.predict(X, verbose=0)
            forecasts.append(pred[0, 0])
            
            # Update sequence
            current_seq = np.append(current_seq[1:], pred[0, 0])
        
        # Inverse transform
        forecasts = self.scaler.inverse_transform(
            np.array(forecasts).reshape(-1, 1)
        ).flatten()
        
        return forecasts
