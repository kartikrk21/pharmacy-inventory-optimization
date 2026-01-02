"""
Uncertainty Quantification for forecasts
Implements confidence intervals and prediction intervals
"""
import numpy as np
from scipy import stats
from typing import Tuple

class UncertaintyQuantifier:
    """
    Quantify uncertainty in forecasts
    Provides confidence and prediction intervals
    """
    
    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level
        self.z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    def calculate_confidence_interval(self, 
                                     forecast: np.ndarray,
                                     std_errors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence intervals
        
        Args:
            forecast: Point forecasts
            std_errors: Standard errors
            
        Returns:
            (lower_bound, upper_bound)
        """
        margin = self.z_score * std_errors
        
        lower_bound = forecast - margin
        upper_bound = forecast + margin
        
        return lower_bound, upper_bound
    
    def bootstrap_intervals(self, 
                           model,
                           data: np.ndarray,
                           n_bootstrap: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate bootstrap confidence intervals
        
        Args:
            model: Fitted forecasting model
            data: Historical data
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            (lower_bound, upper_bound)
        """
        forecasts = []
        
        for _ in range(n_bootstrap):
            # Resample data
            sample = np.random.choice(data, size=len(data), replace=True)
            
            # Fit model and forecast
            model_temp = type(model)()
            model_temp.fit(sample)
            forecast = model_temp.forecast(steps=30)
            forecasts.append(forecast)
        
        forecasts = np.array(forecasts)
        
        # Calculate percentiles
        alpha = 1 - self.confidence_level
        lower_bound = np.percentile(forecasts, alpha/2 * 100, axis=0)
        upper_bound = np.percentile(forecasts, (1 - alpha/2) * 100, axis=0)
        
        return lower_bound, upper_bound
    
    def monte_carlo_intervals(self,
                             forecast: np.ndarray,
                             variance: float,
                             n_simulations: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate intervals using Monte Carlo simulation
        
        Args:
            forecast: Point forecast
            variance: Forecast variance
            n_simulations: Number of simulations
            
        Returns:
            (lower_bound, upper_bound)
        """
        simulations = []
        
        for _ in range(n_simulations):
            # Add random noise
            noise = np.random.normal(0, np.sqrt(variance), len(forecast))
            sim = forecast + noise
            simulations.append(sim)
        
        simulations = np.array(simulations)
        
        # Calculate percentiles
        alpha = 1 - self.confidence_level
        lower_bound = np.percentile(simulations, alpha/2 * 100, axis=0)
        upper_bound = np.percentile(simulations, (1 - alpha/2) * 100, axis=0)
        
        return lower_bound, upper_bound
    
    def quantile_regression_intervals(self,
                                     model,
                                     X: np.ndarray,
                                     quantiles=[0.025, 0.975]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals using quantile regression
        
        Args:
            model: Quantile regression model
            X: Features
            quantiles: Lower and upper quantiles
            
        Returns:
            (lower_bound, upper_bound)
        """
        lower_bound = model.predict(X, quantile=quantiles[0])
        upper_bound = model.predict(X, quantile=quantiles[1])
        
        return lower_bound, upper_bound
    
    def ensemble_uncertainty(self,
                           forecasts: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate uncertainty from ensemble of forecasts
        
        Args:
            forecasts: List of forecast arrays
            
        Returns:
            (mean_forecast, lower_bound, upper_bound)
        """
        forecasts = np.array(forecasts)
        
        mean_forecast = np.mean(forecasts, axis=0)
        std_forecast = np.std(forecasts, axis=0)
        
        lower_bound = mean_forecast - self.z_score * std_forecast
        upper_bound = mean_forecast + self.z_score * std_forecast
        
        return mean_forecast, lower_bound, upper_bound
    
    def calculate_prediction_intervals(self,
                                      forecast: np.ndarray,
                                      residual_variance: float,
                                      horizon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals accounting for forecast horizon
        
        Args:
            forecast: Point forecast
            residual_variance: Model residual variance
            horizon: Forecast horizon (days ahead)
            
        Returns:
            (lower_bound, upper_bound)
        """
        # Variance increases with horizon
        horizon_variance = residual_variance * (1 + 0.1 * horizon)
        
        margin = self.z_score * np.sqrt(horizon_variance)
        
        lower_bound = forecast - margin
        upper_bound = forecast + margin
        
        # Ensure non-negative forecasts
        lower_bound = np.maximum(lower_bound, 0)
        
        return lower_bound, upper_bound

# Example usage
if __name__ == "__main__":
    # Sample time series
    np.random.seed(42)
    t = np.arange(0, 365)
    ts = 100 + 10 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 5, 365)
    
    # Fit ARIMA
    print("Training ARIMA...")
    arima = ARIMAModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    arima.fit(ts)
    arima_forecast = arima.forecast(steps=30)
    print(f"ARIMA forecast mean: {arima_forecast['mean'][:5]}")
    
    # Fit LSTM
    print("\nTraining LSTM...")
    lstm = LSTMModel(lookback=14)
    lstm.fit(ts, epochs=20)
    lstm_forecast = lstm.forecast(ts, steps=30)
    print(f"LSTM forecast: {lstm_forecast[:5]}")
    
    # Uncertainty quantification
    print("\nCalculating uncertainty...")
    uq = UncertaintyQuantifier(confidence_level=0.95)
    
    # Bootstrap intervals
    std_errors = np.std(arima.get_residuals()) * np.ones(30)
    lower, upper = uq.calculate_confidence_interval(arima_forecast['mean'], std_errors)
    print(f"Confidence interval width: {np.mean(upper - lower):.2f}")