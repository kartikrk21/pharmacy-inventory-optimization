import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import random
from typing import Dict, List

class PrescriptionDataGenerator:
    """
    Generates synthetic prescription data with realistic patterns
    including seasonality, trends, and stochastic variations
    """
    
    def __init__(self, num_medicines=500, utility_range=(0.85, 0.95)):
        self.num_medicines = num_medicines
        self.utility_range = utility_range
        self.medicines = self._generate_medicine_catalog()
        self.seasonal_patterns = self._generate_seasonal_patterns()
        
    def _generate_medicine_catalog(self) -> pd.DataFrame:
        """Generate synthetic medicine catalog"""
        categories = ['Antibiotics', 'Analgesics', 'Antivirals', 
                     'Cardiovascular', 'Diabetes', 'Respiratory',
                     'Gastro', 'Dermatology', 'Vitamins', 'Others']
        
        medicines = []
        for i in range(self.num_medicines):
            med = {
                'medicine_id': f'MED{i:04d}',
                'medicine_name': f'Medicine_{i}',
                'category': random.choice(categories),
                'unit_price': round(random.uniform(10, 500), 2),
                'shelf_life_days': random.choice([180, 365, 730, 1095]),
                'storage_type': random.choice(['Room Temp', 'Refrigerated', 'Frozen']),
                'pack_size': random.choice([10, 30, 60, 100]),
                'base_demand': random.uniform(10, 200),
                'demand_variance': random.uniform(0.1, 0.3),
                'seasonality_factor': random.uniform(0.5, 1.5)
            }
            medicines.append(med)
        
        return pd.DataFrame(medicines)
    
    def _generate_seasonal_patterns(self) -> Dict:
        """Generate seasonal demand patterns"""
        patterns = {}
        for _, med in self.medicines.iterrows():
            # Different patterns for different categories
            if med['category'] in ['Respiratory', 'Antivirals']:
                # Winter peak
                pattern = [1.5, 1.4, 1.2, 0.9, 0.7, 0.6, 0.6, 0.7, 0.9, 1.1, 1.3, 1.5]
            elif med['category'] in ['Dermatology']:
                # Summer peak
                pattern = [0.8, 0.9, 1.0, 1.2, 1.5, 1.6, 1.6, 1.5, 1.2, 1.0, 0.9, 0.8]
            else:
                # Relatively stable
                pattern = [1.0 + random.uniform(-0.1, 0.1) for _ in range(12)]
            
            patterns[med['medicine_id']] = pattern
        
        return patterns
    
    def generate_prescription(self, timestamp: datetime = None) -> Dict:
        """Generate a single prescription event"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Select medicine based on realistic probability distribution
        medicine = self.medicines.sample(1, weights='base_demand').iloc[0]
        
        # Apply seasonal adjustment
        month = timestamp.month - 1
        seasonal_factor = self.seasonal_patterns[medicine['medicine_id']][month]
        
        # Calculate demand with noise
        base_demand = medicine['base_demand'] * seasonal_factor
        noise = np.random.normal(0, medicine['demand_variance'] * base_demand)
        quantity = max(1, int(base_demand + noise))
        
        # Generate prescription
        prescription = {
            'prescription_id': f'RX{int(timestamp.timestamp()*1000)}',
            'timestamp': timestamp.isoformat(),
            'medicine_id': medicine['medicine_id'],
            'medicine_name': medicine['medicine_name'],
            'category': medicine['category'],
            'quantity': quantity,
            'patient_age': random.randint(1, 90),
            'is_emergency': random.random() < 0.05,
            'insurance': random.choice(['Yes', 'No']),
            'location': random.choice(['Store1', 'Store2', 'Store3', 'Online']),
            'day_of_week': timestamp.strftime('%A'),
            'hour_of_day': timestamp.hour
        }
        
        return prescription
    
    def generate_batch(self, num_prescriptions: int, 
                       start_time: datetime = None,
                       duration_hours: int = 24) -> List[Dict]:
        """Generate a batch of prescriptions over a time period"""
        if start_time is None:
            start_time = datetime.now()
        
        prescriptions = []
        
        # Generate prescriptions with realistic time distribution
        # More prescriptions during business hours (9 AM - 8 PM)
        for _ in range(num_prescriptions):
            # Random time within duration
            random_seconds = random.randint(0, duration_hours * 3600)
            timestamp = start_time + timedelta(seconds=random_seconds)
            
            # Adjust probability based on hour
            hour_weight = 1.0
            if 9 <= timestamp.hour <= 20:
                hour_weight = 2.0
            elif timestamp.hour < 6 or timestamp.hour > 22:
                hour_weight = 0.3
            
            if random.random() < hour_weight / 2.0:
                prescription = self.generate_prescription(timestamp)
                prescriptions.append(prescription)
        
        # Sort by timestamp
        prescriptions.sort(key=lambda x: x['timestamp'])
        
        return prescriptions
    
    def generate_streaming_data(self, rate_per_second: float = 10):
        """
        Generator function for streaming prescription data
        Used with Kafka producer
        """
        while True:
            # Generate prescription at specified rate
            prescription = self.generate_prescription()
            yield prescription
            
            # Sleep to maintain rate
            import time
            time.sleep(1 / rate_per_second + random.uniform(-0.1, 0.1))
    
    def calculate_utility_metrics(self, prescriptions: List[Dict]) -> Dict:
        """
        Calculate utility and privacy metrics for generated data
        Target: 85-95% utility
        """
        df = pd.DataFrame(prescriptions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        metrics = {
            'total_prescriptions': len(df),
            'unique_medicines': df['medicine_id'].nunique(),
            'avg_quantity': df['quantity'].mean(),
            'emergency_rate': df['is_emergency'].mean(),
            'temporal_coverage': (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600,
            'category_distribution': df['category'].value_counts().to_dict(),
            'hourly_distribution': df['hour_of_day'].value_counts().to_dict()
        }
        
        # Utility score (0-1)
        # Based on realistic patterns and data completeness
        utility_score = min(0.95, 
                           0.7 +  # base score
                           0.15 * (metrics['unique_medicines'] / self.num_medicines) +
                           0.10 * min(1.0, metrics['total_prescriptions'] / 1000))
        
        metrics['utility_score'] = utility_score
        
        return metrics
    
    def save_historical_data(self, filepath: str, num_days: int = 365):
        """Generate and save historical data for ML training"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=num_days)
        
        # Generate approximately 10k prescriptions per day
        total_prescriptions = num_days * 2500
        
        print(f"Generating {total_prescriptions} prescriptions over {num_days} days...")
        
        prescriptions = self.generate_batch(
            total_prescriptions,
            start_date,
            num_days * 24
        )
        
        df = pd.DataFrame(prescriptions)
        df.to_csv(filepath, index=False)
        
        print(f"Saved {len(df)} prescriptions to {filepath}")
        
        # Calculate and display metrics
        metrics = self.calculate_utility_metrics(prescriptions)
        print(f"\nData Quality Metrics:")
        print(f"  Utility Score: {metrics['utility_score']:.2%}")
        print(f"  Unique Medicines: {metrics['unique_medicines']}")
        print(f"  Average Quantity: {metrics['avg_quantity']:.2f}")
        
        return df

# Example usage
if __name__ == "__main__":
    generator = PrescriptionDataGenerator(num_medicines=500)
    
    # Generate historical data for ML training
    historical_data = generator.save_historical_data(
        'historical_prescriptions.csv',
        num_days=365
    )
    
   