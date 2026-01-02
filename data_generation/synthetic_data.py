"""
Synthetic data utilities and CTGAN/SDV generation
Alternative to rule-based generation using generative models
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """
    Generate synthetic prescription data using statistical methods
    Can be extended with CTGAN/SDV for more sophisticated generation
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def fit(self, real_data: pd.DataFrame):
        """
        Learn patterns from real data
        
        Args:
            real_data: Real prescription data to learn from
        """
        logger.info("Learning patterns from real data...")
        
        # Extract statistical properties
        self.statistics = {}
        
        # Numerical columns
        for col in ['quantity', 'patient_age']:
            if col in real_data.columns:
                self.statistics[col] = {
                    'mean': real_data[col].mean(),
                    'std': real_data[col].std(),
                    'min': real_data[col].min(),
                    'max': real_data[col].max()
                }
        
        # Categorical columns
        for col in ['medicine_id', 'category', 'location']:
            if col in real_data.columns:
                value_counts = real_data[col].value_counts(normalize=True)
                self.statistics[col] = {
                    'values': value_counts.index.tolist(),
                    'probabilities': value_counts.values.tolist()
                }
        
        # Temporal patterns
        if 'timestamp' in real_data.columns:
            real_data['hour'] = pd.to_datetime(real_data['timestamp']).dt.hour
            hourly_dist = real_data['hour'].value_counts(normalize=True).sort_index()
            self.statistics['hour'] = {
                'values': hourly_dist.index.tolist(),
                'probabilities': hourly_dist.values.tolist()
            }
        
        # Correlations
        if 'medicine_id' in real_data.columns and 'category' in real_data.columns:
            self.medicine_category_map = (
                real_data.groupby('medicine_id')['category']
                .apply(lambda x: x.mode()[0])
                .to_dict()
            )
        
        logger.info("Pattern learning complete")
    
    def generate(self, num_samples: int) -> pd.DataFrame:
        """
        Generate synthetic data based on learned patterns
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Synthetic prescription data
        """
        logger.info(f"Generating {num_samples} synthetic samples...")
        
        synthetic_data = []
        
        for i in range(num_samples):
            sample = {}
            
            # Generate prescription ID
            sample['prescription_id'] = f"SYNTH{i:06d}"
            
            # Generate medicine ID
            if 'medicine_id' in self.statistics:
                sample['medicine_id'] = np.random.choice(
                    self.statistics['medicine_id']['values'],
                    p=self.statistics['medicine_id']['probabilities']
                )
            
            # Get category from medicine
            if hasattr(self, 'medicine_category_map') and sample['medicine_id'] in self.medicine_category_map:
                sample['category'] = self.medicine_category_map[sample['medicine_id']]
            elif 'category' in self.statistics:
                sample['category'] = np.random.choice(
                    self.statistics['category']['values'],
                    p=self.statistics['category']['probabilities']
                )
            
            # Generate quantity
            if 'quantity' in self.statistics:
                sample['quantity'] = int(np.random.normal(
                    self.statistics['quantity']['mean'],
                    self.statistics['quantity']['std']
                ))
                sample['quantity'] = np.clip(
                    sample['quantity'],
                    self.statistics['quantity']['min'],
                    self.statistics['quantity']['max']
                )
            
            # Generate patient age
            if 'patient_age' in self.statistics:
                sample['patient_age'] = int(np.random.normal(
                    self.statistics['patient_age']['mean'],
                    self.statistics['patient_age']['std']
                ))
                sample['patient_age'] = np.clip(
                    sample['patient_age'],
                    self.statistics['patient_age']['min'],
                    self.statistics['patient_age']['max']
                )
            
            # Generate location
            if 'location' in self.statistics:
                sample['location'] = np.random.choice(
                    self.statistics['location']['values'],
                    p=self.statistics['location']['probabilities']
                )
            
            # Generate timestamp with temporal patterns
            if 'hour' in self.statistics:
                hour = np.random.choice(
                    self.statistics['hour']['values'],
                    p=self.statistics['hour']['probabilities']
                )
            else:
                hour = np.random.randint(0, 24)
            
            # Random date within a range
            days_offset = np.random.randint(0, 365)
            base_date = pd.Timestamp('2024-01-01')
            sample['timestamp'] = (
                base_date + pd.Timedelta(days=days_offset, hours=hour)
            ).isoformat()
            
            # Binary flags
            sample['is_emergency'] = np.random.random() < 0.05
            sample['insurance'] = np.random.choice(['Yes', 'No'], p=[0.7, 0.3])
            
            synthetic_data.append(sample)
        
        df = pd.DataFrame(synthetic_data)
        logger.info("Synthetic data generation complete")
        
        return df
    
    def generate_correlated_samples(self, num_samples: int,
                                   correlation_strength: float = 0.5) -> pd.DataFrame:
        """
        Generate synthetic data with specific correlations
        
        Args:
            num_samples: Number of samples
            correlation_strength: Strength of correlations (0-1)
            
        Returns:
            Correlated synthetic data
        """
        # Generate base samples
        df = self.generate(num_samples)
        
        # Add correlations
        # Example: Older patients -> higher quantities
        if 'patient_age' in df.columns and 'quantity' in df.columns:
            age_effect = (df['patient_age'] - df['patient_age'].mean()) / df['patient_age'].std()
            quantity_adjustment = age_effect * correlation_strength * df['quantity'].std()
            df['quantity'] = df['quantity'] + quantity_adjustment
            df['quantity'] = df['quantity'].clip(lower=1).astype(int)
        
        return df

class CTGANWrapper:
    """
    Wrapper for CTGAN/SDV synthetic data generation
    Note: Requires 'sdv' package installation
    """
    
    def __init__(self):
        try:
            from sdv.tabular import CTGAN
            self.model = CTGAN()
            self.available = True
            logger.info("CTGAN available for synthetic data generation")
        except ImportError:
            self.available = False
            logger.warning("SDV not installed. Use: pip install sdv")
    
    def fit(self, real_data: pd.DataFrame):
        """Fit CTGAN model"""
        if not self.available:
            raise ImportError("SDV package not available")
        
        logger.info("Training CTGAN model...")
        self.model.fit(real_data)
        logger.info("CTGAN training complete")
    
    def generate(self, num_samples: int) -> pd.DataFrame:
        """Generate samples using CTGAN"""
        if not self.available:
            raise ImportError("SDV package not available")
        
        logger.info(f"Generating {num_samples} samples with CTGAN...")
        synthetic_data = self.model.sample(num_samples)
        return synthetic_data

def create_train_test_split(data: pd.DataFrame, 
                           test_size: float = 0.2) -> tuple:
    """
    Create train/test split for validation
    
    Args:
        data: Full dataset
        test_size: Proportion for test set
        
    Returns:
        train_data, test_data
    """
    split_idx = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    return train_data, test_data

def augment_data(data: pd.DataFrame, 
                augmentation_factor: float = 1.5) -> pd.DataFrame:
    """
    Augment existing data with synthetic samples
    
    Args:
        data: Original data
        augmentation_factor: Multiplier for data size
        
    Returns:
        Augmented dataset
    """
    num_synthetic = int(len(data) * (augmentation_factor - 1))
    
    generator = SyntheticDataGenerator()
    generator.fit(data)
    synthetic = generator.generate(num_synthetic)
    
    # Combine
    augmented = pd.concat([data, synthetic], ignore_index=True)
    
    logger.info(f"Augmented data: {len(data)} -> {len(augmented)} samples")
    
    return augmented

# Example usage
if __name__ == "__main__":
    # Create sample real data
    real_data = pd.DataFrame({
        'prescription_id': [f'RX{i:04d}' for i in range(1000)],
        'medicine_id': np.random.choice(['MED001', 'MED002', 'MED003'], 1000),
        'category': np.random.choice(['Antibiotics', 'Analgesics'], 1000),
        'quantity': np.random.randint(1, 10, 1000),
        'patient_age': np.random.randint(1, 90, 1000),
        'location': np.random.choice(['Store1', 'Store2'], 1000),
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1H')
    })
    
    # Generate synthetic data
    generator = SyntheticDataGenerator()
    generator.fit(real_data)
    synthetic_data = generator.generate(500)
    
    print("\nSynthetic Data Sample:")
    print(synthetic_data.head())
    print(f"\nGenerated {len(synthetic_data)} synthetic samples")