"""
Privacy evaluation for synthetic data generation
Implements differential privacy mechanisms and utility metrics
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrivacyEvaluator:
    """
    Evaluate privacy and utility of synthetic prescription data
    Target: 85-95% utility with differential privacy guarantees
    """
    
    def __init__(self, epsilon: float = 1.0):
        """
        Args:
            epsilon: Privacy budget (lower = more private)
        """
        self.epsilon = epsilon
        self.utility_threshold = 0.85
    
    def add_laplace_noise(self, value: float, sensitivity: float) -> float:
        """
        Add Laplace noise for differential privacy
        
        Args:
            value: True value
            sensitivity: Sensitivity of the query
            
        Returns:
            Noisy value
        """
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    def anonymize_age(self, age: int, bins: int = 5) -> str:
        """
        Anonymize age into bins
        
        Args:
            age: Patient age
            bins: Number of age groups
            
        Returns:
            Age group string
        """
        if age < 18:
            return "0-17"
        elif age < 35:
            return "18-34"
        elif age < 50:
            return "35-49"
        elif age < 65:
            return "50-64"
        else:
            return "65+"
    
    def evaluate_utility(self, original_data: pd.DataFrame, 
                        synthetic_data: pd.DataFrame) -> Dict:
        """
        Evaluate utility of synthetic data compared to original
        
        Returns:
            Dictionary with utility metrics
        """
        metrics = {}
        
        # 1. Statistical similarity
        # Compare distributions of key columns
        for col in ['quantity', 'patient_age']:
            if col in original_data.columns and col in synthetic_data.columns:
                orig_mean = original_data[col].mean()
                synth_mean = synthetic_data[col].mean()
                
                orig_std = original_data[col].std()
                synth_std = synthetic_data[col].std()
                
                # Calculate similarity score (0-1)
                mean_similarity = 1 - abs(orig_mean - synth_mean) / orig_mean
                std_similarity = 1 - abs(orig_std - synth_std) / orig_std
                
                metrics[f'{col}_mean_similarity'] = mean_similarity
                metrics[f'{col}_std_similarity'] = std_similarity
        
        # 2. Category distribution similarity
        for col in ['category', 'location']:
            if col in original_data.columns and col in synthetic_data.columns:
                orig_dist = original_data[col].value_counts(normalize=True)
                synth_dist = synthetic_data[col].value_counts(normalize=True)
                
                # Calculate KL divergence
                common_cats = set(orig_dist.index) & set(synth_dist.index)
                kl_div = 0
                for cat in common_cats:
                    if synth_dist[cat] > 0:
                        kl_div += orig_dist[cat] * np.log(orig_dist[cat] / synth_dist[cat])
                
                # Convert to similarity (0-1)
                similarity = np.exp(-kl_div)
                metrics[f'{col}_distribution_similarity'] = similarity
        
        # 3. Temporal patterns
        if 'timestamp' in original_data.columns and 'timestamp' in synthetic_data.columns:
            orig_data = original_data.copy()
            synth_data = synthetic_data.copy()
            
            orig_data['hour'] = pd.to_datetime(orig_data['timestamp']).dt.hour
            synth_data['hour'] = pd.to_datetime(synth_data['timestamp']).dt.hour
            
            orig_hourly = orig_data.groupby('hour').size()
            synth_hourly = synth_data.groupby('hour').size()
            
            # Normalize
            orig_hourly = orig_hourly / orig_hourly.sum()
            synth_hourly = synth_hourly / synth_hourly.sum()
            
            # Calculate correlation
            temporal_similarity = np.corrcoef(orig_hourly, synth_hourly)[0, 1]
            metrics['temporal_pattern_similarity'] = temporal_similarity
        
        # 4. Overall utility score
        utility_score = np.mean([v for v in metrics.values() if isinstance(v, float)])
        metrics['overall_utility'] = utility_score
        
        logger.info(f"Utility evaluation complete: {utility_score:.2%}")
        
        return metrics
    
    def evaluate_privacy_risk(self, synthetic_data: pd.DataFrame) -> Dict:
        """
        Evaluate privacy risks in synthetic data
        
        Returns:
            Dictionary with privacy risk metrics
        """
        risks = {}
        
        # 1. Check for unique identifiers
        if 'prescription_id' in synthetic_data.columns:
            unique_ratio = synthetic_data['prescription_id'].nunique() / len(synthetic_data)
            risks['unique_identifier_ratio'] = unique_ratio
        
        # 2. Check for rare combinations that might be identifying
        if 'medicine_id' in synthetic_data.columns and 'patient_age' in synthetic_data.columns:
            combo_counts = synthetic_data.groupby(['medicine_id', 'patient_age']).size()
            rare_combos = (combo_counts == 1).sum()
            risks['rare_combination_ratio'] = rare_combos / len(combo_counts)
        
        # 3. k-anonymity check (simplified)
        if 'patient_age' in synthetic_data.columns and 'location' in synthetic_data.columns:
            quasi_identifiers = synthetic_data.groupby(['patient_age', 'location']).size()
            min_group_size = quasi_identifiers.min()
            avg_group_size = quasi_identifiers.mean()
            
            risks['min_k_anonymity'] = min_group_size
            risks['avg_k_anonymity'] = avg_group_size
        
        # 4. Overall privacy score (higher = better privacy)
        privacy_score = 1.0
        if 'rare_combination_ratio' in risks:
            privacy_score *= (1 - risks['rare_combination_ratio'])
        
        risks['overall_privacy_score'] = privacy_score
        
        logger.info(f"Privacy evaluation complete: {privacy_score:.2%}")
        
        return risks
    
    def apply_differential_privacy(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply differential privacy to dataset
        
        Args:
            data: Original data
            
        Returns:
            Privacy-protected data
        """
        protected_data = data.copy()
        
        # Add noise to numerical columns
        if 'quantity' in protected_data.columns:
            protected_data['quantity'] = protected_data['quantity'].apply(
                lambda x: max(1, int(self.add_laplace_noise(x, sensitivity=1)))
            )
        
        # Generalize age
        if 'patient_age' in protected_data.columns:
            protected_data['age_group'] = protected_data['patient_age'].apply(
                self.anonymize_age
            )
            # Optionally remove exact age
            # protected_data = protected_data.drop('patient_age', axis=1)
        
        return protected_data
    
    def generate_privacy_report(self, original_data: pd.DataFrame,
                               synthetic_data: pd.DataFrame) -> Dict:
        """
        Generate comprehensive privacy and utility report
        
        Returns:
            Complete evaluation report
        """
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'epsilon': self.epsilon,
            'data_size': {
                'original': len(original_data),
                'synthetic': len(synthetic_data)
            }
        }
        
        # Utility metrics
        utility_metrics = self.evaluate_utility(original_data, synthetic_data)
        report['utility'] = utility_metrics
        
        # Privacy metrics
        privacy_risks = self.evaluate_privacy_risk(synthetic_data)
        report['privacy'] = privacy_risks
        
        # Overall assessment
        report['meets_utility_threshold'] = (
            utility_metrics['overall_utility'] >= self.utility_threshold
        )
        
        report['assessment'] = self._generate_assessment(utility_metrics, privacy_risks)
        
        return report
    
    def _generate_assessment(self, utility_metrics: Dict, 
                            privacy_risks: Dict) -> str:
        """Generate human-readable assessment"""
        utility_score = utility_metrics['overall_utility']
        privacy_score = privacy_risks['overall_privacy_score']
        
        if utility_score >= 0.85 and privacy_score >= 0.80:
            return "EXCELLENT: High utility with strong privacy guarantees"
        elif utility_score >= 0.85:
            return "GOOD: Meets utility threshold, privacy can be improved"
        elif privacy_score >= 0.80:
            return "FAIR: Good privacy, but utility below threshold"
        else:
            return "NEEDS IMPROVEMENT: Both utility and privacy need attention"

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    
    original = pd.DataFrame({
        'prescription_id': [f'RX{i:04d}' for i in range(1000)],
        'medicine_id': np.random.choice(['MED001', 'MED002', 'MED003'], 1000),
        'quantity': np.random.randint(1, 10, 1000),
        'patient_age': np.random.randint(1, 90, 1000),
        'category': np.random.choice(['Antibiotics', 'Analgesics'], 1000),
        'location': np.random.choice(['Store1', 'Store2'], 1000),
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1H')
    })
    
    # Generate synthetic data (simplified)
    synthetic = original.copy()
    synthetic['quantity'] = synthetic['quantity'] + np.random.randint(-1, 2, 1000)
    
    # Evaluate
    evaluator = PrivacyEvaluator(epsilon=1.0)
    report = evaluator.generate_privacy_report(original, synthetic)
    
    print("\nPrivacy & Utility Report:")
    print(f"Overall Utility: {report['utility']['overall_utility']:.2%}")
    print(f"Overall Privacy: {report['privacy']['overall_privacy_score']:.2%}")
    print(f"Assessment: {report['assessment']}")