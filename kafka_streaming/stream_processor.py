"""
Advanced Stream Processing
Implements Flink-style windowing, feature extraction, and real-time analytics
"""
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Callable
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WindowedStreamProcessor:
    """
    Advanced stream processor with multiple window types
    Implements tumbling, sliding, and session windows
    """
    
    def __init__(self, window_type: str = 'sliding',
                 window_size: int = 300,
                 slide_interval: int = 60):
        """
        Args:
            window_type: 'tumbling', 'sliding', or 'session'
            window_size: Window size in seconds
            slide_interval: Slide interval in seconds (for sliding windows)
        """
        self.window_type = window_type
        self.window_size = window_size
        self.slide_interval = slide_interval
        
        # Window storage
        self.windows = defaultdict(lambda: deque())
        self.session_windows = defaultdict(lambda: {})
        
        # Aggregates
        self.current_aggregates = {}
        
        # Statistics
        self.processed_count = 0
        self.last_window_time = None
    
    def process_event(self, event: Dict) -> Dict:
        """
        Process a single event
        
        Args:
            event: Event data
            
        Returns:
            Processing result
        """
        self.processed_count += 1
        timestamp = datetime.fromisoformat(event['timestamp'])
        key = event.get('medicine_id', 'unknown')
        
        # Add to appropriate window
        if self.window_type == 'tumbling':
            self._add_to_tumbling_window(key, event, timestamp)
        elif self.window_type == 'sliding':
            self._add_to_sliding_window(key, event, timestamp)
        elif self.window_type == 'session':
            self._add_to_session_window(key, event, timestamp)
        
        # Check if window should be triggered
        should_trigger = self._should_trigger_window(timestamp)
        
        if should_trigger:
            self.current_aggregates = self.compute_aggregates()
            self.last_window_time = datetime.now()
        
        return {
            'processed': True,
            'event_id': event.get('prescription_id'),
            'window_triggered': should_trigger
        }
    
    def _add_to_sliding_window(self, key: str, event: Dict, timestamp: datetime):
        """Add event to sliding window"""
        self.windows[key].append({
            'timestamp': timestamp,
            'event': event
        })
        
        # Remove old events
        cutoff_time = timestamp - timedelta(seconds=self.window_size)
        while (self.windows[key] and 
               self.windows[key][0]['timestamp'] < cutoff_time):
            self.windows[key].popleft()
    
    def _add_to_tumbling_window(self, key: str, event: Dict, timestamp: datetime):
        """Add event to tumbling window"""
        # Calculate window bucket
        window_id = int(timestamp.timestamp() / self.window_size)
        
        if not hasattr(self, 'tumbling_windows'):
            self.tumbling_windows = defaultdict(lambda: defaultdict(list))
        
        self.tumbling_windows[key][window_id].append(event)
    
    def _add_to_session_window(self, key: str, event: Dict, timestamp: datetime):
        """Add event to session window (session gap = slide_interval)"""
        if key not in self.session_windows or not self.session_windows[key]:
            # Start new session
            self.session_windows[key] = {
                'start': timestamp,
                'last_event': timestamp,
                'events': [event]
            }
        else:
            session = self.session_windows[key]
            time_gap = (timestamp - session['last_event']).total_seconds()
            
            if time_gap > self.slide_interval:
                # Session expired, start new one
                self.session_windows[key] = {
                    'start': timestamp,
                    'last_event': timestamp,
                    'events': [event]
                }
            else:
                # Continue session
                session['last_event'] = timestamp
                session['events'].append(event)
    
    def _should_trigger_window(self, current_time: datetime) -> bool:
        """Determine if window should be triggered"""
        if self.last_window_time is None:
            return True
        
        elapsed = (current_time - self.last_window_time).total_seconds() if isinstance(self.last_window_time, datetime) else (datetime.now() - self.last_window_time).total_seconds()
        
        return elapsed >= self.slide_interval
    
    def compute_aggregates(self) -> Dict:
        """Compute aggregates for all windows"""
        aggregates = {}
        
        if self.window_type == 'session':
            return self._compute_session_aggregates()
        
        for key, window in self.windows.items():
            if not window:
                continue
            
            events = [w['event'] for w in window]
            timestamps = [w['timestamp'] for w in window]
            
            # Basic statistics
            quantities = [e['quantity'] for e in events]
            
            aggregate = {
                'key': key,
                'window_start': min(timestamps),
                'window_end': max(timestamps),
                'count': len(events),
                'sum': sum(quantities),
                'mean': np.mean(quantities),
                'median': np.median(quantities),
                'std': np.std(quantities),
                'min': min(quantities),
                'max': max(quantities)
            }
            
            # Advanced features
            aggregate.update(self._extract_features(events, timestamps))
            
            aggregates[key] = aggregate
        
        return aggregates
    
    def _compute_session_aggregates(self) -> Dict:
        """Compute aggregates for session windows"""
        aggregates = {}
        
        for key, session in self.session_windows.items():
            if not session.get('events'):
                continue
            
            events = session['events']
            quantities = [e['quantity'] for e in events]
            
            aggregate = {
                'key': key,
                'session_start': session['start'],
                'session_end': session['last_event'],
                'session_duration': (
                    session['last_event'] - session['start']
                ).total_seconds(),
                'count': len(events),
                'sum': sum(quantities),
                'mean': np.mean(quantities)
            }
            
            aggregates[key] = aggregate
        
        return aggregates
    
    def _extract_features(self, events: List[Dict], 
                         timestamps: List[datetime]) -> Dict:
        """Extract advanced features from window"""
        features = {}
        
        # Temporal features
        if len(timestamps) > 1:
            time_diffs = []
            for i in range(1, len(timestamps)):
                diff = (timestamps[i] - timestamps[i-1]).total_seconds()
                time_diffs.append(diff)
            
            features['avg_inter_arrival_time'] = np.mean(time_diffs)
            features['std_inter_arrival_time'] = np.std(time_diffs)
        
        # Emergency rate
        emergency_count = sum(1 for e in events if e.get('is_emergency', False))
        features['emergency_rate'] = emergency_count / len(events)
        
        # Necessity score
        base_rate = len(events) / (self.window_size / 3600)  # per hour
        features['necessity_score'] = base_rate * (1 + features['emergency_rate'])
        
        # Location diversity
        locations = set(e.get('location', 'unknown') for e in events)
        features['location_diversity'] = len(locations)
        
        # Age distribution
        ages = [e.get('patient_age', 0) for e in events if 'patient_age' in e]
        if ages:
            features['avg_patient_age'] = np.mean(ages)
            features['age_std'] = np.std(ages)
        
        # Hour distribution
        features['hour_distribution'] = defaultdict(int)
        for e in events:
            hour = e.get('hour_of_day', 0)
            features['hour_distribution'][hour] += 1
        
        return features

class FeatureExtractor:
    """Extract features for ML models from stream data"""
    
    @staticmethod
    def extract_time_features(timestamp: datetime) -> Dict:
        """Extract time-based features"""
        return {
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'day_of_month': timestamp.day,
            'month': timestamp.month,
            'quarter': (timestamp.month - 1) // 3 + 1,
            'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
            'is_business_hours': 1 if 9 <= timestamp.hour <= 17 else 0
        }
    
    @staticmethod
    def extract_prescription_features(prescription: Dict) -> Dict:
        """Extract features from prescription"""
        features = {
            'quantity': prescription.get('quantity', 0),
            'is_emergency': 1 if prescription.get('is_emergency', False) else 0,
            'has_insurance': 1 if prescription.get('insurance') == 'Yes' else 0
        }
        
        # Age category
        age = prescription.get('patient_age', 0)
        features['age_category'] = (
            0 if age < 18 else
            1 if age < 35 else
            2 if age < 50 else
            3 if age < 65 else 4
        )
        
        return features
    
    @staticmethod
    def extract_rolling_features(window_data: List[Dict], 
                                window_sizes: List[int] = [5, 10, 20]) -> Dict:
        """Extract rolling window features"""
        features = {}
        
        if not window_data:
            return features
        
        quantities = [d.get('quantity', 0) for d in window_data]
        
        for size in window_sizes:
            if len(quantities) >= size:
                recent = quantities[-size:]
                features[f'rolling_mean_{size}'] = np.mean(recent)
                features[f'rolling_std_{size}'] = np.std(recent)
                features[f'rolling_max_{size}'] = max(recent)
                features[f'rolling_min_{size}'] = min(recent)
        
        return features

class CoprescriptionGraphBuilder:
    """Build co-prescription graph for association analysis"""
    
    def __init__(self):
        self.graph = defaultdict(lambda: defaultdict(int))
        self.medicine_counts = defaultdict(int)
    
    def add_prescription_group(self, medicines: List[str]):
        """Add a group of co-prescribed medicines"""
        for med in medicines:
            self.medicine_counts[med] += 1
        
        # Add edges for all pairs
        for i, med1 in enumerate(medicines):
            for med2 in medicines[i+1:]:
                self.graph[med1][med2] += 1
                self.graph[med2][med1] += 1
    
    def get_coprescription_score(self, med1: str, med2: str) -> float:
        """Get co-prescription score between two medicines"""
        coprescription_count = self.graph[med1][med2]
        
        # Normalize by individual frequencies
        if self.medicine_counts[med1] > 0 and self.medicine_counts[med2] > 0:
            score = coprescription_count / min(
                self.medicine_counts[med1],
                self.medicine_counts[med2]
            )
        else:
            score = 0
        
        return score
    
    def get_top_coprescriptions(self, medicine_id: str, top_k: int = 5) -> List[tuple]:
        """Get top k co-prescribed medicines"""
        if medicine_id not in self.graph:
            return []
        
        coprescriptions = [
            (med, count) 
            for med, count in self.graph[medicine_id].items()
        ]
        
        # Sort by count
        coprescriptions.sort(key=lambda x: x[1], reverse=True)
        
        return coprescriptions[:top_k]

# Example usage
if __name__ == "__main__":
    # Create processor
    processor = WindowedStreamProcessor(
        window_type='sliding',
        window_size=300,  # 5 minutes
        slide_interval=60  # 1 minute
    )
    
    # Simulate events
    from datetime import datetime
    import time
    
    for i in range(100):
        event = {
            'prescription_id': f'RX{i:04d}',
            'medicine_id': f'MED{(i % 10):03d}',
            'quantity': np.random.randint(1, 10),
            'is_emergency': np.random.random() < 0.1,
            'patient_age': np.random.randint(1, 90),
            'location': f'Store{(i % 3) + 1}',
            'hour_of_day': datetime.now().hour,
            'timestamp': datetime.now().isoformat()
        }
        
        result = processor.process_event(event)
        
        if result['window_triggered']:
            aggregates = processor.current_aggregates
            print(f"\nWindow triggered at event {i}")
            print(f"  Aggregated {len(aggregates)} medicine groups")
            
            if aggregates:
                sample_key = list(aggregates.keys())[0]
                sample = aggregates[sample_key]
                print(f"  Sample: {sample_key}")
                print(f"    Count: {sample['count']}")
                print(f"    Mean quantity: {sample['mean']:.2f}")
                print(f"    Necessity score: {sample.get('necessity_score', 0):.2f}")
        
        time.sleep(0.01)  # Small delay
    
    print(f"\n\nTotal events processed: {processor.processed_count}")