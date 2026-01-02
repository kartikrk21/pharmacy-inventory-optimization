"""
Utility helper functions
"""
from datetime import datetime, timedelta
import numpy as np
import hashlib

def format_currency(amount: float, currency='USD') -> str:
    """Format amount as currency"""
    symbols = {'USD': '$', 'EUR': '€', 'GBP': '£'}
    symbol = symbols.get(currency, '$')
    return f"{symbol}{amount:,.2f}"

def calculate_metrics(orders: list, prescriptions: list) -> dict:
    """Calculate system performance metrics"""
    total_orders = len(orders)
    total_prescriptions = len(prescriptions)
    
    # Fill rate
    filled = sum(1 for p in prescriptions if p.get('filled', True))
    fill_rate = filled / max(total_prescriptions, 1)
    
    # Average latency (simulated)
    latency = np.random.uniform(100, 300)
    
    return {
        'total_orders': total_orders,
        'total_prescriptions': total_prescriptions,
        'fill_rate': fill_rate,
        'avg_latency_ms': latency
    }

def validate_data(data: dict, required_fields: list) -> tuple:
    """Validate data has required fields"""
    missing = [f for f in required_fields if f not in data]
    
    if missing:
        return False, f"Missing fields: {', '.join(missing)}"
    
    return True, "Valid"

def generate_id(prefix: str = 'ID') -> str:
    """Generate unique ID"""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    hash_obj = hashlib.md5(timestamp.encode())
    hash_hex = hash_obj.hexdigest()[:8]
    return f"{prefix}{timestamp[-6:]}{hash_hex}"

def calculate_days_between(date1: str, date2: str) -> int:
    """Calculate days between two ISO format dates"""
    d1 = datetime.fromisoformat(date1)
    d2 = datetime.fromisoformat(date2)
    return abs((d2 - d1).days)

def moving_average(data: list, window: int = 7) -> list:
    """Calculate moving average"""
    if len(data) < window:
        return data
    
    result = []
    for i in range(len(data)):
        if i < window - 1:
            result.append(np.mean(data[:i+1]))
        else:
            result.append(np.mean(data[i-window+1:i+1]))
    
    return result

# Example usage
if __name__ == "__main__":
    print("\n=== SMDP Policy ===")
    smdp = SMDPPolicy()
    demand_dist = np.random.uniform(10, 50, 100)
    smdp.value_iteration(demand_dist)
    action = smdp.get_action(100)
    print(f"Optimal order for inventory=100: {action}")
    
    print("\n=== Batch Optimization ===")
    medicines = [
        {'medicine_id': 'MED001', 'unit_cost': 10, 'current_inventory': 50},
        {'medicine_id': 'MED002', 'unit_cost': 15, 'current_inventory': 30}
    ]
    forecasts = {
        'MED001': np.array([20, 22, 21, 23]),
        'MED002': np.array([15, 16, 15, 17])
    }
    
    optimizer = BatchOptimizer()
    results = optimizer.optimize(medicines, forecasts)
    print(f"Optimized {len(results)} orders")
    
    print("\n=== Helpers ===")
    print(format_currency(1234.56))
    print(generate_id('ORD'))