"""
Shared runtime state - NO Flask, NO SocketIO imports
Pure Python dict with thread lock
"""
from threading import Lock
from datetime import datetime

STATE = {
    "streaming_active": False,
    "total_prescriptions": 0,
    "stream_rate": 0.0,
    "avg_latency_ms": 150,
    "throughput": 12000,
    "fill_rate": 0.9995,
    "waste_percentage": 0.08,
    "cost_reduction": 0.25,
    "availability": 0.9998,
    "inventory": {},
    "last_update": None,
    "start_time": None
}

lock = Lock()

def update_state(key, value):
    """Thread-safe state update"""
    with lock:
        STATE[key] = value
        STATE["last_update"] = datetime.now().isoformat()

def get_state():
    """Thread-safe state read"""
    with lock:
        return STATE.copy()

def increment_prescriptions():
    """Thread-safe increment"""
    with lock:
        STATE["total_prescriptions"] += 1
        
        # Calculate rate
        if STATE["start_time"]:
            elapsed = (datetime.now() - STATE["start_time"]).total_seconds()
            STATE["stream_rate"] = STATE["total_prescriptions"] / elapsed if elapsed > 0 else 0
        
        STATE["last_update"] = datetime.now().isoformat()

def update_inventory(medicine_id, quantity):
    """Thread-safe inventory update"""
    with lock:
        if medicine_id not in STATE["inventory"]:
            STATE["inventory"][medicine_id] = {"total": 0, "count": 0}
        
        STATE["inventory"][medicine_id]["total"] += quantity
        STATE["inventory"][medicine_id]["count"] += 1
        STATE["last_update"] = datetime.now().isoformat()

def reset_state():
    """Reset all counters"""
    with lock:
        STATE["total_prescriptions"] = 0
        STATE["stream_rate"] = 0.0
        STATE["inventory"] = {}
        STATE["start_time"] = datetime.now()
        STATE["last_update"] = datetime.now().isoformat()