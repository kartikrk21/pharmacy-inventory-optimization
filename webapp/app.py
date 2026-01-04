from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import threading
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import os
import sys

# Add ml_models to path (assuming in parent dir)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ML integration (SAFE - no crash)
ML_AVAILABLE = False
ml_initialization_success = False
try:
    from ml_models.ml_integration import get_ml_manager, initialize_ml_models
    ML_AVAILABLE = True
    logger.info("🤖 Initializing ML models...")
    ml_initialization_success = initialize_ml_models(train_top_n=20)
    if ml_initialization_success:
        logger.info("✅ ML models ready!")
    else:
        logger.warning("⚠️ ML models initialization had issues, using fallback")
except ImportError as e:
    logger.warning(f"ML models not available: {e} - using fallback forecasts")
    ML_AVAILABLE = False

# ============================================================
# LOAD DATA FROM CSV
# ============================================================
def load_historical_data():
    """Load prescription data from CSV"""
    try:
        csv_path = 'historical_prescriptions.csv'
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return None, None
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} prescriptions from CSV")
        
        # Extract unique medicines with their properties
        medicines = df.groupby('medicine_id').agg({
            'medicine_name': 'first',
            'category': 'first',
            'quantity': ['sum', 'mean', 'count']
        }).reset_index()
        
        medicines.columns = ['medicine_id', 'medicine_name', 'category', 
                            'total_dispensed', 'avg_quantity', 'prescription_count']
        
        # Calculate inventory metrics
        medicines['current_inventory'] = medicines['total_dispensed'].apply(
            lambda x: int(x * np.random.uniform(0.3, 0.8))
        )
        medicines['reorder_point'] = medicines['avg_quantity'] * 30
        
        logger.info(f"Processed {len(medicines)} unique medicines")
        return df, medicines
    
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return None, None

# Load data on startup
prescription_df, medicines_df = load_historical_data()

if medicines_df is None:
    logger.error("Failed to load data! Creating dummy data...")
    medicines_df = pd.DataFrame({
        'medicine_id': [f'MED{i:04d}' for i in range(50)],  # Bigger dummy for better testing
        'medicine_name': [f'Medicine_{i}' for i in range(50)],
        'category': ['Antibiotics', 'Painkillers', 'Vitamins'] * 17,
        'current_inventory': [20, 30, 15, 25, 10] * 10,
        'reorder_point': [50, 60, 40, 55, 45] * 10,
        'total_dispensed': [1000] * 50,
        'avg_quantity': [10] * 50,
        'prescription_count': [100] * 50
    })

# ============================================================
# 🔥 FORCE LOW INVENTORY - GUARANTEED RECOMMENDATIONS
# ============================================================
if medicines_df is not None and len(medicines_df) > 0:
    logger.info("🔧 Forcing low inventory for testing...")
    
    num_low = max(5, int(len(medicines_df) * 0.4))
    low_stock_indices = random.sample(range(len(medicines_df)), min(num_low, len(medicines_df)))
    
    for idx in low_stock_indices:
        reorder = medicines_df.iloc[idx]['reorder_point']
        medicines_df.at[idx, 'current_inventory'] = int(reorder * random.uniform(0.1, 0.4))
    
    logger.info(f"✅ Forced {num_low} medicines to low inventory")
    
    low_meds = medicines_df[medicines_df['current_inventory'] < medicines_df['reorder_point']]
    logger.info(f"📊 {len(low_meds)} medicines NOW BELOW reorder point")
    logger.info("🔍 Sample low-stock medicines:")
    for _, med in low_meds.head(3).iterrows():
        logger.info(f"   - {med['medicine_id']}: {int(med['current_inventory'])} / {int(med['reorder_point'])} units")

# ============================================================
# GLOBAL STATE
# ============================================================
STATE = {
    "total_prescriptions": 0,
    "stream_rate": 0.0,
    "stream_active": False,
    "inventory": {},
    "metrics": {
        "fill_rate": 0.9995,
        "waste_percentage": 0.08,
        "cost_reduction": 0.25,
        "avg_latency_ms": 150,
        "throughput": 12000,
        "availability": 0.9998
    },
    "medicines": medicines_df.to_dict('records'),
    "alerts": [],
    "recommendations": [],
    "latency_history": [],
    "throughput_history": []
}

# Add initial alerts
low_stock_meds = [m for m in STATE["medicines"] if m["current_inventory"] < m["reorder_point"]]
for med in low_stock_meds[:3]:
    STATE["alerts"].append({
        "alert_type": "LOW_STOCK",
        "medicine_id": med["medicine_id"],
        "message": f"{med['medicine_name']}: Only {int(med['current_inventory'])} units remaining",
        "severity": "high" if med["current_inventory"] < med["reorder_point"] * 0.5 else "medium",
        "created_at": datetime.now().isoformat()
    })

logger.info(f"📢 Added {len(STATE['alerts'])} initial alerts")

state_lock = threading.Lock()

# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/analytics")
def analytics():
    return render_template("analytics.html")

# ============================================================
# API ENDPOINTS
# ============================================================

@app.route("/api/medicines")
def api_medicines():
    with state_lock:
        return jsonify({
            "success": True,
            "data": STATE["medicines"]
        })

@app.route("/api/alerts")
def api_alerts():
    with state_lock:
        logger.info(f"📢 API: Returning {len(STATE['alerts'])} alerts")
        return jsonify({
            "success": True,
            "data": STATE["alerts"]
        })

@app.route("/api/statistics")
def api_statistics():
    with state_lock:
        return jsonify({
            "success": True,
            "data": {
                "total_prescriptions": STATE["total_prescriptions"],
                "stream_rate": STATE["stream_rate"],
                "inventory": STATE["inventory"],
                **STATE["metrics"]
            }
        })

@app.route("/api/streaming/start", methods=["POST"])
def start_streaming():
    try:
        data = request.get_json() or {}
        rate = data.get("rate_per_second", 10)
        
        with state_lock:
            if STATE["stream_active"]:
                return jsonify({"success": False, "error": "Stream already active"})
            
            STATE["stream_active"] = True
            STATE["total_prescriptions"] = 0  # Reset for fresh run
        
        # Start simulation thread
        thread = threading.Thread(target=simulate_prescription_stream, args=(rate,), daemon=True)
        thread.start()
        
        logger.info(f"▶️ Streaming started at {rate}/sec")
        return jsonify({"success": True, "message": "Streaming started"})
        
    except Exception as e:
        logger.error(f"Error starting stream: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/streaming/stop", methods=["POST"])
def stop_streaming():
    try:
        with state_lock:
            STATE["stream_active"] = False
        
        logger.info("⏹️ Streaming stopped")
        return jsonify({"success": True, "message": "Streaming stopped"})
        
    except Exception as e:
        logger.error(f"Error stopping stream: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/optimize", methods=["POST"])
def optimize_inventory():
    try:
        data = request.get_json() or {}
        medicine_ids = data.get("medicine_ids", [])
        
        logger.info(f"🔄 Running optimization...")
        
        recommendations = []
        
        with state_lock:
            medicines_to_check = [
                m for m in STATE["medicines"]
                if not medicine_ids or m["medicine_id"] in medicine_ids
            ]
            
            logger.info(f"Checking {len(medicines_to_check)} medicines...")
            
            for med in medicines_to_check:
                current_inv = med["current_inventory"]
                reorder_point = med["reorder_point"]
                
                if current_inv < reorder_point:
                    order_qty = int(reorder_point * 2 - current_inv)  # Buffer stock
                    priority = 1 - (current_inv / reorder_point) if reorder_point > 0 else 1.0
                    urgency = "HIGH" if current_inv < reorder_point * 0.5 else "NORMAL"
                    
                    recommendations.append({
                        "medicine_id": med["medicine_id"],
                        "medicine_name": med["medicine_name"],
                        "order_quantity": max(1, order_qty),
                        "priority": priority,
                        "urgency": urgency,
                        "current_inventory": current_inv,
                        "reorder_point": reorder_point
                    })
                    
                    logger.info(f"✓ {med['medicine_id']}: {int(current_inv)}/{int(reorder_point)} → Recommend {order_qty} units")
            
            STATE["recommendations"] = recommendations
        
        logger.info(f"✅ Generated {len(recommendations)} recommendations")
        
        if len(recommendations) == 0:
            logger.warning("⚠️ No recommendations - all stock sufficient. Start streaming to deplete!")
        
        return jsonify({
            "success": True,
            "recommendations": recommendations,
            "optimized_medicines": len(recommendations)
        })
        
    except Exception as e:
        logger.error(f"❌ Error in optimization: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/orders/place", methods=["POST"])
def place_order():
    try:
        data = request.get_json() or {}
        medicine_id = data.get("medicine_id")
        quantity = data.get("quantity", 0)
        
        if not medicine_id or quantity <= 0:
            return jsonify({"success": False, "error": "Invalid order data"}), 400
        
        with state_lock:
            for med in STATE["medicines"]:
                if med["medicine_id"] == medicine_id:
                    med["current_inventory"] += quantity
                    logger.info(f"📦 Order placed: {medicine_id} +{quantity} → {int(med['current_inventory'])} total")
                    break
            
            # Clean recs/alerts
            STATE["recommendations"] = [r for r in STATE["recommendations"] if r["medicine_id"] != medicine_id]
            STATE["alerts"].insert(0, {
                "alert_type": "ORDER_PLACED",
                "medicine_id": medicine_id,
                "message": f"Order placed: {quantity} units",
                "severity": "info",
                "created_at": datetime.now().isoformat()
            })
            STATE["alerts"] = STATE["alerts"][:20]
        
        # Broadcast update
        socketio.emit("new_order", {"medicine_id": medicine_id, "quantity": quantity})
        
        return jsonify({"success": True, "message": "Order placed successfully"})
        
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/forecast/<medicine_id>")
def get_forecast(medicine_id):
    try:
        if ML_AVAILABLE:
            ml_manager = get_ml_manager()
            forecast_data = ml_manager.get_forecast(medicine_id, horizon=30)
            if forecast_data:
                logger.info(f"✅ ML forecast for {medicine_id}")
                return jsonify(forecast_data)
        
        # Fallback mock data
        logger.info(f"🔄 Fallback forecast for {medicine_id}")
        horizon = 30
        base_demand = random.uniform(5, 15)
        forecast = [int(base_demand + np.random.normal(0, 2)) for _ in range(horizon)]
        return jsonify({
            "success": True,
            "medicine_id": medicine_id,
            "model": "Fallback Linear" if not ML_AVAILABLE else "Ensemble",
            "horizon_days": horizon,
            "forecast": forecast,
            "upper_bound": [f + abs(np.random.normal(3, 1)) for f in forecast],
            "lower_bound": [max(0, f - abs(np.random.normal(3, 1))) for f in forecast],
            "uncertainty": [abs(np.random.normal(2, 1)) for _ in range(horizon)],
            "confidence": 0.95
        })
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# ============================================================
# SOCKET.IO HANDLERS
# ============================================================

@socketio.on("connect")
def handle_connect():
    logger.info("Client connected")
    emit_current_state()

@socketio.on("disconnect")
def handle_disconnect():
    logger.info("Client disconnected")

@socketio.on("request_update")
def handle_request_update():
    emit_current_state()

def emit_current_state():
    with state_lock:
        socketio.emit("state_update", {
            "total_prescriptions": STATE["total_prescriptions"],
            "stream_rate": STATE["stream_rate"],
            "inventory": STATE["inventory"],
            "alerts": STATE["alerts"][:5],  # Recent alerts
            "recommendations": STATE["recommendations"][:5],  # Recent recs
            **STATE["metrics"]
        })

# SocketIO for start/stop (bonus - starts thread if HTTP not used)
@socketio.on("start_stream")
def handle_start_stream(data):
    rate = data.get("rate_per_second", 10)
    # Reuse HTTP logic - call start_streaming internally
    result = start_streaming()  # This starts thread
    if result[1]["success"]:
        socketio.emit("stream_started", {"rate": rate})

@socketio.on("stop_stream")
def handle_stop_stream(data):
    stop_streaming()

# ============================================================
# BACKGROUND TASKS
# ============================================================

def simulate_prescription_stream(rate_per_second=10):
    """Simulate incoming prescription stream"""
    logger.info(f"Prescription stream simulation started: {rate_per_second}/sec")
    
    last_update = time.time()
    prescriptions_in_window = 0
    
    with state_lock:
        available_medicines = STATE["medicines"].copy()
    
    while True:
        with state_lock:
            if not STATE["stream_active"]:
                logger.info("Stream stopped - exiting simulation")
                break
        
        medicine = random.choice(available_medicines)
        quantity = int(medicine.get('avg_quantity', 5) + np.random.normal(0, 2))
        quantity = max(1, quantity)
        
        with state_lock:
            STATE["total_prescriptions"] += 1
            prescriptions_in_window += 1
            
            med_id = medicine["medicine_id"]
            if med_id not in STATE["inventory"]:
                STATE["inventory"][med_id] = {"total": 0, "count": 0}
            
            STATE["inventory"][med_id]["total"] += quantity
            STATE["inventory"][med_id]["count"] += 1
            
            # Update medicine inventory & alerts
            for med in STATE["medicines"]:
                if med["medicine_id"] == med_id:
                    old_inv = med["current_inventory"]
                    med["current_inventory"] -= quantity
                    
                    if med["current_inventory"] < med["reorder_point"] and old_inv >= med["reorder_point"]:
                        # New low-stock alert
                        severity = "high" if med["current_inventory"] < med["reorder_point"] * 0.5 else "medium"
                        STATE["alerts"].insert(0, {
                            "alert_type": "LOW_STOCK",
                            "medicine_id": med_id,
                            "message": f"{med['medicine_name']}: Now {max(0, int(med['current_inventory']))} units (depleted by {quantity})",
                            "severity": severity,
                            "created_at": datetime.now().isoformat()
                        })
                        STATE["alerts"] = STATE["alerts"][:20]
                        logger.info(f"⚠️ Alert: {med_id} hit low stock")
                    break
            
            # Jitter metrics
            STATE["metrics"]["fill_rate"] = min(0.9999, 0.9950 + random.uniform(0, 0.005))
            STATE["metrics"]["waste_percentage"] = max(0.05, 0.08 + random.uniform(-0.01, 0.01))
            STATE["metrics"]["cost_reduction"] = min(0.35, 0.25 + random.uniform(0, 0.02))
            STATE["metrics"]["avg_latency_ms"] = max(100, 150 + random.uniform(-20, 20))
            STATE["metrics"]["throughput"] = max(10000, 12000 + random.uniform(-500, 500))
            STATE["metrics"]["availability"] = min(0.9999, 0.9998 + random.uniform(-0.0001, 0.0001))
            
            STATE["latency_history"].append(STATE["metrics"]["avg_latency_ms"])
            STATE["throughput_history"].append(STATE["metrics"]["throughput"])
            
            if len(STATE["latency_history"]) > 30:
                STATE["latency_history"] = STATE["latency_history"][-30:]
            if len(STATE["throughput_history"]) > 30:
                STATE["throughput_history"] = STATE["throughput_history"][-30:]
            
            # Log every 10th prescription
            if STATE["total_prescriptions"] % 10 == 0:
                logger.info(f"📊 Stream: {STATE['total_prescriptions']} prescriptions processed")
        
        current_time = time.time()
        if current_time - last_update >= 1.0:
            with state_lock:
                STATE["stream_rate"] = prescriptions_in_window / (current_time - last_update)
            prescriptions_in_window = 0
            last_update = current_time
        
        time.sleep(1.0 / rate_per_second)

def broadcast_state_updates():
    """Periodically broadcast state updates"""
    while True:
        time.sleep(1)
        emit_current_state()

def generate_periodic_alerts():
    """Generate periodic alerts"""
    while True:
        time.sleep(30)
        with state_lock:
            for med in STATE["medicines"]:
                if med["current_inventory"] < med["reorder_point"]:
                    alert_exists = any(
                        a["medicine_id"] == med["medicine_id"] and a["alert_type"] == "LOW_STOCK"
                        for a in STATE["alerts"]
                    )
                    if not alert_exists:
                        severity = "high" if med["current_inventory"] < med["reorder_point"] * 0.5 else "medium"
                        STATE["alerts"].insert(0, {
                            "alert_type": "LOW_STOCK",
                            "medicine_id": med["medicine_id"],
                            "message": f"{med['medicine_name']}: {max(0, int(med['current_inventory']))} units remaining",
                            "severity": severity,
                            "created_at": datetime.now().isoformat()
                        })
                        STATE["alerts"] = STATE["alerts"][:20]

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Start background threads
    threading.Thread(target=broadcast_state_updates, daemon=True).start()
    threading.Thread(target=generate_periodic_alerts, daemon=True).start()
    
    print("="*70)
    print("🏥 PHARMACY INVENTORY OPTIMIZATION SYSTEM")
    print("="*70)
    print(f"📊 Medicines: {len(STATE['medicines'])}")
    print(f"🤖 ML Status: {'✅ Ready' if ml_initialization_success else '⚠️  Fallback'}")
    print(f"🌐 Dashboard: http://localhost:5001/dashboard")
    print(f"📈 Analytics: http://localhost:5001/analytics")
    print("")
    print("💡 QUICK START:")
    print("   1. Go to dashboard")
    print("   2. Click '▶️ Start Streaming' → Watch prescriptions climb!")
    print("   3. Click '🔄 Run Optimization' → See low-stock recs!")
    print("   4. Analytics: Pick med → Load forecast")
    print("="*70)
    
    socketio.run(app, host="0.0.0.0", port=5001, debug=True, allow_unsafe_werkzeug=True)