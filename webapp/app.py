from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import threading
import time
import random
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ============================================================
# GLOBAL STATE - Dynamic Runtime State
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
    "medicines": [],
    "alerts": [],
    "recommendations": [],
    "latency_history": [],
    "throughput_history": []
}

# Lock for thread safety
state_lock = threading.Lock()

# ============================================================
# SAMPLE MEDICINES DATA
# ============================================================
SAMPLE_MEDICINES = [
    {"medicine_id": "MED001", "medicine_name": "Amoxicillin", "category": "Antibiotics", 
     "current_inventory": 250, "reorder_point": 100, "unit_cost": 5.50, "shelf_life": 730},
    {"medicine_id": "MED002", "medicine_name": "Ibuprofen", "category": "Analgesics", 
     "current_inventory": 180, "reorder_point": 150, "unit_cost": 3.20, "shelf_life": 1095},
    {"medicine_id": "MED003", "medicine_name": "Lisinopril", "category": "Cardiovascular", 
     "current_inventory": 90, "reorder_point": 120, "unit_cost": 8.75, "shelf_life": 365},
    {"medicine_id": "MED004", "medicine_name": "Metformin", "category": "Diabetes", 
     "current_inventory": 320, "reorder_point": 200, "unit_cost": 4.30, "shelf_life": 730},
    {"medicine_id": "MED005", "medicine_name": "Amlodipine", "category": "Cardiovascular", 
     "current_inventory": 150, "reorder_point": 100, "unit_cost": 6.80, "shelf_life": 730},
    {"medicine_id": "MED006", "medicine_name": "Omeprazole", "category": "Gastrointestinal", 
     "current_inventory": 75, "reorder_point": 80, "unit_cost": 7.20, "shelf_life": 365},
    {"medicine_id": "MED007", "medicine_name": "Simvastatin", "category": "Cardiovascular", 
     "current_inventory": 200, "reorder_point": 150, "unit_cost": 5.90, "shelf_life": 730},
    {"medicine_id": "MED008", "medicine_name": "Levothyroxine", "category": "Hormones", 
     "current_inventory": 180, "reorder_point": 100, "unit_cost": 9.40, "shelf_life": 365},
    {"medicine_id": "MED009", "medicine_name": "Azithromycin", "category": "Antibiotics", 
     "current_inventory": 60, "reorder_point": 70, "unit_cost": 12.50, "shelf_life": 365},
    {"medicine_id": "MED010", "medicine_name": "Albuterol", "category": "Respiratory", 
     "current_inventory": 140, "reorder_point": 90, "unit_cost": 15.30, "shelf_life": 730},
]

# Initialize state with sample data
STATE["medicines"] = SAMPLE_MEDICINES.copy()

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
                "metrics": STATE["metrics"]
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
            STATE["total_prescriptions"] = 0
            
        # Start streaming in background thread
        thread = threading.Thread(target=simulate_prescription_stream, args=(rate,), daemon=True)
        thread.start()
        
        logger.info(f"Streaming started at {rate}/sec")
        return jsonify({"success": True, "message": "Streaming started"})
        
    except Exception as e:
        logger.error(f"Error starting stream: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/streaming/stop", methods=["POST"])
def stop_streaming():
    try:
        with state_lock:
            STATE["stream_active"] = False
        
        logger.info("Streaming stopped")
        return jsonify({"success": True, "message": "Streaming stopped"})
        
    except Exception as e:
        logger.error(f"Error stopping stream: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/optimize", methods=["POST"])
def optimize_inventory():
    try:
        data = request.get_json() or {}
        medicine_ids = data.get("medicine_ids", [])
        
        # Generate recommendations using simple logic
        recommendations = []
        
        with state_lock:
            for med in STATE["medicines"]:
                if not medicine_ids or med["medicine_id"] in medicine_ids:
                    current_inv = med["current_inventory"]
                    reorder_point = med["reorder_point"]
                    
                    # Simple reorder logic
                    if current_inv < reorder_point:
                        order_qty = reorder_point * 2 - current_inv
                        priority = 1 - (current_inv / reorder_point)
                        urgency = "HIGH" if current_inv < reorder_point * 0.5 else "NORMAL"
                        
                        recommendations.append({
                            "medicine_id": med["medicine_id"],
                            "medicine_name": med["medicine_name"],
                            "order_quantity": order_qty,
                            "priority": priority,
                            "urgency": urgency,
                            "current_inventory": current_inv,
                            "reorder_point": reorder_point
                        })
            
            STATE["recommendations"] = recommendations
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return jsonify({
            "success": True,
            "recommendations": recommendations,
            "optimized_medicines": len(recommendations)
        })
        
    except Exception as e:
        logger.error(f"Error in optimization: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/orders/place", methods=["POST"])
def place_order():
    try:
        data = request.get_json()
        medicine_id = data.get("medicine_id")
        quantity = data.get("quantity", 0)
        
        with state_lock:
            # Update inventory
            for med in STATE["medicines"]:
                if med["medicine_id"] == medicine_id:
                    med["current_inventory"] += quantity
                    break
            
            # Remove from recommendations
            STATE["recommendations"] = [
                r for r in STATE["recommendations"] 
                if r["medicine_id"] != medicine_id
            ]
            
            # Add alert
            STATE["alerts"].insert(0, {
                "alert_type": "ORDER_PLACED",
                "medicine_id": medicine_id,
                "message": f"Order placed for {quantity} units",
                "severity": "info",
                "created_at": datetime.now().isoformat()
            })
            
            # Keep only last 20 alerts
            STATE["alerts"] = STATE["alerts"][:20]
        
        logger.info(f"Order placed: {medicine_id} x {quantity}")
        return jsonify({"success": True, "message": "Order placed successfully"})
        
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/forecast/<medicine_id>")
def get_forecast(medicine_id):
    try:
        # Generate synthetic forecast for demo
        horizon = 30
        base_demand = random.uniform(5, 20)
        
        forecast = []
        lower_bound = []
        upper_bound = []
        uncertainty = []
        
        for i in range(horizon):
            # Add trend and seasonality
            trend = base_demand + i * 0.1
            seasonal = 2 * np.sin(2 * np.pi * i / 7)  # Weekly pattern
            noise = random.gauss(0, 1)
            
            value = max(0, trend + seasonal + noise)
            unc = abs(random.gauss(2, 0.5))
            
            forecast.append(value)
            uncertainty.append(unc)
            lower_bound.append(max(0, value - 1.96 * unc))
            upper_bound.append(value + 1.96 * unc)
        
        return jsonify({
            "medicine_id": medicine_id,
            "horizon_days": horizon,
            "forecast": forecast,
            "uncertainty": uncertainty,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "model": "ensemble"
        })
        
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        return jsonify({"error": str(e)})

# ============================================================
# SOCKET.IO HANDLERS
# ============================================================

@socketio.on("connect")
def handle_connect():
    logger.info("Client connected")
    
    # Send initial state
    with state_lock:
        socketio.emit("state_update", {
            "total_prescriptions": STATE["total_prescriptions"],
            "stream_rate": STATE["stream_rate"],
            "inventory": STATE["inventory"],
            **STATE["metrics"]
        })

@socketio.on("disconnect")
def handle_disconnect():
    logger.info("Client disconnected")

@socketio.on("request_update")
def handle_request_update():
    with state_lock:
        socketio.emit("state_update", {
            "total_prescriptions": STATE["total_prescriptions"],
            "stream_rate": STATE["stream_rate"],
            "inventory": STATE["inventory"],
            **STATE["metrics"]
        })

# ============================================================
# BACKGROUND TASKS
# ============================================================

def simulate_prescription_stream(rate_per_second=10):
    """Simulate incoming prescription stream"""
    logger.info(f"Prescription stream simulation started: {rate_per_second}/sec")
    
    last_update = time.time()
    prescriptions_in_window = 0
    
    while True:
        with state_lock:
            if not STATE["stream_active"]:
                break
        
        # Simulate prescription
        medicine = random.choice(SAMPLE_MEDICINES)
        quantity = random.randint(1, 5)
        
        with state_lock:
            # Update prescription count
            STATE["total_prescriptions"] += 1
            prescriptions_in_window += 1
            
            # Update inventory tracking
            med_id = medicine["medicine_id"]
            if med_id not in STATE["inventory"]:
                STATE["inventory"][med_id] = {"total": 0, "count": 0}
            
            STATE["inventory"][med_id]["total"] += quantity
            STATE["inventory"][med_id]["count"] += 1
            
            # Update medicine inventory
            for med in STATE["medicines"]:
                if med["medicine_id"] == med_id:
                    med["current_inventory"] -= quantity
                    
                    # Generate alert if low
                    if med["current_inventory"] < med["reorder_point"] * 0.5:
                        alert_exists = any(
                            a["medicine_id"] == med_id and a["alert_type"] == "LOW_STOCK"
                            for a in STATE["alerts"]
                        )
                        
                        if not alert_exists:
                            STATE["alerts"].insert(0, {
                                "alert_type": "LOW_STOCK",
                                "medicine_id": med_id,
                                "message": f"Critical: Only {med['current_inventory']} units remaining",
                                "severity": "high",
                                "created_at": datetime.now().isoformat()
                            })
                            STATE["alerts"] = STATE["alerts"][:20]
                    break
            
            # Update metrics dynamically
            STATE["metrics"]["fill_rate"] = min(0.9999, 0.9950 + random.uniform(0, 0.005))
            STATE["metrics"]["waste_percentage"] = max(0.05, 0.08 + random.uniform(-0.01, 0.01))
            STATE["metrics"]["cost_reduction"] = min(0.35, 0.25 + random.uniform(0, 0.02))
            STATE["metrics"]["avg_latency_ms"] = max(100, 150 + random.uniform(-20, 20))
            STATE["metrics"]["throughput"] = max(10000, 12000 + random.uniform(-500, 500))
            STATE["metrics"]["availability"] = min(0.9999, 0.9998 + random.uniform(-0.0001, 0.0001))
            
            # Track latency and throughput for charts
            STATE["latency_history"].append(STATE["metrics"]["avg_latency_ms"])
            STATE["throughput_history"].append(STATE["metrics"]["throughput"])
            
            # Keep only last 30 points
            if len(STATE["latency_history"]) > 30:
                STATE["latency_history"] = STATE["latency_history"][-30:]
            if len(STATE["throughput_history"]) > 30:
                STATE["throughput_history"] = STATE["throughput_history"][-30:]
        
        # Calculate stream rate
        current_time = time.time()
        if current_time - last_update >= 1.0:
            with state_lock:
                STATE["stream_rate"] = prescriptions_in_window / (current_time - last_update)
            
            prescriptions_in_window = 0
            last_update = current_time
        
        # Sleep to maintain rate
        time.sleep(1.0 / rate_per_second)

def broadcast_state_updates():
    """Periodically broadcast state updates to all connected clients"""
    while True:
        time.sleep(1)
        
        with state_lock:
            # Send state update
            socketio.emit("state_update", {
                "total_prescriptions": STATE["total_prescriptions"],
                "stream_rate": STATE["stream_rate"],
                "inventory": STATE["inventory"],
                **STATE["metrics"]
            })
            
            # Send metrics update with chart data
            if len(STATE["latency_history"]) > 0:
                socketio.emit("metrics_update", {
                    "latency": STATE["latency_history"][-1],
                    "throughput": STATE["throughput_history"][-1]
                })

def generate_periodic_alerts():
    """Generate periodic alerts for demo purposes"""
    while True:
        time.sleep(30)  # Every 30 seconds
        
        with state_lock:
            # Check for low stock
            for med in STATE["medicines"]:
                if med["current_inventory"] < med["reorder_point"]:
                    alert_exists = any(
                        a["medicine_id"] == med["medicine_id"] and 
                        a["alert_type"] == "LOW_STOCK"
                        for a in STATE["alerts"]
                    )
                    
                    if not alert_exists:
                        severity = "high" if med["current_inventory"] < med["reorder_point"] * 0.5 else "medium"
                        
                        STATE["alerts"].insert(0, {
                            "alert_type": "LOW_STOCK",
                            "medicine_id": med["medicine_id"],
                            "message": f"{med['medicine_name']}: {med['current_inventory']} units remaining",
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
    
    logger.info("Starting Flask-SocketIO server...")
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)