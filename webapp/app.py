from flask import Flask, render_template, jsonify
import threading
import time
from utils.runtime_state import STATE, lock
from utils.runtime_state import STATE
from kafka_streaming.consumer import PrescriptionStreamProcessor
from flask_socketio import SocketIO




app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/api/statistics")
def api_statistics():
    return jsonify({
        "success": True,
        "data": {
            "total_prescriptions": STATE["total_prescriptions"],
            "stream_rate": STATE["stream_rate"],
            "inventory": STATE["inventory"]
        }
    })


@socketio.on("connect")
def handle_connect():
    print("Client connected")


def start_kafka_consumer():
    consumer = PrescriptionStreamProcessor(socketio)
    consumer.run()

threading.Thread(
    target=start_kafka_consumer,
    daemon=True
).start()



def broadcast_state():
    while True:
        time.sleep(1)
        with lock:
            socketio.emit("state_update", {
                "total_prescriptions": STATE["total_prescriptions"],
                "stream_rate": STATE["stream_rate"],
                "inventory": STATE["inventory"]
            })

if __name__ == "__main__":
    threading.Thread(
        target=start_kafka_consumer,
        daemon=True
    ).start()

    socketio.run(app, host="0.0.0.0", port=5001)

