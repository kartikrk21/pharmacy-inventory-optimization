import json
import time
import threading
import logging
from kafka import KafkaConsumer
from utils.runtime_state import STATE, lock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrescriptionStreamProcessor:
    def __init__(
        self,
        socketio,
        topic="prescriptions",
        bootstrap_servers="localhost:29093",
        group_id="pharmacy-consumer-group",
    ):
        self.socketio = socketio
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.consumer = None

        threading.Thread(target=self._reset_rate_loop, daemon=True).start()

    def _reset_rate_loop(self):
        while True:
            time.sleep(1)
            with lock:
                STATE["stream_rate"] = 0

    def connect(self):
        self.consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=[self.bootstrap_servers],
            group_id=self.group_id,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
        )
        logger.info("Kafka consumer connected")

    def run(self):
        self.connect()
        logger.info("Consumer running...")

        for msg in self.consumer:
            data = msg.value

            with lock:
                STATE["total_prescriptions"] += 1
                STATE["stream_rate"] += 1

                med = data["medicine_id"]
                qty = data["quantity"]
                inv = STATE["inventory"].setdefault(med, {
                                    "medicine_id": med,
                                    "category": data.get("category", "Unknown"),
                                    "total_quantity": 0,
                                    "count": 0,
                                })

                inv["total_quantity"] += qty
                inv["count"] += 1
                inv["avg_per_rx"] = round(inv["total_quantity"] / inv["count"], 2)


                payload = {
                    "total_prescriptions": STATE["total_prescriptions"],
                    "stream_rate": STATE["stream_rate"],
                    "inventory": STATE["inventory"]
                }

            # 🔥 THIS IS WHAT YOU WERE MISSING
            self.socketio.emit("state_update", payload)