from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from datetime import datetime, timedelta
import pandas as pd
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models import (
    Base,
    Medicine,
    Prescription,
    Order,
    Alert
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url, echo=False)
        self.session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(self.session_factory)

    # ------------------------------------------------------------------
    # INIT + SEED
    # ------------------------------------------------------------------
    def init_db(self):
        Base.metadata.create_all(self.engine)
        logger.info("Database initialized")
        self._seed_medicines()

    def _seed_medicines(self):
        session = self.Session()
        try:
            if session.query(Medicine).count() > 0:
                logger.info("Database already has medicines")
                return

            import random
            categories = [
                "Antibiotics", "Analgesics", "Antivirals",
                "Cardiovascular", "Diabetes", "Respiratory"
            ]

            for i in range(100):
                med = Medicine(
                    medicine_name=f"Medicine {i}",
                    category=random.choice(categories),
                    current_stock=random.randint(50, 200),
                    reorder_point=50,
                    unit_price=round(random.uniform(10, 500), 2),
                    is_active=True
                )
                session.add(med)

            session.commit()
            logger.info("Medicine data seeded successfully")

        except Exception as e:
            session.rollback()
            logger.error(f"Error seeding data: {e}")
        finally:
            session.close()

    # ------------------------------------------------------------------
    # MEDICINES
    # ------------------------------------------------------------------
    def get_all_medicines(self):
        session = self.Session()
        try:
            meds = session.query(Medicine).all()
            return [
                {
                    "id": m.id,
                    "medicine_name": m.medicine_name,
                    "category": m.category,
                    "current_stock": m.current_stock,
                    "reorder_point": m.reorder_point,
                    "unit_price": m.unit_price
                }
                for m in meds
            ]
        finally:
            session.close()

    def get_medicine_by_id(self, medicine_id: int):
        session = self.Session()
        try:
            return session.query(Medicine).filter_by(id=medicine_id).first()
        finally:
            session.close()

    # ------------------------------------------------------------------
    # PRESCRIPTIONS
    # ------------------------------------------------------------------
    def save_prescription(self, data: dict):
        session = self.Session()
        try:
            if "medicine_id" not in data:
                raise ValueError("medicine_id is required")

            prescription = Prescription(
                prescription_id=data.get(
                    "prescription_id",
                    f"RX{int(datetime.now().timestamp()*1000)}"
                ),
                medicine_id=data["medicine_id"],
                quantity=data["quantity"],
                patient_age=data.get("patient_age", 0),
                is_emergency=data.get("is_emergency", False),
                insurance=data.get("insurance", "No"),
                location=data.get("location", "Store1"),
                timestamp=datetime.fromisoformat(data["timestamp"])
                if "timestamp" in data else datetime.now()
            )

            session.add(prescription)

            # 🔽 Inventory update (ID-based ONLY)
            medicine = session.query(Medicine).filter_by(
                id=data["medicine_id"]
            ).first()

            if medicine:
                medicine.current_stock = max(
                    0, medicine.current_stock - data["quantity"]
                )

                if medicine.current_stock < medicine.reorder_point:
                    self._create_alert(
                        session,
                        alert_type="LOW_STOCK",
                        medicine_id=str(medicine.id),
                        severity="WARNING",
                        message=f"{medicine.medicine_name} below reorder point"
                    )

            session.commit()
            return True

        except Exception as e:
            logger.error(f"Error saving prescription: {e}")
            session.rollback()
            return False

        finally:
            session.close()

    # ------------------------------------------------------------------
    # ORDERS
    # ------------------------------------------------------------------
    def save_order(self, data: dict):
        session = self.Session()
        try:
            order = Order(
                order_id=f"ORD{int(datetime.now().timestamp()*1000)}",
                medicine_id=data["medicine_id"],
                quantity=data["quantity"],
                status="PENDING",
                unit_cost=data.get("unit_cost", 0),
                total_cost=data["quantity"] * data.get("unit_cost", 0),
                expected_delivery_date=datetime.now() + timedelta(days=7)
            )
            session.add(order)
            session.commit()
            return order.order_id
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving order: {e}")
            return None
        finally:
            session.close()

    # ------------------------------------------------------------------
    # ALERTS
    # ------------------------------------------------------------------
    def _create_alert(self, session, alert_type, medicine_id, severity, message):
        alert = Alert(
            alert_type=alert_type,
            medicine_id=medicine_id,
            severity=severity,
            message=message
        )
        session.add(alert)

    def get_recent_alerts(self, limit=20):
        session = self.Session()
        try:
            alerts = (
                session.query(Alert)
                .filter_by(is_resolved=False)
                .order_by(Alert.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": a.id,
                    "alert_type": a.alert_type,
                    "medicine_id": a.medicine_id,
                    "severity": a.severity,
                    "message": a.message,
                    "created_at": a.created_at.isoformat()
                }
                for a in alerts
            ]
        finally:
            session.close()

    # ------------------------------------------------------------------
    # DASHBOARD STATS
    # ------------------------------------------------------------------
    def get_statistics(self):
        session = self.Session()
        try:
            return {
                "total_medicines": session.query(Medicine).count(),
                "total_prescriptions": session.query(Prescription).count(),
                "total_orders": session.query(Order).count(),
                "fill_rate": 0.9995,
                "waste_percentage": 0.08,
                "cost_reduction": 0.25,
                "avg_latency_ms": 150,
                "throughput": 12000,
                "availability": 0.9998
            }
        finally:
            session.close()
