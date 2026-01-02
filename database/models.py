"""
SQLAlchemy database models
Defines the schema for all database tables
"""
from sqlalchemy import (
    Column, Integer, String, Float, DateTime,
    Boolean, Text, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


# =========================
# Medicine
# =========================
class Medicine(Base):
    __tablename__ = "medicines"

    id = Column(Integer, primary_key=True, autoincrement=True)

    medicine_name = Column(String, nullable=False, unique=True)
    category = Column(String, nullable=False)

    current_stock = Column(Integer, default=0)
    reorder_point = Column(Integer, default=50)

    unit_price = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    prescriptions = relationship("Prescription", back_populates="medicine")
    orders = relationship("Order", back_populates="medicine")

    def __repr__(self):
        return f"<Medicine(name={self.medicine_name}, stock={self.current_stock})>"


# =========================
# Prescription
# =========================
class Prescription(Base):
    __tablename__ = "prescriptions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prescription_id = Column(String(100), unique=True, nullable=False, index=True)

    medicine_id = Column(Integer, ForeignKey("medicines.id"), nullable=False, index=True)

    quantity = Column(Integer, nullable=False)
    patient_age = Column(Integer)
    is_emergency = Column(Boolean, default=False)
    insurance = Column(String(20))
    location = Column(String(100))
    day_of_week = Column(String(20))
    hour_of_day = Column(Integer)

    timestamp = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    medicine = relationship("Medicine", back_populates="prescriptions")

    def __repr__(self):
        return f"<Prescription(rx={self.prescription_id}, med={self.medicine_id})>"


# =========================
# Order
# =========================
class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(100), unique=True, nullable=False, index=True)

    medicine_id = Column(Integer, ForeignKey("medicines.id"), nullable=False, index=True)

    quantity = Column(Float, nullable=False)
    status = Column(String(50), default="PENDING", index=True)
    priority = Column(String(20), default="NORMAL")

    unit_cost = Column(Float)
    total_cost = Column(Float)
    supplier = Column(String(200))

    expected_delivery_date = Column(DateTime)
    actual_delivery_date = Column(DateTime)

    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(100))

    medicine = relationship("Medicine", back_populates="orders")

    def __repr__(self):
        return f"<Order(order={self.order_id}, status={self.status})>"


# =========================
# Optimization Result
# =========================
class OptimizationResult(Base):
    __tablename__ = "optimization_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    medicine_id = Column(Integer, index=True)

    optimization_type = Column(String(50))
    current_inventory = Column(Float)
    safety_stock = Column(Float)
    reorder_point = Column(Float)
    economic_order_quantity = Column(Float)
    optimal_order_quantity = Column(Float)
    total_expected_cost = Column(Float)

    priority = Column(Float)
    urgency = Column(String(50))
    rationale = Column(Text)

    is_executed = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


# =========================
# Alert
# =========================
class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_type = Column(String(50), index=True)

    medicine_id = Column(Integer, index=True)
    severity = Column(String(20), default="INFO")

    message = Column(Text, nullable=False)
    is_resolved = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)


# =========================
# Performance Metric
# =========================
class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_type = Column(String(50), index=True)
    metric_value = Column(Float, nullable=False)
    target_value = Column(Float)
    unit = Column(String(20))
    timestamp = Column(DateTime, default=datetime.utcnow)


# =========================
# Forecast
# =========================
class Forecast(Base):
    __tablename__ = "forecasts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    medicine_id = Column(Integer, index=True)

    forecast_date = Column(DateTime, nullable=False)
    forecast_quantity = Column(Float, nullable=False)

    lower_bound = Column(Float)
    upper_bound = Column(Float)
    model_type = Column(String(50))
    confidence_level = Column(Float, default=0.95)

    created_at = Column(DateTime, default=datetime.utcnow)


# =========================
# Inventory Transaction
# =========================
class InventoryTransaction(Base):
    __tablename__ = "inventory_transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(100), unique=True, nullable=False)

    medicine_id = Column(Integer, index=True)
    transaction_type = Column(String(50))
    quantity_change = Column(Float)

    inventory_before = Column(Float)
    inventory_after = Column(Float)

    reference_id = Column(String(100))
    timestamp = Column(DateTime, default=datetime.utcnow)


# =========================
# User
# =========================
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default="USER")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
