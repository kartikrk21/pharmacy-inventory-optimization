from database.db_manager import DatabaseManager
from config.config import Config
from database.models import Medicine
from datetime import datetime
import random

db = DatabaseManager(Config.DATABASE_URL)
session = db.Session()

medicine = session.query(Medicine).first()
session.close()

if not medicine:
    raise RuntimeError("No medicines found in DB")

medicine_id = medicine.id

for _ in range(30):
    db.save_prescription({
        "medicine_id": medicine_id,
        "quantity": random.randint(1, 5),
        "timestamp": datetime.now().isoformat(),
        "location": "Store1"
    })

print("Inserted prescriptions")
