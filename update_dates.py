from app import db, Patient
from datetime import datetime

patients = Patient.query.all()

for patient in patients:
    if isinstance(patient.date_diagnosed, str):
        patient.date_diagnosed = datetime.fromisoformat(patient.date_diagnosed).date()

db.session.commit()
