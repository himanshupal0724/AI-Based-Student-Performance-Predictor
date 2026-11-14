from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf

# Create FastAPI application
app = FastAPI(
    title="Student Performance Prediction API",
    description="API for predicting student performance using ANN"
)

# Load trained model and preprocessing files
model = tf.keras.models.load_model(
    "models/student_performance.keras"
)

scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")


# Define input structure
class StudentData(BaseModel):
    age: float
    attendance_pct: float
    study_hours_per_week: float
    previous_score: float
    assignment_completion_pct: float
    internal_assessment_score: float
    participation_score: float
    sleep_hours_per_day: float
    extracurricular_hours_per_week: float
    failed_subjects: int


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Student Performance Prediction API is running!"
    }


# Prediction endpoint
@app.post("/predict")
def predict(student: StudentData):

    # Input features in the SAME order used during training
    input_data = np.array([[
        student.age,
        student.attendance_pct,
        student.study_hours_per_week,
        student.previous_score,
        student.assignment_completion_pct,
        student.internal_assessment_score,
        student.participation_score,
        student.sleep_hours_per_day,
        student.extracurricular_hours_per_week,
        student.failed_subjects
    ]])

    # Scale input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    probabilities = model.predict(input_scaled, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = float(probabilities[predicted_index] * 100)

    # Return JSON response
    return {
        "predicted_performance": predicted_class,
        "confidence": round(confidence, 2),
        "probabilities": {
            class_name: round(float(probability * 100), 2)
            for class_name, probability in zip(
                label_encoder.classes_,
                probabilities
            )
        }
    }