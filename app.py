import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------
# Load Model and Preprocessing Files
# -----------------------------
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model(
        "models/student_performance.keras"
    )
    scaler = joblib.load("models/scaler.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")

    return model, scaler, label_encoder


model, scaler, label_encoder = load_resources()

# -----------------------------
# Application Header
# -----------------------------
st.title("🎓 AI-Based Student Performance Predictor")
st.write(
    "Enter the student's academic and behavioral information "
    "to predict their expected performance category."
)

st.divider()

# -----------------------------
# Student Input Fields
# -----------------------------
st.subheader("Student Information")

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=20
)

attendance_pct = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

study_hours = st.number_input(
    "Study Hours per Week",
    min_value=0.0,
    value=10.0
)

previous_score = st.number_input(
    "Previous Score",
    min_value=0.0,
    max_value=100.0,
    value=60.0
)

assignment_completion = st.number_input(
    "Assignment Completion (%)",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

internal_assessment = st.number_input(
    "Internal Assessment Score",
    min_value=0.0,
    max_value=100.0,
    value=60.0
)

participation_score = st.number_input(
    "Participation Score",
    min_value=0.0,
    max_value=10.0,
    value=5.0
)

sleep_hours = st.number_input(
    "Sleep Hours per Day",
    min_value=0.0,
    max_value=24.0,
    value=7.0
)

extracurricular_hours = st.number_input(
    "Extracurricular Hours per Week",
    min_value=0.0,
    value=3.0
)

failed_subjects = st.number_input(
    "Failed Subjects",
    min_value=0,
    value=0,
    step=1
)

# -----------------------------
# Prediction
# -----------------------------
st.divider()

if st.button("🔮 Predict Performance", use_container_width=True):

    # IMPORTANT:
    # Feature order must be exactly the same as training data
    input_data = np.array([[
        age,
        attendance_pct,
        study_hours,
        previous_score,
        assignment_completion,
        internal_assessment,
        participation_score,
        sleep_hours,
        extracurricular_hours,
        failed_subjects
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Make prediction
    probabilities = model.predict(input_scaled, verbose=0)

    predicted_index = np.argmax(probabilities[0])
    predicted_class = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = probabilities[0][predicted_index] * 100

    # Display Result
    st.success(
        f"Predicted Performance: {predicted_class}"
    )

    st.metric(
        "Prediction Confidence",
        f"{confidence:.2f}%"
    )

    # Show all probabilities
    st.subheader("Class Probabilities")

    for class_name, probability in zip(
        label_encoder.classes_,
        probabilities[0]
    ):
        st.write(
            f"**{class_name}:** {probability * 100:.2f}%"
        )