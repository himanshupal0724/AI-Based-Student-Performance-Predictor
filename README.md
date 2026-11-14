# 🎓 AI-Based Student Performance Risk Predictor

## 📌 Project Overview

The AI-Based Student Performance Risk Predictor is a Machine Learning and Deep Learning project that predicts a student's performance category using academic and behavioral information.

The system uses an **Artificial Neural Network (ANN)** to classify students into performance categories:

- Low
- Medium
- High

The project also includes a Logistic Regression baseline model for comparison, a Streamlit prediction application, and a FastAPI-based prediction API.

---

## 🎯 Project Objective

The objective of this project is to build and evaluate an Artificial Neural Network that learns patterns from student-related data and predicts the expected student performance category.

The project demonstrates the complete Machine Learning workflow:

1. Data Understanding
2. Data Preprocessing
3. Baseline Model Development
4. ANN Design and Training
5. Model Evaluation
6. Model Saving
7. Prediction Application
8. API Development

---

## 📊 Dataset Features

The model uses the following input features:

- Age
- Attendance Percentage
- Study Hours per Week
- Previous Score
- Assignment Completion Percentage
- Internal Assessment Score
- Participation Score
- Sleep Hours per Day
- Extracurricular Hours per Week
- Failed Subjects

The target variable is:

**Performance Category**

The possible prediction classes are:

**Low | Medium | High**

The `student_id` column is excluded from training because it is only an identifier.

---

## 🧠 ANN Architecture

The project uses a Feed-Forward Artificial Neural Network with the following architecture:

```text
Input Layer
     ↓
Dense Layer (32 neurons, ReLU)
     ↓
Dense Layer (16 neurons, ReLU)
     ↓
Output Layer (3 neurons, Softmax)
     ↓
Low / Medium / High
```

The ANN uses:

- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Output Activation:** Softmax

---

## ⚙️ Data Preprocessing

The following preprocessing steps are performed:

- Checked missing values and duplicate records
- Removed the `student_id` column
- Separated input features and target variable
- Encoded target labels using `LabelEncoder`
- Split the dataset into training and testing data
- Applied `StandardScaler` for feature scaling

The same scaler is saved and reused during prediction to ensure consistency.

---

## 📈 Model Evaluation

The ANN model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix

A Logistic Regression model is also used as a baseline for comparison with the ANN.

---

## 🖥️ Streamlit Application

The Streamlit application allows users to enter student information and receive:

- Predicted Performance Category
- Prediction Confidence
- Probability for each performance class

### Run the application

```bash
streamlit run app.py
```

---

## 🔌 FastAPI Prediction API

The project includes a FastAPI endpoint for model predictions.

### Run the API

```bash
uvicorn api:app --reload
```

After starting the API, open:

`http://127.0.0.1:8000/docs`

Use the interactive documentation to test the `/predict` endpoint.

---

## 📁 Project Structure

```text
student-performance-ann/
│
├── data/
│   └── student_data.csv
│
├── models/
│   ├── student_performance.keras
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── notebooks/
│   └── ann_training.ipynb
│
├── app.py
├── api.py
├── requirements.txt
└── README.md
```

---

## 📦 Installation

Clone or download the project and install the required packages:

```bash
pip install -r requirements.txt
```

---

## 🚀 Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- FastAPI
- Uvicorn
- Matplotlib
- Seaborn

---

## 👨‍💻 Author

**Himanshu Pal**

BCA Student | AI & Machine Learning