# Student Depression Prediction

An end-to-end machine learning system that predicts a student's depression risk based on academic, lifestyle, and demographic factors, deployed as a real-time Flask web application.

## Overview

Student mental health is shaped by a combination of academic pressure, sleep habits, diet, financial stress, and other lifestyle factors. This project implements a full ML pipeline:

1. **Data preprocessing** — cleaning and encoding a student mental health dataset
2. **Model training** — training and comparing four classification algorithms
3. **Model selection** — Logistic Regression selected for deployment based on strong AUC performance (~0.92) and interpretability
4. **Deployment** — a Flask web app serving real-time predictions through a simple form-based UI

## Features

- Web form collecting key inputs: age, gender, city, academic pressure, CGPA, sleep duration, dietary habits, degree, work/study hours, financial stress, and family history of mental illness
- Real-time depression risk prediction with a probability score
- Graceful handling of unseen categorical values (e.g. new cities) via safe label encoding, avoiding crashes on out-of-distribution input
- Server-side input validation for realistic ranges (age, CGPA, work/study hours)

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Machine Learning | scikit-learn (Logistic Regression), pandas, NumPy, joblib |
| Frontend | HTML, CSS |

## Project Structure

```
├── app.py                          # Flask application and prediction logic
├── depression_model.pkl            # Trained Logistic Regression model
├── student_depression_dataset.csv  # Training dataset
├── static/
│   └── style.css                   # Styling
└── templates/
    └── index.html                  # Web form UI
```

## Getting Started

### Prerequisites
- Python 3.8+

### Installation

```bash
git clone https://github.com/ABHINABO/Analysing_Student_Depression_Using_Machine_Learning.git
cd Analysing_Student_Depression_Using_Machine_Learning
pip install flask pandas numpy scikit-learn joblib
```

### Run the app

```bash
python app.py
```

Then open `http://localhost:5000` in your browser.

## Model Details

Four classification models were trained and evaluated during development. Logistic Regression was selected for final deployment based on:
- **AUC ~0.92** on the held-out test set
- **Interpretability**, which is especially important for a sensitive use case like mental health screening, where understanding *why* a prediction was made matters as much as the prediction itself

## Interactive Model Explainer

An interactive breakdown of how the trained model computes its prediction — showing the real coefficients from `depression_model.pkl` and how each factor (academic pressure, financial stress, sleep, etc.) moves the score — is available at [`docs/logistic_regression_explainer.html`](docs/logistic_regression_explainer.html). Download it and open it in your browser to explore it (GitHub's file preview only shows the raw code, not the live interactive version).

## Disclaimer

This is an academic minor project built for educational purposes and **is not a diagnostic or clinical tool**. Predictions are based on a public dataset and general patterns, not individual clinical assessment. If you or someone you know is struggling with mental health, please reach out to a qualified mental health professional or a local support helpline.

## Team

- Abhinabo Acharjee
- Suveer Mishra

Supervised by Dr. Yang Saring, NIT Arunachal Pradesh
