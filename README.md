# Insurance Cost Prediction — ZenML MLOps Pipeline

This repository contains a **production-grade end-to-end MLOps pipeline** built using **ZenML** for predicting insurance costs.

The project demonstrates how to design, train, evaluate, and register machine learning models using **best MLOps practices**, including reproducibility, modularity, and experiment tracking.

---

## Project Overview

The pipeline performs the following steps:

1. **Data Ingestion**
   - Loads and validates raw insurance data
   - Tracks dataset as a versioned artifact

2. **Data Preprocessing**
   - Train–test split
   - Numerical scaling using `StandardScaler`
   - Categorical encoding using `OneHotEncoder`
   - Implemented via `ColumnTransformer`

3. **Model Selection**
   - Automated model selection using **Optuna**
   - Compares:
     - Linear Regression
     - Random Forest Regressor
   - Uses cross-validated RMSE for optimization

4. **Model Training**
   - Trains the best model on processed data

5. **Model Evaluation**
   - Metrics:
     - RMSE
     - MAE
     - R² score

6. **Model Registration**
   - Registers model only if RMSE meets a defined quality threshold

All steps are orchestrated and tracked using **ZenML**.

---

## Tech Stack

- **Python 3.12**
- **ZenML**
- **Scikit-learn**
- **Optuna**
- **Pandas / NumPy**
- **Git & GitHub**

---
## Project Structure
mlops-insurance/
│
├── src/
│ ├── steps/
│ │ ├── data_loader.py
│ │ ├── data_preprocessor.py
│ │ ├── model_selection.py
│ │ ├── model_trainer.py
│ │ ├── model_evaluator.py
│ │ └── register_model.py
│ │
│ └── pipelines/
│ └── training_pipeline.py
│
├── data/
│ └── raw/
│ └── insurance.csv
│
├── config/
├── notebooks/
├── run.py
├── requirements.txt
└── README.md


---

## ▶️ How to Run the Pipeline

### Create and activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
zenml login --local
python run.py




