# 🏥 Medical Insurance Cost Prediction

A professional MLOps project using ZenML to predict medical insurance costs based on patient information.

## 📋 Project Overview

This project builds a machine learning pipeline to predict insurance costs using features like:
- Age
- Sex
- BMI (Body Mass Index)
- Number of children
- Smoking status
- Region

## 🗂️ Project Structure

```
Medical_Insurance_Cost/
│
├── data/
│   ├── raw/                    # Original insurance.csv
│   └── processed/              # Processed data (auto-generated)
│
├── src/
│   ├── steps/                  # Individual pipeline steps
│   │   ├── data_loader.py      # Load data
│   │   ├── data_preprocessor.py # Preprocess data
│   │   ├── model_trainer.py    # Train model
│   │   └── model_evaluator.py  # Evaluate model
│   │
│   └── pipelines/
│       └── training_pipeline.py # Main pipeline
│
├── config/
│   └── config.yaml             # Configuration file
│
├── notebooks/                  # Jupyter notebooks
├── run.py                      # Main execution script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd Medical_Insurance_Cost

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize ZenML
zenml init
```

### 2. Prepare Data

Place your `insurance.csv` file in the `data/raw/` folder:
```
data/raw/insurance.csv
```

### 3. Run Pipeline

```bash
python run.py
```

## 📊 Pipeline Steps

The pipeline consists of 4 main steps:

1. **Data Loading** (`data_loader.py`)
   - Loads insurance data from CSV
   - Validates data structure
   - Displays basic statistics

2. **Data Preprocessing** (`data_preprocessor.py`)
   - Handles missing values
   - Encodes categorical variables (sex, smoker, region)
   - Splits into train/test sets (80/20)
   - Scales numerical features

3. **Model Training** (`model_trainer.py`)
   - Trains Random Forest Regressor
   - Uses 100 trees with max depth of 10
   - Outputs training R² score

4. **Model Evaluation** (`model_evaluator.py`)
   - Calculates RMSE, MAE, R² Score, MAPE
   - Shows sample predictions
   - Provides performance interpretation

## 🎯 Model Performance

Expected metrics:
- **R² Score**: ~0.75-0.85 (model explains 75-85% of variance)
- **RMSE**: ~$4,000-$6,000 (average prediction error)
- **MAE**: ~$2,500-$4,000 (average absolute error)

## 🔧 Configuration

Edit `config/config.yaml` to change pipeline parameters:

```yaml
preprocessing:
  test_size: 0.2        # Change test set size
  
model:
  n_estimators: 100     # Number of trees
  max_depth: 10         # Tree depth
```

## 📈 Next Steps

1. **View Pipeline in ZenML Dashboard**
   ```bash
   zenml up
   ```

2. **Experiment with Hyperparameters**
   - Modify `config/config.yaml`
   - Try different models (XGBoost, Linear Regression)

3. **Feature Engineering**
   - Create new features (age groups, BMI categories)
   - Feature importance analysis

4. **Model Deployment**
   - Save model for inference
   - Create prediction API
   - Deploy to production

## 🛠️ Troubleshooting

### Common Issues

**1. Module not found error**
```bash
# Make sure you're in the project root
cd Medical_Insurance_Cost
python run.py
```

**2. Data file not found**
```bash
# Ensure data is in correct location
ls data/raw/insurance.csv
```

**3. ZenML not initialized**
```bash
zenml init
```

## 📚 Learn More

- [ZenML Documentation](https://docs.zenml.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [MLOps Best Practices](https://ml-ops.org/)

## 📝 License

This project is for educational purposes.

## 👤 Author

Created as an MLOps learning project using ZenML.