from zenml import pipeline
from src.steps.data_loader import ingest_data
from src.steps.data_preprocessor import preprocess_data
from src.steps.model_selection import select_best_model
from src.steps.model_trainer import train_model
from src.steps.model_evaluator import evaluate_model
from src.steps.register_model import register_model


@pipeline
def training_pipeline():
    """
    End-to-end training pipeline
    """

    # 1. Ingest data
    df = ingest_data(file_path="data/raw/insurance.csv")

    # 2. Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # 3. Model selection (Optuna)
    best_model, best_params, best_score = select_best_model(
        X_train, y_train
    )

    # 4. Train final model
    trained_model = train_model(best_model, X_train, y_train)

    # 5. Evaluate model
    metrics = evaluate_model(trained_model, X_test, y_test)

    # 6. Register model
    register_model(trained_model, metrics)
