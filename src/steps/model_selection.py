from abc import ABC, abstractmethod
from typing import Tuple, Dict
from sklearn.base import BaseEstimator
import pandas as pd
from zenml import step
from typing_extensions import Annotated
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import optuna

class ModelSelector(ABC):

    @abstractmethod
    def optimize(
        self, X_train, y_train
    ) -> Tuple[BaseEstimator, Dict, float]:
        """
        Returns:
        - best model
        - best hyperparameters
        - best CV score
        """
        pass


class OptunaModelSelector(ModelSelector):

    def optimize(self, X_train, y_train):

        def objective(trial):

            model_name = trial.suggest_categorical(
                "model", ["linear", "rf"]
            )

            if model_name == "linear":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(
                    n_estimators=trial.suggest_int("n_estimators", 50, 300),
                    max_depth=trial.suggest_int("max_depth", 3, 20),
                    random_state=42,
                )

            score = cross_val_score(
                model,
                X_train,
                y_train,
                cv=5,
                scoring="neg_root_mean_squared_error"
            ).mean()

            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)

        best_params = study.best_params

        if best_params["model"] == "linear":
            best_model = LinearRegression()
        else:
            best_model = RandomForestRegressor(
                **{k: v for k, v in best_params.items() if k != "model"},
                random_state=42
            )

        return best_model, best_params, study.best_value


@step
def select_best_model(
    X_train,
    y_train,
) -> Tuple[
    Annotated[BaseEstimator, "best_model"],
    Annotated[Dict, "best_params"],
    Annotated[float, "best_score"],
]:

    selector = OptunaModelSelector()
    return selector.optimize(X_train, y_train)
