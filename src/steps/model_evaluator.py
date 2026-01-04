from abc import ABC, abstractmethod
from typing import Dict
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from typing_extensions import Annotated
from zenml import step

import logging
logger = logging.getLogger(__name__)


class ModelEvaluator(ABC):

    @abstractmethod
    def evaluate(
        self,
        model: BaseEstimator,
        X_test,
        y_test
    ) -> Dict[str, float]:
        """
        Evaluate model on test data and return metrics
        """
        pass

class RegressionEvaluator(ModelEvaluator):

    def evaluate(self, model, X_test, y_test):

        predictions = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2,
        }

        logger.info(f"Evaluation metrics: {metrics}")

        return metrics


@step
def evaluate_model(
    model: BaseEstimator,
    X_test,
    y_test,
) -> Annotated[Dict[str, float], "evaluation_metrics"]:
    """
    ZenML step for evaluating a trained regression model.
    """

    evaluator = RegressionEvaluator()
    metrics = evaluator.evaluate(model, X_test, y_test)

    return metrics
