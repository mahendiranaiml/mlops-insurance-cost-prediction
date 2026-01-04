from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from zenml import step
from typing_extensions import Annotated
from typing import Dict

import logging
logger = logging.getLogger(__name__)


class ModelRegistrar(ABC):

    @abstractmethod
    def register(
        self,
        model: BaseEstimator,
        metrics: Dict[str, float],
    ) -> BaseEstimator:
        pass

class ZenMLModelRegistrar(ModelRegistrar):

    def __init__(self, rmse_threshold: float = 5000):
        self.rmse_threshold = rmse_threshold

    def register(self, model, metrics):

        rmse = metrics.get("rmse")

        if rmse is None:
            raise ValueError("RMSE metric not found.")

        if rmse > self.rmse_threshold:
            raise RuntimeError(
                f"Model rejected: RMSE {rmse:.2f} exceeds threshold {self.rmse_threshold}"
            )

        logger.info(
            f"Model accepted: RMSE {rmse:.2f} below threshold {self.rmse_threshold}"
        )

        return model


@step(enable_cache=False)
def register_model(
    model: BaseEstimator,
    metrics: Dict[str, float],
) -> Annotated[BaseEstimator, "registered_model"]:
    """
    Registers the model if evaluation metrics meet acceptance criteria.
    """

    registrar = ZenMLModelRegistrar(rmse_threshold=5000)
    approved_model = registrar.register(model, metrics)

    return approved_model

