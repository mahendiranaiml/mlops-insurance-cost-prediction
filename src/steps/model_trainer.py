from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from typing_extensions import Annotated
from zenml import step
import pandas as pd


class ModelTrainer(ABC):

    @abstractmethod
    def train(
        self,
        model: BaseEstimator,
        X_train,
        y_train
    ) -> BaseEstimator:
        pass


class SklearnTrainer(ModelTrainer):

    def train(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        return model

@step
def train_model(
    model: BaseEstimator,
    X_train,
    y_train,
) -> Annotated[BaseEstimator, "trained_model"]:

    trainer = SklearnTrainer()
    trained_model = trainer.train(model, X_train, y_train)

    return trained_model
