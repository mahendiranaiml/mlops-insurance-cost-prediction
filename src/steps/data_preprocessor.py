from abc import ABC, abstractmethod
from typing import Tuple
import logging
from sklearn.pipeline import Pipeline

import pandas as pd
from sklearn.model_selection import train_test_split
from zenml import step
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from typing_extensions import Annotated

logger = logging.getLogger(__name__)

# ===============================
# 1️⃣ Abstract Preprocessor
# ===============================

class Preprocessor(ABC):
    """
    Abstract base class for preprocessing steps.
    """

    @abstractmethod
    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        pass

    @abstractmethod
    def build_pipeline(self) -> Pipeline:
        """
        Create and return an sklearn preprocessing pipeline
        """
        pass

    @abstractmethod
    def fit_transform(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass


# ===============================
# 2️⃣ Concrete Implementation
# ===============================
class TrainTestPreprocessor(Preprocessor):

    def __init__(self):
        self.pipeline = None

        self.num_cols = ["age", "bmi", "children"]
        self.cat_cols = ["sex", "smoker", "region"]

    # -----------------------------
    # 1️⃣ Split Data
    # -----------------------------
    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

        X = df.drop(columns=["charges"])
        y = df["charges"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=34
        )

        logger.info("Train-test split completed.")

        return X_train, X_test, y_train, y_test

    # -----------------------------
    # 2️⃣ Build sklearn Pipeline
    # -----------------------------
    def build_pipeline(self) -> Pipeline:

        numeric_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler())
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        self.pipeline = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.num_cols),
                ("cat", categorical_pipeline, self.cat_cols),
            ]
        )

        logger.info("Sklearn preprocessing pipeline created.")

        return self.pipeline

    # -----------------------------
    # 3️⃣ Fit & Transform
    # -----------------------------
    def fit_transform(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        if self.pipeline is None:
            raise RuntimeError("Pipeline has not been built yet.")

        X_train_transformed = self.pipeline.fit_transform(X_train)
        X_test_transformed = self.pipeline.transform(X_test)

        logger.info("Data transformed using sklearn pipeline.")

        return X_train_transformed, X_test_transformed



# ===============================
# 3️⃣ ZenML Step Wrapper
# ===============================

@step
def preprocess_data(
    df: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    preprocessor = TrainTestPreprocessor()

    X_train, X_test, y_train, y_test = preprocessor.split(df)

    X_train, X_test = preprocessor.fit_transform(X_train, X_test)

    return X_train, X_test, y_train, y_test


