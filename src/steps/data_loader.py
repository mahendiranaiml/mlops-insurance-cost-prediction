from abc import ABC, abstractmethod
import logging
import pandas as pd
from zenml import step
from typing_extensions import Annotated

# ===============================
# Logger Configuration
# ===============================
logger = logging.getLogger(__name__)

# ===============================
# 1️⃣ Abstract Interface (Contract)
# ===============================

class DataLoader(ABC):
    """Abstract base class for all data loaders"""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load data and return a DataFrame"""
        pass


# ===============================
# 2️⃣ Concrete Implementation
# ===============================

class CSVDataLoader(DataLoader):
    """Loads data from a CSV file"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.file_path)

            logger.info(f"Data loaded successfully from {self.file_path}")
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")

            print(f"\n{'='*60}")
            print("✓ Data Loaded Successfully!")
            print(f"{'='*60}")
            print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"Columns: {', '.join(df.columns)}")
            print("\nFirst few rows:")
            print(df.head())
            print(f"{'='*60}\n")

            return df

        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
            raise

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise


# ===============================
# 3️⃣ ZenML Step Wrapper
# ===============================

@step
def ingest_data(file_path: str) -> Annotated[pd.DataFrame, "raw_data"]:
    """
    ZenML step for ingesting raw data
    """
    loader = CSVDataLoader(file_path)
    df = loader.load()
    return df
