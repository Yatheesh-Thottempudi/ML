import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging

class DataIngestion:
    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv(os.path.join("notebook/data", "stud.csv"))
            os.makedirs("artifacts", exist_ok=True)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv("artifacts/train.csv", index=False)
            test_set.to_csv("artifacts/test.csv", index=False)

            return "artifacts/train.csv", "artifacts/test.csv"

        except Exception as e:
            raise CustomException(e, sys)
