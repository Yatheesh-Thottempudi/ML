import sys
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

if __name__ == "__main__":
    try:
        logging.info("Training pipeline started")

        # 1️⃣ Data Ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        # 2️⃣ Data Transformation
        transformation = DataTransformation()
        preprocessor = transformation.get_data_transformer_object()

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        target_column = "math_score"

        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]

        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        save_object(
            file_path="artifacts/preprocessor.pkl",
            obj=preprocessor
        )

        # 3️⃣ Model Training
        trainer = ModelTrainer()
        trainer.initiate_model_trainer(
            X_train, y_train, X_test, y_test
        )

        logging.info("Training pipeline completed successfully")

    except Exception as e:
        raise CustomException(e, sys)
