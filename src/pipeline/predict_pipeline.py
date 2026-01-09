import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def predict(self, features: pd.DataFrame):
        try:
            model = load_object("artifacts/model.pkl")
            preprocessor = load_object("artifacts/preprocessor.pkl")

            data_scaled = preprocessor.transform(features)
            predictions = model.predict(data_scaled)

            return predictions

        except Exception as e:
            raise CustomException(e, sys)
