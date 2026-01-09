import sys
from src.exception import CustomException
from src.utils import save_object, evaluate_models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

class ModelTrainer:
    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "XGBoost": XGBRegressor(),
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models)
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            save_object("artifacts/model.pkl", best_model)

        except Exception as e:
            raise CustomException(e, sys)
