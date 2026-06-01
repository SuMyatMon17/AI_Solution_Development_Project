import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class ModelTraining:

    def __init__(self, data_path="data/processed_gas_monitoring.csv"):
        self.data_path = data_path

    def load_processed_data(self):
        df = pd.read_csv(self.data_path)
        return df

    def split_data(self, df):
        X = df.drop(columns=["Activity Level"])
        y = df["Activity Level"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        return X_train, X_test, y_train, y_test

    def train_models(self, X_train, y_train):
        models = {
            "Decision Tree": DecisionTreeClassifier(
                random_state=42,
                max_depth=8
            ),

            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            ),

            "XGBoost": XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                eval_metric="mlogloss"
            )
        }

        trained_models = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            trained_models[model_name] = model

        return trained_models

    def evaluate_models(self, trained_models, X_test, y_test):
        best_model = None
        best_score = 0
        best_model_name = ""

        for model_name, model in trained_models.items():
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            print("\n==============================")
            print(model_name)
            print("==============================")
            print("Accuracy:", accuracy)
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))

            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                best_model_name = model_name

        print("\nBest Model:", best_model_name)
        print("Best Accuracy:", best_score)

        return best_model, best_model_name

    def save_model(self, model, model_name):
        joblib.dump(
            model,
            "saved_model/best_model.pkl"
        )

        print("\nBest model saved to saved_model/best_model.pkl")

    def run_training(self):
        df = self.load_processed_data()

        X_train, X_test, y_train, y_test = self.split_data(df)

        trained_models = self.train_models(X_train, y_train)

        best_model, best_model_name = self.evaluate_models(
            trained_models,
            X_test,
            y_test
        )

        self.save_model(best_model, best_model_name)


if __name__ == "__main__":
    trainer = ModelTraining()
    trainer.run_training()