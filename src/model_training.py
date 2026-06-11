import os
import yaml
import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class ModelTraining:
    def __init__(self, config):
        self.config = config
        self.data_path = config["data"]["processed_path"]
        self.random_state = config["training"]["random_state"]
        self.scaler = StandardScaler()

    def load_processed_data(self):
        df = pd.read_csv(self.data_path)
        print(f"Loaded processed data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def split_data(self, df):
        X = df.drop(columns=["Activity Level"])
        y = df["Activity Level"]

        return train_test_split(
            X,
            y,
            test_size=self.config["training"]["test_size"],
            random_state=self.random_state,
            stratify=y
        )

    def scale_features(self, X_train, X_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    def get_sample_weights(self, y_train):
        classes = np.unique(y_train)

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train
        )

        class_weight_dict = dict(zip(classes, class_weights))
        sample_weights = y_train.map(class_weight_dict)

        print("\nClass weights used for imbalance:")
        print(class_weight_dict)

        return sample_weights

    def build_models(self, num_classes):
        return {
            "Decision Tree": DecisionTreeClassifier(
                random_state=self.random_state,
                class_weight=self.config["models"]["decision_tree"]["class_weight"]
            ),

            "Random Forest": RandomForestClassifier(
                random_state=self.random_state,
                class_weight=self.config["models"]["random_forest"]["class_weight"],
                n_estimators=self.config["models"]["random_forest"]["n_estimators"]
            ),

            "XGBoost": XGBClassifier(
                random_state=self.random_state,
                eval_metric=self.config["models"]["xgboost"]["eval_metric"],
                objective=self.config["models"]["xgboost"]["objective"],
                num_class=num_classes
            ),

            "Logistic Regression": LogisticRegression(
                random_state=self.random_state,
                class_weight=self.config["models"]["logistic_regression"]["class_weight"],
                max_iter=self.config["models"]["logistic_regression"]["max_iter"]
            )
        }

    def xgboost_cv(self, model, X_train, y_train, sample_weights, cv):
        scores = []

        y_train = y_train.reset_index(drop=True)
        sample_weights = sample_weights.reset_index(drop=True)

        for train_idx, val_idx in cv.split(X_train, y_train):
            fold_model = clone(model)

            fold_model.fit(
                X_train[train_idx],
                y_train.iloc[train_idx],
                sample_weight=sample_weights.iloc[train_idx]
            )

            y_pred = fold_model.predict(X_train[val_idx])

            score = f1_score(
                y_train.iloc[val_idx],
                y_pred,
                average=self.config["evaluation"]["f1_average"]
            )

            scores.append(score)

        return np.array(scores)

    def run_training(self):
        df = self.load_processed_data()

        X_train, X_test, y_train, y_test = self.split_data(df)

        X_train_scaled, X_test_scaled = self.scale_features(
            X_train,
            X_test
        )

        sample_weights = self.get_sample_weights(y_train)

        num_classes = len(np.unique(y_train))

        models = self.build_models(num_classes)

        cv = StratifiedKFold(
            n_splits=self.config["training"]["cv_folds"],
            shuffle=True,
            random_state=self.random_state
        )

        print("\n--- Training & Testing 4 Balanced Models ---")

        artifacts = {}

        for name, model in models.items():

            if name == "XGBoost":
                scores = self.xgboost_cv(
                    model,
                    X_train_scaled,
                    y_train,
                    sample_weights,
                    cv
                )

                model.fit(
                    X_train_scaled,
                    y_train,
                    sample_weight=sample_weights
                )

            else:
                scores = cross_val_score(
                    model,
                    X_train_scaled,
                    y_train,
                    cv=cv,
                    scoring=self.config["evaluation"]["scoring_metric"]
                )

                model.fit(
                    X_train_scaled,
                    y_train
                )

            y_pred = model.predict(X_test_scaled)

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            macro_f1 = f1_score(
                y_test,
                y_pred,
                average=self.config["evaluation"]["f1_average"]
            )

            print(f"\nModel: {name}")
            print(f"CV Macro F1: {scores.mean():.4f}")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test Macro F1: {macro_f1:.4f}")

            print(
                classification_report(
                    y_test,
                    y_pred,
                    labels=self.config["classification"]["labels"],
                    target_names=self.config["classification"]["target_names"]
               )
            )

            artifacts[name] = {
                "model": model,
                "accuracy": accuracy,
                "macro_f1": macro_f1
            }

        model_dir = self.config["output"]["model_dir"]

        os.makedirs(model_dir, exist_ok=True)

        joblib.dump(
            self.scaler,
            f"{model_dir}/scaler.pkl"
        )

        joblib.dump(
            {
                "X_test": X_test_scaled,
                "y_test": y_test,
                "X_train": X_train_scaled,
                "y_train": y_train,
                "models": artifacts,
                "sample_weights": sample_weights
            },
            f"{model_dir}/training_artifacts.pkl"
    )
        print(f"\nTraining completed. Artifacts saved to {model_dir}/training_artifacts.pkl")


if __name__ == "__main__":
    config = load_config("config.yaml")
    trainer = ModelTraining(config)
    trainer.run_training()