import os
import argparse
import yaml
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier


def load_config(config_path="config.yaml"):
    """Load pipeline configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class ModelTraining:
    """Manages the end-to-end training stage: data splitting, feature scaling,
    cross-validation, model evaluation, and artifact persistence.

    Accepts a config dict so hyperparameters can be changed without touching code.
    """

    def __init__(self, config):
        self.config = config
        self.data_path = config["data"]["processed_path"]
        self.scaler = StandardScaler()

    def load_processed_data(self):
        """Read the preprocessed CSV produced by the preprocessing stage."""
        df = pd.read_csv(self.data_path)
        print(f"Loaded processed data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def split_data(self, df):
        """Separate features from the target and apply an 80/20 stratified train/test split.

        Stratification ensures the class distribution of Activity Level is preserved
        in both splits, which matters when classes may be imbalanced.
        """
        X = df.drop(columns=["Activity Level"])
        y = df["Activity Level"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config["training"]["test_size"],
            random_state=self.config["training"]["random_state"],
            stratify=y
        )

        print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
        return X_train, X_test, y_train, y_test

    def scale_features(self, X_train, X_test):
        """Fit a StandardScaler on the training set and transform both splits.

        Fitting only on X_train prevents the test set's distribution from leaking
        into the scaler, which would give falsely optimistic evaluation results.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def build_models(self):
        """Instantiate the three candidate classifiers using hyperparameters from config."""
        cfg = self.config["models"]
        random_state = self.config["training"]["random_state"]

        return {
            "Decision Tree": DecisionTreeClassifier(
                random_state=random_state,
                **cfg["decision_tree"]
            ),
            "Random Forest": RandomForestClassifier(
                random_state=random_state,
                **cfg["random_forest"]
            ),
            "XGBoost": XGBClassifier(
                random_state=random_state,
                eval_metric="mlogloss",
                **cfg["xgboost"]
            )
        }

    def cross_validate(self, models, X_train_scaled, y_train):
        """Run stratified K-fold cross-validation on the training set for each model.

        K-fold CV gives a more reliable accuracy estimate than a single train/test split
        by averaging results across multiple folds.

        Returns a dict mapping model name to its array of fold scores.
        """
        n_folds = self.config["training"]["cv_folds"]
        cv = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=self.config["training"]["random_state"]
        )

        cv_results = {}
        print(f"\n--- {n_folds}-Fold Cross-Validation (on Training Set) ---")

        for name, model in models.items():
            scores = cross_val_score(
                model, X_train_scaled, y_train, cv=cv, scoring="accuracy"
            )
            cv_results[name] = scores
            print(f"\n{name}:")
            print(f"  Fold Scores : {np.round(scores, 4)}")
            print(f"  Mean Acc    : {scores.mean():.4f}")
            print(f"  Std Dev     : {scores.std():.4f}")

        return cv_results

    def train_models(self, models, X_train_scaled, y_train):
        """Fit each model on the full (scaled) training set after cross-validation."""
        trained_models = {}
        print("\n--- Training on Full Training Set ---")

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            trained_models[name] = model
            print(f"  Trained: {name}")

        return trained_models

    def evaluate_models(self, trained_models, X_test_scaled, y_test):
        """Evaluate each trained model on the held-out test set.

        The model with the highest test accuracy is selected as the champion
        to be passed to the optimization stage.

        Returns the best model object, its name, and its test accuracy.
        """
        best_model = None
        best_score = 0
        best_model_name = ""

        print("\n--- Test Set Evaluation ---")

        for name, model in trained_models.items():
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            print(f"\n{'='*32}")
            print(f"  {name}")
            print(f"{'='*32}")
            print(f"  Test Accuracy : {accuracy:.4f}")
            print("\n  Classification Report:")
            print(classification_report(y_test, y_pred))
            print("  Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred))

            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                best_model_name = name

        print(f"\nBest Model    : {best_model_name}")
        print(f"Test Accuracy : {best_score:.4f}")

        return best_model, best_model_name, best_score

    def save_artifacts(self, best_model, best_model_name, best_score, X_test_scaled, y_test):
        """Save the champion model, fitted scaler, and test-set artifacts to disk.

        The scaler must be saved alongside the model so that any future prediction
        input can be transformed using the same parameters used during training.
        The test artifacts are passed to the Optimization stage for further tuning.
        """
        os.makedirs("saved_model", exist_ok=True)

        joblib.dump(best_model, "saved_model/best_model.pkl")
        joblib.dump(self.scaler, "saved_model/scaler.pkl")
        joblib.dump(
            {
                "X_test": X_test_scaled,
                "y_test": y_test,
                "model_name": best_model_name,
                "accuracy": best_score
            },
            "saved_model/training_artifacts.pkl"
        )

        print(f"\nSaved: saved_model/best_model.pkl  ({best_model_name})")
        print("Saved: saved_model/scaler.pkl")
        print("Saved: saved_model/training_artifacts.pkl")

    def run_training(self):
        """Execute the full training pipeline in sequence:
        load → split → scale → cross-validate → train → evaluate → save.
        """
        df = self.load_processed_data()

        X_train, X_test, y_train, y_test = self.split_data(df)

        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)

        models = self.build_models()

        self.cross_validate(models, X_train_scaled, y_train)

        trained_models = self.train_models(models, X_train_scaled, y_train)

        best_model, best_model_name, best_score = self.evaluate_models(
            trained_models, X_test_scaled, y_test
        )

        self.save_artifacts(best_model, best_model_name, best_score, X_test_scaled, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ML models for elderly activity level prediction."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = ModelTraining(config)
    trainer.run_training()
