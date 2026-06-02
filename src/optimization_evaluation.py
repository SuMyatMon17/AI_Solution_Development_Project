import os
import yaml
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    accuracy_score
)

# Loads the shared configuration file
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class ModelOptimization:
    def __init__(self, config):
        self.config = config
        self.artifacts_path = "saved_model/training_artifacts.pkl"
        self.scaler_path = "saved_model/scaler.pkl"
        
        # Ensure results directory exists for plots
        os.makedirs("results", exist_ok=True)

        # Safety check: Ensure the training member has generated the baseline first
        if not os.path.exists(self.artifacts_path):
            raise FileNotFoundError("Run training.py first to generate necessary artifacts!")

    # Loads data and reconstructs the training set to prevent leakage
    def load_data_and_scaler(self):
        print("--- Loading Pipeline Artifacts ---")
        
        # Loading artifacts (X_test, y_test, and baseline metrics)
        artifacts = joblib.load(self.artifacts_path)
        self.X_test_scaled = artifacts["X_test"]
        self.y_test = artifacts["y_test"]
        self.baseline_acc = artifacts["accuracy"]
        
        # Loading the fitted scaler
        self.scaler = joblib.load(self.scaler_path)
        
        # Load full processed data
        df = pd.read_csv(self.config["data"]["processed_path"])
        
        # Reconstruct Training Set: Use all data NOT in the test set
        test_indices = self.y_test.index
        train_df = df.drop(index=test_indices)
        
        self.X_train = train_df.drop(columns=["Activity Level"])
        self.y_train = train_df["Activity Level"]
        
        # Scale the training data using the existing scaler
        self.X_train_scaled = self.scaler.transform(self.X_train)
        
        print(f"Baseline accuracy from Training member: {self.baseline_acc:.4f}")

    # Optimizes XGBoost hyperparameters using GridSearchCV
    def run_grid_search(self):
        print("\n--- Starting Optimization (Grid Search) ---")
        
        xgb = XGBClassifier(
            random_state=self.config["training"]["random_state"],
            eval_metric="mlogloss"
        )

        # Define the parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0]
        }

        # 3-Fold Cross-Validation on training data only
        grid_search = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            cv=3, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(self.X_train_scaled, self.y_train)
        self.best_model = grid_search.best_estimator_
        
        print(f"Optimization Complete!")
        print(f"Best Parameters: {grid_search.best_params_}")

    # Generates the required reports and visuals for the project
    def produce_evaluation_assets(self):
        print("\n--- Final Model Evaluation ---")
        
        # Predict on the clean test set
        y_pred = self.best_model.predict(self.X_test_scaled)
        
        # 1. Textual Report
        print(classification_report(self.y_test, y_pred, target_names=['Low', 'Moderate', 'High']))
        
        # 2. Confusion Matrix Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            self.y_test, y_pred, 
            display_labels=['Low', 'Moderate', 'High'], 
            cmap='Blues', ax=ax
        )
        plt.title("Confusion Matrix: Activity Recognition")
        plt.savefig("results/confusion_matrix.png")
        print("Saved: results/confusion_matrix.png")

        # 3. Feature Importance Plot
        self.plot_key_features()

    def plot_key_features(self):
        importances = self.best_model.feature_importances_
        # Get feature names from the original dataframe columns
        features = self.X_train.columns
        
        feat_df = pd.DataFrame({'Sensor': features, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Sensor', data=feat_df.head(10), palette='viridis')
        plt.title("Top 10 Key Features for Activity Categorization")
        plt.tight_layout()
        plt.savefig("results/feature_importance.png")
        print("Saved: results/feature_importance.png")
        
        print("\nTop 3 Key Features Identified:")
        print(feat_df.head(3))

    # Saves the final optimized model
    def save_final_system(self):
        joblib.dump(self.best_model, "saved_model/final_optimized_model.pkl")
        print("\nPipeline Complete. Final model ready in 'saved_model/'.")

# Main execution
if __name__ == "__main__":
    config = load_config("config.yaml")
    optimizer = ModelOptimization(config)
    optimizer.load_data_and_scaler()
    optimizer.run_grid_search()
    optimizer.produce_evaluation_assets()
    optimizer.save_final_system()