import os
import yaml
import joblib
import pandas as pd
import numpy as np

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def load_config(config_path="config.yaml"):
    """Load project configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class ModelOptimizerSMOTE:
    def __init__(self, config):
        self.config = config
        self.random_state = config["training"]["random_state"]
        
        # Load the artifacts from training.py
        artifacts_path = config["output"]["artifacts_file"]
        if not os.path.exists(artifacts_path):
            raise FileNotFoundError(f"Missing {artifacts_path}. Please run training.py first.")
            
        artifacts = joblib.load(artifacts_path)
        
        # Pull sets
        self.X_train = artifacts["X_train"]
        self.y_train = artifacts["y_train"]
        self.X_test = artifacts["X_test"]
        self.y_test = artifacts["y_test"]
        self.base_models = artifacts["models"]

    def get_pipelines_and_params(self):
        """
        Wraps models in a SMOTE pipeline and provides high-yield 
        parameter grids to balance classes and avoid severe overfitting.
        """
        pipelines = {
            "Decision Tree": Pipeline([
                ("smote", SMOTE(random_state=self.random_state)),
                ("model", DecisionTreeClassifier(random_state=self.random_state))
            ]),
            "Random Forest": Pipeline([
                ("smote", SMOTE(random_state=self.random_state)),
                ("model", RandomForestClassifier(random_state=self.random_state))
            ]),
            "XGBoost": Pipeline([
                ("smote", SMOTE(random_state=self.random_state)),
                ("model", XGBClassifier(
                    random_state=self.random_state,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    num_class=3
                ))
            ]),
            "Logistic Regression": Pipeline([
                ("smote", SMOTE(random_state=self.random_state)),
                ("model", LogisticRegression(random_state=self.random_state, max_iter=2000))
            ])
        }

        # Optimized hyperparameter grids to speed up search loops
        # FIXED: Removed 'model__penalty' to resolve the SAGA solver warning flood
        param_grids = {
            "Decision Tree": {
                "smote__k_neighbors": [5],
                "model__max_depth": [4, 6, 8],
                "model__min_samples_leaf": [4, 10]
            },
            "Random Forest": {
                "smote__k_neighbors": [5],
                "model__n_estimators": [150],
                "model__max_depth": [6, 10, 14],
                "model__min_samples_leaf": [4, 8],
                "model__max_features": ["sqrt"]
            },
            "XGBoost": {
                "smote__k_neighbors": [5],
                "model__n_estimators": [150],
                "model__max_depth": [4, 6],
                "model__learning_rate": [0.05, 0.15],
                "model__subsample": [0.8],
                "model__colsample_bytree": [0.8]
            },
            "Logistic Regression": {
                "smote__k_neighbors": [5],
                "model__C": [0.1, 1.0, 10.0],
                "model__solver": ["saga"]
            }
        }

        return pipelines, param_grids

    def run_optimization(self):
        """Search for best parameters using SMOTE cross-validation."""
        pipelines, param_grids = self.get_pipelines_and_params()
        tuned_results = {}

        cv = StratifiedKFold(
            n_splits=self.config["training"]["cv_folds"],
            shuffle=True,
            random_state=self.random_state
        )

        print("\n--- Starting SMOTE Fine-Tuning & Re-Training ---")
        
        for name in pipelines.keys():
            print(f"Optimizing {name} with SMOTE...")
            
            grid_search = GridSearchCV(
                estimator=pipelines[name],
                param_grid=param_grids[name],
                cv=cv,
                scoring='f1_macro',
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(self.X_train, self.y_train)

            best_model = grid_search.best_estimator_
            
            # Predict on train set to calculate training tracking metrics
            y_train_pred = best_model.predict(self.X_train)
            train_acc = accuracy_score(self.y_train, y_train_pred)
            train_f1 = f1_score(self.y_train, y_train_pred, average='macro')
            train_rec = recall_score(self.y_train, y_train_pred, average='macro')

            tuned_results[name] = {
                "model": best_model,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "train_rec": train_rec,
                "best_params": grid_search.best_params_
            }
            print(f"-> Best Parameters Found: {grid_search.best_params_}")

        return tuned_results

    def generate_visualizations(self, tuned_results, df_summary):
        """Creates and saves three separate individual metric charts and a confusion matrix."""
        output_dir = "plots"
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n--- Generating and Saving Separated Plots to ./{output_dir}/ ---")

        models = df_summary["Model"].tolist()
        x = np.arange(len(models))
        width = 0.35

        # ---------------------------------------------------------
        # CHART 1: MACRO F1 GAP
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, df_summary["Train F1"].tolist(), width, label='Train Macro F1', color='#4A90E2')
        rects2 = ax.bar(x + width/2, df_summary["Test F1"].tolist(), width, label='Test Macro F1', color='#E24A4A')
        
        ax.set_ylabel('Macro F1 Score', fontsize=12)
        ax.set_title('Model Optimization Comparison: Generalization Variance Gap (F1)', fontsize=14, pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        for rect in rects1 + rects2:
            ax.annotate(f'{rect.get_height():.4f}', xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/macro_f1_gap.png", dpi=300)
        plt.close()
        print("[Saved]: macro_f1_gap.png")

        # ---------------------------------------------------------
        # CHART 2: MACRO RECALL GAP
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, df_summary["Train Recall"].tolist(), width, label='Train Macro Recall', color='#2CA02C')
        rects2 = ax.bar(x + width/2, df_summary["Test Recall"].tolist(), width, label='Test Macro Recall', color='#9467BD')
        
        ax.set_ylabel('Macro Recall Score', fontsize=12)
        ax.set_title('Model Optimization Comparison: Sensitivity Performance (Recall)', fontsize=14, pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        for rect in rects1 + rects2:
            ax.annotate(f'{rect.get_height():.4f}', xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/recall_gap.png", dpi=300)
        plt.close()
        print("[Saved]: recall_gap.png")

        # ---------------------------------------------------------
        # CHART 3: ACCURACY GAP
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, df_summary["Train Acc"].tolist(), width, label='Train Accuracy', color='#FF7F0E')
        rects2 = ax.bar(x + width/2, df_summary["Test Acc"].tolist(), width, label='Test Accuracy', color='#17BECF')
        
        ax.set_ylabel('Accuracy Score', fontsize=12)
        ax.set_title('Model Optimization Comparison: Overall Accuracy Baseline', fontsize=14, pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        for rect in rects1 + rects2:
            ax.annotate(f'{rect.get_height():.4f}', xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/accuracy_gap.png", dpi=300)
        plt.close()
        print("[Saved]: accuracy_gap.png")

        # ---------------------------------------------------------
        # PLOT 4: CONFUSION MATRIX HEATMAP (Best Model)
        # ---------------------------------------------------------
        best_model_name = df_summary.iloc[0]["Model"]
        best_pipeline = tuned_results[best_model_name]["model"]
        y_test_pred = best_pipeline.predict(self.X_test)

        cm = confusion_matrix(self.y_test, y_test_pred)
        plt.figure(figsize=(6.5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=["Low", "Moderate", "High"],
                    yticklabels=["Low", "Moderate", "High"],
                    annot_kws={"size": 12})
        
        plt.title(f'Confusion Matrix: {best_model_name} (SMOTE Optimized)', fontsize=13, pad=12)
        plt.ylabel('Actual Label Class', fontsize=11)
        plt.xlabel('Predicted Label Class', fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix_best.png", dpi=300)
        plt.close()
        print(f"[Saved]: confusion_matrix_best.png")

    def final_evaluation(self, tuned_results):
        """Final Testing and Comparison Report with Recall metrics printed as a table."""
        print("\n" + "="*85)
        print("FINAL EVALUATION REPORT: TRAINING VS TESTING (ACCURACY, F1 & RECALL WITH SMOTE)")
        print("="*85)

        summary_list = []

        for name, data in tuned_results.items():
            model = data["model"]
            
            y_test_pred = model.predict(self.X_test)
            test_acc = accuracy_score(self.y_test, y_test_pred)
            test_f1 = f1_score(self.y_test, y_test_pred, average='macro')
            test_rec = recall_score(self.y_test, y_test_pred, average='macro')
            
            print(f"\n>>> {name} Final Performance:")
            print(f"TRAIN: Acc={data['train_acc']:.4f}, Macro F1={data['train_f1']:.4f}, Macro Recall={data['train_rec']:.4f}")
            print(f"TEST:  Acc={test_acc:.4f}, Macro F1={test_f1:.4f}, Macro Recall={test_rec:.4f}")
            
            print(classification_report(
                self.y_test, 
                y_test_pred, 
                target_names=["Low", "Moderate", "High"]
            ))
            
            summary_list.append({
                "Model": name,
                "Train Acc": data['train_acc'],
                "Test Acc": test_acc,
                "Train F1": data['train_f1'],
                "Test F1": test_f1,
                "Train Recall": data['train_rec'],
                "Test Recall": test_rec
            })

        # Generate structural summary framework
        df_summary = pd.DataFrame(summary_list)
        column_order = ["Model", "Train Acc", "Test Acc", "Train F1", "Test F1", "Train Recall", "Test Recall"]
        df_summary = df_summary[column_order].sort_values(by="Test F1", ascending=False)
        
        print("\n--- Summary Table (Ordered by Test Macro F1) ---")
        print(df_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        # Save all individual charts to drive automatically
        self.generate_visualizations(tuned_results, df_summary)


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    optimizer = ModelOptimizerSMOTE(cfg)
    
    # 1. Run optimization with SMOTE pipeline
    final_models = optimizer.run_optimization()
    
    # 2. Print summary table directly into terminal and save PNGs
    optimizer.final_evaluation(final_models)