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
        
        # Pull evaluation configuration variables
        self.scoring_metric = config["evaluation"].get("scoring_metric", "f1_macro")
        self.f1_average = config["evaluation"].get("f1_average", "macro")
        self.labels = config["classification"].get("labels", [0, 1, 2])
        self.target_names = config["classification"].get("target_names", ["Low", "Moderate", "High"])
        self.output_plots_dir = config["output"].get("plots_dir", "plots")
        
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

        # Parametric loading extracts search spaces from configuration mapping matrix
        param_grids = self.config["optimization_grids"]

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

        print(f"\n--- Starting SMOTE Fine-Tuning & Re-Training [Metric: {self.scoring_metric}] ---")
        
        for name in pipelines.keys():
            print(f"Optimizing {name} with SMOTE...")
            
            grid_search = GridSearchCV(
                estimator=pipelines[name],
                param_grid=param_grids[name],
                cv=cv,
                scoring=self.scoring_metric,
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(self.X_train, self.y_train)

            best_model = grid_search.best_estimator_
            
            # Predict on train set to calculate training tracking metrics
            y_train_pred = best_model.predict(self.X_train)
            train_acc = accuracy_score(self.y_train, y_train_pred)
            train_f1 = f1_score(self.y_train, y_train_pred, average=self.f1_average)
            train_rec = recall_score(self.y_train, y_train_pred, average=self.f1_average)

            tuned_results[name] = {
                "model": best_model,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "train_rec": train_rec,
                "best_params": grid_search.best_params_
            }
            print(f"-> Best Parameters Found: {grid_search.best_params_}")

        return tuned_results

    def generate_feature_importance(self, best_model_name, best_pipeline, output_dir):
        """
        Extracts structural weights from the top tree model, prints real column 
        names to the terminal, and saves a custom horizontal visualization.
        """
        raw_model = best_pipeline.named_steps["model"]
        
        if hasattr(raw_model, "feature_importances_"):
            print(f"\n--- Feature Importance Ranking Leaderboard ({best_model_name}) ---")
            
            # Explicitly mapping back your exact database schema layout
            feature_names = [
                "Temperature",
                "Humidity",
                "CO2_InfraredSensor",
                "CO2_ElectroChemicalSensor",
                "CO_GasSensor",
                "Ambient Light Level",
                "Activity Level",
                "CO2_Average",
                "CO2_Divergence",
                "MetalOxide_PCA",
                "Time of Day_evening",
                "Time of Day_morning",
                "Time of Day_night",
                "HVAC Operation Mode_eco_mode",
                "HVAC Operation Mode_heating_active",
                "HVAC Operation Mode_maintenance_mode",
                "HVAC Operation Mode_off",
                "HVAC Operation Mode_ventilation_only"
            ]
                
            importances = raw_model.feature_importances_
            
            # Sort everything cleanly in descending order
            indices = np.argsort(importances)[::-1]
            sorted_features = [feature_names[i] for i in indices]
            sorted_weights = importances[indices]

            # Output clean layout table directly inside your terminal window
            print(f"{'Rank':<6}{'Sensor Feature Name':<40}{'Importance Score':<15}")
            print("-" * 61)
            for rank, (feat, weight) in enumerate(zip(sorted_features, sorted_weights), 1):
                print(f"{rank:<6}{feat:<40}{weight:<15.4f}")
            print("-" * 61)

            # Generate horizontal bar plot with true text axis attributes
            plt.figure(figsize=(11, 7))
            sns.barplot(x=sorted_weights, y=sorted_features, color="#4CAF50")
            
            plt.title(f'Sensor Feature Importance Structural Weights ({best_model_name})', fontsize=13, pad=15)
            plt.xlabel('Relative Information Gains / Importance Score', fontsize=11)
            plt.grid(axis='x', linestyle='--', alpha=0.4)
            plt.tight_layout()
            
            plt.savefig(f"{output_dir}/sensor_feature_importance.png", dpi=300)
            plt.close()
            print(f"[Saved]: sensor_feature_importance.png dropped in ./{output_dir}/")
        else:
            print(f"\n[Notice]: Top model configuration ({best_model_name}) does not compute feature weights natively. Skipping.")

    def generate_visualizations(self, tuned_results, df_summary):
        """Creates and saves three separate individual metric charts and a confusion matrix."""
        output_dir = self.output_plots_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n--- Generating and Saving Separated Plots to ./{output_dir}/ ---")

        models = df_summary["Model"].tolist()
        x = np.arange(len(models))
        width = 0.35

        # ---------------------------------------------------------
        # CHART 1: MACRO F1 GAP
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, df_summary["Train F1"].tolist(), width, label='Train F1', color='#4A90E2')
        rects2 = ax.bar(x + width/2, df_summary["Test F1"].tolist(), width, label='Test F1', color='#E24A4A')
        
        ax.set_ylabel('F1 Score', fontsize=12)
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
        # CHART 2: RECALL GAP
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, df_summary["Train Recall"].tolist(), width, label='Train Recall', color='#2CA02C')
        rects2 = ax.bar(x + width/2, df_summary["Test Recall"].tolist(), width, label='Test Recall', color='#9467BD')
        
        ax.set_ylabel('Recall Score', fontsize=12)
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
                    xticklabels=self.target_names,
                    yticklabels=self.target_names,
                    annot_kws={"size": 12})
        
        plt.title(f'Confusion Matrix: {best_model_name} (SMOTE Optimized)', fontsize=13, pad=12)
        plt.ylabel('Actual Label Class', fontsize=11)
        plt.xlabel('Predicted Label Class', fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix_best.png", dpi=300)
        plt.close()
        print(f"[Saved]: confusion_matrix_best.png")

        # AUTO-TRIGGER FEATURE PLOT AND TERMINAL PRINTOUT
        self.generate_feature_importance(best_model_name, best_pipeline, output_dir)

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
            test_f1 = f1_score(self.y_test, y_test_pred, average=self.f1_average)
            test_rec = recall_score(self.y_test, y_test_pred, average=self.f1_average)
            
            print(f"\n>>> {name} Final Performance:")
            print(f"TRAIN: Acc={data['train_acc']:.4f}, F1={data['train_f1']:.4f}, Recall={data['train_rec']:.4f}")
            print(f"TEST:  Acc={test_acc:.4f}, F1={test_f1:.4f}, Recall={test_rec:.4f}")
            
            print(classification_report(
                self.y_test, 
                y_test_pred, 
                labels=self.labels,
                target_names=self.target_names
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
        
        # CHANGED PART: Safely sorts the table based on configuration routing rules
        sort_column = "Test F1"
        if "recall" in self.scoring_metric:
            sort_column = "Test Recall"
        elif "accuracy" in self.scoring_metric:
            sort_column = "Test Acc"
            
        df_summary = df_summary[column_order].sort_values(by=sort_column, ascending=False)
        
        print(f"\n--- Summary Table (Ordered by {sort_column}) ---")
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