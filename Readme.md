# ElderGuard Analytics – Activity Level Prediction

## Group Information
### Group Members

| Name              | Admin Number | Contribution                            |
|----------------   |--------------|-----------------------------------------|
| Su Myat Mon       | 244009Q      | Data Preprocessing + Extra Effort in EDA|
| Tong Gia Linh     | 244914E      | Model Training                          |
| Mya Phu Pwint Soe | 243843G      | Model Optimization & Evaluation         |

---

# Project Overview

This project develops a machine learning pipeline to predict elderly activity levels using environmental and gas sensor data collected from a smart-home monitoring system.

The objective is to classify activity levels into:
- Low Activity
- Moderate Activity
- High Activity

The solution follows a complete machine learning workflow consisting of:
1. Data Preprocessing
2. Feature Engineering
3. Model Training
4. Hyperparameter Optimization
5. Model Evaluation
6. Docker Containerization

---

# Project Structure

```text
AI_Solution_Development_Project/
│
├── data/
│   ├── gas_monitoring.db
│   └── processed_gas_monitoring.csv
│
├── plots/
│   ├── accuracy_gap.png
│   ├── confusion_matrix_best.png
│   ├── macro_f1_gap.png
│   ├── recall_gap.png
│   └── sensor_feature_importance.png
|
├── saved_model/
│   ├── scaler.pkl
│   └── training_artifacts.pkl
│
├── src/
│   ├── preprocessing_data.py
│   ├── model_training.py
│   └── optimization_evaluation.py
│
├── config.yaml
├── requirements.txt
├── Dockerfile
├── run.sh
└── README.md

```

---

# File Ownership

Each member contributed at least one Python file that can be assessed for code quality.

| Python File | Owner |
| --- | --- |
| preprocessing_data.py + Extra Effort in EDA | Su Myat Mon |
| model_training.py | Tong Gia Linh |
| optimization_evaluation.py | Mya Phu Pwint Soe |

---

# Centralized Configuration Management

To maximize reproducibility and eliminate code repetition, this entire pipeline is parametrically driven by a single `config.yaml` file located at the project root.

By altering parameters inside `config.yaml`, changes flow seamlessly downstream through all modules without modifying code files:

* **Global Parameters:** Easily modify test splits (`test_size`), random seed values (`random_state`), or cross-validation strategies (`cv_folds`).
* **Dynamic Pipeline Tuning Metric:** Adjusting the `scoring_metric` property (e.g., changing from `f1_macro` to `recall_macro` or `accuracy`) alters the evaluation scoring of baseline modules, re-targets the optimization search criteria for `GridSearchCV`, and automatically re-orders the final model selection summary table.
* **Hyperparameter Spaces:** Adjust search spaces for all four algorithms inside a unified `optimization_grids` block instead of tracking down variable declarations across multiple execution files.

---

# Pipeline Architecture & Detailed Component Breakdown

The pipeline operates sequentially as an integrated end-to-end orchestration system. Below is the detailed breakdown of the technical implementation within each component script:

### 1. Data Preprocessing & Feature Engineering (`src/preprocessing_data.py`)

This script handles data extraction, rigorous data cleaning, and feature generation to transform raw sensor readings into a clean modeling matrix:

* **Database Ingestion:** Connects dynamically to the SQLite database path defined in `config.yaml` to extract the raw historical gas monitoring records.
* **Missing Value & Integrity Check:** Inspects the dataset for null values or structural gaps to prevent script compilation errors downstream.
* **Outlier Treatment (IQR Clipping):** Implements an automated Interquartile Range (IQR) thresholding algorithm. Sensor values falling outside `[Q1 - 1.5*IQR]` and `[Q3 + 1.5*IQR]` are safely clipped to the upper or lower boundary limits, preserving rows while neutralizing extreme electrical or environmental telemetry spikes.
* **Feature Engineering:**
* Computes `CO2_Average` to combine information from the Infrared and Electrochemical sensors into a single robust measure.
* Calculates `CO2_Divergence` (`|Infrared - Electrochemical|`) to quantify variance or anomalies between the two sensing units.
* Applies Principal Component Analysis (`PCA` with `n_components=1`) across all four highly correlated Metal Oxide sensors to capture their shared information variance while dropping dimensions and eliminating multicollinearity.


* **Data Export:** Saves the final engineering matrix out to `data/processed_gas_monitoring.csv` for downstream ingestion.

### 2. Baseline Model Training (`src/model_training.py`)

This module establishes baseline model benchmarks by tracking multi-class classification behaviors across four structurally distinct algorithms before hyperparameter fine-tuning:

* **Parametric Environment Setup:** Initializes random seeds, test sizes, split stratification configurations, and directory targets cleanly out of the centralized configuration matrix.
* **Feature Scaling:** Fits a scikit-learn `StandardScaler` to the training features to compute column-wise means and standard deviations, shifting distributions to center around zero with unit variance. The fitted state is saved to serialize transformed mappings onto incoming test sets.
* **Automated Class Imbalance Balancing:** Computes analytical sample weights via scikit-learn's `compute_class_weight(class_weight="balanced")`. This matches training data frequencies and outputs an active sample weight array, penalizing models proportionally if they misclassify less frequent minority activity states.
* **Stratified Validation Strategy:** Instantiates a `StratifiedKFold` loop utilizing the configuration-defined `cv_folds` variable. This ensures every validation split accurately mirrors the master multi-class activity level target distribution.
* **Custom XGBoost Cross-Validation Loop:** Contains an explicitly engineered validation method (`xgboost_cv`) constructed to accurately feed the custom calculated sample weight arrays directly into individual data fold partitions during internal fitting cycles, an option not supported natively by scikit-learn's basic `cross_val_score` wrapper.
* **State Preservation Transfer Packaging:** Packaged data matrices, standard scalers, sample weight arrays, and initial trained model structures are compiled and saved into a binary transport layer file (`saved_model/training_artifacts.pkl`).

### 3. Hyperparameter Tuning & Post-Optimization Diagnostic Evaluation (`src/optimization_evaluation.py`)

This script executes the final fine-tuning phase by isolating search grids over synthetic sample structures and outputting analytical diagnostic visualizations:

* **Artifact De-serialization:** Restores training/testing sets and baseline model wrappers straight from the intermediate `.pkl` state file.
* **GridSearchCV Over Isolated Pipelines:** Wraps algorithm architectures inside an `imblearn.pipeline.Pipeline` consisting of a synthetic oversampling block (`SMOTE`) followed by the algorithm model. This guarantees that synthetic generation happens strictly inside individual cross-validation splits, preventing data leakage.
* **Dynamic Metric Execution:** Executes grid sweeps across the hyperparameter blocks loaded from `optimization_grids` using the target `scoring_metric` defined in the configuration room.
* **Diagnostic Report Generation:** Compiles evaluation logs, classification tables, and automatically saves five production-ready analytical visualization charts tracking performance trade-offs directly to the targeted plots directory.

---

# How To Run The Pipeline

## Option 1 – Using Docker

### Build Docker Image

```bash
docker build -t elderguard-pipeline .

```

### Run Container

```bash
docker run --rm elderguard-pipeline

```

The pipeline will automatically execute:

```text
Preprocessing
      ↓
Model Training
      ↓
SMOTE Optimization
      ↓
Final Evaluation

```

---

## Option 2 – Local Execution

### Step 1

Install dependencies:

```bash
pip install -r requirements.txt

```

### Step 2

Run preprocessing:

```bash
python src/preprocessing_data.py

```

### Step 3

Run model training:

```bash
python src/model_training.py

```

### Step 4

Run optimization and evaluation:

```bash
python src/optimization_evaluation.py

```

---

# Docker Development Environment

The project uses Docker to ensure a consistent and reproducible development environment.

### Docker Version

* Docker Engine
* Python 3.11 Slim Image

### Start Docker Environment

Build image:

```bash
docker build -t elderguard-pipeline .

```

Run container:

```bash
docker run --rm elderguard-pipeline

```

This eliminates dependency issues and ensures the same execution environment across different machines.

---

# Summary of Key EDA Findings

Exploratory Data Analysis revealed several important observations.

### 1. Strong Correlation Between Metal Oxide Sensors

The four Metal Oxide sensor readings were highly correlated with one another. This indicates that the sensors measure similar environmental characteristics and may contain redundant information.

### 2. Correlation Between CO₂ Sensors

The Infrared and Electrochemical CO₂ sensors showed moderate to strong positive correlation. This suggests both sensors capture similar CO₂ concentration trends.

### 3. Class Imbalance

The Activity Level classes were not perfectly balanced. Certain activity categories appeared less frequently than others. This justified the use of:

* Class Weights
* SMOTE Oversampling
during model development.

### 4. Presence of Outliers

Several sensor measurements contained extreme values. Outlier treatment was performed using the IQR clipping method to reduce the impact of abnormal readings.

---

# Feature Engineering Justification

Several engineered features were introduced to improve predictive performance.

## CO2_Average

```text
(CO2_InfraredSensor + CO2_ElectroChemicalSensor) / 2

```

### Justification

Combines information from both CO₂ sensors into a single feature representing overall CO₂ concentration.

---

## CO2_Divergence

```text
|CO2_InfraredSensor - CO2_ElectroChemicalSensor|

```

### Justification

Measures disagreement between the two sensors. Large divergence may indicate unusual environmental conditions.

---

## MetalOxide_PCA

Principal Component Analysis (PCA) was applied to:

* MetalOxideSensor_Unit1
* MetalOxideSensor_Unit2
* MetalOxideSensor_Unit3
* MetalOxideSensor_Unit4

### Justification

EDA showed strong correlation among these sensors. PCA reduces redundancy while preserving most of the information contained in the four sensors. This reduces dimensionality and helps prevent overfitting.

---

# Models Used

Four machine learning models were trained and compared.

## Decision Tree

### Reason

* Easy to interpret
* Handles non-linear relationships
* Provides baseline performance

---

## Random Forest

### Reason

* Reduces overfitting through bagging
* Robust to noisy data
* Captures complex feature interactions

---

## Logistic Regression

### Reason

* Strong baseline model
* Computationally efficient
* Provides a linear benchmark for comparison

---

## XGBoost

### Reason

* State-of-the-art ensemble model
* Often achieves high predictive performance
* Handles non-linear relationships effectively

---

# Hyperparameter Tuning

GridSearchCV was used together with Stratified K-Fold Cross Validation.

### Why GridSearchCV?

GridSearchCV systematically searches combinations of hyperparameters and selects the configuration that produces the best cross-validation performance.

### Why Stratified K-Fold?

Stratification preserves class distributions across folds and provides a more reliable estimate of model performance.

### Why SMOTE?

EDA revealed class imbalance. SMOTE creates synthetic minority class samples and helps models learn underrepresented activity classes more effectively.

---

# Evaluation Metrics

Several evaluation metrics were used.

## Accuracy

Measures overall prediction correctness.

### Limitation

Accuracy may be misleading when class distributions are imbalanced.

---

## Macro F1 Score

Primary evaluation metric.

### Justification

The project predicts three activity levels. Macro F1 evaluates each class equally regardless of class size. This prevents dominant classes from hiding poor performance on minority classes.

---

## Macro Recall

Measures how effectively each activity class is detected.

### Justification

In an elderly monitoring system, missing activity patterns may be more problematic than producing occasional false alarms. High recall ensures activity classes are detected as consistently as possible.

---

# Output Files

Generated after pipeline execution.

### Processed Dataset

```text
data/processed_gas_monitoring.csv

```

### Saved Artifacts

```text
saved_model/scaler.pkl
saved_model/training_artifacts.pkl

```

### Visual Diagnostics & Plots

Generated dynamically inside the directory defined by the configuration (`plots/`):

* `macro_f1_gap.png`: Visualizes variance and generalization gaps between validation training and test splits.
* `recall_gap.png`: Illustrates specific sensitivity variations across models to track safety/omission trade-offs.
* `accuracy_gap.png`: Provides a visual benchmark tracking overall raw correctness behaviors.
* `confusion_matrix_best.png`: A multi-class heatmap breakdown mapping specific misclassification errors for your top-performing optimized pipeline model.
* `sensor_feature_importance.png`: Structural information-gain weight distributions mapping which gas and environmental sensors are driving predictions.

### Console Outputs

* Classification Reports
* Accuracy Scores
* Macro F1 Scores
* Macro Recall Scores
* Hyperparameter Search Results
* Final Model Comparison Table

---

# Technologies Used

* Python 3.11
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Imbalanced-Learn (SMOTE)
* SQLite
* Docker
* Joblib
* YAML