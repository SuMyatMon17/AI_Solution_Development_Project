import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class DataPreparation:
    
    #Data Ingestion
    def __init__(self, db_path="data/gas_monitoring.db"):
        self.db_path = db_path
        self.outlier_summary = {}
        os.makedirs("preprocessing_plots/outliers", exist_ok=True)

    def load_data(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql("SELECT * FROM gas_monitoring", conn)
        conn.close()
        return df

    def clean_categorical_columns(self, df):

        #Standardize categorical values by converting text to lowercase
        # removing leading/trailing spaces to ensure consistency
        df["Time of Day"] = df["Time of Day"].str.lower().str.strip()
        df["HVAC Operation Mode"] = df["HVAC Operation Mode"].str.lower().str.strip()

        df["Ambient Light Level"] = df["Ambient Light Level"].str.lower().str.strip().str.replace("_", " ", regex=False)

        df["Activity Level"] = df["Activity Level"].str.lower().str.strip().str.replace("_", " ", regex=False).replace({
            "lowactivity": "low activity",
            "moderateactivity": "moderate activity"
        # # regex=False ensures "_" is replaced as a normal character not interpreted as a regular expression.
        })

        return df

    def handle_duplicates(self, df):
        #Duplicate records were removed to prevent repeated observations from biasing statistical analysis 
        #and machine learning model training.
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)

        print("Duplicates removed:", before - after)
        return df

    def handle_missing_values(self, df):
        numeric_cols = df.select_dtypes(include=np.number).columns
        categorical_cols = df.select_dtypes(include=["object", "str"]).columns

        # Replace missing numerical values with the median.
        # The median is less sensitive to extreme values and
        # therefore more robust than the mean.
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

        return df
    

    def handle_outliers(self, df):
        numeric_cols = df.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            if col != "Session ID":
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1

                # Using IQR method to define lower and upper limit
                # 1.5 is used as A smaller multiplier (e.g., 1.0) would classify more values as
                # outliers, making the method overly sensitive.
                #A larger multiplier (e.g., 2.0 or 3.0) would classify fewer
                # values as outliers, potentially missing genuinely unusual values
                lower_limit = q1 - 1.5 * iqr
                upper_limit = q3 + 1.5 * iqr

                # Outliers Statistics 
                lower_clipped = (df[col] < lower_limit).sum()
                upper_clipped = (df[col] > upper_limit).sum()

                self.outlier_summary[col] = {
                    "lower_limit": lower_limit,
                    "upper_limit": upper_limit,
                    "lower_clipped": lower_clipped,
                    "upper_clipped": upper_clipped
                }

                df[col] = df[col].clip(lower_limit, upper_limit)

        return df
    
    def compare_outlier_effect(self, df_before, df_after):

        numeric_cols = df_before.select_dtypes(include=np.number).columns

        print("\n" + "=" * 60)
        print("OUTLIER CLIPPING IMPACT ANALYSIS")
        print("=" * 60)

        for col in numeric_cols:

            if col == "Session ID":
                continue
            # Calculate summary statistics before and after clipping
            # to evaluate whether the distribution changed significantly.
            before_mean = df_before[col].mean()
            after_mean = df_after[col].mean()

            before_median = df_before[col].median()
            after_median = df_after[col].median()

            before_std = df_before[col].std()
            after_std = df_after[col].std()

            # Count how many values were changed by clipping.
            clipped_count = (df_before[col] != df_after[col]).sum()
            clipped_percentage = (clipped_count / len(df_before)) * 100

            print(f"\nColumn: {col}")
            print("-" * 40)
            print(f"Values Clipped: {clipped_count}")
            print(f"Percentage Clipped: {clipped_percentage:.2f}%")

            lower_limit = self.outlier_summary[col]["lower_limit"]
            upper_limit = self.outlier_summary[col]["upper_limit"]

            print(f"Lower Limit: {lower_limit:.2f}")
            print(f"Upper Limit: {upper_limit:.2f}")

            # Display whether outliers were mainly low-end or high-end values.
            lower_clipped = self.outlier_summary[col]["lower_clipped"]
            upper_clipped = self.outlier_summary[col]["upper_clipped"]

            print(f"Values Clipped to Lower Limit: {lower_clipped}")
            print(f"Values Clipped to Upper Limit: {upper_clipped}")

            print(f"\nMean:")
            print(f"Before = {before_mean:.2f}")
            print(f"After  = {after_mean:.2f}")

            print(f"\nMedian:")
            print(f"Before = {before_median:.2f}")
            print(f"After  = {after_median:.2f}")

            print(f"\nStandard Deviation:")
            print(f"Before = {before_std:.2f}")
            print(f"After  = {after_std:.2f}")

            # Calculate the percentage change in mean after clipping.
            # Mean is sensitive to extreme values, so this shows whether
            # clipping significantly changed the feature distribution.
            if before_mean != 0:
                mean_change_pct = (
                    abs(after_mean - before_mean) / abs(before_mean)
                ) * 100
            else:
                mean_change_pct = 0

            # Impact Assessment Criteria:
            # < 1%  = Negligible Impact
            # 1%-5% = Small Impact
            # > 5%  = Noticeable Impact
            # This assessment is based on how much the mean changed,
            # not only how many values were clipped.
            if mean_change_pct < 1:
                impact = "Negligible Impact"
            elif mean_change_pct < 5:
                impact = "Small Impact"
            else:
                impact = "Noticeable Impact"

            print(f"\nMean Change Percentage: {mean_change_pct:.2f}%")
            print(f"Assessment: {impact}")

            # Boxplot Comparison
            plt.figure(figsize=(10, 4))

            plt.subplot(1, 2, 1)
            plt.boxplot(df_before[col])
            plt.title(f"{col}\nBefore Clipping")

            plt.subplot(1, 2, 2)
            plt.boxplot(df_after[col])
            plt.title(f"{col}\nAfter Clipping")

            plt.tight_layout()
            plt.savefig(
                f"preprocessing_plots/outliers/{col}_boxplot_comparison.png",
                bbox_inches="tight"
            )
            plt.close()

            # Histogram Comparison
            plt.figure(figsize=(8, 4))

            plt.hist(df_before[col], bins=30, alpha=0.5, label="Before")
            plt.hist(df_after[col], bins=30, alpha=0.5, label="After")

            plt.title(f"{col} Distribution")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.legend()

            plt.savefig(
                f"preprocessing_plots/outliers/{col}_histogram_comparison.png",
                bbox_inches="tight"
            )
            plt.close()

    def engineer_co2_features(self, df):
        # Calculate the average reading from both CO₂ sensors
        # to represent overall CO₂ concentration.
        df["CO2_Average"] = (
            df["CO2_InfraredSensor"] + df["CO2_ElectroChemicalSensor"]
        ) / 2

        # Measure the difference between the two CO₂ sensors.
        # Larger differences may capture additional activity-related patterns.
        df["CO2_Divergence"] = abs(
            df["CO2_InfraredSensor"] - df["CO2_ElectroChemicalSensor"]
        )

        return df

    def apply_metal_oxide_pca(self, df):
        metal_cols = [
            "MetalOxideSensor_Unit1",
            "MetalOxideSensor_Unit2",
            "MetalOxideSensor_Unit3",
            "MetalOxideSensor_Unit4"
        ]
        # Standardize Metal Oxide sensor readings before PCA
        # because PCA is affected by differences in feature scale.
        scaler = StandardScaler()
        metal_scaled = scaler.fit_transform(df[metal_cols])

        # Reduce four related Metal Oxide sensor features into one
        # principal component to reduce redundancy.
        pca = PCA(n_components=1)
        df["MetalOxide_PCA"] = pca.fit_transform(metal_scaled)

        # Display the variance of each Metal Oxide sensor to
        # the principal component. Larger absolute loading values
        # indicate a stronger influence on the PCA feature.
        print("\nMetal Oxide PCA Explained Variance:")
        print(pca.explained_variance_ratio_[0])

        loadings = pd.DataFrame(
            pca.components_.T,
            index=metal_cols,
            columns=["PC1 Loading"]
        )

        print("\nMetal Oxide PCA Loadings:")
        print(loadings)

        df = df.drop(columns=metal_cols)

        return df

    def feature_engineering(self, df):

        # Define ordinal mappings for Ambient Light Level because
        # the categories have a natural order from very dim to very bright.
        light_map = {
            "very dim": 0,
            "dim": 1,
            "moderate": 2,
            "bright": 3,
            "very bright": 4
        }
        
        # Encode the target variable into numerical classes
        # required for machine learning model training.
        activity_map = {
            "low activity": 0,
            "moderate activity": 1,
            "high activity": 2
        }

        df["Ambient Light Level"] = df["Ambient Light Level"].map(light_map)
        df["Activity Level"] = df["Activity Level"].map(activity_map)

        # Apply one-hot encoding to nominal categorical features.
        # drop_first=True removes one category from each feature
        # to reduce multicollinearity and avoid the dummy variable trap.
        df = pd.get_dummies(
            df,
            columns=[
                "Time of Day",
                "HVAC Operation Mode"
            ],
            drop_first=True,
            dtype=int
        )

        return df

    def prepare_data(self):
        df = self.load_data()

        print("Raw Dataset Shape:", df.shape)

        df = self.clean_categorical_columns(df)
        df = self.handle_missing_values(df)
        df_before_outliers = df.copy()

        df = self.handle_outliers(df)

        self.compare_outlier_effect(
            df_before_outliers,
            df
        )
        df = self.engineer_co2_features(df)
        df = self.apply_metal_oxide_pca(df)
        df = self.feature_engineering(df)
        df = self.handle_duplicates(df)
        df = df.drop(columns=["Session ID"])
        
        

        print("\nPrepared Dataset Shape:", df.shape)

        return df


if __name__ == "__main__":
    preparation = DataPreparation()

    df_prepared = preparation.prepare_data()

    df_prepared.to_csv(
        "data/processed_gas_monitoring.csv",
        index=False
    )

    print("\nPrepared dataset saved successfully.")
    print("\nFirst 5 rows:")
    print(df_prepared.head())