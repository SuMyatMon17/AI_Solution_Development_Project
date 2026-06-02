import sqlite3
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class DataPreparation:

    def __init__(self, db_path="data/gas_monitoring.db"):
        self.db_path = db_path

    def load_data(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql("SELECT * FROM gas_monitoring", conn)
        conn.close()
        return df

    def clean_categorical_columns(self, df):
        df["Time of Day"] = df["Time of Day"].str.lower().str.strip()
        df["HVAC Operation Mode"] = df["HVAC Operation Mode"].str.lower().str.strip()

        df["Ambient Light Level"] = df["Ambient Light Level"].str.lower().str.strip().str.replace("_", " ", regex=False)

        df["Activity Level"] = df["Activity Level"].str.lower().str.strip().str.replace("_", " ", regex=False).replace({
            "lowactivity": "low activity",
            "moderateactivity": "moderate activity"
        })

        return df

    def handle_duplicates(self, df):
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)

        print("Duplicates removed:", before - after)
        return df

    def handle_missing_values(self, df):
        numeric_cols = df.select_dtypes(include=np.number).columns
        categorical_cols = df.select_dtypes(include=["object", "str"]).columns

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

                lower_limit = q1 - 1.5 * iqr
                upper_limit = q3 + 1.5 * iqr

                df[col] = df[col].clip(lower_limit, upper_limit)

        return df

    def engineer_co2_features(self, df):
        df["CO2_Average"] = (
            df["CO2_InfraredSensor"] + df["CO2_ElectroChemicalSensor"]
        ) / 2

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

        scaler = StandardScaler()
        metal_scaled = scaler.fit_transform(df[metal_cols])

        pca = PCA(n_components=1)
        df["MetalOxide_PCA"] = pca.fit_transform(metal_scaled)

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
        light_map = {
            "very dim": 0,
            "dim": 1,
            "moderate": 2,
            "bright": 3,
            "very bright": 4
        }

        activity_map = {
            "low activity": 0,
            "moderate activity": 1,
            "high activity": 2
        }

        df["Ambient Light Level"] = df["Ambient Light Level"].map(light_map)
        df["Activity Level"] = df["Activity Level"].map(activity_map)

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
        df = self.handle_outliers(df)
        df = self.engineer_co2_features(df)
        df = self.apply_metal_oxide_pca(df)
        df = self.feature_engineering(df)

        df = df.drop(columns=["Session ID"])
        df = self.handle_duplicates(df)
        

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