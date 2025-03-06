import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.decomposition import PCA
import numpy as np
import os
import joblib

class Imputer:
    def __init__(self, df, output_path="data/"):
        self.df = df
        self.output_path = output_path
        self.numeric_cols = self.df.select_dtypes(include=['number']).columns
        os.makedirs(self.output_path, exist_ok=True)

    def apply_imputations(self):
        """Apply multiple imputation methods and save results."""
        print("Applying imputations...")
        
        # Ignoring the column with patient_id
        self.numeric_cols = self.numeric_cols.drop(['patient_id', 'In-hospital_death'], errors='ignore')

        imputers = {
            "mean": SimpleImputer(strategy='mean'),
            "median": SimpleImputer(strategy='median'),
            "knn": KNNImputer(n_neighbors=20, weights='uniform', metric='nan_euclidean'),
            "iterative": IterativeImputer(max_iter=10, random_state=42)
        }

        for method, imputer in imputers.items():
            df_imputed = self.df.copy()
            df_imputed[self.numeric_cols] = imputer.fit_transform(self.df[self.numeric_cols])
            file_path = f"{self.output_path}{method}_imputed.csv"
            df_imputed.to_csv(file_path, index=False)
            print(f"Saved {method} imputed dataset at {file_path}.")

    def process(self, apply_pca=False, n_components=0.95):
        """Execute imputation and optionally apply PCA."""
        self.apply_imputations()
        
        if apply_pca:
            for method in ["mean", "median", "knn", "iterative"]:
                file_path = f"{self.output_path}{method}_imputed.csv"
                df_imputed = pd.read_csv(file_path)
                
                print(f"Applying PCA on {method} imputed dataset...")
                pca = PCA(n_components=n_components)
                df_numeric = df_imputed[self.numeric_cols].dropna(axis=1)
                df_pca = pca.fit_transform(df_numeric)
                
                # Save PCA model
                pca_model_path = f"{self.output_path}{method}_pca_model.pkl"
                joblib.dump(pca, pca_model_path)
                print(f"Saved PCA model at {pca_model_path}")
                
                reduced_columns = [f"PC{i+1}" for i in range(df_pca.shape[1])]
                df_pca = pd.DataFrame(df_pca, columns=reduced_columns)
                df_pca.insert(0, "patient_id", df_imputed["patient_id"])  # Retain patient_id
                df_pca.insert(1, "In-hospital_death", df_imputed["In-hospital_death"])  # Retain target variable
                
                pca_file_path = f"{self.output_path}{method}_imputed_pca.csv"
                df_pca.to_csv(pca_file_path, index=False)
                print(f"PCA reduced {df_numeric.shape[1]} features to {df_pca.shape[1]} features. Saved at {pca_file_path}.")
        
        print("Processing completed.")
    
    def apply_pca_to_test_data(self, test_df, method):
        """Apply pre-trained PCA to test dataset."""
        pca_model_path = f"{self.output_path}{method}_pca_model.pkl"
        if not os.path.exists(pca_model_path):
            raise FileNotFoundError(f"PCA model for {method} imputation not found. Run process() first.")
        
        print(f"Applying saved PCA model from {pca_model_path} to test data...")
        pca = joblib.load(pca_model_path)
        test_numeric = test_df[self.numeric_cols].dropna(axis=1)
        test_pca = pca.transform(test_numeric)
        
        reduced_columns = [f"PC{i+1}" for i in range(test_pca.shape[1])]
        test_pca_df = pd.DataFrame(test_pca, columns=reduced_columns)
        test_pca_df.insert(0, "patient_id", test_df["patient_id"])  # Retain patient_id
        test_pca_df.insert(1, "In-hospital_death", test_df["In-hospital_death"])  # Retain target variable
        
        test_pca_file_path = f"{self.output_path}{method}_test_pca.csv"
        test_pca_df.to_csv(test_pca_file_path, index=False)
        print(f"PCA applied to test data and saved at {test_pca_file_path}.")
