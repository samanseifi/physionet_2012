from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class Imputer:
    def __init__(self, df, output_path="data/"):
        self.df = df
        self.output_path = output_path
        self.numeric_cols = self.df.select_dtypes(include=['number']).columns

    def apply_imputations(self):
        """Apply multiple imputation methods and save results."""
        print("Applying imputations...")

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

    def process(self):
        """Execute imputation."""
        self.apply_imputations()
        print("Imputation completed.")
