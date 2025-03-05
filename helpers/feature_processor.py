import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.decomposition import PCA


def FeatureProcess(data, percent_of_missing=0.6, low_variance_threshold=0.01, correlation_threshold=0.95):
    
    # replace -1 values with nan in age, height, weight, gender
    data['age'] = data['age'].replace(-1, pd.NA)
    data['height'] = data['height'].replace(-1, pd.NA)
    data['weight'] = data['weight'].replace(-1, pd.NA)

    # Step 2: Drop completely empty numeric columns
    data = data.dropna(axis=1, how='all')
    print(f"Number of columns after dropping empty columns: {data.shape[1]}")

    # Step 3: Print how many missing values in percentage are in each column
    missing_percentage = (data.isna()).mean() * 100

    # print the non zero ones
    print("Columns with missing values in percentage:")
    print(missing_percentage[missing_percentage > percent_of_missing * 100])

    data = data.loc[:, missing_percentage[missing_percentage < percent_of_missing * 100].index]
    print(f"Number of columns after removing columns with more than missing values: {data.shape[1]}")

    # Step 4: Remove columns with low variance
    # Create a VarianceThreshold object
    selector = VarianceThreshold(threshold=low_variance_threshold)
    # Fit the selector to the data
    selector.fit(data)
    # Get the columns that are retained
    retained_columns = data.columns[selector.get_support()]
    # Create a new DataFrame with the retained columns
    data = data[retained_columns]
    print(f"Number of columns after removing low variance columns: {data.shape[1]}")

    # Step 4: Remove highly correlated columns
    # Compute the correlation matrix
    correlation_matrix = data.corr().abs()
    # Select upper triangle of correlation matrix
    upper = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    # Find index of feature columns with correlation greater than percent_of_missing
    to_drop = [column for column in upper.columns if any(upper[column] > coorelation_threshold)]
    
    # Drop the features
    data = data.drop(data[to_drop], axis=1)
    print(f"Number of columns after removing highly correlated columns: {data.shape[1]}")
    
    return data


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



