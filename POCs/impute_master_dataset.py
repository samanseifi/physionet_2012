import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.decomposition import PCA

# Load dataset
file_path = "data/patients_master_table.csv"
df = pd.read_csv(file_path)

# Drop unnecessary columns
columns_to_drop =["_iqr", "_cv", "_percent_change", "_num_outliers", "_slope", "_first_value", "_last_value", "_std"]
df_filtered = df.drop(columns=[col for col in df.columns if any(sub in col for sub in columns_to_drop)], errors='ignore')

print(f"Initial feature count: {df_filtered.shape[1]}")

# Identify numeric columns
numeric_cols = df_filtered.select_dtypes(include=['number']).columns

# Drop completely empty numeric columns
df_filtered = df_filtered.dropna(axis=1, how='all')

# Drop columns with more than 80% missing values
threshold = 0.8 * len(df_filtered)
columns_before = df_filtered.columns
df_filtered = df_filtered.dropna(thresh=threshold, axis=1)
dropped_columns_missing = set(columns_before) - set(df_filtered.columns)

print(f"Dropped {len(dropped_columns_missing)} columns due to >80% missing values:\n{dropped_columns_missing}")

# Ensure only existing numeric columns are selected
numeric_cols = [col for col in numeric_cols if col in df_filtered.columns]

# Convert non-numeric values to NaN in numeric columns and treat -1 as missing
df_filtered[numeric_cols] = df_filtered[numeric_cols].apply(pd.to_numeric, errors='coerce')
df_filtered[numeric_cols] = df_filtered[numeric_cols].replace(-1, np.nan)

# -------------------------------------
# STEP 1: REMOVE LOW VARIANCE FEATURES
# -------------------------------------
var_thresh = VarianceThreshold(threshold=0.01)  # Remove near-constant features
columns_before = df_filtered.columns
df_filtered = pd.DataFrame(var_thresh.fit_transform(df_filtered), columns=columns_before[var_thresh.get_support()])
dropped_low_variance = set(columns_before) - set(df_filtered.columns)

print(f"Dropped {len(dropped_low_variance)} low-variance features:\n{dropped_low_variance}")

# -------------------------------------
# STEP 2: REMOVE HIGHLY CORRELATED FEATURES
# -------------------------------------
correlation_matrix = df_filtered.corr().abs()
upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Drop features with correlation > 0.9
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
df_filtered = df_filtered.drop(columns=to_drop, errors='ignore')

print(f"Dropped {len(to_drop)} highly correlated features (correlation > 0.9):\n{to_drop}")

# -------------------------------------
# STEP 3: SELECT FEATURES USING MUTUAL INFORMATION
# -------------------------------------
# Drop rows with NaN for feature selection step
df_no_na = df_filtered.dropna()

# Compute Mutual Information for Feature Selection
target_col = "target"  # Change this to your actual target column name
if target_col in df_no_na.columns:
    X = df_no_na.drop(columns=[target_col])
    y = df_no_na[target_col]
    mi_scores = mutual_info_regression(X, y)
    mi_scores_df = pd.DataFrame({"Feature": X.columns, "MI Score": mi_scores})
    mi_scores_df = mi_scores_df.sort_values(by="MI Score", ascending=False)

    # Keep the top 20 features
    top_features = mi_scores_df.head(20)["Feature"].tolist()
    df_filtered = df_filtered[top_features + [target_col]]

    print(f"Selected top {len(top_features)} features based on mutual information:\n{top_features}")

# -------------------------------------
# STEP 4: OPTIONAL PCA DIMENSIONALITY REDUCTION
# -------------------------------------
apply_pca = False  # Set to True if PCA is needed
if apply_pca:
    pca = PCA(n_components=0.95)  # Retain 95% variance
    df_filtered = pd.DataFrame(pca.fit_transform(df_filtered), columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    print(f"PCA applied: Reduced to {pca.n_components_} components.")

# -------------------------------------
# FINAL FEATURE COUNT BEFORE IMPUTATION
# -------------------------------------
print(f"Final feature count before imputation: {df_filtered.shape[1]}")

# -------------------------------------
# IMPUTATION METHODS
# -------------------------------------
# Get the updated list of numeric columns after feature reduction
updated_numeric_cols = df_filtered.select_dtypes(include=['number']).columns

# Define imputers
mean_imputer = SimpleImputer(strategy='mean')
median_imputer = SimpleImputer(strategy='median')
knn_imputer = KNNImputer(n_neighbors=20, weights='uniform', metric='nan_euclidean')
iterative_imputer = IterativeImputer(max_iter=10, random_state=42)

# Apply imputations
df_mean_imputed = df_filtered.copy()
df_mean_imputed[updated_numeric_cols] = mean_imputer.fit_transform(df_filtered[updated_numeric_cols])

df_median_imputed = df_filtered.copy()
df_median_imputed[updated_numeric_cols] = median_imputer.fit_transform(df_filtered[updated_numeric_cols])

df_knn_imputed = df_filtered.copy()
df_knn_imputed[updated_numeric_cols] = knn_imputer.fit_transform(df_filtered[updated_numeric_cols])

df_iterative_imputed = df_filtered.copy()
df_iterative_imputed[updated_numeric_cols] = iterative_imputer.fit_transform(df_filtered[updated_numeric_cols])

# Save the imputed datasets
df_mean_imputed.to_csv("data/mean_imputed.csv", index=False)
df_median_imputed.to_csv("data/median_imputed.csv", index=False)
df_knn_imputed.to_csv("data/knn_imputed.csv", index=False)
df_iterative_imputed.to_csv("data/iterative_imputed.csv", index=False)

print("\nFeature selection, reduction, and imputation completed and saved.")
