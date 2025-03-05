import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve
from scipy.stats import chi2_contingency
from sklearn.calibration import calibration_curve

# Load imputed datasets
mean_imputed_path = "data/mean_imputed.csv"
median_imputed_path = "data/median_imputed.csv"
knn_imputed_path = "data/knn_imputed.csv"
iterative_imputed_path = "data/iterative_imputed.csv"

df_mean_imputed = pd.read_csv(mean_imputed_path)
df_median_imputed = pd.read_csv(median_imputed_path)
df_knn_imputed = pd.read_csv(knn_imputed_path)
df_iterative_imputed = pd.read_csv(iterative_imputed_path)

# Function to train and evaluate models
def train_and_evaluate(df, method):
    print(f"\nEvaluating model using {method} imputed data")

    X = df.drop(columns=['In-hospital_death', 'SAPS-I', 'SOFA', 'Length_of_stay', 'Survival'], errors='ignore')
    y = df['In-hospital_death']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, penalty='l1', solver='liblinear', C=1,  random_state=42),
        # 'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        # 'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        # 'NeuralNet': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Compute Sensitivity (Se) and PPV
        TP = cm[1, 1]
        FN = cm[1, 0]
        FP = cm[0, 1]
        
        Se = TP / (TP + FN) if (TP + FN) > 0 else 0  # Sensitivity (Recall)
        PPV = TP / (TP + FP) if (TP + FP) > 0 else 0  # Positive Predictive Value (Precision)
        
        # Compute χ² Calibration Statistic (Avoiding Zero Elements)
        num_bins = min(10, len(np.unique(y_probs)))  # Dynamically adjust bins
        observed, _ = np.histogram(y_test, bins=num_bins, range=(0, 1))
        predicted_bins, _ = np.histogram(y_probs, bins=num_bins, range=(0, 1))

        # Ensure nonzero values by adding a small constant
        observed = observed + 1e-6
        predicted_bins = predicted_bins + 1e-6

        chi2_stat, p_value, _, _ = chi2_contingency([observed, predicted_bins])

        # Compute H-Statistic (Sum of Squared Differences)
        prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=num_bins)
        H_statistic = np.sum((prob_true - prob_pred) ** 2)

        # Find the best threshold (maximize min(Se, PPV))
        precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
        optimal_idx = np.argmax(np.minimum(precision[:-1], recall[:-1]))
        optimal_threshold = thresholds[optimal_idx]

        print(f"{name} Accuracy: {acc:.4f}")
        print(f"Sensitivity (Se): {Se:.4f}, PPV: {PPV:.4f}, Chi-Square: {chi2_stat:.4f}, H-Statistic: {H_statistic:.4f}")
        print(f"Optimal Mortality Threshold: {optimal_threshold:.2f}")
        print(classification_report(y_test, y_pred))

        results.append((method, name, acc, Se, PPV, chi2_stat, H_statistic, optimal_threshold))
        
        # Plot confusion matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {name} ({method} Imputed Data)")
        plt.show()

    return results

# Train and evaluate models on different imputed datasets
results = []
results.extend(train_and_evaluate(df_mean_imputed, "Mean"))
results.extend(train_and_evaluate(df_median_imputed, "Median"))
results.extend(train_and_evaluate(df_knn_imputed, "KNN"))
results.extend(train_and_evaluate(df_iterative_imputed, "Itreative"))

# Convert results to DataFrame for visualization
results_df = pd.DataFrame(results, columns=["Imputation Method", "Model", "Accuracy", "Se", "PPV", "Chi2", "H-Statistic", "Optimal Threshold"])

# Plot Accuracy, Se, and PPV Comparison
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x="Imputation Method", y="Accuracy", hue="Model")
plt.ylim(0, 1)
plt.title("Model Accuracy Comparison Across Imputation Methods")
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x="Imputation Method", y="Se", hue="Model")
plt.ylim(0, 1)
plt.title("Sensitivity (Se) Comparison Across Models")
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x="Imputation Method", y="PPV", hue="Model")
plt.ylim(0, 1)
plt.title("Positive Predictive Value (PPV) Comparison Across Models")
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

print("Model evaluation completed for all imputed datasets.")
