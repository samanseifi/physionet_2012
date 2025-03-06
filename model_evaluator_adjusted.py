import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, classification_report
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from scipy.stats import chi2_contingency
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
from collections import Counter


class MortalityModelEvaluatorAdjusted:
    def __init__(self, imputed_files, target_column="In-hospital_death"):
        self.imputed_files = imputed_files
        self.target_column = target_column
        self.datasets = self.load_datasets()
        self.results = []
        self.feature_importance_data = {}  # Store feature importance separately

    def load_datasets(self):
        datasets = {}
        for method, path in self.imputed_files.items():
            datasets[method] = pd.read_csv(path)
        return datasets

    def train_and_evaluate(self, df, method):
        print(f"\nEvaluating models using {method} imputed data")

        # Define features and target
        X = df.drop(columns=[self.target_column, 'SAPS-I', 'SOFA', 'Length_of_stay', 'Survival'], errors='ignore')
        y = df[self.target_column]
        feature_names = X.columns.tolist()

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Standardization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Compute class imbalance ratio
        class_counts = Counter(y_train)
        imbalance_ratio = class_counts[0] / class_counts[1] if class_counts[1] != 0 else 1  # Avoid division by zero

        class_weights = {0: 1, 1: imbalance_ratio}

        # Define models including SVM & Neural Net
        models = {
            'Logistic Regression': LogisticRegression(max_iter=2000, penalty='l2', class_weight=class_weights, solver='lbfgs', C=0.5, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, class_weight=class_weights, random_state=42),
            'Decision Tree': DecisionTreeClassifier(max_depth=10, min_samples_split=5, class_weight=class_weights, random_state=42),
            'Extra Trees': ExtraTreesClassifier(n_estimators=300, max_depth=10, min_samples_split=5, class_weight=class_weights, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, min_samples_split=5, random_state=42),
            'SVM': SVC(probability=True, kernel='rbf', C=0.5, gamma='scale', class_weight=class_weights, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, scale_pos_weight=imbalance_ratio, eval_metric='logloss', colsample_bytree=0.8, subsample=0.8, random_state=42),
            'Neural Net': MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', alpha=0.0001, max_iter=1000, early_stopping=True, random_state=42)
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_probs = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            TP = cm[1, 1]
            FN = cm[1, 0]
            FP = cm[0, 1]
            Se = TP / (TP + FN) if (TP + FN) > 0 else 0  # Sensitivity (Recall)
            PPV = TP / (TP + FP) if (TP + FP) > 0 else 0  # Precision
            f1 = 2 * (PPV * Se) / (PPV + Se) if (PPV + Se) > 0 else 0  # F1-score

            # Compute H-Statistic
            prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)
            H_statistic = np.sum((prob_true - prob_pred) ** 2)

            # Compute D (range between top and bottom deciles)
            π10, π1 = np.percentile(y_probs, [90, 10])
            D = π10 - π1 if π10 > π1 else 1e-6  # Avoid division by zero

            # Compute Event-2 Score
            event_2_score = H_statistic / D

            event_1_performance = min(Se, PPV)

            print(f"{name} - Accuracy: {acc:.4f}, Sensitivity: {Se:.4f}, Precision: {PPV:.4f}, Event-1: {event_1_performance:.4f}")

            self.results.append((method, name, acc, Se, PPV, f1, H_statistic, event_1_performance, event_2_score))

            # Store feature importance for tree-based models
            if hasattr(model, "feature_importances_"):
                self.feature_importance_data.setdefault(method, {})[name] = (model.feature_importances_, feature_names)

    def evaluate_all(self):
        for method, df in self.datasets.items():
            self.train_and_evaluate(df, method)

        self.results_df = pd.DataFrame(
            self.results, 
            columns=["Imputation Method", "Model", "Accuracy", "Se", "PPV", "F1-score", "H-Statistic", "Event-1", "Event-2"]
        )

        self.results_df = self.results_df.sort_values(by="Event-2", ascending=False)

    def run(self):
        self.evaluate_all()
        print("\nFinal Ranked Model Performance:")
        print(self.results_df)

class FeatureImportanceVisualizer:
    def __init__(self, feature_importance_data, top_n_features=10):
        self.feature_importance_data = feature_importance_data
        self.top_n_features = top_n_features

    def plot_feature_importance(self):
        for method, model_importances in self.feature_importance_data.items():
            for model_name, (importances, feature_names) in model_importances.items():
                sorted_idx = np.argsort(importances)[::-1][:self.top_n_features]
                top_features = np.array(feature_names)[sorted_idx]
                top_importances = np.array(importances)[sorted_idx]

                print(f"\nTop {self.top_n_features} Features for {model_name} ({method} Imputation):")
                print(top_features)

                plt.figure(figsize=(10, 6))
                sns.barplot(x=top_importances, y=top_features, palette="viridis")
                plt.title(f"Top {self.top_n_features} Feature Importance for {model_name} ({method} Imputation)")
                plt.xlabel("Importance Score")
                plt.ylabel("Feature Name")
                plt.show()

    def run(self):
        self.plot_feature_importance()


