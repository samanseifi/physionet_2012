import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve
from scipy.stats import chi2_contingency
from sklearn.calibration import calibration_curve


class MortalityModelEvaluator:
    def __init__(self, imputed_files, target_column="In-hospital_death"):
        """
        Initialize the class with imputed datasets and target column.
        """
        self.imputed_files = imputed_files
        self.target_column = target_column
        self.datasets = self.load_datasets()
        self.results = []

    def load_datasets(self):
        """
        Load and store imputed datasets.
        """
        datasets = {}
        for method, path in self.imputed_files.items():
            datasets[method] = pd.read_csv(path)
        return datasets

    def train_and_evaluate(self, df, method):
        """
        Train and evaluate ML models using a given imputed dataset.
        """
        print(f"\nEvaluating models using {method} imputed data")

        # Define features and target
        X = df.drop(columns=['In-hospital_death', 'SAPS-I', 'SOFA', 'Length_of_stay', 'Survival'], errors='ignore')
        y = df[self.target_column]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Standardization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Define models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, penalty='l1', class_weight='balanced', solver='liblinear', C=1, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'AdaBoost': AdaBoostClassifier(n_estimators=200, random_state=42),
            'Bagging': BaggingClassifier(n_estimators=200, random_state=42),
            'Extra Trees': ExtraTreesClassifier(n_estimators=200, random_state=42),
            'GaussianNB': GaussianNB(),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
            'SVM': SVC(probability=True, kernel='rbf', C=1, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42),
            'Neural Net': MLPClassifier(hidden_layer_sizes=(50,50,), max_iter=500, random_state=42)
        }

        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_probs = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            # Compute Sensitivity (Se) and PPV
            TP = cm[1, 1]
            FN = cm[1, 0]
            FP = cm[0, 1]
            Se = TP / (TP + FN) if (TP + FN) > 0 else 0  # Sensitivity (Recall)
            PPV = TP / (TP + FP) if (TP + FP) > 0 else 0  # Positive Predictive Value (Precision)

            # Compute χ² Calibration Statistic
            num_bins = min(10, len(np.unique(y_probs)))
            observed, _ = np.histogram(y_test, bins=num_bins, range=(0, 1))
            predicted_bins, _ = np.histogram(y_probs, bins=num_bins, range=(0, 1))

            observed = observed + 1e-6  # Avoid zero values
            predicted_bins = predicted_bins + 1e-6

            chi2_stat, p_value, _, _ = chi2_contingency([observed, predicted_bins])

            # Compute H-Statistic
            prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=num_bins)
            H_statistic = np.sum((prob_true - prob_pred) ** 2)

            # Find optimal threshold based on min(Sensitivity, PPV)
            precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
            optimal_idx = np.argmax(np.minimum(precision[:-1], recall[:-1]))
            optimal_threshold = thresholds[optimal_idx]

            event_1_performance = min(Se, PPV)  # Ranking metric

            print(f"{name} - Accuracy: {acc:.4f}, Sensitivity: {Se:.4f}, PPV: {PPV:.4f}, Event-1: {event_1_performance:.4f}")
            print(classification_report(y_test, y_pred))

            self.results.append((method, name, acc, Se, PPV, chi2_stat, H_statistic, optimal_threshold, event_1_performance))

    def evaluate_all(self):
        """
        Train and evaluate models on all imputed datasets.
        """
        for method, df in self.datasets.items():
            self.train_and_evaluate(df, method)

        # Convert results to DataFrame
        self.results_df = pd.DataFrame(
            self.results, 
            columns=["Imputation Method", "Model", "Accuracy", "Se", "PPV", "Chi2", "H-Statistic", "Optimal Threshold", "Event-1"]
        )

        # Rank models based on Event-1 performance
        self.results_df = self.results_df.sort_values(by="Event-1", ascending=False)

    def visualize_results(self):
        """
        Plot model performance metrics across different imputation methods.
        """
        metrics = ["Accuracy", "Se", "PPV", "Event-1"]
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            sns.barplot(data=self.results_df, x="Imputation Method", y=metric, hue="Model")
            plt.ylim(0, 1)
            plt.title(f"{metric} Comparison Across Models")
            plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()

    def run(self):
        """
        Execute the entire workflow.
        """
        self.evaluate_all()
        self.visualize_results()
        print("\nFinal Ranked Model Performance:")
        print(self.results_df)



