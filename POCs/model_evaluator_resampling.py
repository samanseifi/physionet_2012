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
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, confusion_matrix, precision_recall_curve
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from scipy.stats import chi2_contingency
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from collections import Counter

class MortalityModelEvaluatorResampling:
    def __init__(self, imputed_files, target_column="In-hospital_death", resampling_method="none"):
        """
        Initialize the class with imputed datasets, target column, and resampling strategy.
        
        resampling_method options:
        - "none": No resampling (default)
        - "smote": SMOTE (Synthetic Minority Over-sampling)
        - "over": Random Oversampling
        - "under": Random Undersampling
        - "hybrid": SMOTETomek (Combination of SMOTE and undersampling)
        """
        self.imputed_files = imputed_files
        self.target_column = target_column
        self.resampling_method = resampling_method
        self.datasets = self.load_datasets()
        self.results = []

    def load_datasets(self):
        datasets = {}
        for method, path in self.imputed_files.items():
            datasets[method] = pd.read_csv(path)
        return datasets

    def apply_resampling(self, X_train, y_train):
        """
        Apply the selected resampling method to handle class imbalance.
        """
        print(f"Applying {self.resampling_method} resampling...")
        
        if self.resampling_method == "smote":
            resampler = SMOTE(random_state=42)
        elif self.resampling_method == "over":
            resampler = RandomOverSampler(random_state=42)
        elif self.resampling_method == "under":
            resampler = RandomUnderSampler(random_state=42)
        elif self.resampling_method == "hybrid":
            resampler = SMOTETomek(random_state=42)
        else:
            return X_train, y_train  # No resampling

        X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
        print(f"New class distribution: {Counter(y_resampled)}")
        return X_resampled, y_resampled

    def train_and_evaluate(self, df, method):
        """
        Train and evaluate ML models using a given imputed dataset with resampling.
        """
        print(f"\nEvaluating models using {method} imputed data with {self.resampling_method} resampling")

        # Define features and target
        X = df.drop(columns=['In-hospital_death', 'SAPS-I', 'SOFA', 'Length_of_stay', 'Survival'], errors='ignore')
        y = df[self.target_column]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Apply resampling strategy
        X_train, y_train = self.apply_resampling(X_train, y_train)

        # Standardization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Compute class imbalance ratio
        class_counts = Counter(y_train)
        imbalance_ratio = class_counts[0] / class_counts[1] if class_counts[1] != 0 else 1  # To avoid division by zero

        # Define models with class weighting
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, penalty='l1', class_weight='balanced', solver='liblinear', C=1, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
            'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'AdaBoost': AdaBoostClassifier(n_estimators=200, random_state=42),
            'Bagging': BaggingClassifier(n_estimators=200, random_state=42),
            'Extra Trees': ExtraTreesClassifier(n_estimators=200, random_state=42),
            'GaussianNB': GaussianNB(),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
            'SVM': SVC(probability=True, kernel='rbf', C=1, random_state=42, class_weight='balanced'),
            'XGBoost': XGBClassifier(n_estimators=200, scale_pos_weight=imbalance_ratio, eval_metric='logloss', random_state=42),
            'Neural Net': MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42)
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_probs = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            # Compute Sensitivity (Recall) and Precision
            TP = cm[1, 1]
            FN = cm[1, 0]
            FP = cm[0, 1]
            Se = TP / (TP + FN) if (TP + FN) > 0 else 0
            PPV = TP / (TP + FP) if (TP + FP) > 0 else 0
            f1 = 2 * (PPV * Se) / (PPV + Se) if (PPV + Se) > 0 else 0

            # Compute χ² Calibration Statistic
            num_bins = min(10, len(np.unique(y_probs)))
            prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=num_bins)
            H_statistic = np.sum((prob_true - prob_pred) ** 2)

            # Find optimal threshold based on min(Sensitivity, Precision)
            precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
            optimal_idx = np.argmax(np.minimum(precision[:-1], recall[:-1]))
            optimal_threshold = thresholds[optimal_idx]

            event_1_performance = min(Se, PPV)

            print(f"{name} - Accuracy: {acc:.4f}, Sensitivity: {Se:.4f}, Precision: {PPV:.4f}, F1-score: {f1:.4f}, Event-1: {event_1_performance:.4f}")
            print(classification_report(y_test, y_pred))

            self.results.append((method, self.resampling_method, name, acc, Se, PPV, f1, H_statistic, optimal_threshold, event_1_performance))

    def run(self):
        for method, df in self.datasets.items():
            self.train_and_evaluate(df, method)

        # Convert results to DataFrame
        self.results_df = pd.DataFrame(
            self.results, 
            columns=["Imputation Method", "Resampling Method", "Model", "Accuracy", "Se", "PPV", "F1-score", "H-Statistic", "Optimal Threshold", "Event-1"]
        )

        print("\nFinal Ranked Model Performance:")
        print(self.results_df)
