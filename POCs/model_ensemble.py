import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve
from scipy.stats import chi2_contingency
from sklearn.calibration import calibration_curve

class MortalityModelEnsemble:
    def __init__(self, imputed_files, target_column="In-hospital_death"):
        self.imputed_files = imputed_files
        self.target_column = target_column
        self.datasets = self.load_datasets()
        self.results = []
        self.top_models = None

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
        Train models without grid search and evaluate them.
        """
        print(f"\nEvaluating models using {method} imputed data")

        X = df.drop(columns=['In-hospital_death', 'SAPS-I', 'SOFA', 'Length_of_stay', 'Survival'], errors='ignore')
        y = df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Standardization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Define models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, solver='liblinear', random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'Neural Net': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_probs = model.predict_proba(X_test)[:, 1]

            cm = confusion_matrix(y_test, y_pred)
            TP, FN, FP = cm[1, 1], cm[1, 0], cm[0, 1]
            Se = TP / (TP + FN) if (TP + FN) > 0 else 0
            PPV = TP / (TP + FP) if (TP + FP) > 0 else 0
            event_1_performance = min(Se, PPV)

            self.results.append((method, name, model, Se, PPV, event_1_performance))

    def evaluate_all(self):
        """
        Train and evaluate models on all imputed datasets.
        """
        for method, df in self.datasets.items():
            self.train_and_evaluate(df, method)

        self.results_df = pd.DataFrame(self.results, columns=["Imputation Method", "Model", "Trained Model", "Se", "PPV", "Event-1"])
        self.results_df = self.results_df.sort_values(by="Event-1", ascending=False)

        # Select top 10 models for grid search
        self.top_models = self.results_df.head(10)

    def grid_search_top_models(self):
        """
        Perform Grid Search on the top 10 models.
        """
        grid_results = []
        for _, row in self.top_models.iterrows():
            method, model_name, model = row["Imputation Method"], row["Model"], row["Trained Model"]
            df = self.datasets[method]

            X = df.drop(columns=['In-hospital_death', 'SAPS-I', 'SOFA', 'Length_of_stay', 'Survival'], errors='ignore')
            y = df[self.target_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            param_grid = {
                'Logistic Regression': {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2']},
                'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10]},
                'Gradient Boosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
                'SVM': {'C': [0.1, 1], 'kernel': ['linear', 'rbf']},
                'XGBoost': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
                'Neural Net': {'hidden_layer_sizes': [(50,), (100,)], 'learning_rate_init': [0.001, 0.01]}
            }

            grid_search = GridSearchCV(model, param_grid[model_name], cv=3, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            grid_results.append((method, model_name, best_params, best_model))

        self.grid_results_df = pd.DataFrame(grid_results, columns=["Imputation Method", "Model", "Best Params", "Tuned Model"])

    def build_ensemble(self):
        """
        Create an ensemble of the top tuned models.
        """
        top_tuned_models = self.grid_results_df["Tuned Model"].tolist()[:5]
        ensemble_model = VotingClassifier(estimators=[(f"Model_{i}", model) for i, model in enumerate(top_tuned_models)], voting='soft')
        return ensemble_model

    def run(self):
        """
        Execute the entire workflow.
        """
        self.evaluate_all()
        print("\nTop 10 Models Based on Event-1 Performance:")
        print(self.top_models)

        self.grid_search_top_models()
        print("\nGrid Search Completed. Best models found.")

        ensemble_model = self.build_ensemble()
        print("\nEnsemble Model Built.")
        return ensemble_model


