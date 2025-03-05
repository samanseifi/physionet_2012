import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

class MortalityNeuralNet:
    def __init__(self, imputed_files, target_column="In-hospital_death"):
        self.imputed_files = imputed_files
        self.target_column = target_column
        self.datasets = self.load_datasets()
        self.history = None  # Store training history

    def load_datasets(self):
        """
        Load and preprocess imputed datasets safely.
        """
        datasets = {}
        for method, path in self.imputed_files.items():
            if not os.path.exists(path):
                print(f"⚠️ Warning: File not found {path}. Skipping...")
                continue
            
            try:
                df = pd.read_csv(path)

                # Convert all columns to numeric (handling errors)
                df = df.apply(pd.to_numeric, errors='coerce')

                # Check for NaN values after conversion
                if df.isnull().sum().sum() > 0:
                    print(f"⚠️ Warning: NaN values found in {method} dataset after conversion. Filling with mean.")
                    df.fillna(df.mean(), inplace=True)

                datasets[method] = df

            except Exception as e:
                print(f"❌ Error loading {method} dataset from {path}: {e}")
        
        return datasets

    def build_model(self, input_dim):
        """
        Define a deep neural network model with best practices.
        """
        model = keras.Sequential([
            Input(shape=(input_dim,)),  # Input layer
            Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),  
            BatchNormalization(),  # Normalizes activations
            Dropout(0.3),  # Prevents overfitting
            Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1, activation='sigmoid')  # Output layer (binary classification)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001), 
            loss='binary_crossentropy',  
            metrics=['accuracy', keras.metrics.AUC(name="AUC"), keras.metrics.Precision(), keras.metrics.Recall()]
        )

        return model

    def train_and_evaluate(self, df, method):
        """
        Train and evaluate the neural network on a given imputed dataset.
        """
        print(f"\n🔹 Training Neural Network using {method} imputed data")

        # Drop non-feature columns and separate target variable
        drop_cols = ['In-hospital_death', 'SAPS-I', 'SOFA', 'Length_of_stay', 'Survival']
        X = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
        y = df[self.target_column]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Standardization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Build model
        model = self.build_model(input_dim=X_train.shape[1])

        # Callbacks for training
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

        # Train model
        self.history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, lr_scheduler],
            verbose=1
        )

        # Evaluate model
        evaluation = model.evaluate(X_test, y_test, verbose=0)
        print(f"✅ {method} Dataset - Test Accuracy: {evaluation[1]:.4f}, AUC: {evaluation[2]:.4f}, Precision: {evaluation[3]:.4f}, Recall: {evaluation[4]:.4f}")

        # Store training history and plot results
        self.plot_training_history(method)

    def plot_training_history(self, method):
        """
        Plot training vs validation loss and other metrics.
        """
        plt.figure(figsize=(12, 5))

        # Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Val Loss')
        plt.title(f'Training vs Validation Loss ({method})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # AUC Plot
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['AUC'], label='Train AUC')
        plt.plot(self.history.history['val_AUC'], label='Val AUC')
        plt.title(f'AUC Score ({method})')
        plt.xlabel('Epochs')
        plt.ylabel('AUC')
        plt.legend()

        plt.show()

    def evaluate_all(self):
        """
        Train and evaluate the neural network on all imputed datasets.
        """
        for method, df in self.datasets.items():
            self.train_and_evaluate(df, method)

    def run(self):
        """
        Execute the entire workflow.
        """
        self.evaluate_all()


import pandas as pd
import numpy as np
import os

file_paths = [
    "data/mean_imputed.csv",
    "data/median_imputed.csv",
    "data/knn_imputed.csv",
    "data/iterative_imputed.csv"
]

for file in file_paths:
    if not os.path.exists(file):
        print(f"❌ File not found: {file}")
        continue

    try:
        # Load without automatic type conversion
        df = pd.read_csv(file, dtype=str, encoding='utf-8')

        print(f"\n✅ Successfully loaded {file}")
        print(df.head())  # Show first few rows
        print(df.dtypes)  # Show column types

        # Convert to numeric, forcing errors to NaN
        df = df.apply(pd.to_numeric, errors='coerce')

        # Check for NaN values
        if df.isnull().sum().sum() > 0:
            print(f"⚠️ Warning: NaN values found in {file}. Filling with mean.")
            df.fillna(df.mean(), inplace=True)

        # Save fixed version
        fixed_path = file.replace(".csv", "_fixed.csv")
        df.to_csv(fixed_path, index=False)
        print(f"✅ Fixed file saved as {fixed_path}")

    except Exception as e:
        print(f"❌ Error loading {file}: {e}")
