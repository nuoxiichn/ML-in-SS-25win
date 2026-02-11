import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mlflow
import mlflow.sklearn
import mlflow.keras
import random

# Machine Learning Imports
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Deep Learning Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# --- 设置随机种子以保证结果可复现 ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# ----------------------------------

# ---------------------------------------------------------
# 1. Data Preparation
# ---------------------------------------------------------
print("Fetching dataset...")
website_phishing = fetch_ucirepo(id=379)
X = website_phishing.data.features
y = website_phishing.data.targets.iloc[:, 0]

# Split for ML
X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(X, y, test_size=0.2, random_state=42)

# Prepare for DL (Requires numeric mapping and one-hot encoding)
y_mapped = y.values + 1
y_dl = tf.keras.utils.to_categorical(y_mapped, num_classes=3)
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(X.values, y_dl, test_size=0.2, random_state=42)

results = {} # Store results for final comparison

# ---------------------------------------------------------
# 2. MLflow Setup
# ---------------------------------------------------------
mlflow.set_experiment("Phishing_ML_vs_DL_Comprehensive")

with mlflow.start_run(run_name="Ultimate_Comparison_Parent"):
    
    # --- SECTION A: Traditional Machine Learning ---
    ml_models = {
        "Extra_Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
        "Gradient_Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    for name, model in ml_models.items():
        with mlflow.start_run(run_name=f"ML_{name}", nested=True):
            model.fit(X_train_ml, y_train_ml)
            y_pred = model.predict(X_test_ml)
            acc = accuracy_score(y_test_ml, y_pred)
            results[name] = acc
            
            mlflow.log_param("type", "ML")
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, f"model_{name}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test_ml, y_pred)
            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f"CM: {name}")
            plt.savefig(f"cm_{name}.png")
            mlflow.log_artifact(f"cm_{name}.png")
            plt.close()

    # --- SECTION B: Deep Learning (TensorFlow/Keras) ---
    def get_dl_model(type):
        model = Sequential()
        if type == "DL_Baseline":
            model.add(Dense(64, activation='relu', input_shape=(X_train_dl.shape[1],)))
            model.add(Dense(32, activation='relu'))
        elif type == "DL_Deep":
            model.add(Dense(128, activation='relu', input_shape=(X_train_dl.shape[1],)))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(64, activation='relu'))
        elif type == "DL_Regularized":
            model.add(Dense(128, activation='relu', input_shape=(X_train_dl.shape[1],)))
            model.add(Dropout(0.3))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.2))
        
        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    dl_configs = ["DL_Baseline", "DL_Deep", "DL_Regularized"]
    for name in dl_configs:
        with mlflow.start_run(run_name=name, nested=True):
            model = get_dl_model(name)
            history = model.fit(X_train_dl, y_train_dl, epochs=50, batch_size=32, verbose=0, validation_split=0.1)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # 图1：Accuracy 变化
            ax1.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
            ax1.plot(history.history['val_accuracy'], label='Test Accuracy', color='orange')
            ax1.set_title(f'{name} - Accuracy Curves')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True)

            # 图2：Loss 变化
            ax2.plot(history.history['loss'], label='Train Loss', color='blue')
            ax2.plot(history.history['val_loss'], label='Test Loss', color='orange')
            ax2.set_title(f'{name} - Loss Curves')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Loss (Categorical Crossentropy)')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            curve_plot = f"learning_curves_{name}.png"
            plt.savefig(curve_plot)
            mlflow.log_artifact(curve_plot) # 记录到 MLflow
            plt.close()
            
            y_pred_dl = np.argmax(model.predict(X_test_dl), axis=1)
            y_true_dl = np.argmax(y_test_dl, axis=1)
            acc = accuracy_score(y_true_dl, y_pred_dl)
            results[name] = acc
            
            mlflow.log_param("type", "DL")
            mlflow.log_metric("accuracy", acc)
            mlflow.keras.log_model(model, f"model_{name}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_true_dl, y_pred_dl)
            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
            plt.title(f"CM: {name}")
            plt.savefig(f"cm_{name}.png")
            mlflow.log_artifact(f"cm_{name}.png")
            plt.close()

    # --- SECTION C: Final Comparison Plot (Optimized for Visual Impact) ---
    # 1. Sort results by accuracy in descending order
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    
    plt.figure(figsize=(14, 7))
    
    # 2. Assign colors based on sorted keys
    colors = ['skyblue' if 'DL' not in k else 'salmon' for k in sorted_results.keys()]
    
    # 3. Create the bar plot
    bars = plt.bar(sorted_results.keys(), sorted_results.values(), color=colors, edgecolor='black', alpha=0.8)
    
    # 4. Add text labels on top of bars for precision
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.ylabel("Accuracy Score", fontsize=12)
    plt.title("Model Performance Comparison: Ranked ML vs DL", fontsize=14, pad=20)
    
    # 5. Zoom in on the Y-axis to make differences visible
    # We set the bottom to slightly less than the minimum accuracy found
    min_acc = min(sorted_results.values())
    plt.ylim(min_acc - 0.01, 1.0) 
    
    plt.xticks(rotation=30, ha='right', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save and Log
    plt.savefig("final_comparison.png")
    mlflow.log_artifact("final_comparison.png")
    plt.show()

print("\nAll experiments done. Use 'mlflow ui' to see details.")