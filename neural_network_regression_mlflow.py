import os
import io
import time
import hashlib
import warnings
import platform
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)
tf.random.set_seed(42)

PATH_PROJECT: str = os.path.dirname(os.path.abspath(__file__))
PATH_CSV = os.path.join(PATH_PROJECT, "Results_CSV")
PATH_PNG = os.path.join(PATH_PROJECT, "Results_PNG")

# Create directories if they don't exist
for path in [PATH_CSV, PATH_PNG]:
    os.makedirs(path, exist_ok=True)

# Set MLflow experiment
EXPERIMENT_NAME = "airfoil-noise-neural-network"

# =========================================================
# Utils
# =========================================================
def dataset_hash(df: pd.DataFrame) -> str:
    """Hash del dataset para trazabilidad."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.md5(csv_bytes).hexdigest()

def df_info_str(df: pd.DataFrame) -> str:
    """Captura df.info() como texto."""
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()

def save_and_log_fig(fig: plt.Figure, path: str):
    """Guarda y registra figura en MLflow."""
    fig.savefig(path, dpi=120, bbox_inches="tight")
    mlflow.log_artifact(path)
    plt.close(fig)

def remove_outliers_iqr_df(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Elimina outliers aplicando IQR."""
    filt = pd.Series(True, index=df.index)
    for c in columns:
        Q1 = df[c].quantile(0.25)
        Q3 = df[c].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        filt &= (df[c] >= lower) & (df[c] <= upper)
    return df.loc[filt].copy()

def r2_adjusted(y_true, y_pred, n, p):
    """Calcula R² ajustado."""
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.where(y_true == 0, np.finfo(float).eps, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom))

def plot_history(history, title="", save_path_prefix=""):
    """Visualiza historial de entrenamiento (loss y métricas)."""
    # Loss plot
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(history.history["loss"], label="Train Loss", linewidth=2)
    ax1.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    ax1.set_title(f"Loss - {title}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    save_and_log_fig(fig1, f"{save_path_prefix}_loss.png")

    # MAE plot if available
    if "mae" in history.history:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.plot(history.history["mae"], label="Train MAE", linewidth=2)
        ax2.plot(history.history["val_mae"], label="Val MAE", linewidth=2)
        ax2.set_title(f"MAE - {title}")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MAE")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        save_and_log_fig(fig2, f"{save_path_prefix}_mae.png")

def baseline_model(input_dim, learning_rate=0.001):
    """Modelo baseline simple (similar al notebook)."""
    model = Sequential()
    model.add(Dense(input_dim, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))  # Output layer para regresión
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model

def deep_model(input_dim, learning_rate=0.001):
    """Modelo profundo con más capas."""
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model

def deep_model_with_dropout(input_dim, learning_rate=0.001, dropout_rate=0.3):
    """Modelo profundo con Dropout para regularización."""
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout_rate * 0.7))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model

def evaluate_and_log_model(model, model_name, X_train, X_val, X_test, 
                           y_train, y_val, y_test, history=None):
    """Evalúa modelo y registra métricas/gráficos en MLflow."""
    
    # Predicciones
    y_pred_train = model.predict(X_train, verbose=0).flatten()
    y_pred_val = model.predict(X_val, verbose=0).flatten()
    y_pred_test = model.predict(X_test, verbose=0).flatten()
    
    # Métricas
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    train_mae = mean_absolute_error(y_train, y_pred_train)
    val_mae = mean_absolute_error(y_val, y_pred_val)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    train_mape = mape(y_train, y_pred_train)
    val_mape = mape(y_val, y_pred_val)
    test_mape = mape(y_test, y_pred_test)
    
    train_r2 = r2_score(y_train, y_pred_train)
    val_r2 = r2_score(y_val, y_pred_val)
    test_r2 = r2_score(y_test, y_pred_test)
    
    test_r2_adj = r2_adjusted(y_test, y_pred_test, len(y_test), X_test.shape[1])
    
    # Log métricas
    mlflow.log_metrics({
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'test_mae': test_mae,
        'train_mape': train_mape,
        'val_mape': val_mape,
        'test_mape': test_mape,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'test_r2': test_r2,
        'test_r2_adjusted': test_r2_adj
    })
    
    # Plot: Real vs Predicted (Test)
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(y_test, y_pred_test, alpha=0.6, color="tab:blue", edgecolors="k")
    y_min, y_max = float(np.min(y_test)), float(np.max(y_test))
    ax1.plot([y_min, y_max], [y_min, y_max], color="red", linestyle="--", linewidth=2)
    ax1.set_title(f"Real vs Predicted - {model_name}")
    ax1.set_xlabel("Real Values")
    ax1.set_ylabel("Predicted Values")
    ax1.grid(True, alpha=0.3)
    save_and_log_fig(fig1, os.path.join(PATH_PNG, f'{model_name}_predictions_vs_actual.png'))
    
    # Plot: Residuals
    residuals = y_test - y_pred_test
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(y_pred_test, residuals, alpha=0.6, color="purple", edgecolors="k")
    ax2.axhline(0, color="red", linestyle="--", linewidth=2)
    ax2.set_title(f"Residuals - {model_name}")
    ax2.set_xlabel("Predicted Values")
    ax2.set_ylabel("Residuals")
    ax2.grid(True, alpha=0.3)
    save_and_log_fig(fig2, os.path.join(PATH_PNG, f'{model_name}_residuals.png'))
    
    # Plot: Residuals Distribution
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color="orange", bins=30, ax=ax3)
    ax3.axvline(0, color="red", linestyle="--", linewidth=2)
    ax3.set_title(f"Residuals Distribution - {model_name}")
    ax3.set_xlabel("Residuals")
    ax3.set_ylabel("Frequency")
    ax3.grid(True, alpha=0.3)
    save_and_log_fig(fig3, os.path.join(PATH_PNG, f'{model_name}_residuals_distribution.png'))
    
    # Log model
    signature = infer_signature(X_train, y_pred_train)
    mlflow.keras.log_model(model, artifact_path="model", signature=signature)
    
    print(f"\n{model_name} Results:")
    print(f"  Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
    print(f"  Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}, Test R²: {test_r2:.4f}")
    print(f"  Test R² Adjusted: {test_r2_adj:.4f}")
    
    return {
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_mape': test_mape,
        'test_r2': test_r2,
        'test_r2_adjusted': test_r2_adj
    }

def main():
    # =========================================================
    # Configurar MLflow
    # =========================================================
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name="Neural_Network_Pipeline") as parent_run:
        # Tags del run
        mlflow.set_tags({
            "project": "airfoil_noise_neural_network",
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "keras_version": keras.__version__,
            "mlflow_version": mlflow.__version__,
            "model_type": "neural_network_regression",
            "author": "neural_network_script"
        })
        
        ####################################################################################################################
        ############################################### LOAD DATA ##########################################################
        ####################################################################################################################
        
        print("Loading data from UCI...")
        airfoil_self_noise = fetch_ucirepo(id=291)
        X = airfoil_self_noise.data.features
        y = airfoil_self_noise.data.targets
        
        data = pd.concat([X, y], axis=1)
        target_vble = y.columns[0]
        
        # Trazabilidad del dataset
        d_hash = dataset_hash(data)
        mlflow.set_tag("dataset_hash", d_hash)
        mlflow.set_tag("dataset_source", "UCI Airfoil Self-Noise")
        
        mlflow.log_param("dataset_name", "Airfoil Self-Noise")
        mlflow.log_param("dataset_id", 291)
        mlflow.log_param("original_shape", str(data.shape))
        mlflow.log_param("target_variable", target_vble)
        
        print(f"Dataset shape: {data.shape}")
        print(data.head())
        
        ####################################################################################################################
        ################################################### OUTLIERS #######################################################
        ####################################################################################################################
        
        numeric_cols = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
        
        original_size = len(data)
        data_filt = remove_outliers_iqr_df(data, numeric_cols)
        cleaned_size = len(data_filt)
        outliers_removed = original_size - cleaned_size
        
        mlflow.log_params({
            "n_rows_raw": original_size,
            "n_rows_after_outliers": cleaned_size,
            "outliers_removed": outliers_removed,
            "outlier_removal_method": "IQR"
        })
        
        print(f"Dataset shape after removing outliers: {data_filt.shape}")
        
        ####################################################################################################################
        ######################################## TRAIN/VAL/TEST SPLIT ######################################################
        ####################################################################################################################
        
        X = data_filt.drop(columns=[target_vble, 'attack-angle'])
        y = data_filt[target_vble]
        X.columns = X.columns.str.strip()
        
        # Split train+val / test
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Split train / val
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, random_state=42)
        
        print(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")
        
        mlflow.log_params({
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "n_features": X.shape[1]
        })
        
        ####################################################################################################################
        ############################################### NORMALIZATION ######################################################
        ####################################################################################################################
        
        # Normalizar features (importante para redes neuronales)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Convertir target a numpy arrays
        y_train = y_train.values
        y_val = y_val.values
        y_test = y_test.values
        
        input_dim = X_train_scaled.shape[1]
        
        ####################################################################################################################
        ########################################### MODEL 1: BASELINE ######################################################
        ####################################################################################################################
        
        print("\n" + "="*80)
        print("Training Model 1: Baseline Neural Network")
        print("="*80)
        
        with mlflow.start_run(run_name="Baseline_NN", nested=True):
            mlflow.log_params({
                "model_type": "baseline",
                "layers": "input->100->1",
                "activation": "relu",
                "optimizer": "adam",
                "learning_rate": 0.001
            })
            
            model1 = baseline_model(input_dim, learning_rate=0.001)
            
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            history1 = model1.fit(
                X_train_scaled, y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=100,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )
            
            mlflow.log_param("epochs_trained", len(history1.history['loss']))
            
            # Plot training history
            plot_history(history1, "Baseline", 
                        os.path.join(PATH_PNG, "Baseline_NN"))
            
            # Evaluate
            metrics1 = evaluate_and_log_model(
                model1, "Baseline_NN",
                X_train_scaled, X_val_scaled, X_test_scaled,
                y_train, y_val, y_test,
                history1
            )
        
        ####################################################################################################################
        ########################################### MODEL 2: DEEP MODEL ####################################################
        ####################################################################################################################
        
        print("\n" + "="*80)
        print("Training Model 2: Deep Neural Network")
        print("="*80)
        
        with mlflow.start_run(run_name="Deep_NN", nested=True):
            mlflow.log_params({
                "model_type": "deep",
                "layers": "512->512->256->256->128->1",
                "activation": "relu",
                "optimizer": "adam",
                "learning_rate": 0.001
            })
            
            model2 = deep_model(input_dim, learning_rate=0.001)
            
            early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            
            history2 = model2.fit(
                X_train_scaled, y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=150,
                batch_size=32,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            mlflow.log_param("epochs_trained", len(history2.history['loss']))
            
            # Plot training history
            plot_history(history2, "Deep Model", 
                        os.path.join(PATH_PNG, "Deep_NN"))
            
            # Evaluate
            metrics2 = evaluate_and_log_model(
                model2, "Deep_NN",
                X_train_scaled, X_val_scaled, X_test_scaled,
                y_train, y_val, y_test,
                history2
            )
        
        ####################################################################################################################
        ################################# MODEL 3: DEEP MODEL WITH DROPOUT #################################################
        ####################################################################################################################
        
        print("\n" + "="*80)
        print("Training Model 3: Deep Neural Network with Dropout")
        print("="*80)
        
        with mlflow.start_run(run_name="Deep_NN_Dropout", nested=True):
            mlflow.log_params({
                "model_type": "deep_with_dropout",
                "layers": "512->Dropout(0.3)->512->Dropout(0.3)->256->Dropout(0.2)->128->1",
                "activation": "relu",
                "optimizer": "adam",
                "learning_rate": 0.001,
                "dropout_rate": 0.3
            })
            
            model3 = deep_model_with_dropout(input_dim, learning_rate=0.001, dropout_rate=0.3)
            
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)
            
            history3 = model3.fit(
                X_train_scaled, y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=200,
                batch_size=32,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            mlflow.log_param("epochs_trained", len(history3.history['loss']))
            
            # Plot training history
            plot_history(history3, "Deep + Dropout", 
                        os.path.join(PATH_PNG, "Deep_NN_Dropout"))
            
            # Evaluate
            metrics3 = evaluate_and_log_model(
                model3, "Deep_NN_Dropout",
                X_train_scaled, X_val_scaled, X_test_scaled,
                y_train, y_val, y_test,
                history3
            )
        
        ####################################################################################################################
        ########################################### MODEL COMPARISON #######################################################
        ####################################################################################################################
        
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        results = pd.DataFrame({
            "Model": ["Baseline NN", "Deep NN", "Deep NN + Dropout"],
            "Test RMSE": [metrics1['test_rmse'], metrics2['test_rmse'], metrics3['test_rmse']],
            "Test MAE": [metrics1['test_mae'], metrics2['test_mae'], metrics3['test_mae']],
            "Test R²": [metrics1['test_r2'], metrics2['test_r2'], metrics3['test_r2']],
            "Test R² Adj": [metrics1['test_r2_adjusted'], metrics2['test_r2_adjusted'], metrics3['test_r2_adjusted']],
        }).sort_values("Test RMSE")
        
        print(results.to_string(index=False))
        
        # Save comparison
        comparison_path = os.path.join(PATH_CSV, "nn_model_comparison.csv")
        results.to_csv(comparison_path, index=False)
        mlflow.log_artifact(comparison_path)
        
        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # RMSE comparison
        axes[0].bar(results["Model"], results["Test RMSE"], color=['tab:blue', 'tab:orange', 'tab:green'])
        axes[0].set_title("Test RMSE Comparison")
        axes[0].set_ylabel("RMSE")
        axes[0].tick_params(axis='x', rotation=15)
        axes[0].grid(True, alpha=0.3)
        
        # R² comparison
        axes[1].bar(results["Model"], results["Test R²"], color=['tab:blue', 'tab:orange', 'tab:green'])
        axes[1].set_title("Test R² Comparison")
        axes[1].set_ylabel("R²")
        axes[1].tick_params(axis='x', rotation=15)
        axes[1].grid(True, alpha=0.3)
        
        fig.tight_layout()
        save_and_log_fig(fig, os.path.join(PATH_PNG, "nn_model_comparison.png"))
        
        print("\n" + "="*80)
        print("MLflow experiment completed successfully!")
        print(f"Check MLflow UI with: mlflow ui --port 5001")
        print("="*80)

if __name__ == "__main__":
    main()
