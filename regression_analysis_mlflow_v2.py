import os
import io
import json
import time
import hashlib
import warnings
import platform
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from ucimlrepo import fetch_ucirepo
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

from sklearn import __version__ as sklearn_version
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance

from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import make_scorer

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)

PATH_PROJECT: str = os.path.dirname(os.path.abspath(__file__))
PATH_CSV = os.path.join(PATH_PROJECT, "Results_CSV")
PATH_PNG = os.path.join(PATH_PROJECT, "Results_PNG")

# Create directories if they don't exist
for path in [PATH_CSV, PATH_PNG]:
    os.makedirs(path, exist_ok=True)

# Set MLflow experiment
EXPERIMENT_NAME = "airfoil-noise-regression"

# =========================================================
# Utils
# =========================================================
def dataset_hash(df: pd.DataFrame) -> str:
    """Hash estable del dataset para trazabilidad."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.md5(csv_bytes).hexdigest()

def df_info_str(df: pd.DataFrame) -> str:
    """Captura df.info() como texto."""
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()

def save_and_log_fig(fig: plt.Figure, path: str):
    fig.savefig(path, dpi=120, bbox_inches="tight")
    mlflow.log_artifact(path)
    plt.close(fig)

def remove_outliers_iqr_df(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Elimina outliers aplicando IQR de forma conjunta (AND) para las columnas indicadas."""
    filt = pd.Series(True, index=df.index)
    for c in columns:
        Q1 = df[c].quantile(0.25)
        Q3 = df[c].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        filt &= (df[c] >= lower) & (df[c] <= upper)
    return df.loc[filt].copy()

def scatter_plots_comparison(data_before, data_after, target_column, out_path_prefix="scatter"):
    """Compara pares (feature vs target) antes vs después. Registra figuras."""
    numeric_columns = [
        col for col in data_before.columns
        if col != target_column and pd.api.types.is_numeric_dtype(data_before[col])
    ]
    n_cols = 2
    n_rows = (len(numeric_columns) + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for i, column in enumerate(numeric_columns):
        ax = axes[i]
        ax.scatter(data_before[column], data_before[target_column], alpha=0.35, label='Before', color='tab:blue')
        ax.scatter(data_after[column], data_after[target_column], alpha=0.35, label='After', color='tab:orange')
        ax.set_title(f"{column} vs {target_column}")
        ax.set_xlabel(column)
        ax.set_ylabel(target_column)
        ax.grid(True, alpha=0.3)
        ax.legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    save_and_log_fig(fig, os.path.join(PATH_PNG, f"{out_path_prefix}_before_after.png"))

def calculate_pvalues(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula matriz de p-values de Pearson. Maneja columnas constantes.
    H0: no hay correlación lineal. Si p-value < 0.05, se rechaza H0.
    """
    cols = data.columns
    pvalues = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for c1 in cols:
        for c2 in cols:
            if c1 == c2:
                pvalues.loc[c1, c2] = 0.0
            else:
                x, y = data[c1], data[c2]
                # Si alguna es constante, pearsonr falla → p=1.0
                if x.nunique() < 2 or y.nunique() < 2:
                    pvalues.loc[c1, c2] = 1.0
                else:
                    _, p = pearsonr(x, y)
                    pvalues.loc[c1, c2] = p
    return pvalues

def r2_adjusted(y_true, y_pred, n, p):
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.where(y_true == 0, np.finfo(float).eps, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom))

def log_feature_importances(model, X_train: pd.DataFrame, prefix="final"):
    """
    Registra importancias por atributo (árbol si existe, si no permutación).
    """
    # 1) Importancias internas (si existen)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        imp_df = pd.DataFrame({
            "feature": X_train.columns,
            "importance": importances
        }).sort_values("importance", ascending=False)
        csv_path = os.path.join(PATH_CSV, f"{prefix}_feature_importances_tree.csv")
        imp_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)

        # Plot
        fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(imp_df))))
        sns.barplot(data=imp_df, y="feature", x="importance", ax=ax, color="tab:green")
        ax.set_title("Feature Importances (tree-based model)")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        fig.tight_layout()
        save_and_log_fig(fig, os.path.join(PATH_PNG, f"{prefix}_feature_importances_tree.png"))

    # 2) Permutation importance (más general)
    try:
        r = permutation_importance(model, X_train, model.predict(X_train), scoring="neg_mean_squared_error",
                                   n_repeats=10, random_state=42)
        perm_df = pd.DataFrame({
            "feature": X_train.columns,
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std
        }).sort_values("importance_mean", ascending=False)
        csv_path = os.path.join(PATH_CSV, f"{prefix}_feature_importances_permutation.csv")
        perm_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)

        fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(perm_df))))
        sns.barplot(data=perm_df, y="feature", x="importance_mean", ax=ax, color="tab:purple")
        ax.set_title("Permutation Feature Importances (train)")
        ax.set_xlabel("Mean Importance (↓ MSE)")
        ax.set_ylabel("Feature")
        fig.tight_layout()
        save_and_log_fig(fig, os.path.join(PATH_PNG, f"{prefix}_feature_importances_permutation.png"))
    except Exception as e:
        error_path = os.path.join(PATH_CSV, f"{prefix}_perm_importance_error.txt")
        with open(error_path, 'w') as f:
            f.write(f"Permutation importance failed: {str(e)}")
        mlflow.log_artifact(error_path)

def log_model_with_mlflow(model_name, model, X_train, X_test, y_train, y_test, params=None, tags=None):
    """
    Registra un modelo con MLflow, incluyendo parámetros, métricas y artefactos.
    """
    start = time.perf_counter()
    with mlflow.start_run(run_name=model_name, nested=True):
        if tags:
            mlflow.set_tags(tags)
        if params:
            mlflow.log_params(params)

        # Entrenar (en caso no esté entrenado)
        if hasattr(model, "fit") and not hasattr(model, "_is_fitted"):
            model.fit(X_train, y_train)
            setattr(model, "_is_fitted", True)

        # Predicciones
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Métricas
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_mape = mape(y_train, y_pred_train)
        test_mape = mape(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_r2_adj = r2_adjusted(y_test, y_pred_test, len(y_test), X_test.shape[1])

        mlflow.log_metrics({
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_r2_adjusted': test_r2_adj
        })

        # Residuales y plots
        residuals = y_test - y_pred_test

        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.scatter(y_test, y_pred_test, alpha=0.6, color="tab:blue", edgecolors="k")
        y_min, y_max = float(np.min(y_test)), float(np.max(y_test))
        ax1.plot([y_min, y_max], [y_min, y_max], color="red", linestyle="--", linewidth=2)
        ax1.set_title("Comparison: Real vs Predicted Values (Test)")
        ax1.set_xlabel("Real Values")
        ax1.set_ylabel("Predicted Values")
        ax1.grid(True, alpha=0.3)
        save_and_log_fig(fig1, os.path.join(PATH_PNG, f'{model_name}_predictions_vs_actual.png'))

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.scatter(y_pred_test, residuals, alpha=0.6, color="purple", edgecolors="k")
        ax2.axhline(0, color="red", linestyle="--", linewidth=2)
        ax2.set_title("Residuals Plot (Test)")
        ax2.set_xlabel("Predicted Values")
        ax2.set_ylabel("Residuals (Real - Predicted)")
        ax2.grid(True, alpha=0.3)
        save_and_log_fig(fig2, os.path.join(PATH_PNG, f'{model_name}_residuals.png'))

        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.histplot(residuals, kde=True, color="orange", bins=30, ax=ax3)
        ax3.axvline(0, color="red", linestyle="--", linewidth=2)
        ax3.set_title("Residuals Distribution (Test)")
        ax3.set_xlabel("Residuals")
        ax3.set_ylabel("Frequency")
        ax3.grid(True, alpha=0.3)
        save_and_log_fig(fig3, os.path.join(PATH_PNG, f'{model_name}_residuals_distribution.png'))

        # Importancias
        log_feature_importances(model, X_train, prefix=model_name)

        # Log del modelo con firma e input de ejemplo
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.head(5)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        # Metadatos de datos
        mlflow.log_params({
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X_train.shape[1]
        })

        # Tiempo
        elapsed = time.perf_counter() - start
        mlflow.log_metric("run_seconds", elapsed)

        # Devolver métricas clave
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_r2_adjusted': test_r2_adj,
            'run_seconds': elapsed
        }

def main():
    # =========================================================
    # Configurar MLflow
    # =========================================================
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="Airfoil_Pipeline") as parent_run:
        # Tags del run
        mlflow.set_tags({
            "project": "airfoil_noise_prediction",
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "sklearn_version": sklearn_version,
            "mlflow_version": mlflow.__version__,
            "preprocessing": "IQR_outlier_removal",
            "author": "user_script"
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
        
        # Save and log data info
        info_path = os.path.join(PATH_CSV, "raw_data_info.txt")
        with open(info_path, 'w') as f:
            f.write(df_info_str(data))
        mlflow.log_artifact(info_path)
        
        data_head_path = os.path.join(PATH_CSV, "raw_head20.csv")
        data.head(20).to_csv(data_head_path, index=False)
        mlflow.log_artifact(data_head_path)
        
        # Log dataset info
        mlflow.log_param("dataset_name", "Airfoil Self-Noise")
        mlflow.log_param("dataset_id", 291)
        mlflow.log_param("original_shape", str(data.shape))
        mlflow.log_param("target_variable", target_vble)
        
        print("First 5 rows:")
        print(data.head())
        print("Info:")
        print(data.info())
        print("Description:")
        print(data.describe())
        
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
            "n_cols": data.shape[1],
            "outliers_removed": outliers_removed,
            "outlier_removal_method": "IQR",
            "iqr_columns_count": len(numeric_cols),
        })
        mlflow.set_tag("iqr_columns", json.dumps(numeric_cols))
        
        print(f"Dataset shape after removing outliers: {data_filt.shape}")
        
        # Create and log comparison plots
        scatter_plots_comparison(data, data_filt, target_vble, out_path_prefix="scatter")
        
        ####################################################################################################################
        ################################################### CORRELATION ####################################################
        ####################################################################################################################
        
        correlation_matrix = data_filt.corr(numeric_only=True)
        pvalues_matrix = calculate_pvalues(data_filt[numeric_cols])
        
        corr_path = os.path.join(PATH_CSV, "correlation_matrix.csv")
        pvalues_path = os.path.join(PATH_CSV, "pvalues_matrix.csv")
        correlation_matrix.to_csv(corr_path)
        pvalues_matrix.to_csv(pvalues_path)
        mlflow.log_artifact(corr_path)
        mlflow.log_artifact(pvalues_path)
        
        # Heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(22, 9))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                    cbar=True, annot_kws={"size": 8}, ax=axes[0])
        axes[0].set_title("Heatmap: Correlation")
        
        sns.heatmap(pvalues_matrix, annot=False, cmap="coolwarm", cbar=True, ax=axes[1])
        axes[1].set_title("Heatmap: p-values")
        fig.tight_layout()
        save_and_log_fig(fig, os.path.join(PATH_PNG, "heatmaps_corr_pvalues.png"))
        
        # Pares con alta correlación
        threshold = 0.75
        high_correlation_pairs = [
            (var1, var2, correlation_matrix.loc[var1, var2])
            for var1 in correlation_matrix.columns
            for var2 in correlation_matrix.columns
            if var1 != var2 and abs(correlation_matrix.loc[var1, var2]) > threshold
        ]
        
        # Save and log high correlation pairs
        pairs_path = os.path.join(PATH_CSV, "high_correlation_pairs.txt")
        with open(pairs_path, 'w') as f:
            f.write("\n".join([f"{v1} - {v2}: {val:.3f}" for v1, v2, val in high_correlation_pairs]))
        mlflow.log_artifact(pairs_path)
        mlflow.log_param("correlation_threshold", threshold)
        
        print("High correlation variable pairs:")
        for var1, var2, corr_value in high_correlation_pairs:
            print(f"{var1} and {var2}: Correlation = {corr_value:.2f}")
        
        ####################################################################################################################
        ############################################### MODEL EXECUTION ####################################################
        ####################################################################################################################
        
        X = data_filt.drop(columns=[target_vble])
        y = data_filt[target_vble]
        X.columns = X.columns.str.strip()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Log datasets
        train_x_path = os.path.join(PATH_CSV, "X_train.csv")
        train_y_path = os.path.join(PATH_CSV, "y_train.csv")
        test_x_path = os.path.join(PATH_CSV, "X_test.csv")
        test_y_path = os.path.join(PATH_CSV, "y_test.csv")
        
        X_train.to_csv(train_x_path, index=False)
        y_train.to_csv(train_y_path, index=False)
        X_test.to_csv(test_x_path, index=False)
        y_test.to_csv(test_y_path, index=False)
        
        mlflow.log_artifact(train_x_path)
        mlflow.log_artifact(train_y_path)
        mlflow.log_artifact(test_x_path)
        mlflow.log_artifact(test_y_path)
        
        # Log train/test split info
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size_actual", len(X_test))
        mlflow.log_param("n_features", X.shape[1])
        
        models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
                'Extra Trees': ExtraTreesRegressor(n_estimators=200, max_depth=8, random_state=42),
                'GBR': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42),
                'ADA': AdaBoostRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
                 }
        
        print('############# CROSS VALIDATION AND MODEL COMPARISON ###############')
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_summary_rows = []
        
        for name, model in models.items():
            with mlflow.start_run(run_name=f"CV - {name}", nested=True):
                # RMSE (neg MSE → RMSE)
                neg_mse_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
                rmse_scores = np.sqrt(-neg_mse_scores)
                
                # R2 ajustado
                r2_adj_scores = cross_val_score(model, X_train, y_train, cv=kfold,
                                                scoring=make_scorer(r2_adjusted, n=len(y_train), p=X_train.shape[1]))
                
                mlflow.log_metrics({
                    "cv_rmse_mean": rmse_scores.mean(),
                    "cv_rmse_std": rmse_scores.std(),
                    "cv_r2_adj_mean": r2_adj_scores.mean(),
                    "cv_r2_adj_std": r2_adj_scores.std()
                })
                
                # Guardar por fold
                cv_df = pd.DataFrame({
                    "rmse": rmse_scores,
                    "r2_adj": r2_adj_scores
                })
                path_cv = os.path.join(PATH_CSV, f"cv_{name.replace(' ', '_')}.csv")
                cv_df.to_csv(path_cv, index=False)
                mlflow.log_artifact(path_cv)
                
                cv_summary_rows.append({
                    "model": name,
                    "cv_rmse_mean": rmse_scores.mean(),
                    "cv_rmse_std": rmse_scores.std(),
                    "cv_r2_adj_mean": r2_adj_scores.mean(),
                    "cv_r2_adj_std": r2_adj_scores.std()
                })
                
                print(f"{name} - CV RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}, "
                      f"CV Adjusted R2: {r2_adj_scores.mean():.4f} ± {r2_adj_scores.std():.4f}")
        
        cv_summary_path = os.path.join(PATH_CSV, "cv_summary.csv")
        pd.DataFrame(cv_summary_rows).to_csv(cv_summary_path, index=False)
        mlflow.log_artifact(cv_summary_path)
        
        # Evaluación en Test
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_mae = mean_absolute_error(y_test, y_pred)
            test_m = mape(y_test, y_pred)
            test_r2_adj = r2_adjusted(y_test, y_pred, len(y_test), X_test.shape[1])
            
            print(f"{name} - Test RMSE: {test_rmse:.4f}, Test Adjusted R2: {test_r2_adj:.4f}")
        
        ####################################################################################################################
        ############################################### HYPERPARAMETER TUNING ##############################################
        ####################################################################################################################
        
        gbr_model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 150, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [2, 3, 4, 5, 6]
        }
        
        with mlflow.start_run(run_name="Tuning - GBR RandomSearch", nested=True):
            # Save and log parameter grid
            param_grid_path = os.path.join(PATH_CSV, "gbr_param_space.json")
            with open(param_grid_path, 'w') as f:
                json.dump(param_grid, f, indent=2)
            mlflow.log_artifact(param_grid_path)
            
            random_search = RandomizedSearchCV(
                estimator=gbr_model,
                param_distributions=param_grid,
                n_iter=50,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
            
            print("Starting Randomized Search for GBR...")
            t0 = time.perf_counter()
            random_search.fit(X_train, y_train)
            t1 = time.perf_counter()
            
            best_params = random_search.best_params_
            best_neg_mse = random_search.best_score_
            best_rmse = np.sqrt(-best_neg_mse)
            
            mlflow.log_params(best_params)
            mlflow.log_metrics({
                "best_neg_mse_cv": best_neg_mse,
                "best_rmse_cv": best_rmse,
                "tuning_seconds": (t1 - t0)
            })
            
            # Resultados completos de CV
            cv_results_path = os.path.join(PATH_CSV, "gbr_random_search_cv_results.csv")
            cvres = pd.DataFrame(random_search.cv_results_)
            cvres.to_csv(cv_results_path, index=False)
            mlflow.log_artifact(cv_results_path)
            
            print(f"Best parameters found: {random_search.best_params_}")
            print(f"Best score (neg_MSE): {random_search.best_score_}")
            print(f"Best score (RMSE): {np.sqrt(-random_search.best_score_)}")
        
        ####################################################################################################################
        #################################################### FINAL MODEL ##################################################
        ####################################################################################################################
        
        gbr_model_final = GradientBoostingRegressor(
            n_estimators=best_params['n_estimators'],
            learning_rate=best_params['learning_rate'],
            max_depth=best_params['max_depth'],
            random_state=42
        )
        gbr_model_final.fit(X_train, y_train)
        
        final_params = {
            'n_estimators': best_params['n_estimators'],
            'learning_rate': best_params['learning_rate'],
            'max_depth': best_params['max_depth'],
            'model_type': 'GradientBoostingRegressor_Final'
        }
        
        final_metrics = log_model_with_mlflow(
            'GradientBoosting_Final',
            gbr_model_final,
            X_train, X_test, y_train, y_test,
            params=final_params,
            tags={"stage": "final_model"}
        )
        
        # Mostrar resultados finales por consola
        print("****** FINAL MODEL VALIDATION RESULTS *******")
        print(f'RMSE (dB): {final_metrics["test_rmse"]:.2f}')
        print(f'MAE (dB): {final_metrics["test_mae"]:.2f}')
        print(f'MAPE (%): {final_metrics["test_mape"]*100:.2f}')
        print(f'R2 (-): {final_metrics["test_r2"]:.2f}')
        print(f'R2 Adjusted (-): {final_metrics["test_r2_adjusted"]:.2f}')
        
        print("MLflow experiment completed successfully!")
        print(f"Check MLflow UI with: mlflow ui --port 5001")

if __name__ == "__main__":
    main()