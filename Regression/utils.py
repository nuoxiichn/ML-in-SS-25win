"""
工具函数模块
包含数据处理、可视化和指标计算的辅助函数
"""
import os
import io
import hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import mlflow


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
    """保存图表并记录到 MLflow"""
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


def scatter_plots_comparison(data_before, data_after, target_column, path_png, out_path_prefix="scatter"):
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
    save_and_log_fig(fig, os.path.join(path_png, f"{out_path_prefix}_before_after.png"))


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
    """计算调整后的 R² 值"""
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def mape(y_true, y_pred):
    """计算平均绝对百分比误差 (MAPE)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.where(y_true == 0, np.finfo(float).eps, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom))


def plot_training_history(history, model_name, path_png):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title(f'{model_name} - Training History (Loss)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(history.history['mae'], label='Training MAE')
    axes[1].plot(history.history['val_mae'], label='Validation MAE')
    axes[1].set_title(f'{model_name} - Training History (MAE)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    path = os.path.join(path_png, f'{model_name}_training_history.png')
    save_and_log_fig(fig, path)
