"""
主程序文件
Airfoil Self-Noise 数据集神经网络回归分析
"""
import os
import warnings
import platform

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
import mlflow
import mlflow.keras

from sklearn import __version__ as sklearn_version
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf

from Regression.utils import (
    dataset_hash,
    df_info_str,
    save_and_log_fig,
    remove_outliers_iqr_df,
    scatter_plots_comparison,
    calculate_pvalues
)
from nn_config import (
    build_neural_network,
    get_nn_configs,
    log_nn_model_with_mlflow
)

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)
tf.random.set_seed(42)

# 路径配置
PATH_PROJECT: str = os.path.dirname(os.path.abspath(__file__))
PATH_CSV = os.path.join(PATH_PROJECT, "Results_CSV")
PATH_PNG = os.path.join(PATH_PROJECT, "Results_PNG")

# 创建目录
for path in [PATH_CSV, PATH_PNG]:
    os.makedirs(path, exist_ok=True)

# MLflow 实验名称
EXPERIMENT_NAME = "airfoil-noise-neural-network"


def main():
    """主函数"""
    # =========================================================
    # 配置 MLflow
    # =========================================================
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name="Airfoil_Neural_Network_Pipeline") as parent_run:
        # 设置标签
        mlflow.set_tags({
            "project": "airfoil_noise_prediction_nn",
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "sklearn_version": sklearn_version,
            "keras_version": keras.__version__,
            "tensorflow_version": tf.__version__,
            "mlflow_version": mlflow.__version__,
            "preprocessing": "IQR_outlier_removal_and_normalization",
            "model_type": "Neural_Network",
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
        
        # 数据集追踪
        d_hash = dataset_hash(data)
        mlflow.set_tag("dataset_hash", d_hash)
        mlflow.set_tag("dataset_source", "UCI Airfoil Self-Noise")
        
        # 保存和记录数据信息
        info_path = os.path.join(PATH_CSV, "nn_raw_data_info.txt")
        with open(info_path, 'w') as f:
            f.write(df_info_str(data))
        mlflow.log_artifact(info_path)
        
        data_head_path = os.path.join(PATH_CSV, "nn_raw_head20.csv")
        data.head(20).to_csv(data_head_path, index=False)
        mlflow.log_artifact(data_head_path)
        
        mlflow.log_param("dataset_name", "Airfoil Self-Noise")
        mlflow.log_param("dataset_id", 291)
        mlflow.log_param("original_shape", str(data.shape))
        mlflow.log_param("target_variable", target_vble)
        
        print(f"Dataset shape: {data.shape}")
        print("First 5 rows:")
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
            "n_cols": data.shape[1],
            "outliers_removed": outliers_removed,
            "outlier_removal_method": "IQR",
            "iqr_columns_count": len(numeric_cols),
        })
        
        print(f"Dataset shape after removing outliers: {data_filt.shape}")
        
        # 对比图
        scatter_plots_comparison(data, data_filt, target_vble, PATH_PNG, out_path_prefix="nn_scatter")
        
        ####################################################################################################################
        ################################################### CORRELATION ####################################################
        ####################################################################################################################
        
        correlation_matrix = data_filt.corr(numeric_only=True)
        pvalues_matrix = calculate_pvalues(data_filt[numeric_cols])
        
        corr_path = os.path.join(PATH_CSV, "nn_correlation_matrix.csv")
        pvalues_path = os.path.join(PATH_CSV, "nn_pvalues_matrix.csv")
        correlation_matrix.to_csv(corr_path)
        pvalues_matrix.to_csv(pvalues_path)
        mlflow.log_artifact(corr_path)
        mlflow.log_artifact(pvalues_path)
        
        # 热图
        fig, axes = plt.subplots(1, 2, figsize=(22, 9))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                    cbar=True, annot_kws={"size": 8}, ax=axes[0])
        axes[0].set_title("Heatmap: Correlation")
        
        sns.heatmap(pvalues_matrix, annot=False, cmap="coolwarm", cbar=True, ax=axes[1])
        axes[1].set_title("Heatmap: p-values")
        fig.tight_layout()
        save_and_log_fig(fig, os.path.join(PATH_PNG, "nn_heatmaps_corr_pvalues.png"))
        
        ####################################################################################################################
        ############################################ PREPARE DATA FOR NN ###################################################
        ####################################################################################################################
        
        # 移除高度相关的特征（与目标变量相关性高的特征）
        X = data_filt.drop(columns=[target_vble, 'attack-angle'])
        y = data_filt[target_vble]
        X.columns = X.columns.str.strip()
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 数据标准化（使用训练集的均值和标准差）
        X_train_mean = X_train.mean()
        X_train_std = X_train.std()
        
        X_train_norm = (X_train - X_train_mean) / X_train_std
        X_test_norm = (X_test - X_train_mean) / X_train_std
        
        # 保存标准化参数
        norm_params = pd.DataFrame({
            'mean': X_train_mean,
            'std': X_train_std
        })
        norm_params_path = os.path.join(PATH_CSV, "nn_normalization_params.csv")
        norm_params.to_csv(norm_params_path)
        mlflow.log_artifact(norm_params_path)
        
        # 记录数据集
        train_x_path = os.path.join(PATH_CSV, "nn_X_train.csv")
        train_y_path = os.path.join(PATH_CSV, "nn_y_train.csv")
        test_x_path = os.path.join(PATH_CSV, "nn_X_test.csv")
        test_y_path = os.path.join(PATH_CSV, "nn_y_test.csv")
        
        X_train.to_csv(train_x_path, index=False)
        y_train.to_csv(train_y_path, index=False)
        X_test.to_csv(test_x_path, index=False)
        y_test.to_csv(test_y_path, index=False)
        
        mlflow.log_artifact(train_x_path)
        mlflow.log_artifact(train_y_path)
        mlflow.log_artifact(test_x_path)
        mlflow.log_artifact(test_y_path)
        
        mlflow.log_params({
            "test_size": 0.2,
            "train_size": len(X_train),
            "test_size_actual": len(X_test),
            "n_features": X.shape[1],
            "normalization": "z-score (mean/std)"
        })
        
        print(f"Training set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        print(f"Number of features: {X_train_norm.shape[1]}")
        
        ####################################################################################################################
        ############################################ NEURAL NETWORK MODELS #################################################
        ####################################################################################################################
        
        n_features = X_train_norm.shape[1]
        
        # 获取神经网络配置
        nn_configs = get_nn_configs()
        
        results_summary = []
        
        for config in nn_configs:
            print(f"\n{'='*80}")
            print(f"Training {config['name']}...")
            print(f"{'='*80}")
            
            # 构建模型
            model = build_neural_network(
                input_dim=n_features,
                hidden_units=config['hidden_units'],
                num_layers=config['num_layers']
            )
            
            # 打印模型结构
            print("\nModel Architecture:")
            model.summary()
            
            # 训练模型
            history = model.fit(
                X_train_norm,
                y_train,
                validation_split=0.3,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                verbose=1
            )
            
            # 记录模型和结果
            params = {
                'hidden_units': config['hidden_units'],
                'num_layers': config['num_layers'],
                'epochs': config['epochs'],
                'batch_size': config['batch_size'],
                'optimizer': 'adam',
                'loss_function': 'mean_squared_error',
                'activation': 'relu'
            }
            
            metrics = log_nn_model_with_mlflow(
                config['name'],
                model,
                X_train, X_test, y_train, y_test,
                X_train_norm, X_test_norm,
                history,
                PATH_PNG,
                params=params,
                tags={'model_architecture': config['name']}
            )
            
            results_summary.append({
                'model': config['name'],
                **metrics
            })
            
            print(f"\n{config['name']} Results:")
            print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
            print(f"  Test MAE: {metrics['test_mae']:.4f}")
            print(f"  Test R²: {metrics['test_r2']:.4f}")
            print(f"  Test R² Adjusted: {metrics['test_r2_adjusted']:.4f}")
        
        ####################################################################################################################
        ############################################### SUMMARY ############################################################
        ####################################################################################################################
        
        # 保存汇总结果
        results_df = pd.DataFrame(results_summary)
        results_path = os.path.join(PATH_CSV, "nn_models_summary.csv")
        results_df.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path)
        
        # 找到最佳模型
        best_model_idx = results_df['test_r2'].idxmax()
        best_model_name = results_df.loc[best_model_idx, 'model']
        best_model_r2 = results_df.loc[best_model_idx, 'test_r2']
        
        print("\n" + "="*80)
        print("NEURAL NETWORK MODELS COMPARISON")
        print("="*80)
        print(results_df.to_string(index=False))
        print("\n" + "="*80)
        print(f"Best Model: {best_model_name} with Test R² = {best_model_r2:.4f}")
        print("="*80)
        
        mlflow.log_params({
            'best_model': best_model_name,
            'best_test_r2': best_model_r2
        })
        
        print(f"\nMLflow experiment completed successfully!")
        print(f"Experiment name: {EXPERIMENT_NAME}")
        print(f"Check MLflow UI with: mlflow ui --port 5001")


if __name__ == "__main__":
    main()
