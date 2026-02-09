"""
神经网络配置和模型管理模块
包含模型构建、训练和 MLflow 记录功能
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.keras
from keras.models import Sequential
from keras.layers import Dense

from Regression.utils import (
    r2_adjusted, 
    mape, 
    plot_training_history, 
    save_and_log_fig
)


def build_neural_network(input_dim, hidden_units=50, num_layers=2):
    """
    构建神经网络回归模型
    
    参数:
        input_dim: 输入特征数量
        hidden_units: 隐藏层神经元数量
        num_layers: 隐藏层数量
    """
    model = Sequential()
    
    # 第一个隐藏层
    model.add(Dense(hidden_units, activation='relu', input_shape=(input_dim,)))
    
    # 额外的隐藏层
    for _ in range(num_layers - 1):
        model.add(Dense(hidden_units, activation='relu'))
    
    # 输出层
    model.add(Dense(1))
    
    # 编译模型
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    return model


def get_nn_configs():
    """
    获取神经网络配置列表
    可以根据需要修改或扩展配置
    """
    return [
        {
            'name': 'NN_2Layer_50Units',
            'hidden_units': 50,
            'num_layers': 2,
            'epochs': 100,
            'batch_size': 32
        },
        {
            'name': 'NN_3Layer_50Units',
            'hidden_units': 50,
            'num_layers': 3,
            'epochs': 100,
            'batch_size': 32
        },
        {
            'name': 'NN_2Layer_100Units',
            'hidden_units': 100,
            'num_layers': 2,
            'epochs': 100,
            'batch_size': 32
        },
        {
            'name': 'NN_3Layer_100Units',
            'hidden_units': 100,
            'num_layers': 3,
            'epochs': 150,
            'batch_size': 32
        }
    ]


def log_nn_model_with_mlflow(model_name, model, X_train, X_test, y_train, y_test, 
                              X_train_norm, X_test_norm, history, path_png, params=None, tags=None):
    """
    使用 MLflow 记录神经网络模型
    """
    start = time.perf_counter()
    with mlflow.start_run(run_name=model_name, nested=True):
        if tags:
            mlflow.set_tags(tags)
        if params:
            mlflow.log_params(params)
        
        # 预测
        y_pred_train = model.predict(X_train_norm, verbose=0).flatten()
        y_pred_test = model.predict(X_test_norm, verbose=0).flatten()
        
        # 计算指标
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
            'test_r2_adjusted': test_r2_adj,
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        })
        
        # 绘制训练历史
        plot_training_history(history, model_name, path_png)
        
        # 残差和预测图
        residuals = y_test - y_pred_test
        
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.scatter(y_test, y_pred_test, alpha=0.6, color="tab:blue", edgecolors="k")
        y_min, y_max = float(np.min(y_test)), float(np.max(y_test))
        ax1.plot([y_min, y_max], [y_min, y_max], color="red", linestyle="--", linewidth=2)
        ax1.set_title("Comparison: Real vs Predicted Values (Test)")
        ax1.set_xlabel("Real Values")
        ax1.set_ylabel("Predicted Values")
        ax1.grid(True, alpha=0.3)
        save_and_log_fig(fig1, os.path.join(path_png, f'{model_name}_predictions_vs_actual.png'))
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.scatter(y_pred_test, residuals, alpha=0.6, color="purple", edgecolors="k")
        ax2.axhline(0, color="red", linestyle="--", linewidth=2)
        ax2.set_title("Residuals Plot (Test)")
        ax2.set_xlabel("Predicted Values")
        ax2.set_ylabel("Residuals (Real - Predicted)")
        ax2.grid(True, alpha=0.3)
        save_and_log_fig(fig2, os.path.join(path_png, f'{model_name}_residuals.png'))
        
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.histplot(residuals, kde=True, color="orange", bins=30, ax=ax3)
        ax3.axvline(0, color="red", linestyle="--", linewidth=2)
        ax3.set_title("Residuals Distribution (Test)")
        ax3.set_xlabel("Residuals")
        ax3.set_ylabel("Frequency")
        ax3.grid(True, alpha=0.3)
        save_and_log_fig(fig3, os.path.join(path_png, f'{model_name}_residuals_distribution.png'))
        
        # 记录模型 - Keras 模型不需要 signature 和 input_example
        mlflow.keras.log_model(
            model,
            artifact_path="model"
        )
        
        # 数据集信息
        mlflow.log_params({
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X_train.shape[1]
        })
        
        # 运行时间
        elapsed = time.perf_counter() - start
        mlflow.log_metric("run_seconds", elapsed)
        
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
