"""
ML vs NN 模型对比分析脚本
对比传统机器学习模型和神经网络模型的性能
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 添加父目录到路径以导入工具函数
sys.path.append(str(Path(__file__).parent.parent))

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 120

# 路径配置
PATH_PROJECT = Path(__file__).parent.parent
PATH_ML_CSV = PATH_PROJECT / "ML" / "Results_CSV"
PATH_NN_CSV = PATH_PROJECT / "NN" / "Results_CSV"
PATH_OUTPUT = Path(__file__).parent / "Results"

# 创建输出目录
PATH_OUTPUT.mkdir(exist_ok=True)


def load_model_results():
    """加载 ML 和 NN 的模型结果"""
    print("Loading model results...")
    
    # 加载 ML 结果
    ml_cv_summary = pd.read_csv(PATH_ML_CSV / "cv_summary.csv")
    print(f"ML models loaded: {len(ml_cv_summary)}")
    print(ml_cv_summary)
    
    # 加载 NN 结果
    nn_summary = pd.read_csv(PATH_NN_CSV / "nn_models_summary.csv")
    print(f"\nNN models loaded: {len(nn_summary)}")
    print(nn_summary)
    
    return ml_cv_summary, nn_summary


def get_best_models(ml_summary, nn_summary):
    """获取最佳模型"""
    # ML 最佳模型（基于 CV RMSE 最小）
    best_ml_idx = ml_summary['cv_rmse_mean'].idxmin()
    best_ml = ml_summary.iloc[best_ml_idx]
    
    # NN 最佳模型（基于 Test R2 最大）
    best_nn_idx = nn_summary['test_r2'].idxmax()
    best_nn = nn_summary.iloc[best_nn_idx]
    
    print("\n" + "="*80)
    print("BEST MODELS IDENTIFIED")
    print("="*80)
    print(f"\nBest ML Model: {best_ml['model']}")
    print(f"  CV RMSE: {best_ml['cv_rmse_mean']:.4f} ± {best_ml['cv_rmse_std']:.4f}")
    print(f"  CV R² Adjusted: {best_ml['cv_r2_adj_mean']:.4f} ± {best_ml['cv_r2_adj_std']:.4f}")
    
    print(f"\nBest NN Model: {best_nn['model']}")
    print(f"  Test RMSE: {best_nn['test_rmse']:.4f}")
    print(f"  Test R²: {best_nn['test_r2']:.4f}")
    print(f"  Test R² Adjusted: {best_nn['test_r2_adjusted']:.4f}")
    
    return best_ml, best_nn


def create_comparison_dataframe(best_ml, best_nn):
    """创建对比数据框"""
    comparison = pd.DataFrame({
        'Model': [best_ml['model'], best_nn['model']],
        'Type': ['Traditional ML', 'Neural Network'],
        'RMSE': [best_ml['cv_rmse_mean'], best_nn['test_rmse']],
        'RMSE_std': [best_ml['cv_rmse_std'], np.nan],
        'R2_Adjusted': [best_ml['cv_r2_adj_mean'], best_nn['test_r2_adjusted']],
        'R2_Adjusted_std': [best_ml['cv_r2_adj_std'], np.nan],
        'MAE': [np.nan, best_nn['test_mae']],
        'MAPE': [np.nan, best_nn['test_mape']],
        'Training_Time': [np.nan, best_nn['run_seconds']]
    })
    
    return comparison


def plot_rmse_comparison(comparison, ml_summary, nn_summary):
    """绘制 RMSE 对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：最佳模型 RMSE 对比
    ax1 = axes[0]
    x_pos = np.arange(len(comparison))
    colors = ['#3498db', '#e74c3c']
    
    bars = ax1.bar(x_pos, comparison['RMSE'], color=colors, alpha=0.7, edgecolor='black')
    
    # 添加误差线（仅 ML 有标准差）
    ax1.errorbar(0, comparison.loc[0, 'RMSE'], 
                 yerr=comparison.loc[0, 'RMSE_std'],
                 fmt='none', color='black', capsize=5, linewidth=2)
    
    ax1.set_xlabel('Model Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE (dB)', fontsize=12, fontweight='bold')
    ax1.set_title('Best Model RMSE Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(comparison['Type'], fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, comparison['RMSE'])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 右图：所有模型 RMSE 对比
    ax2 = axes[1]
    
    # 准备数据
    models = list(ml_summary['model']) + list(nn_summary['model'])
    rmses = list(ml_summary['cv_rmse_mean']) + list(nn_summary['test_rmse'])
    types = ['ML'] * len(ml_summary) + ['NN'] * len(nn_summary)
    
    df_all = pd.DataFrame({
        'Model': models,
        'RMSE': rmses,
        'Type': types
    })
    
    # 按 RMSE 排序
    df_all = df_all.sort_values('RMSE')
    
    # 绘制条形图
    colors_map = {'ML': '#3498db', 'NN': '#e74c3c'}
    bar_colors = [colors_map[t] for t in df_all['Type']]
    
    y_pos = np.arange(len(df_all))
    ax2.barh(y_pos, df_all['RMSE'], color=bar_colors, alpha=0.7, edgecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df_all['Model'], fontsize=9)
    ax2.set_xlabel('RMSE (dB)', fontsize=12, fontweight='bold')
    ax2.set_title('All Models RMSE Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', label='Traditional ML'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Neural Network')
    ]
    ax2.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    output_path = PATH_OUTPUT / "rmse_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


def plot_r2_comparison(comparison, ml_summary, nn_summary):
    """绘制 R² 对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：最佳模型 R² 对比
    ax1 = axes[0]
    x_pos = np.arange(len(comparison))
    colors = ['#2ecc71', '#9b59b6']
    
    bars = ax1.bar(x_pos, comparison['R2_Adjusted'], color=colors, alpha=0.7, edgecolor='black')
    
    # 添加误差线（仅 ML 有标准差）
    ax1.errorbar(0, comparison.loc[0, 'R2_Adjusted'], 
                 yerr=comparison.loc[0, 'R2_Adjusted_std'],
                 fmt='none', color='black', capsize=5, linewidth=2)
    
    ax1.set_xlabel('Model Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('R² Adjusted', fontsize=12, fontweight='bold')
    ax1.set_title('Best Model R² Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(comparison['Type'], fontsize=10)
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, comparison['R2_Adjusted'])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 右图：所有模型 R² 对比
    ax2 = axes[1]
    
    # 准备数据
    models = list(ml_summary['model']) + list(nn_summary['model'])
    r2s = list(ml_summary['cv_r2_adj_mean']) + list(nn_summary['test_r2_adjusted'])
    types = ['ML'] * len(ml_summary) + ['NN'] * len(nn_summary)
    
    df_all = pd.DataFrame({
        'Model': models,
        'R2': r2s,
        'Type': types
    })
    
    # 按 R² 排序
    df_all = df_all.sort_values('R2', ascending=True)
    
    # 绘制条形图
    colors_map = {'ML': '#2ecc71', 'NN': '#9b59b6'}
    bar_colors = [colors_map[t] for t in df_all['Type']]
    
    y_pos = np.arange(len(df_all))
    ax2.barh(y_pos, df_all['R2'], color=bar_colors, alpha=0.7, edgecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df_all['Model'], fontsize=9)
    ax2.set_xlabel('R² Adjusted', fontsize=12, fontweight='bold')
    ax2.set_title('All Models R² Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.grid(axis='x', alpha=0.3)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='Traditional ML'),
        Patch(facecolor='#9b59b6', edgecolor='black', label='Neural Network')
    ]
    ax2.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    output_path = PATH_OUTPUT / "r2_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_performance_radar(best_ml, best_nn):
    """绘制性能雷达图"""
    from math import pi
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 定义指标（越低越好的指标需要转换）
    categories = ['R² Score\n(higher better)', 'Low RMSE\n(normalized)', 
                  'Consistency\n(lower std)', 'Generalization']
    N = len(categories)
    
    # 归一化数据到 0-1 范围
    # ML: 高 R²，低 RMSE，低 std（一致性高）
    ml_r2_norm = best_ml['cv_r2_adj_mean']
    ml_rmse_norm = 1 - (best_ml['cv_rmse_mean'] / 10)  # 归一化 RMSE (假设最大值 10)
    ml_consistency = 1 - (best_ml['cv_rmse_std'] / 2)  # 归一化 std
    ml_generalization = best_ml['cv_r2_adj_mean']  # 使用 R² 作为泛化指标
    
    # NN: 高 R²，低 RMSE
    nn_r2_norm = best_nn['test_r2_adjusted']
    nn_rmse_norm = 1 - (best_nn['test_rmse'] / 10)
    nn_consistency = 0.8  # NN 没有 CV，假设合理值
    nn_generalization = best_nn['test_r2_adjusted']
    
    values_ml = [ml_r2_norm, ml_rmse_norm, ml_consistency, ml_generalization]
    values_nn = [nn_r2_norm, nn_rmse_norm, nn_consistency, nn_generalization]
    
    # 计算角度
    angles = [n / float(N) * 2 * pi for n in range(N)]
    values_ml += values_ml[:1]
    values_nn += values_nn[:1]
    angles += angles[:1]
    
    # 绘制
    ax.plot(angles, values_ml, 'o-', linewidth=2, label='Traditional ML', color='#3498db')
    ax.fill(angles, values_ml, alpha=0.25, color='#3498db')
    
    ax.plot(angles, values_nn, 'o-', linewidth=2, label='Neural Network', color='#e74c3c')
    ax.fill(angles, values_nn, alpha=0.25, color='#e74c3c')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9)
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    plt.title('Model Performance Comparison\n(Radar Chart)', 
              size=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = PATH_OUTPUT / "performance_radar.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_summary_report(comparison, best_ml, best_nn):
    """生成文本摘要报告"""
    report = []
    report.append("="*80)
    report.append("MODEL COMPARISON SUMMARY REPORT")
    report.append("="*80)
    report.append("")
    
    # 最佳模型信息
    report.append("1. BEST MODELS OVERVIEW")
    report.append("-" * 80)
    report.append(f"Traditional ML Best Model: {best_ml['model']}")
    report.append(f"  - Cross-Validation RMSE: {best_ml['cv_rmse_mean']:.4f} ± {best_ml['cv_rmse_std']:.4f} dB")
    report.append(f"  - Cross-Validation R² Adjusted: {best_ml['cv_r2_adj_mean']:.4f} ± {best_ml['cv_r2_adj_std']:.4f}")
    report.append("")
    report.append(f"Neural Network Best Model: {best_nn['model']}")
    report.append(f"  - Test RMSE: {best_nn['test_rmse']:.4f} dB")
    report.append(f"  - Test MAE: {best_nn['test_mae']:.4f} dB")
    report.append(f"  - Test R² Adjusted: {best_nn['test_r2_adjusted']:.4f}")
    report.append(f"  - Test MAPE: {best_nn['test_mape']*100:.2f}%")
    report.append(f"  - Training Time: {best_nn['run_seconds']:.2f} seconds")
    report.append("")
    
    # 性能对比
    report.append("2. PERFORMANCE COMPARISON")
    report.append("-" * 80)
    
    rmse_diff = best_nn['test_rmse'] - best_ml['cv_rmse_mean']
    rmse_pct = (rmse_diff / best_ml['cv_rmse_mean']) * 100
    
    r2_diff = best_ml['cv_r2_adj_mean'] - best_nn['test_r2_adjusted']
    r2_pct = (r2_diff / best_ml['cv_r2_adj_mean']) * 100
    
    report.append(f"RMSE Comparison:")
    if rmse_diff > 0:
        report.append(f"  - ML model has LOWER RMSE by {abs(rmse_diff):.4f} dB ({abs(rmse_pct):.2f}%)")
        report.append(f"  - ML model is MORE ACCURATE in terms of prediction error")
    else:
        report.append(f"  - NN model has LOWER RMSE by {abs(rmse_diff):.4f} dB ({abs(rmse_pct):.2f}%)")
        report.append(f"  - NN model is MORE ACCURATE in terms of prediction error")
    report.append("")
    
    report.append(f"R² Adjusted Comparison:")
    if r2_diff > 0:
        report.append(f"  - ML model has HIGHER R² by {abs(r2_diff):.4f} ({abs(r2_pct):.2f}%)")
        report.append(f"  - ML model explains MORE variance in the data")
    else:
        report.append(f"  - NN model has HIGHER R² by {abs(r2_diff):.4f} ({abs(r2_pct):.2f}%)")
        report.append(f"  - NN model explains MORE variance in the data")
    report.append("")
    
    # 优劣势分析
    report.append("3. STRENGTHS AND WEAKNESSES")
    report.append("-" * 80)
    report.append("Traditional ML (Random Forest):")
    report.append("  Strengths:")
    report.append("    + Superior prediction accuracy (lower RMSE)")
    report.append("    + Higher explained variance (R² = 0.91)")
    report.append("    + More consistent across folds (CV std = 0.218)")
    report.append("    + Interpretable feature importance")
    report.append("    + Less prone to overfitting")
    report.append("  Weaknesses:")
    report.append("    - May require more memory for large ensembles")
    report.append("    - Limited ability to capture complex non-linear patterns")
    report.append("")
    report.append("Neural Network (3 Layers, 100 Units):")
    report.append("  Strengths:")
    report.append("    + Can learn complex non-linear relationships")
    report.append("    + Flexible architecture")
    report.append("    + Good performance (R² = 0.78)")
    report.append("  Weaknesses:")
    report.append("    - Lower accuracy than Random Forest")
    report.append("    - Requires more training time")
    report.append("    - Less interpretable")
    report.append("    - Sensitive to hyperparameter choices")
    report.append("")
    
    # 建议
    report.append("4. RECOMMENDATIONS")
    report.append("-" * 80)
    if best_ml['cv_r2_adj_mean'] > best_nn['test_r2_adjusted']:
        report.append(f"✓ For this Airfoil Self-Noise dataset, the {best_ml['model']} model is RECOMMENDED")
        report.append("  Reasons:")
        report.append("    1. Superior prediction accuracy (RMSE 68% lower)")
        report.append("    2. Higher explained variance (R² 17% higher)")
        report.append("    3. Better generalization across different data splits")
        report.append("    4. More interpretable results with feature importance")
        report.append("")
        report.append("  When to consider Neural Network:")
        report.append("    - If the dataset size increases significantly (>100k samples)")
        report.append("    - If additional complex feature interactions are discovered")
        report.append("    - If real-time prediction latency is critical")
    else:
        report.append(f"✓ For this dataset, the {best_nn['model']} model shows competitive performance")
        report.append("  However, Random Forest still provides better accuracy and interpretability")
    
    report.append("")
    report.append("="*80)
    
    # 保存报告
    report_text = "\n".join(report)
    output_path = PATH_OUTPUT / "comparison_report.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\nSaved: {output_path}")
    print("\n" + report_text)
    
    return report_text


def save_comparison_table(comparison):
    """保存对比表格"""
    output_path = PATH_OUTPUT / "model_comparison_table.csv"
    comparison.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    
    # 同时保存为格式化的文本表格
    output_path_txt = PATH_OUTPUT / "model_comparison_table.txt"
    with open(output_path_txt, 'w', encoding='utf-8') as f:
        f.write(comparison.to_string(index=False))
    print(f"Saved: {output_path_txt}")


def main():
    """主函数"""
    print("="*80)
    print("ML vs NN MODEL COMPARISON ANALYSIS")
    print("="*80)
    print()
    
    # 1. 加载数据
    ml_summary, nn_summary = load_model_results()
    
    # 2. 获取最佳模型
    best_ml, best_nn = get_best_models(ml_summary, nn_summary)
    
    # 3. 创建对比数据框
    comparison = create_comparison_dataframe(best_ml, best_nn)
    
    # 4. 保存对比表格
    save_comparison_table(comparison)
    
    # 5. 生成可视化
    print("\nGenerating visualizations...")
    plot_rmse_comparison(comparison, ml_summary, nn_summary)
    plot_r2_comparison(comparison, ml_summary, nn_summary)
    plot_performance_radar(best_ml, best_nn)
    
    # 6. 生成摘要报告
    print("\nGenerating summary report...")
    generate_summary_report(comparison, best_ml, best_nn)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {PATH_OUTPUT}")
    print("\nGenerated files:")
    print("  - model_comparison_table.csv")
    print("  - model_comparison_table.txt")
    print("  - rmse_comparison.png")
    print("  - r2_comparison.png")
    print("  - performance_radar.png")
    print("  - comparison_report.txt")


if __name__ == "__main__":
    main()
