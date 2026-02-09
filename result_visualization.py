import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_cv_results(csv_path):
    df = pd.read_csv(csv_path)
    # 设置配色和风格
    sns.set_theme(style="whitegrid")
    metrics = ['cv_rmse_mean', 'cv_r2_adj_mean']
    colors = ['#4C72B0', '#55A868'] # 经典深蓝与自然绿

    for i, metric in enumerate(metrics):
        plt.figure(figsize=(10, 6))
        # 按照指标值排序，让图表更有序
        df_sorted = df.sort_values(by=metric, ascending=(metric == 'cv_rmse_mean'))
        
        ax = sns.barplot(x='model', y=metric, data=df_sorted, palette='viridis', hue='model', legend=False)
        
        # 添加误差棒 (Standard Deviation)
        std_col = metric.replace('mean', 'std')
        plt.errorbar(x=range(len(df_sorted)), y=df_sorted[metric], 
                     yerr=df_sorted[std_col], fmt='none', c='black', capsize=5)

        plt.title(f'Model Comparison: {metric.replace("_", " ").title()}', fontsize=15, pad=20)
        plt.ylabel(metric.replace("_", " ").upper(), fontsize=12)
        plt.xlabel('Algorithms', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def main():
    csv_path = input("Please enter the path to the CSV file: ")  # 确保路径正确
    plot_cv_results(csv_path)

if __name__ == "__main__":    main()