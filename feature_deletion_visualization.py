import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_feature_optimization_comparison():
    # 假设数据（你需要根据 MLflow 的结果填入）
    data = {
        'Model': ['RF', 'RF', 'GBR', 'GBR'],
        'Condition': ['Before (All)', 'After (Dropped)', 'Before (All)', 'After (Dropped)'],
        'Adj_R2': [0.909516, 0.909516, 0.995527, 0.885527] # 示例数值
    }
    df_comp = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Adj_R2', hue='Condition', data=df_comp, palette='Paired')
    plt.ylim(0.8, 1.0) # 聚焦差异点
    plt.title("Impact of Feature Selection on Adjusted R2", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()