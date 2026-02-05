import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import make_scorer

# Function to remove outliers using IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Function to create scatter plots comparison
def scatter_plots_comparison(data_before, data_after, target_column):
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
        ax.scatter(data_before[column], data_before[target_column], alpha=0.4, label='Before', color='blue')
        ax.scatter(data_after[column], data_after[target_column], alpha=0.4, label='After', color='orange')

        ax.set_title(f"{column} vs {target_column}")
        ax.set_xlabel(column)
        ax.set_ylabel(target_column)
        ax.grid(True)
        ax.legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# Function to calculate p-values matrix
def calculate_pvalues(data):
    cols = data.columns
    pvalues = pd.DataFrame(index=cols, columns=cols)
    for col1 in cols:
        for col2 in cols:
            if col1 == col2:
                pvalues.loc[col1, col2] = 0
            else:
                try:
                    _, p_value = pearsonr(data[col1], data[col2])
                    pvalues.loc[col1, col2] = p_value
                except:
                    pvalues.loc[col1, col2] = np.nan
    return pvalues.astype(float)

# Function to calculate Adjusted R2
def r2_adjusted(y_true, y_pred, n, p):
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def main():
    ####################################################################################################################
    ############################################### LOAD DATA ##########################################################
    ####################################################################################################################

    print("Loading data from UCI...")
    airfoil_self_noise = fetch_ucirepo(id=291)
    X = airfoil_self_noise.data.features
    y = airfoil_self_noise.data.targets
    
    # Concatenate to creating a single DataFrame for processing
    data = pd.concat([X, y], axis=1)

    print("First 5 rows:")
    print(data.head())
    print("Info:")
    print(data.info())
    print("Description:")
    print(data.describe())

    # Identify target variable
    target_vble = y.columns[0] # 'Scaled sound pressure level'

    ####################################################################################################################
    ################################################### OUTLIERS #######################################################
    ####################################################################################################################

    data_filt = data.copy()
    # Apply outlier removal to each numeric column
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            data_filt = remove_outliers_iqr(data_filt, column)

    print(f"Dataset shape after removing outliers: {data_filt.shape}")

    # Scatter plots comparison
    # Note: Depending on the environment, this might block execution until window is closed.
    scatter_plots_comparison(data, data_filt, target_vble)

    ####################################################################################################################
    ################################################### CORRELATION ####################################################
    ####################################################################################################################

    correlation_matrix = data_filt.corr()
    pvalues_matrix = calculate_pvalues(data_filt)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, annot_kws={"size": 10}, ax=axes[0])
    axes[0].set_title("Heatmap: Correlation")
    sns.heatmap(pvalues_matrix, annot=True, fmt=".4f", cmap="coolwarm", cbar=True, annot_kws={"size": 10}, ax=axes[1])
    axes[1].set_title("Heatmap: p-values")
    plt.tight_layout()
    plt.show()

    threshold = 0.75
    high_correlation_pairs = [
        (var1, var2, correlation_matrix.loc[var1, var2])
        for var1 in correlation_matrix.columns
        for var2 in correlation_matrix.columns
        if var1 != var2 and abs(correlation_matrix.loc[var1, var2]) > threshold
    ]
    print("High correlation variable pairs:")
    for var1, var2, corr_value in high_correlation_pairs:
        print(f"{var1} and {var2}: Correlation = {corr_value:.2f}")

    ####################################################################################################################
    ############################################### MODEL EXECUTION ####################################################
    ####################################################################################################################

    # Initialize models
    models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Extra Trees': ExtraTreesRegressor(n_estimators=100, max_depth=5),
            'GBR': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3),
            'ADA': AdaBoostRegressor(n_estimators=100, learning_rate=0.1)
             }

    # Separate features and target
    X = data_filt.drop(columns=[target_vble])
    y = data_filt[target_vble]
    
    # Clean feature names to avoid issues with some models/plotters
    X.columns = X.columns.str.strip()
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    print('############# CROSS VALIDATION (K-Fold with Train Set) ###############')
    results = {}
    for name, model in models.items():
        print(f"Evaluating model: {name}")

        n = len(y_train)
        p = X_train.shape[1]

        neg_mse_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-neg_mse_scores)

        # Using a wrapper for r2_adjusted to pass to cross_val_score
        # make_scorer's arguments are passed to the scorer function
        # Note: 'r2_adjusted' function signature: (y_true, y_pred, n, p)
        # However, make_scorer passes y_true, y_pred. We need to pre-bind n and p or use a custom class/partial.
        # But for simplicity in this context, standard R2 is often enough for initial check, 
        # but let's stick to the user's reference code's intent. 
        # The reference code uses make_scorer(r2_adjusted, n=n, p=p).
        
        r2_scorer = make_scorer(r2_adjusted, greater_is_better=True, n=n, p=p)
        r2_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring=r2_scorer)

        results[name] = {
            'RMSE Mean': rmse_scores.mean(),
            'RMSE Std': rmse_scores.std(),
            'R2 Adjusted Mean': r2_scores.mean(),
            'R2 Adjusted Std': r2_scores.std()
        }

        print(
            f"{name} - RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}, Adjusted R2: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")

    print('############# CROSS VALIDATION (Test Set) ###############')
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2_adjusted = r2_adjusted(y_test, y_pred, len(y_test), X_test.shape[1])

        print(f"Test Results for {name}:")
        print(f"RMSE: {test_rmse:.4f}, Adjusted R2: {test_r2_adjusted:.4f}\n")

    ####################################################################################################################
    ############################################### HYPERPARAMETER TUNING ##############################################
    ####################################################################################################################

    gbr_model = GradientBoostingRegressor()
    param_grid = {
                    'n_estimators': [50, 100, 150, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 4, 5, 6]
    }

    print("Starting Randomized Search for GBR...")
    random_search = RandomizedSearchCV(estimator=gbr_model, param_distributions=param_grid,
                                       n_iter=50, cv=5, scoring='neg_mean_squared_error',
                                       n_jobs=-1, verbose=1, random_state=42)

    random_search.fit(X_train, y_train)

    print(f"Best parameters found: {random_search.best_params_}")
    print(f"Best score (neg_MSE): {random_search.best_score_}")
    print(f"Best score (RMSE): {np.sqrt(-random_search.best_score_)}")

    ###################################################################################################################
    #################################################### FINAL MODEL ##################################################
    ###################################################################################################################

    gbr_model_final = GradientBoostingRegressor(
        n_estimators = random_search.best_params_['n_estimators'],
        learning_rate = random_search.best_params_['learning_rate'],
        max_depth = random_search.best_params_['max_depth']
    )
    
    gbr_model_final.fit(X_train, y_train)
    y_pred = gbr_model_final.predict(X_test)
    
    final_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    final_test_r2_adjusted = r2_adjusted(y_test, y_pred, len(y_test), X_test.shape[1])
    
    print(f"****** FINAL MODEL VALIDATION RESULTS *******")
    print(f'RMSE (dB): {final_test_rmse:.2f}')
    print(f'R2 (-): {final_test_r2_adjusted:.2f}')

    # Visualizations
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color="b", edgecolors="k")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", linewidth=2)
    plt.title("Comparison: Real vs Predicted Values")
    plt.xlabel("Real Values")
    plt.ylabel("Predicted Values")
    plt.grid(True)
    plt.show()

    residuals = y_test - y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, color="purple", edgecolors="k")
    plt.axhline(0, color="red", linestyle="--", linewidth=2)
    plt.title("Residuals Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (Real - Predicted)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color="orange", bins=30)
    plt.axvline(0, color="red", linestyle="--", linewidth=2)
    plt.title("Residuals Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    print("Main function executed successfully")

if __name__ == "__main__":
    main()
