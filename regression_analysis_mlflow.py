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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import make_scorer

# Set MLflow experiment
EXPERIMENT_NAME = "airfoil-noise-regression"

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
    plt.savefig('data_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    return 'data_comparison.png'

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

def log_correlation_analysis(correlation_matrix, pvalues_matrix):
    """Log correlation analysis to MLFlow"""
    # Save correlation heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, annot_kws={"size": 10}, ax=axes[0])
    axes[0].set_title("Heatmap: Correlation")
    sns.heatmap(pvalues_matrix, annot=True, fmt=".4f", cmap="coolwarm", cbar=True, annot_kws={"size": 10}, ax=axes[1])
    axes[1].set_title("Heatmap: p-values")
    plt.tight_layout()
    plt.savefig('correlation_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    
    # Log as artifact
    mlflow.log_artifact('correlation_analysis.png')
    
    # Log high correlation pairs as metrics
    threshold = 0.75
    high_correlation_pairs = [
        (var1, var2, correlation_matrix.loc[var1, var2])
        for var1 in correlation_matrix.columns
        for var2 in correlation_matrix.columns
        if var1 != var2 and abs(correlation_matrix.loc[var1, var2]) > threshold
    ]
    
    mlflow.log_param("high_correlation_threshold", threshold)
    mlflow.log_param("high_correlation_pairs_count", len(high_correlation_pairs))
    
    return high_correlation_pairs

def evaluate_models_with_mlflow(models, X_train, X_test, y_train, y_test):
    """Evaluate multiple models and log results to MLFlow"""
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    model_results = {}
    
    for name, model in models.items():
        with mlflow.start_run(run_name=f"Model_{name}", nested=True):
            print(f"Evaluating model: {name}")
            
            # Log model parameters
            mlflow.log_params(model.get_params())
            
            n = len(y_train)
            p = X_train.shape[1]
            
            # Cross validation
            neg_mse_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-neg_mse_scores)
            
            r2_scorer = make_scorer(r2_adjusted, greater_is_better=True, n=n, p=p)
            r2_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring=r2_scorer)
            
            # Log cross-validation metrics
            mlflow.log_metric("cv_rmse_mean", rmse_scores.mean())
            mlflow.log_metric("cv_rmse_std", rmse_scores.std())
            mlflow.log_metric("cv_r2_adjusted_mean", r2_scores.mean())
            mlflow.log_metric("cv_r2_adjusted_std", r2_scores.std())
            
            # Test set evaluation
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_r2_adjusted = r2_adjusted(y_test, y_pred, len(y_test), X_test.shape[1])
            
            # Log test metrics
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("test_r2_adjusted", test_r2_adjusted)
            
            # Log model
            mlflow.sklearn.log_model(model, f"model_{name.lower().replace(' ', '_')}")
            
            model_results[name] = {
                'RMSE Mean': rmse_scores.mean(),
                'RMSE Std': rmse_scores.std(),
                'R2 Adjusted Mean': r2_scores.mean(),
                'R2 Adjusted Std': r2_scores.std(),
                'Test RMSE': test_rmse,
                'Test R2 Adjusted': test_r2_adjusted
            }
            
            print(f"{name} - CV RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}, "
                  f"CV Adjusted R2: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
            print(f"{name} - Test RMSE: {test_rmse:.4f}, Test Adjusted R2: {test_r2_adjusted:.4f}\n")
    
    return model_results

def hyperparameter_tuning_with_mlflow(X_train, y_train):
    """Perform hyperparameter tuning and log to MLFlow"""
    with mlflow.start_run(run_name="Hyperparameter_Tuning_GBR", nested=True):
        gbr_model = GradientBoostingRegressor()
        param_grid = {
                        'n_estimators': [50, 100, 150, 200],
                        'learning_rate': [0.01, 0.05, 0.1, 0.2],
                        'max_depth': [3, 4, 5, 6]
        }
        
        print("Starting Randomized Search for GBR...")
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
        
        random_search.fit(X_train, y_train)
        
        # Log best parameters and scores
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("best_cv_score", random_search.best_score_)
        mlflow.log_metric("best_cv_rmse", np.sqrt(-random_search.best_score_))
        
        # Log search space
        mlflow.log_params({"param_grid": str(param_grid)})
        mlflow.log_param("n_iter", 50)
        
        print(f"Best parameters found: {random_search.best_params_}")
        print(f"Best score (neg_MSE): {random_search.best_score_}")
        print(f"Best score (RMSE): {np.sqrt(-random_search.best_score_)}")
        
        return random_search

def create_final_model_visualizations(y_test, y_pred, model_name="Final_GBR"):
    """Create and log final model visualizations"""
    
    # Real vs Predicted plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color="b", edgecolors="k")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", linewidth=2)
    plt.title("Comparison: Real vs Predicted Values")
    plt.xlabel("Real Values")
    plt.ylabel("Predicted Values")
    plt.grid(True)
    plt.savefig('real_vs_predicted.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    
    # Residuals plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, color="purple", edgecolors="k")
    plt.axhline(0, color="red", linestyle="--", linewidth=2)
    plt.title("Residuals Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (Real - Predicted)")
    plt.grid(True)
    plt.savefig('residuals_plot.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    
    # Residuals distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color="orange", bins=30)
    plt.axvline(0, color="red", linestyle="--", linewidth=2)
    plt.title("Residuals Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig('residuals_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    
    # Log all plots
    mlflow.log_artifact('real_vs_predicted.png')
    mlflow.log_artifact('residuals_plot.png')
    mlflow.log_artifact('residuals_distribution.png')

def main():
    # Initialize MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name="Airfoil_Noise_Analysis"):
        
        ####################################################################################################################
        ############################################### LOAD DATA ##########################################################
        ####################################################################################################################
        
        print("Loading data from UCI...")
        airfoil_self_noise = fetch_ucirepo(id=291)
        X = airfoil_self_noise.data.features
        y = airfoil_self_noise.data.targets
        
        data = pd.concat([X, y], axis=1)
        
        # Log dataset info
        mlflow.log_param("dataset_name", "Airfoil Self-Noise")
        mlflow.log_param("dataset_id", 291)
        mlflow.log_param("original_shape", str(data.shape))
        
        print("First 5 rows:")
        print(data.head())
        print("Info:")
        print(data.info())
        print("Description:")
        print(data.describe())
        
        target_vble = y.columns[0]
        mlflow.log_param("target_variable", target_vble)
        
        ####################################################################################################################
        ################################################### OUTLIERS #######################################################
        ####################################################################################################################
        
        data_filt = data.copy()
        original_size = len(data_filt)
        
        for column in data.columns:
            if pd.api.types.is_numeric_dtype(data[column]):
                data_filt = remove_outliers_iqr(data_filt, column)
        
        cleaned_size = len(data_filt)
        outliers_removed = original_size - cleaned_size
        
        # Log cleaning info
        mlflow.log_param("original_dataset_size", original_size)
        mlflow.log_param("cleaned_dataset_size", cleaned_size)
        mlflow.log_param("outliers_removed", outliers_removed)
        mlflow.log_param("outlier_removal_method", "IQR")
        
        print(f"Dataset shape after removing outliers: {data_filt.shape}")
        
        # Create and log comparison plots
        comparison_plot = scatter_plots_comparison(data, data_filt, target_vble)
        mlflow.log_artifact(comparison_plot)
        
        ####################################################################################################################
        ################################################### CORRELATION ####################################################
        ####################################################################################################################
        
        correlation_matrix = data_filt.corr()
        pvalues_matrix = calculate_pvalues(data_filt)
        
        high_correlation_pairs = log_correlation_analysis(correlation_matrix, pvalues_matrix)
        
        print("High correlation variable pairs:")
        for var1, var2, corr_value in high_correlation_pairs:
            print(f"{var1} and {var2}: Correlation = {corr_value:.2f}")
        
        ####################################################################################################################
        ############################################### MODEL EXECUTION ####################################################
        ####################################################################################################################
        
        models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Extra Trees': ExtraTreesRegressor(n_estimators=100, max_depth=5),
                'GBR': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3),
                'ADA': AdaBoostRegressor(n_estimators=100, learning_rate=0.1)
                 }
        
        X = data_filt.drop(columns=[target_vble])
        y = data_filt[target_vble]
        
        X.columns = X.columns.str.strip()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Log train/test split info
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size_actual", len(X_test))
        mlflow.log_param("n_features", X.shape[1])
        
        print('############# CROSS VALIDATION AND MODEL COMPARISON ###############')
        model_results = evaluate_models_with_mlflow(models, X_train, X_test, y_train, y_test)
        
        ####################################################################################################################
        ############################################### HYPERPARAMETER TUNING ##############################################
        ####################################################################################################################
        
        random_search = hyperparameter_tuning_with_mlflow(X_train, y_train)
        
        ####################################################################################################################
        #################################################### FINAL MODEL ##################################################
        ####################################################################################################################
        
        with mlflow.start_run(run_name="Final_Best_Model", nested=True):
            gbr_model_final = GradientBoostingRegressor(
                n_estimators = random_search.best_params_['n_estimators'],
                learning_rate = random_search.best_params_['learning_rate'],
                max_depth = random_search.best_params_['max_depth'],
                random_state=42
            )
            
            gbr_model_final.fit(X_train, y_train)
            y_pred = gbr_model_final.predict(X_test)
            
            final_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            final_test_r2_adjusted = r2_adjusted(y_test, y_pred, len(y_test), X_test.shape[1])
            
            # Log final model
            mlflow.log_params(gbr_model_final.get_params())
            mlflow.log_metric("final_test_rmse", final_test_rmse)
            mlflow.log_metric("final_test_r2_adjusted", final_test_r2_adjusted)
            
            # Log the final model
            mlflow.sklearn.log_model(gbr_model_final, "final_model")
            
            print(f"****** FINAL MODEL VALIDATION RESULTS *******")
            print(f'RMSE (dB): {final_test_rmse:.2f}')
            print(f'R2 (-): {final_test_r2_adjusted:.2f}')
            
            # Create and log visualizations
            create_final_model_visualizations(y_test, y_pred)
        
        print("MLflow experiment completed successfully!")
        print(f"Check MLflow UI with: mlflow ui")

if __name__ == "__main__":
    main()