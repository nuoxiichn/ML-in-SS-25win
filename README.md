# ML-in-SS-25win

## Introduction

This project is used for learning machine learning algorithms, using the NASA Airfoil Self-Noise dataset.

### Dataset Information

The dataset comes from the UCI Machine Learning Repository (ID: 291).
It comprises different size NACA 0012 airfoils at various wind tunnel speeds and angles of attack. The span of the airfoil and the observer position were the same in all of the experiments.

- **Data Source**: NASA
- **UCI Link**: [Airfoil Self-Noise Data Set](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise)

### Variables

| Type | Name | Description | Units |
| --- | --- | --- | --- |
| **Input** | Frequency | Frequency | Hertzs (Hz) |
| **Input** | Angle of attack | Angle of attack | Degrees (°) |
| **Input** | Chord length | Chord length | Meters (m) |
| **Input** | Free-stream velocity | Free-stream velocity | Meters per second (m/s) |
| **Input** | Suction side displacement thickness | Suction side displacement thickness | Meters (m) |
| **Output** | Scaled sound pressure level | Scaled sound pressure level | Decibels (dB) |

## Set up

It is required to install `ucimlrepo` and `pandas` libraries to load and process data.

```bash
pip install ucimlrepo pandas scikit-learn matplotlib seaborn scipy mlflow
```

## Model Building and Analysis

This project provides two comprehensive analysis scripts:

- `regression_analysis.py`: Standard regression analysis pipeline
- `regression_analysis_mlflow.py`: **MLFlow-enabled** version with experiment tracking (recommended)

### Pipeline Steps

1.  **Data Loading**: Fetches the NASA Airfoil Self-Noise dataset directly from the UCI repository.
2.  **Data Cleaning**: Automatically applies the IQR (Interquartile Range) method to remove outliers from the dataset.
3.  **EDA & Visualization**:
    *   Generates scatter plots comparing data distribution before and after cleaning.
    *   Computes and visualizes the correlation matrix and p-values to understand feature relationships.
4.  **Model Training**: Trains and evaluates multiple regression models using K-Fold Cross Validation:
    *   Linear Regression
    *   Random Forest Regressor
    *   Extra Trees Regressor
    *   Gradient Boosting Regressor (GBR)
    *   AdaBoost Regressor
5.  **Hyperparameter Tuning**: optimize the Gradient Boosting Regressor using `RandomizedSearchCV`.
6.  **Final Evaluation**: Evaluates the best model on the test set and visualizes residuals.

### Usage

**Standard Analysis:**
```bash
python regression_analysis.py
```

**MLFlow-Enabled Analysis (Recommended):**
```bash
python regression_analysis_mlflow.py
```

After running the MLFlow version, launch the MLFlow UI to explore results:
```bash
mlflow ui
```
Then visit `http://localhost:5000` to view the experiment tracking dashboard.

## MLFlow Features

The MLFlow-enabled version provides:

- **Experiment Tracking**: Automatic logging of all model parameters, metrics, and artifacts
- **Model Registry**: Versioned model storage with metadata
- **Visualization Artifacts**: Automatically saved plots (correlation heatmaps, residuals, etc.)
- **Comparison Dashboard**: Easy comparison between different model runs
- **Reproducibility**: Complete parameter and environment tracking


 