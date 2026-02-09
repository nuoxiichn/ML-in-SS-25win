import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

# ---------------------------------------------------------
# 1. Import Data
# ---------------------------------------------------------
print("Fetching Phishing Websites dataset...")
website_phishing = fetch_ucirepo(id=379) 
X = website_phishing.data.features 
y = website_phishing.data.targets.iloc[:, 0]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 2. MLflow Experiment Setup
# ---------------------------------------------------------
mlflow.set_experiment("Phishing_Website_Classification")

# Define models to test
models = {
    "Extra_Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    "Gradient_Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# ---------------------------------------------------------
# 3. Execution with Parent-Child Runs
# ---------------------------------------------------------
with mlflow.start_run(run_name="Algorithm_Comparison_Parent"):
    
    print("Starting Parent Run. Training child models...")
    
    for name, model in models.items():
        # Start a nested Child Run for each algorithm
        with mlflow.start_run(run_name=name, nested=True):
            print(f"Running Child: {name}")
            
            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            
            # Log Parameters and Metrics
            mlflow.log_param("algorithm", name)
            mlflow.log_param("n_estimators", 100)
            mlflow.log_metric("accuracy", acc)
            
            # Log the Model itself
            mlflow.sklearn.log_model(model, f"{name}_model")
            
            # Generate and Log Confusion Matrix Plot as Artifact
            plt.figure(figsize=(6, 4))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f"Confusion Matrix: {name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            
            # Save plot locally then log to MLflow
            plot_filename = f"cm_{name}.png"
            plt.savefig(plot_filename)
            mlflow.log_artifact(plot_filename)
            plt.close()
            
            # Clean up local file
            if os.path.exists(plot_filename):
                os.remove(plot_filename)

    print("\nAll runs completed. To see results, run 'mlflow ui' in your terminal.")

# ---------------------------------------------------------
# 4. Local Final Visualization (Optional)
# ---------------------------------------------------------
# (Correlation heatmap remains for local insight)
plt.figure(figsize=(10, 8))
sns.heatmap(pd.concat([X, y], axis=1).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()