import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

# ---------------------------------------------------------
# 1. Load and Preprocess Data
# ---------------------------------------------------------
print("Loading dataset...")
website_phishing = fetch_ucirepo(id=379)
X = website_phishing.data.features.values.astype(np.float32)
y = (website_phishing.data.targets.values.flatten() + 1).astype(np.int64) # Map -1,0,1 to 0,1,2

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert to PyTorch Tensors
def create_loader(X_data, y_data, batch_size=32, shuffle=True):
    dataset = TensorDataset(torch.tensor(X_data), torch.tensor(y_data))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_loader = create_loader(X_train, y_train)
val_tensor = torch.tensor(X_val)
val_labels = torch.tensor(y_val)
test_tensor = torch.tensor(X_test)

num_features = X.shape[1]
num_classes = 3

# ---------------------------------------------------------
# 2. Define Model Architectures (PyTorch style)
# ---------------------------------------------------------
class PhishingModel(nn.Module):
    def __init__(self, architecture_type):
        super(PhishingModel, self).__init__()
        if architecture_type == "Baseline_MLP":
            self.layers = nn.Sequential(
                nn.Linear(num_features, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, num_classes)
            )
        elif architecture_type == "Deep_MLP":
            self.layers = nn.Sequential(
                nn.Linear(num_features, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, num_classes)
            )
        elif architecture_type == "Regularized_MLP":
            self.layers = nn.Sequential(
                nn.Linear(num_features, 128), nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, num_classes)
            )

    def forward(self, x):
        return self.layers(x)

# ---------------------------------------------------------
# 3. Training Loop Helper
# ---------------------------------------------------------
def train_model(model, train_loader, epochs, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        train_acc = correct / total
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_tensor)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_acc = (val_predicted == val_labels).sum().item() / val_labels.size(0)
        
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
    return history

# ---------------------------------------------------------
# 4. MLflow Experiment Execution
# ---------------------------------------------------------
mlflow.set_experiment("Phishing_PyTorch_Comparison")

configs = {
    "Baseline_MLP": 50,
    "Deep_MLP": 50,
    "Regularized_MLP": 80
}

with mlflow.start_run(run_name="PyTorch_Architecture_Comparison"):
    for name, epochs in configs.items():
        with mlflow.start_run(run_name=name, nested=True):
            print(f"Training {name} for {epochs} epochs...")
            
            model = PhishingModel(name)
            history = train_model(model, train_loader, epochs)
            
            # Final Evaluation
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_tensor)
                _, y_pred = torch.max(test_outputs, 1)
                test_acc = accuracy_score(y_test, y_pred.numpy())
            
            print(f"Test Accuracy: {test_acc:.4f}")
            
            # Log Metrics & Model
            mlflow.log_param("model_type", name)
            mlflow.log_param("epochs", epochs)
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.pytorch.log_model(model, f"model_{name}")
            
            # Plot 1: Accuracy Curves
            plt.figure(figsize=(8, 5))
            plt.plot(history['train_acc'], label='Train Acc')
            plt.plot(history['val_acc'], label='Val Acc')
            plt.title(f"Accuracy Curve: {name}")
            plt.legend()
            acc_plot = f"acc_{name}.png"
            plt.savefig(acc_plot)
            mlflow.log_artifact(acc_plot)
            plt.close()
            
            # Plot 2: Confusion Matrix
            cm = confusion_matrix(y_test, y_pred.numpy())
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
            plt.title(f"Confusion Matrix: {name}")
            cm_plot = f"cm_{name}.png"
            plt.savefig(cm_plot)
            mlflow.log_artifact(cm_plot)
            plt.close()

print("\nSuccess! Run 'mlflow ui' to see the PyTorch results.")