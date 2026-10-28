import copy
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
Path("artifacts").mkdir(exist_ok=True)
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Using device:", device)
dataset = load_breast_cancer()
X = dataset.data.astype(np.float32)
y = dataset.target.astype(np.float32)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

class CsvDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(
            features, dtype=torch.float32
        )
        self.labels = torch.tensor(
            labels, dtype=torch.float32
        )
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        return self.features[index], self.labels[index]

train_dataset = CsvDataset(X_train, y_train)
val_dataset = CsvDataset(X_val, y_val)
test_dataset = CsvDataset(X_test, y_test)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

class MLP(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x).squeeze(1)

model = MLP(X_train.shape[1]).to(device)
parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)
print("Trainable parameters:", parameter_count)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

def evaluate_model(model, data_loader):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            predictions = (
                torch.sigmoid(logits) >= 0.5
            ).float()
            all_predictions.extend(
                predictions.cpu().numpy()
            )
            all_labels.extend(labels.cpu().numpy())
    average_loss = total_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    return average_loss, accuracy

best_val_accuracy = -1.0
best_val_loss = float("inf")
best_epoch = 0
best_weights = None
patience = 7
epochs_without_improvement = 0
train_losses = []
val_accuracies = []

for epoch in range(40):
    model.train()
    running_loss = 0.0
    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(labels)
    train_loss = running_loss / len(train_dataset)
    val_loss, val_accuracy = evaluate_model(
        model, val_loader
    )
    train_losses.append(train_loss)
    val_accuracies.append(val_accuracy)
    print(
        f"Epoch {epoch + 1:02d}/40 | "
        f"Train loss: {train_loss:.4f} | "
        f"Val loss: {val_loss:.4f} | "
        f"Val accuracy: {val_accuracy:.4f}"
    )
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1
        best_weights = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Early stopping triggered.")
        break

model.load_state_dict(best_weights)
test_loss, test_accuracy = evaluate_model(
    model, test_loader
)
print("\nBest epoch:", best_epoch)
print("Best validation accuracy:", best_val_accuracy)
print("Test accuracy:", test_accuracy)
torch.save(
    model.state_dict(),
    "artifacts/mlp_state_dict.pt"
)
joblib.dump(scaler, "artifacts/scaler.joblib")
reloaded_model = MLP(X_train.shape[1]).to(device)
reloaded_model.load_state_dict(
    torch.load(
        "artifacts/mlp_state_dict.pt",
        map_location=device,
        weights_only=True
    )
)
reloaded_model.eval()
_, reloaded_accuracy = evaluate_model(
    reloaded_model, test_loader
)
print("Reloaded model accuracy:", reloaded_accuracy)
assert abs(test_accuracy - reloaded_accuracy) < 1e-12
print("Reloaded accuracy matches the original model.")
logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)
logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)
logistic_accuracy = accuracy_score(
    y_test, logistic_predictions
)
print("Logistic regression accuracy:", logistic_accuracy)

plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="Training loss")
plt.xlabel("Epoch")
plt.ylabel("BCE loss")
plt.title("PyTorch MLP Training Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reports/task2_training_loss.png")
plt.show()
