import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model.mlp_model import MLPModel

# 1. Load dataset
df = pd.read_csv("data/dataset.csv")

# 2. Define inputs and outputs
X = df[["cell_size_mm", "strut_diameter_mm", "porosity", "gradation_index"]].values
y = df[["E_xx_GPa", "E_yy_GPa"]].values

# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Standardize data
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

# 5. Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 6. Build model
model = MLPModel(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 7. Train model
epochs = 500
loss_history = []

for epoch in range(epochs):
    model.train()
    pred = model(X_train)
    loss = criterion(pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

# 8. Evaluate model
model.eval()
with torch.no_grad():
    test_pred = model(X_test)
    test_loss = criterion(test_pred, y_test)

print("Training completed.")
print("Test MSE:", test_loss.item())

# 9. Save results text
with open("results/training_results.txt", "w") as f:
    f.write("Week 5 basic reproduction result\n")
    f.write("Model: Basic MLP\n")
    f.write("Task: Input parameters to material-related outputs\n")
    f.write("Training status: Completed\n")
    f.write(f"Test MSE: {test_loss.item()}\n")

# 10. Plot training loss curve
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.savefig("results/loss_curve.png")
plt.show()

# 11. Plot predicted vs true values for E_xx_GPa
plt.figure(figsize=(6, 6))
plt.scatter(y_test[:, 0].numpy(), test_pred[:, 0].numpy())
plt.xlabel("True E_xx_GPa (scaled)")
plt.ylabel("Predicted E_xx_GPa (scaled)")
plt.title("Predicted vs True Values (E_xx_GPa)")
plt.grid(True)
plt.savefig("results/pred_vs_true_exx.png")
plt.show()

# 12. Plot bar chart of test MSE
plt.figure(figsize=(6, 4))
plt.bar(["Test MSE"], [test_loss.item()])
plt.ylabel("MSE")
plt.title("Test Error")
plt.grid(True, axis="y")
plt.savefig("results/test_mse_bar.png")
plt.show()