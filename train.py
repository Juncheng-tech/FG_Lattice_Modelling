import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model.mlp_model import MLPModel

# Load dataset from CSV file
df = pd.read_csv("data/dataset.csv")

# Define input features and target outputs
X = df[["cell_size_mm", "strut_diameter_mm", "porosity", "gradation_index"]].values
y = df[["E_xx_GPa", "E_yy_GPa"]].values

# Split data into training+validation and test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# Split training+validation into training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42
)
# 0.1765 of 85% ≈ 15% of total

# Standardize input and output data
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_train)
X_val = x_scaler.transform(X_val)
X_test = x_scaler.transform(X_test)

y_train = y_scaler.fit_transform(y_train)
y_val = y_scaler.transform(y_val)
y_test = y_scaler.transform(y_test)

# Convert numpy arrays to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Initialize MLP model, loss function, and optimizer
hidden_dim = 100

model = MLPModel(
    input_dim=X_train.shape[1],
    output_dim=y_train.shape[1],
    hidden_dim=hidden_dim
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Gradient-related weight
# Set to 0.0 for now because there is no true gradient label yet.
lambda_grad = 0.0

# Training process
epochs = 500
train_loss_history = []
val_loss_history = []

for epoch in range(epochs):
    # Training step
    model.train()

    # Clone training input and enable gradient tracking
    X_train_batch = X_train.clone().detach().requires_grad_(True)

    # Forward pass
    train_pred = model(X_train_batch)

    # Output loss
    train_loss_output = criterion(train_pred, y_train)

    # Gradient of outputs with respect to inputs
    grad_pred = torch.autograd.grad(
        outputs=train_pred,
        inputs=X_train_batch,
        grad_outputs=torch.ones_like(train_pred),
        create_graph=True
    )[0]

    # Temporary gradient-related term
    # This is only a placeholder for the simplified DANN prototype.
    train_loss_grad = torch.mean(grad_pred ** 2)

    # Total loss
    train_loss = train_loss_output + lambda_grad * train_loss_grad

    # Backpropagation
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # Validation step
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val)

    # Save loss values
    train_loss_history.append(train_loss.item())
    val_loss_history.append(val_loss.item())

    # Print progress
    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Train Loss: {train_loss.item():.6f}, "
            f"Output Loss: {train_loss_output.item():.6f}, "
            f"Grad Term: {train_loss_grad.item():.6f}, "
            f"Val Loss: {val_loss.item():.6f}"
        )

# Final test evaluation
model.eval()

# Enable gradient tracking on test input
X_test_grad = X_test.clone().detach().requires_grad_(True)

# Test prediction
test_pred = model(X_test_grad)
test_loss = criterion(test_pred, y_test)

# Compute gradient on test set
test_grad = torch.autograd.grad(
    outputs=test_pred,
    inputs=X_test_grad,
    grad_outputs=torch.ones_like(test_pred),
    create_graph=False
)[0]

print(f"Training completed for hidden_dim={hidden_dim}.")
print("Final Validation Loss:", val_loss.item())
print("Test MSE:", test_loss.item())
print(f"Gradient shape on test set: {test_grad.shape}")
print("Example gradient (first sample):", test_grad[0])

# Save training results to text file
with open(f"results/training_results_hidden_{hidden_dim}.txt", "w") as f:
    f.write("Week 6 hidden neuron comparison result\n")
    f.write(f"Model: Simplified DANN prototype (hidden_dim={hidden_dim})\n")
    f.write("Task: Input parameters to target outputs\n")
    f.write("Training status: Completed\n")
    f.write("Derivative-aware setting: gradient computed by autograd\n")
    f.write(f"Final Validation Loss: {val_loss.item()}\n")
    f.write(f"Test MSE: {test_loss.item()}\n")
    f.write(f"Gradient shape on test set: {tuple(test_grad.shape)}\n")
    f.write(f"Example gradient (first sample): {test_grad[0].tolist()}\n")

# Plot and save training loss curve
plt.figure(figsize=(8, 5))
plt.plot(train_loss_history, label="Training Loss")
plt.plot(val_loss_history, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig(f"results/loss_curve_hidden_{hidden_dim}.png")
plt.show()

# Plot predicted vs true values for E_xx_GPa
plt.figure(figsize=(6, 6))
plt.scatter(y_test[:, 0].numpy(), test_pred[:, 0].detach().numpy())
plt.xlabel("True E_xx_GPa (scaled)")
plt.ylabel("Predicted E_xx_GPa (scaled)")
plt.title("Predicted vs True Values (E_xx_GPa)")
plt.grid(True)
plt.savefig(f"results/pred_vs_true_exx_hidden_{hidden_dim}.png")
plt.show()

# Plot test MSE bar chart
plt.figure(figsize=(6, 4))
plt.bar(["Test MSE"], [test_loss.item()])
plt.ylabel("MSE")
plt.title("Test Error")
plt.grid(True, axis="y")
plt.savefig(f"results/test_mse_bar_hidden_{hidden_dim}.png")
plt.show()