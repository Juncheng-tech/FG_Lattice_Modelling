import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model.mlp_model import MLPModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(BASE_DIR, "data", "dataset.csv")
hidden_dim = 100
model_path = os.path.join(BASE_DIR, "results", f"mlp_hidden_{hidden_dim}.pt")

optim_steps = 200
learning_rate = 0.01
custom_initial_design = None  # set manually if needed

# =========================
# Load and process data
# =========================
df = pd.read_csv(dataset_path)

input_cols = ["cell_size_mm", "strut_diameter_mm", "porosity", "gradation_index"]
output_cols = ["E_xx_GPa", "E_yy_GPa"]

X = df[input_cols].values
y = df[output_cols].values

# Use the same split as training
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)

x_scaler = StandardScaler()
y_scaler = StandardScaler()
x_scaler.fit(X_train)
y_scaler.fit(y_train)

# Input bounds and scaling stats
x_min = torch.tensor(X.min(axis=0), dtype=torch.float32)
x_max = torch.tensor(X.max(axis=0), dtype=torch.float32)
x_mean = torch.tensor(x_scaler.mean_, dtype=torch.float32)
x_std = torch.tensor(x_scaler.scale_, dtype=torch.float32)
y_mean = torch.tensor(y_scaler.mean_, dtype=torch.float32)
y_std = torch.tensor(y_scaler.scale_, dtype=torch.float32)

# =========================
# Load trained model
# =========================
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

model = MLPModel(input_dim=4, output_dim=2, hidden_dim=hidden_dim)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# =========================
# Initial design
# =========================
if custom_initial_design is None:
    design = torch.tensor(X.mean(axis=0), dtype=torch.float32, requires_grad=True)
else:
    design = torch.tensor(custom_initial_design, dtype=torch.float32, requires_grad=True)

# =========================
# History storage
# =========================
loss_history = []
design_history = []
output_history = []

# =========================
# Optimization loop
# Objective: minimize (E_xx - E_yy)^2
# =========================
for step in range(optim_steps):
    if design.grad is not None:
        design.grad.zero_()

    # Normalize input
    x_norm = (design - x_mean) / x_std
    y_pred_norm = model(x_norm.unsqueeze(0))

    # Convert back to physical values
    y_pred = y_pred_norm * y_std + y_mean
    e_xx = y_pred[0, 0]
    e_yy = y_pred[0, 1]

    # Objective function
    loss = (e_xx - e_yy) ** 2

    # Update design variables
    loss.backward()
    with torch.no_grad():
        design -= learning_rate * design.grad
        design.clamp_(min=x_min, max=x_max)

    # Record history
    loss_history.append(loss.item())
    design_history.append(design.detach().numpy())
    output_history.append(y_pred.detach().numpy()[0])

    if (step + 1) % 20 == 0:
        print(f"Step [{step+1}/{optim_steps}] | Loss: {loss.item():.6f} | E_xx: {e_xx.item():.4f} | E_yy: {e_yy.item():.4f}")

# =========================
# Final results
# =========================
final_design = design.detach()
final_pred = model((final_design - x_mean) / x_std) * y_std + y_mean
initial_x = torch.tensor(X.mean(axis=0), dtype=torch.float32)
initial_pred = model(((initial_x - x_mean) / x_std).unsqueeze(0)) * y_std + y_mean
print("\nOptimization finished.")
print("Initial design:", X.mean(axis=0))
print("Final design:", final_design.numpy())
print("Initial output:", initial_pred.detach().numpy()[0])
print("Final output:", final_pred.detach().numpy()[0])

# =========================
# Save output log
# =========================
os.makedirs("results", exist_ok=True)
with open("results/optimization_result.txt", "w") as f:
    f.write("Optimization: Minimize (E_xx - E_yy)^2\n")
    f.write(f"Steps: {optim_steps}, Learning rate: {learning_rate}\n\n")

    f.write("Initial design:\n")
    for name, val in zip(input_cols, X.mean(axis=0)):
        f.write(f"{name}: {val:.4f}\n")

    f.write("\nFinal design:\n")
    for name, val in zip(input_cols, final_design.numpy()):
        f.write(f"{name}: {val:.4f}\n")

    f.write(f"\nInitial loss: {loss_history[0]:.6f}\n")
    f.write(f"Final loss: {loss_history[-1]:.6f}\n")

# =========================
# Plot results
# =========================
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Optimization Loss Curve")
plt.grid(True)
plt.savefig("results/optim_loss.png")
plt.show()

plt.figure(figsize=(10, 6))
for i, name in enumerate(input_cols):
    plt.plot([d[i] for d in design_history], label=name)
plt.xlabel("Iteration")
plt.ylabel("Design Variable Value")
plt.title("Design Variables During Optimization")
plt.legend()
plt.grid(True)
plt.savefig("results/optim_design.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot([out[0] for out in output_history], label="E_xx_GPa")
plt.plot([out[1] for out in output_history], label="E_yy_GPa")
plt.xlabel("Iteration")
plt.ylabel("Predicted Modulus")
plt.title("Predicted Outputs During Optimization")
plt.legend()
plt.grid(True)
plt.savefig("results/optim_output.png")
plt.show()