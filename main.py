import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from train import *
# Step 1: create dummy input data
# Example: 100 samples, 4 input features
X = np.random.rand(100, 4)

# Step 2: create dummy output data
# Example: 100 samples, 2 output targets
y = np.random.rand(100, 2)

# Step 3: split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: build simple neural network
model = MLPRegressor(
    hidden_layer_sizes=(16, 16),
    max_iter=1000,
    random_state=42
)

# Step 5: train model
model.fit(X_train, y_train)

# Step 6: predict
y_pred = model.predict(X_test)

# Step 7: evaluate
mse = mean_squared_error(y_test, y_pred)
print("Model training completed.")
print("MSE:", mse)
print("Week 5 basic model training finished.")