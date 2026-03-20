import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Generate dummy data for initial testing
# 100 samples with 4 input design parameters
X = np.random.rand(100, 4)

# Corresponding output: 2 target mechanical responses
y = np.random.rand(100, 2)

# Split dataset into training and testing sets (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a basic MLP regression model
model = MLPRegressor(hidden_layer_sizes=(16, 16), max_iter=1000, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)

print("Model training completed.")
print("MSE:", mse)
print("Week 5 basic model training finished.")