import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Define a simple model
model = Sequential([
    Dense(10, activation='relu', input_shape=(5,)),
    Dense(1, activation='linear')
])

# Compile the model without specifying an optimizer
try:
    model.compile(loss='mse')
    # Print the default optimizer
    print("Default optimizer:", model.optimizer.get_config())
except Exception as e:
    print("Error:", e)
    print("No default optimizer is used. Specify one explicitly in `model.compile()`.")
