import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Define a simple model with a Dense layer without specifying activation
try:
    model = Sequential([
        Dense(10, input_shape=(5,))
    ])
    # Check the default activation function
    default_activation = model.layers[0].activation
    # Get the name of the activation function
    activation_name = default_activation.__name__ if default_activation else "None"
    print("Default activation function:", activation_name)
except Exception as e:
    print("Error:", e)
    print("No default activation function is set by default.")
