import tensorflow as tf
import numpy as np

# Check for GPU
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print("GPU detected:", gpu_devices)
else:
    print("No GPU detected.")

# Create a small dummy model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(100,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')

# Generate random data
x_train = np.random.random((1000, 100))
y_train = tf.keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), 10)

# Train for a few epochs
model.fit(x_train, y_train, epochs=5, batch_size=32)
