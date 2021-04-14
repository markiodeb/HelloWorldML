
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Next, create the simplest possible neural network. It has one layer, that layer has one neuron, and the input shape to it is only one value.
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# First, here's how to tell it to use mean_squared_error for the loss and stochastic gradient descent (
# sgd) for the optimizer. You don't need to understand the math for those yet, but you can see that they work!
model.compile(optimizer='sgd', loss='mean_squared_error')

# A python library called NumPy provides lots of array type data structures to do this. Specify the values as an array in NumPy with np.array[].
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# Train the neural network
model.fit(xs, ys, epochs=500)

# Use the model
print(model.predict([10.0]))