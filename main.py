import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# Generate random input features and output labels
# Input will be 2d numpy array of size 1000 x 2 of two random numbers (0-1)
# Output will be numpy array of size 1000 will be sum of both corresponding numbers
np.random.seed(42)
x_train = np.random.rand(1000, 2)
y_train = x_train[:, 0] + x_train[:, 1]

# Print sample data
print("Sample input features:")
print(x_train[:5])
print("Sample output labels:")
print(y_train[:5])

# Define the model using Keras sequential API
# Add hidden layer of 10 neurons and another layer of 1 output neuron
# Uses ReLU activation function
model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu'))
model.add(Dense(1))

# Compile the model with specified loss function (mean squared error) 
# Will be trained using Adam optimizer
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model using fit() on input features and output labels
# Model is trained for specified number of epochs and batch size
model_history = model.fit(x_train, y_train, epochs=100, batch_size=32)

# Generate test data
x_test = np.random.rand(100, 2)
y_test = x_test[:, 0] + x_test[:, 1]

# Evaluate the model on test data using mean squared error then print to console
loss = model.evaluate(x_test, y_test)
print("Test loss:", loss)

# Plot training loss over time
plt.plot(model_history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# Plot predicted output against true output
y_pred = model.predict(x_test)
plt.scatter(y_test, y_pred)
plt.title('Predicted vs True Output')
plt.ylabel('Predicted Output')
plt.xlabel('True Output')
plt.show()