import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
# Define the RNN model
model = Sequential()
# Add a SimpleRNN layer with 128 units
model.add(SimpleRNN(128, input_shape=(28, 28), activation='relu'))
# Add a Dense layer with 10 units (for 10 digits) and softmax activation
model.add(Dense(10, activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
metrics=['accuracy'])
# Reshape the input data to match the RNN input shape
x_train = x_train.reshape((x_train.shape[0], 28, 28))
x_test = x_test.reshape((x_test.shape[0], 28, 28))
# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
# Evaluate the model on the test set
accuracy = model.evaluate(x_test, y_test)[1]
print(f"Test Accuracy: {accuracy}")
