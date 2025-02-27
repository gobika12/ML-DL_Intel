from sklearn.linear_model import LogisticRegression
import numpy as np
# Training data: x values and corresponding binary y labels
# Features
x = np.array([[1], [2], [3], [4], [5]])
# Labels (binary classification)
y = np.array([0, 0, 0, 1, 1])
# Create and train the model
model = LogisticRegression()
model.fit(x, y)
# Make predictions
test_data = np.array([[2.5], [3.5]])
predictions = model.predict(test_data)
print("Predictions for inputs {}: {}".format(test_data.flatten(),
predictions))
