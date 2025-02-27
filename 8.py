import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Sample data
X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]]) # 4 samples, 2 features
y = np.array([1, 0, 0, 1]) # Corresponding labels
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=42)
# Create and fit the model
model = RandomForestClassifier(n_estimators=100, random_state=42) 
#Set the number of decision trees
model.fit(X_train, y_train)
# Predict the output
y_pred = model.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
