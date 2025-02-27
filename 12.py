import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
file_path ='/home/test4/dataset/iris.csv'  # Use raw string format (r"")
iris_df = pd.read_csv(file_path)

# Inspect the dataset
print(iris_df.head())

# Separate features and labels
X = iris_df.iloc[:, :-1].values  # All columns except the last
y = iris_df.iloc[:, -1].values   # Last column (species)

# Encode categorical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Convert species names into numerical labels

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate different models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.2f}")

