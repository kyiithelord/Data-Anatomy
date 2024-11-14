import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
data = pd.read_csv('data/dataset.csv')

# Encoding categorical data for the 'Location' column
le_location = LabelEncoder()
data['Location'] = le_location.fit_transform(data['Location'])

# Features and target variable
X = data[['Age', 'Income', 'Location']]  # Features
y = data['Purchase']  # Target variable

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'models/train_model.pkl')

print("Model trained and saved successfully.")
