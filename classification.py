############################################
#
#  Data Science:  Classification Model
#
#  Written By : BARA AHMAD MOHAMMED
#
#############################################
# TODO : IMPORT ALL NEEDED LIBRARIES
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
import os

# Ensure the 'images' directory exists
os.makedirs('images', exist_ok=True)

# Load dataset
df = pd.read_csv("hw6.data.csv")

# Split dataset into training and testing sets
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Labels
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=42)

# Train kNN classifier on training set
kNN = KNeighborsClassifier()
kNN.fit(X_train, y_train)

# Test classifier on testing set
y_pred = kNN.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score:", accuracy)

# Save trained model
joblib.dump(kNN, 'kNN_model_HW6.pkl')

# Visualization
# Assuming binary classification for simplicity

# Plotting true labels
plt.scatter(X_test[y_test == 0].iloc[:, 0], X_test[y_test == 0].iloc[:, 1], color='blue', label='Class 0 (Actual)')
plt.scatter(X_test[y_test == 1].iloc[:, 0], X_test[y_test == 1].iloc[:, 1], color='red', label='Class 1 (Actual)')

# Plotting predicted labels
plt.scatter(X_test[y_pred == 0].iloc[:, 0], X_test[y_pred == 0].iloc[:, 1], color='lightblue', label='Class 0 (Predicted)')
plt.scatter(X_test[y_pred == 1].iloc[:, 0], X_test[y_pred == 1].iloc[:, 1], color='lightcoral', label='Class 1 (Predicted)')

plt.title('Classification Model Visualization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.savefig('images/classification_visualization.png')
plt.show()
