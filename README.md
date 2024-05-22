
# Data Science: Classification Model

This project involves building a classification model using the K-Nearest Neighbors (kNN) algorithm to classify data into different categories. The dataset used for this classification task is loaded from a CSV file, split into training and testing sets, and used to train a kNN classifier. The trained model is then evaluated on the testing set, and its accuracy score is calculated.

## Project Overview

The classification model project includes the following steps:

1. **Data Loading**: Loading the dataset from a CSV file using the `pandas` library.

2. **Data Preprocessing**: Splitting the dataset into features (X) and target labels (y) and further dividing them into training and testing sets using the `train_test_split` function from `sklearn.model_selection`.

3. **Model Training**: Training a K-Nearest Neighbors classifier on the training set using the `KNeighborsClassifier` class from `sklearn.neighbors`.

4. **Model Evaluation**: Testing the trained classifier on the testing set and calculating the accuracy score using the `accuracy_score` function from `sklearn.metrics`.

5. **Model Persistence**: Saving the trained model to a file using the `joblib.dump` function from the `joblib` library.

## Setup

Ensure you have the necessary dependencies installed to run the project:

```bash
pip install pandas scikit-learn joblib


Output

Upon execution, the script will output the accuracy score of the trained kNN classifier on the testing set. Additionally, the trained model will be saved to a file (kNN_model_HW6.pkl) for future use.
Accuracy score: 1.0

Running the Code

To execute the classification model code, run the provided Python script (classification.py) in your preferred Python environment. Ensure that you have the required dataset accessible.
##  Visualization

see images folder

##  License

This project is licensed under the MIT License. See the LICENSE file for details.
