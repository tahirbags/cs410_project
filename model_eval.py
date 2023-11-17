import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def load_and_preprocess_data(csv_path):
    data = pd.read_csv(csv_path)
   
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(data['text'])
    y = data['target']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def load_model(model_path, class_name):
    model_directory, model_file = os.path.split(model_path)
    model_name = model_file.replace('.py', '')
    sys.path.insert(0, model_directory)

    model_module = __import__(model_name)
    model_class = getattr(model_module, class_name)
    return model_class()

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    return {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, average='binary'),
        'recall': recall_score(y_test, predictions, average='binary'),
        'f1_score': f1_score(y_test, predictions, average='binary')
    }

if __name__ == '__main__':
    # Path to your training dataset
    csv_path = 'train.csv'

    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(csv_path)

    # List of models to evaluate - each entry is a (file_path, class_name) tuple
    models_to_evaluate = [
        ('models/model1.py', 'LogisticModel'),
        ('models/RandomForestClassifier.py', 'RandomForestClassifierModel'), #TB
        ('models/AdaBoostClassifier.py', 'AdaBoostClassifierModel'), #TB
        ('models/DecisionTreeClassifier.py', 'DecisionTreeClassifierModel'), #TB
        ('models/KNeighborsClassifier.py', 'KNeighborsClassifierModel'), #TB
        ('models/GaussianNB.py', 'GaussianNBModel'), #TB
        ('models/GradientBoostingClassifier.py', 'GradientBoostingClassifierModel'), #TB
#        ('models/svc.py', 'SVCModel'), #TB
        # Add more models here
    ]

    start_time = time.time()

   # Evaluate each model
    for model_path, class_name in models_to_evaluate:
        model = load_model(model_path, class_name)
        model.train(X_train, y_train)
        results = evaluate_model(model, X_test, y_test)
        print(f'Results for {class_name}: {results}')
        print("Elapsed: {} seconds".format(round(time.time() - start_time, 4)))



# # Function to read a CSV  file and return the data
# def read_csv(file_path, target_column):
#     data = pd.read_csv(file_path)
#     return data[target_column]

# # Paths to the CSV files
# true_labels_test_path = 'example_true_labels_for_test.csv'  # Replace with the true labels for the test.csv
# predictions_file_path = 'example_predictions.csv'  # Replace with predictions from the model


# true_labels_test = pd.read_csv(true_labels_test_path)
# predictions_data = pd.read_csv(predictions_file_path)

# merged_data = pd.merge(true_labels_test, predictions_data, on='id')

# # Calculate metrics

# target_column_name = 'target'
# prediction_column_name = 'prediction'
# accuracy = accuracy_score(merged_data[target_column_name], merged_data[prediction_column_name])
# precision = precision_score(merged_data[target_column_name], merged_data[prediction_column_name], zero_division=0)
# recall = recall_score(merged_data[target_column_name], merged_data[prediction_column_name], zero_division=0)
# f1 = f1_score(merged_data[target_column_name], merged_data[prediction_column_name], zero_division=0)
# conf_matrix = confusion_matrix(merged_data[target_column_name], merged_data[prediction_column_name])

# # Output the results
# print(f'Confusion Matrix:\n{conf_matrix}')
# print(f'Accuracy: {accuracy:.4f}')
# print(f'Precision: {precision:.4f}')
# print(f'Recall: {recall:.4f}')
# print(f'F1 Score: {f1:.4f}')



