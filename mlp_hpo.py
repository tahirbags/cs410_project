import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_curve
import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import re
from sklearn.metrics import silhouette_score
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB


'''
Hyperparameter selection:

1. Run mlp_hpo.py

2. Using GridSearchCV to find the optimal hyperparameters

3. Set parameter_space to list hyperparameters to tune

4. n_jobs -1 is for using all the CPU cores available, cv=5 is for cross validation, here it means 5-folds Stratified K-fold cross validation

5. Print best parameters

6. Update best parameters in mlp.py

'''

#load and pre-process the data
def load_and_preprocess_data(csv_path):
    data = pd.read_csv(csv_path)
   
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(data['text'])
    y = data['target']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # Path to your training dataset
    csv_path = 'train.csv'

    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(csv_path)

    # Start time
    start_time = time.time()
    #HPO started
    print('Hyperparameter tuning in progress..this could take a 10-15 mins to complete')

    mlp_model = MLPClassifier(max_iter=10000)
    parameter_space = {
    'hidden_layer_sizes': [(10,30,10),(20,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.05, 0.01],
    'learning_rate': ['constant','adaptive'],
    }

    clf = GridSearchCV(mlp_model, parameter_space, n_jobs=-1, cv=5)
    clf.fit(X_train, y_train) # X is train samples and y is the corresponding labels
    print('Best parameters found:\n', clf.best_params_)
    
    # calculate time elapsed for models to generate predictions
    print("Elapsed: {} seconds".format(round(time.time() - start_time, 4)))