import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import numpy as np
import time

def load_and_preprocess_data(csv_path):
    data = pd.read_csv(csv_path)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(data['text']).todense()  # GaussianNB requires dense input
    y = data['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_preprocess_data('train.csv')

    start_time = time.time()

    gnb_params = {'var_smoothing': np.logspace(0, -9, num=100)}
    gnb_model = GaussianNB()
    gnb_clf = GridSearchCV(gnb_model, gnb_params, cv=5, n_jobs=-1, verbose=2)
    gnb_clf.fit(X_train, y_train)

    print('Best parameters for GaussianNB:', gnb_clf.best_params_)
    print("Elapsed: {} seconds".format(round(time.time() - start_time, 4)))
