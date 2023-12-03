import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import time

def load_and_preprocess_data(csv_path):
    data = pd.read_csv(csv_path)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(data['text'])
    y = data['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_preprocess_data('train.csv')

    start_time = time.time()

    dt_params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_clf = GridSearchCV(dt_model, dt_params, cv=5, n_jobs=-1, verbose=2)
    dt_clf.fit(X_train, y_train)

    print('Best parameters for DecisionTreeClassifier:', dt_clf.best_params_)
    print("Elapsed: {} seconds".format(round(time.time() - start_time, 4)))
