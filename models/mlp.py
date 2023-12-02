from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

'''
Hyperparameter selection:

1. Run mlp_hpo.py

2. Using GridSearchCV to find the optimal hyperparameters

3. Set parameter_space to list hyperparameters to tune

4. n_jobs -1 is for using all the CPU cores available, cv=5 is for cross validation, here it means 5-folds Stratified K-fold cross validation

5. Print best parameters

6. Update best parameters in mlp.py

'''

class MLPClassifierModel:
    def __init__(self):
        self.model = MLPClassifier(activation ='tanh', alpha=0.05, learning_rate='adaptive', solver='sgd')

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
