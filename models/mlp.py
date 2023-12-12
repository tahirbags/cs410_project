from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

class MLPClassifierModel:
    """
   A class used to represent a Multi-layer Perceptron Classifier Model

   Attributes
   ----------
   model :
       an instance of the MLPClassifier Class from sklearn

   Methods
   -------
   train(X_train, y_train)
       Trains the model using given training data
   predict(X_test)
       Predicts labels for given test data
   """

    def __init__(self):
        """
        Hyperparameter selection:
        
        1. Run mlp_hpo.py
        
        2. Using GridSearchCV to find the optimal hyperparameters
        
        3. Set parameter_space to list hyperparameters to tune
        
        4. n_jobs -1 is for using all the CPU cores available, cv=5 is for cross validation, here it means 5-folds Stratified K-fold cross validation
        
        5. Print best parameters
        
        6. Update best parameters in mlp.py
        
        7. Test the "best" model against the test data set
        
        8. Re-run steps 1 - 7 as needed

        Best parameters found:
         {'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (20,), 'learning_rate': 'constant', 'solver': 'sgd'}

        
        """
        self.model = MLPClassifier(max_iter = 1000, activation = 'relu', solver='adam', learning_rate='constant', alpha=0.05, hidden_layer_sizes=(1,))

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
