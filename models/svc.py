from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class SVCModel:
    """
   A class used to represent a Support Vector Classification Model

   Attributes
   ----------
   model :
       an instance of the LinearSVC Class from sklearn

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

        List of parameters that can be trained:  
        dict_keys(['C', 'class_weight', 'dual', 'fit_intercept', 'intercept_scaling', 'loss', 'max_iter', 'multi_class', 
        'penalty', 'random_state', 'tol', 'verbose'])

        Best parameters found:
         {'fit_intercept': 1, 'loss': 'hinge', 'multi_class': 'ovr', 'penalty': 'l2'}
        """
        self.model = LinearSVC(max_iter=1000, loss='hinge', verbose = 0, C=1.0, dual=True)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
