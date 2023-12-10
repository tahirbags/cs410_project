from sklearn.linear_model import LogisticRegression
"""
   A class used to represent a Logistic Regression model. This is used as the baseline model to compare with more complex models


   Attributes
   ----------
   model :
       an instance of the LogisticRegression Class from sklearn

   Methods
   -------
   train(X_train, y_train)
       Trains the model using given training data
   predict(X_test)
       Predicts labels for given test data
   """

class LogisticModel:
    def __init__(self):
        """
         Hyperparameter selection:

         List of parameters that can be trained:  
         dict_keys(['C', 'class_weight', 'dual', 'fit_intercept', 'intercept_scaling', 'l1_ratio', 'max_iter', 'multi_class', 
                     'n_jobs', 'penalty', 'random_state', 'solver', 'tol', 'verbose', 'warm_start'])
         
        """
       
        self.model = LogisticRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
