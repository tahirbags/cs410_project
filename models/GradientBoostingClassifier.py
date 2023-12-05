from sklearn.ensemble import GradientBoostingClassifier

class GradientBoostingClassifierModel:
    """
   A class used to represent a Gradient Boosted Classfier Model

   Attributes
   ----------
   model :
       an instance of the GradientBoostingClassifier Class from sklearn

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
        """
        self.model = GradientBoostingClassifier(n_estimators = 100, learning_rate = 1.0, max_depth = 1, random_state = 0)
        #self.n_estimators = 100
        #self.learning_rate = 1.0
        #self.max_depth = 1
        #self.random_state = 0

    def train(self, X_train, y_train):
        self.model.fit(X_train.todense(), y_train)

    def predict(self, X_test):
        return self.model.predict(X_test.todense())
