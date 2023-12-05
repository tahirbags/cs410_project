from sklearn.naive_bayes import GaussianNB

class GaussianNBModel:
    """
   A class used to represent a Gaussian Naive Bayes Classifier model

   Attributes
   ----------
   model :
       an instance of the GaussianNB Class from scikit-learn

   Methods
   -------
   train(X_train, y_train)
       Trains the model using given training data
   predict(X_test)
       Predicts labels for given test data
   """
    def __init__(self):
        self.model = GaussianNB()

    def train(self, X_train, y_train):
        self.model.fit(X_train.todense(), y_train)

    def predict(self, X_test):
        return self.model.predict(X_test.todense())
