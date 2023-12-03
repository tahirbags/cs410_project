from sklearn.naive_bayes import GaussianNB

class GaussianNBModel:
    def __init__(self):
        # Initialize the GaussianNB with the best parameter
        self.model = GaussianNB(var_smoothing=1e-9)

    def train(self, X_train, y_train):
        # GaussianNB requires dense input
        self.model.fit(X_train.todense(), y_train)

    def predict(self, X_test):
        # GaussianNB requires dense input for prediction as well
        return self.model.predict(X_test.todense())
