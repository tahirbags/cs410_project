from sklearn.naive_bayes import GaussianNB

class GaussianNBModel:
    def __init__(self):
        self.model = GaussianNB()

    def train(self, X_train, y_train):
        self.model.fit(X_train.todense(), y_train)

    def predict(self, X_test):
        return self.model.predict(X_test.todense())
