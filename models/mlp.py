from sklearn.neural_network import MLPClassifier

class MLPClassifierModel:
    def __init__(self):
        self.model = MLPClassifier(alpha=1, max_iter=1000, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
