from sklearn.neighbors import KNeighborsClassifier

class KNeighborsClassifierModel:
    def __init__(self):
        self.model = KNeighborsClassifier()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
