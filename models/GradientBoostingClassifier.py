from sklearn.ensemble import GradientBoostingClassifier

class GradientBoostingClassifierModel:
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators = 100, learning_rate = 1.0, max_depth = 1, random_state = 0)
        #self.n_estimators = 100
        #self.learning_rate = 1.0
        #self.max_depth = 1
        #self.random_state = 0

    def train(self, X_train, y_train):
        self.model.fit(X_train.todense(), y_train)

    def predict(self, X_test):
        return self.model.predict(X_test.todense())
