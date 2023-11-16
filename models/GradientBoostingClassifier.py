from sklearn.ensemble import GradientBoostingClassifier

class GradientBoostingClassifierModel:
    def __init__(self):
        self.model.n_estimators = 100
        self.model.learning_rate = 1.0
        self.model.max_depth = 1
        self.model.random_state = 0
        self.model = GradientBoostingClassifierModel()

    def train(self, X_train, y_train):
        self.model.fit(X_train.todense(), y_train)

    def predict(self, X_test):
        return self.model.predict(X_test.todense())
