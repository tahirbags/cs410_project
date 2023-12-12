from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

class AdaBoostClassifierModel:
    def __init__(self):
        self.model = AdaBoostClassifier(base_estimator=RandomForestClassifier(), learning_rate=0.1)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
