from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

class RandomForestClassifierModel:
    def __init__(self):
        self.model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
