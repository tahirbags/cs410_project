from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

class AdaBoostClassifierModel:
    def __init__(self):
        self.model = AdaBoostClassifier(n_estimators=500, max_features="sqrt", min_samples_split=20, criterion="gini", random_state=42), learning_rate=0.1, n_estimators=50)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
