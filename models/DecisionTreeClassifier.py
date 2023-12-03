from sklearn.tree import DecisionTreeClassifier

class DecisionTreeClassifierModel:
    def __init__(self):
        # Initialize the DecisionTreeClassifier with the best parameters
        self.model = DecisionTreeClassifier(criterion='entropy', max_depth=10,
                                            min_samples_split=5, min_samples_leaf=2, 
                                            random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)