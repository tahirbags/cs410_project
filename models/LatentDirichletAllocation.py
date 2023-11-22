from sklearn.decomposition import LatentDirichletAllocation

class LDAModel:
    def __init__(self, n_components=5):
        self.model = LatentDirichletAllocation(n_components=n_components, random_state=42)

    def train(self, X_train, y_train=None):
        # Note that y_train is not used because LDA is unsupervised
        self.model.fit(X_train)

    def predict(self, X_test):
        # This method will return the topic distribution for each document
        return self.model.transform(X_test)
