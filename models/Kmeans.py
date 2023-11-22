from sklearn.cluster import KMeans

class KMeansModel:
    def __init__(self, n_clusters=5):
        self.model = KMeans(n_clusters=n_clusters, random_state=42)

    def train(self, X_train, y_train=None):
        # Note that y_train is not used because KMeans is unsupervised
        self.model.fit(X_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
