from sklearn.cluster import KMeans

"""
There are the main aspects I played around with: 

Number of Clusters (n_clusters): (2 to 11)
Initialization Method (init): (kmeans++ or random)
Number of Initialization Runs (n_init): (5,10,15 or 20))
Maximum Iterations (max_iter): (200,300,400)
Tolerance (tol): (0.0001, 0.0005, 0.001)
"""

class KMeansModel:
    #optimally chosen values to get Silhouette score of ~0.77
    def __init__(self, n_clusters=4, init='k-means++', n_init=10, max_iter=300, tol=0.0001):
        """
        Initialize the KMeans model with specified parameters.
        :param n_clusters: int, the number of clusters to form.
        :param init: method for initialization, defaults to 'k-means++'.
        :param n_init: number of time the k-means algorithm will be run with different centroid seeds.
        :param max_iter: maximum number of iterations of the k-means algorithm for a single run.
        :param tol: relative tolerance with regards to inertia to declare convergence.
        """
        self.model = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol, random_state=42)

    def train(self, X_train, y_train=None):
        # Note that y_train is not used because KMeans is unsupervised
        self.model.fit(X_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
