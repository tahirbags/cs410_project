from sklearn.cluster import KMeans

class KMeansModel:
  """
  A class used to represent a K-Means Clustering Model

  Attributes
  ----------
  model :
      an instance of the KMeans Class from sklearn.cluster

  Methods
  -------
  train(X_train, y_train=None)
      Trains the model using given training data, y_train is None as KMeans is an unsupervised model
  predict(X_test)
      Predicts labels for given test data
  """

  def __init__(self, n_clusters=4, init='k-means++', n_init=10, max_iter=300, tol=0.0001):
    """
    Hyperparameter selection:
    Tuning was done by selecting the values from the below options which maximized the Silhouette score (0.77)
        Number of Clusters (n_clusters): (2 to 11)
        Initialization Method (init): (kmeans++ or random)
        Number of Initialization Runs (n_init): (5,10,15 or 20))
        Maximum Iterations (max_iter): (200,300,400)
        Tolerance (tol): (0.0001, 0.0005, 0.001)
    """
    self.model = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol, random_state=42)

  def train(self, X_train, y_train=None):
    self.model.fit(X_train)

  def predict(self, X_test):
    return self.model.predict(X_test)
