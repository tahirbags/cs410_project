from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

'''
Hyperparameter selection:


n_neighbors=100:
    With a large and diverse dataset, using a higher number of neighbors can help in making more stable and generalizable predictions by considering more samples around each query point.
    A higher number of neighbors can smooth out the effects of noisy labels or outliers, leading to more robust predictions.

weights='uniform':
    Each neighbor contributes equally to the classification, regardless of their distance to the query point.
    Uniform weights are simpler and involve fewer assumptions about the data, which can be beneficial if there's no clear rationale for using distance-based weighting.
     
algorithm='auto':
    This setting allows scikit-learn to automatically choose the most efficient algorithm based on the dataset's structure, leading to potentially faster computations without needing to manually test each algorithm.

leaf_size=30:
    The default value is often a good balance between memory usage and query speed.
    
p=2, metric='minkowski':
    With p=2, the Minkowski metric becomes equivalent to the Euclidean metric, which is effective for many datasets, particularly those where features represent spatial or geometric properties.
    uclidean distance is the most commonly used distance metric for KNN, making it a good starting point.



'''

class KNeighborsClassifierModel:
    def __init__(self, n_neighbors=100, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski'):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
