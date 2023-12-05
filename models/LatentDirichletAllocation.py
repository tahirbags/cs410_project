from sklearn.decomposition import LatentDirichletAllocation

class LDAModel:
    """
   A class used to represent a Latent Dirichlet Allocation (LDA) Model

   Attributes
   ----------
   model :
       an instance of the LatentDirichletAllocation Class from sklearn.decomposition

   Methods
   -------
   train(X_train, y_train=None)
       Trains the model using given training data, y_train is None as this is an unsupervised model
   predict(X_test)
       Predicts labels for given test data
   """

    def __init__(self, n_components=5, learning_method='online', learning_decay=0.7, max_iter=10, doc_topic_prior=None, topic_word_prior=None):
        """
        Hyperparameter selection:
        
        """
        self.model = LatentDirichletAllocation(n_components=n_components, learning_method=learning_method, 
                                               learning_decay=learning_decay, max_iter=max_iter, 
                                               doc_topic_prior=doc_topic_prior, topic_word_prior=topic_word_prior,
                                               random_state=42)

    def train(self, X_train):
        self.model.fit(X_train)

    def predict(self, X_test):
        return self.model.transform(X_test)
