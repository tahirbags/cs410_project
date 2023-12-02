from sklearn.decomposition import LatentDirichletAllocation

class LDAModel:
    #optimally chosen values to get perplexity score of ~62
    def __init__(self, n_components=5, learning_method='online', learning_decay=0.7, max_iter=10, doc_topic_prior=None, topic_word_prior=None):
        """
        Initialize the LDA model with specified parameters.
        :param n_components: int, the number of topics.
        :param learning_method: 'batch' or 'online', the method used for optimization.
        :param learning_decay: float, learning rate decay for 'online' learning method.
        :param max_iter: int, the maximum number of iterations.
        :param doc_topic_prior: float, prior of document topic distribution (alpha).
        :param topic_word_prior: float, prior of topic word distribution (beta).
        """
        self.model = LatentDirichletAllocation(n_components=n_components, learning_method=learning_method, 
                                               learning_decay=learning_decay, max_iter=max_iter, 
                                               doc_topic_prior=doc_topic_prior, topic_word_prior=topic_word_prior,
                                               random_state=42)

    def train(self, X_train):
        # Note that y_train is not used because LDA is unsupervised
        self.model.fit(X_train)

    def predict(self, X_test):
        # This method will return the topic distribution for each document in X_test
        return self.model.transform(X_test)
