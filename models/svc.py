from sklearn.svm import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class SVCModel: #Support Vector Classification
    def __init__(self):
        self.model = svm.SVC (gamma='auto')

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
