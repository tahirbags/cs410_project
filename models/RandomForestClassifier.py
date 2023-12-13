from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier


class RandomForestClassifierModel:
    """
   A class used to represent a Random Forest Classifier Model

   Attributes
   ----------
   model :
       an instance of the RandomForestClassifier Class from sklearn
       
   Methods
   -------
   train(X_train, y_train)
       Trains the model using given training data
   predict(X_test)
       Predicts labels for given test data
   """

    def __init__(self):
        """
        Hyperparameter selection:
        Using only default random forest parameters, get F-measure of 0.7147, slightly lower than base level of logistic regression, 
        which has F-measure of .7325. 
        
        First, upped n_estimators to 500 and got an F-measure of 0.7179, a slight increase, but with a marked increase in runtime. 
        In order to aid in further hyperparameter tuning, dropped nn_estimators back down to 100.
        
        Next modified max_features to None, and this dropped F-measure to 0.6978, so tried log2 and got 0.7149, which was essentially 
        no change. Decided to keep default of sqrt.
        
        Next, increased min_samples_split to 10 and got 0.7253. Further tuning achieved a maximum F-measure of 0.7385 with 
        min_samples_split = 20.
        
        Finally, checked other loss criterion. For entropy, got 0.7173, which was a decrease, so decided to stick with default of gini. 
        
        Finally, re-increased n_estimators to 500, for final F-measure of 0.8080
        """
        self.model = RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_split=20, criterion="gini", random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
