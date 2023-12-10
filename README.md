# Leaderboard Competition Creation using Natural Language Processing with disaster tweets dataset

#### Team: "HAAAT"
#### Members: *H*arish Venkata (hkv2), *A*brielle Agron (aa106), *A*ryan Gandhi (aryang6), *A*shwin Saxena (ashwins2), *T*ahir Bagasrawala (tahirib2)

<font size=5>[Demo Video Link](https://youtu.be/8bDu_KHNICA)</font>  <! --- update link and video presentation -->

## Project Description
*Leaderboard Competition* evaluates several generative and descriptive classifiers against [Kaggle’s Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/data) to predict which tweets are about real disasters and which are not.

This project falls under Theme: Leaderboard Competition.


## Project Structure and Source Code
*Leaderboard Competition* soure code consists of following:
- /docs: Documentation for final project presentation, and project progress status report
- /models: 1l models (a combination of discriminative and generative classifiers) were developed, tuned and tested against the dataset.
  1. AdaBoostClassifer.py: [AdaBoost classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases. 
  2. DecisionTreeClassifier.py: [Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation.
  3. GaussianNB.py: A [Gaussian Naive Bayes Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) has a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features given the value of the class variable.
  4. GradientBoostingClassifier.py: A [Gradient Boosting Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions.
  5. KNeighborsClassifier.py: A [K-Neighbors Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) implements the kth-nearest neighbor's vote.
  6. Kmeans.py: A [K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) model clusters data by trying to separate samples in n groups of equal variance, minimizing a criterion known as the inertia or within-cluster sum-of-squares.
  7. LatentDirichletAllocation.py: A [Latent Dirichlet Allocation (LDA)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html) model is a generative probabilistic model for collections of discrete dataset such as text corpora. It is also a topic model that is used for discovering abstract topics from a collection of documents.
  8. RandomForestClassifier.py: A [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) is a meta estimator that fits several decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
  9 mlp.py: A [Multi-Layer Perceptron (MLP)](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) optimizes the log-loss function using LBFGS or stochastic gradient descent.
  10. LogisticRegression.py: A [Logistic Regression (baseline)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) model (also known as logit or MaxEnt) is a popular for classification problems and is our baseline for this project.
  11. svc.py: A [C-Support Vector Classification](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) is a linear support vector classification model.
- model_eval.py: Contains the main function that evaluates these models and plots a Precision-Recall curve to compare the model outcomes.
- mlp_hpo.py: Contains a helper function to run hyper-parameter tuning (HPO) on the MLP Classifier model.
- sample_submission.csv, test.csv, train.csv is the evaluation, testing and training data set respectively. 


## Team Contributions
All five members worked together to research various models (e.g. Random Forest, MLP, Gaussian Naive Bayes, etc) and then implement then train these models against the data set. We then tuned the hyperparameters of the models to ensure we get the *best* fit. We then deployed the models to the leaderboard to compare which models performance against each other. We tested the code thouroughly to ensure performance.
Lastly, we planned and recorded the demo video together.

## Install and Usage
### To Start
1. Download this project (cs410_proj) and unzip the directory.
2. Ensure you have a version of Python 3.0+ installed on your computer.

### To Add new models
4. Open `model_eval.py`
5. Create a new model `<your_model_here>.py` in /models folder
6. Add a tuple for model file path and Model Class in the section of the code in `model_eval.py`

Here's an example:

```
#List of models to evaluate - each entry is a (file_path, class_name) tuple
    models_to_evaluate = [
        ('models/model1.py', 'LogisticModel'),
        ('models/RandomForestClassifier.py', 'RandomForestClassifierModel'), #TB
        ('models/AdaBoostClassifier.py', 'AdaBoostClassifierModel'), #TB
        ('models/DecisionTreeClassifier.py', 'DecisionTreeClassifierModel'), #TB
        ('models/KNeighborsClassifier.py', 'KNeighborsClassifierModel'), #TB

        ('models/Kmeans.py', 'KMeansModel'),
        ('models/LatentDirichletAllocation.py', 'LDAModel'),

        ('models/GaussianNB.py', 'GaussianNBModel'), #TB
        ('models/GradientBoostingClassifier.py', 'GradientBoostingClassifierModel'), #TB
        ('models/svc.py', 'SVCModel'), #TB
        ('models/mlp.py', 'MLPClassifierModel'), #TB

# Add more models here
```

### To Evaluate all models and view comparison of all models in a Precision-Recall plot
4. Open Terminal and run `model_eval.py`
   
Here's an example:

```
python3 model_eval.py
```
### To Train Hyperparameters for the LDA and K-Means Model



### To Train Hyperparameters for the MLP Model
1. Open `mlp_hpo.py`
2. Update `parameter_space` with the list of hyperparamters that would like to optimize in the model 

Here's an example:

```
#add parameters that you would like to optimize here
parameter_space = {
    'hidden_layer_sizes': [(10,30,10),(20,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.05, 0.01],
    'learning_rate': ['constant','adaptive'],
    }
```
3. Open Terminal and run `mlp_hpo.py`

Here's an example:

```
python3 mlp_hpo.py
``` 
5. Once you've confirmed your "best" parameters, update them in `mlp.py`
6. Repeat Steps 1-5 until you've got the best precision, recall, F1 score. 



