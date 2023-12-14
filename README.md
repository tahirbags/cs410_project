# Leaderboard Competition Creation using Natural Language Processing with disaster tweets dataset

#### Team: "HAAAT" for CS410 Group Project
#### Members: *H*arish Venkata (hkv2), *A*brielle Agron (aa106), *A*ryan Gandhi (aryang6), *A*shwin Saxena (ashwins2), *T*ahir Bagasrawala (tahirib2)

<font size=5>[README file](https://github.com/nilakrishnan/CS_410_Final_Project/blob/main/README.md#project-description)</font> inspired by this previous project.

<font size=5>[Demo Video Link](https://github.com/tahirbags/cs410_project_public/blob/main/docs/CS410%20Final%20Demo.mov)</font>

## Project Description
*Leaderboard Competition* evaluates several generative and descriptive classifiers against [Kaggle’s Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/data) to predict which tweets are about real disasters and which are not.

This project falls under Theme: Leaderboard Competition.


## Project Structure and Source Code
*Leaderboard Competition* soure code consists of following:
- /docs: Documentation for final project presentation, and project progress status report
- /models: 11 total models (incl. baseline), a combination of discriminative and generative classifiers, were developed, tuned and tested against the dataset.
  1. **AdaBoostClassifer.py**: [AdaBoost classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases. 
  2. **DecisionTreeClassifier.py**: [Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation.
  3. **GaussianNB.py**: A [Gaussian Naive Bayes Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) has a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features given the value of the class variable.
  4. **GradientBoostingClassifier.py**: A [Gradient Boosting Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions.
  5. **KNeighborsClassifier.py**: A [K-Neighbors Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) implements the kth-nearest neighbor's vote.
  6. **Kmeans.py**: A [K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) model clusters data by trying to separate samples in n groups of equal variance, minimizing a criterion known as the inertia or within-cluster sum-of-squares.
  7. **LatentDirichletAllocation.py**: A [Latent Dirichlet Allocation (LDA)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html) model is a generative probabilistic model for collections of discrete dataset such as text corpora. It is also a topic model that is used for discovering abstract topics from a collection of documents.
  8. **RandomForestClassifier.py**: A [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) is a meta estimator that fits several decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
  9. **mlp.py**: A [Multi-Layer Perceptron (MLP)](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) optimizes the log-loss function using LBFGS or stochastic gradient descent.
  10. **model1.py**: A [Logistic Regression (baseline)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) model (also known as logit or MaxEnt) is a popular for classification problems and is our **baseline model** for this project.
  11. **svc.py**: A [C-Support Vector Classification](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) is a linear support vector classification model.
- model_eval.py: Contains the function that evaluates all 11 models and plots a Precision-Recall curve to compare model results.
- hpo_tune.py: Is an interactive helper function to run hyper-parameter tuning (HPO) for any model based on user selection.
- sample_submission.csv, test.csv, train.csv is the evaluation, testing and training data set respectively. 


## Team Contributions
All five members worked together to research various models (e.g. Random Forest, MLP, Gaussian Naive Bayes, etc) and then implement then train these models against the data set. We then tuned the hyperparameters of the models to ensure we get the *best* fit. We then deployed the models to the leaderboard to compare which models performance against each other. We tested the code thouroughly to ensure performance.
Lastly, we planned and recorded the demo video together.

## Install and Usage
### 1. To Start
- Download this project (cs410_proj) and unzip the directory.
- Download Python 3.6+. **Note: We faced some compatibility issues with Python 3.11.x. Please try switching Python 3.6.x if you are having trouble.**

- Create (and activate) a virtual env 
```
python -m venv myvenv
source myvenv/bin/activate
```
- Install scikit learn, pandas, matplotlib
```
pip install scikit-learn
pip install pandas
pip install matplotlib
```

### 2. To add new NLP Models
- Open `model_eval.py`
- Create a new model `<your_model_here>.py` in /models folder
- Add a tuple for model file path and Model Class in the section of the code in `model_eval.py`

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

### 3. To optimize Hyperparameters (HPO) for all NLP Models

- Open `hpo_tune.py`
- Choose a model that you want to tune

```
  # Pick a model you want to tune (Example of MLP Classifer below)
  mlp_model = MLPClassifier(max_iter=10000)
```
  
- Choose parameters for the model to update. The list of parameters for that model can be idenfied via the following:

```
  # List of parameters that can be trained for this model selection
    print("List of parameters that can be trained: ", mlp_model.get_params().keys())
```
                 
- Update `parameter_space` with the list of hyperparamters that would like to optimize in the model 

```
#Example parameters for MLP Classifier model
parameter_space = {
    'hidden_layer_sizes': [(10,30,10),(20,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.05, 0.01],
    'learning_rate': ['constant','adaptive'],
    }
```

- Once you've updated parameters for the model you'd like to tune, Open Terminal and run `hpo_tune.py`
- **ATTENTION: HPO Tuning can take up to 10 - 15 mins to complete, depending on your computer processor speed and how many parameters you select for tuning.**

```
python3 hpo_tune.py
```
- Choose a model that you would like to tune

```
[CASE sensitive] Choose one model to tune:
    [MLP]
    [AdaBoost]
    [DecisionTree]
    [NaiveBayes]
    [Logistic]
    [GradientBoosting]
    [KNeighbors]
    [svc]
    [RandomForest]
Which model would like to tune?
```
- The model tuning will begin (usually takes a few minutes to run and complete)
- Once you've confirmed your "best" parameters, update them in `mlp.py`
Here's an example:

```
#Example results for best parameters for MLP Classifier model
Best parameters found:
 {'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (20,), 'learning_rate': 'constant', 'solver': 'sgd'}
``` 

### 4. To evaluate and compare all NLP Models in a Precision-Recall plot
- Open Terminal and run `model_eval.py`
   
Here's an example:

```
python3 model_eval.py
```

- Run the model with the new parameters against the test dataset and compare precision, recall, F1 scores.
- Repeat Steps 1-4 until you've got the best precision, recall, F1 score. 
