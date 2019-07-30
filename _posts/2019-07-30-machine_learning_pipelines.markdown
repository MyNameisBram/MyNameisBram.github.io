---
layout: post
title:      "Machine learning pipelines "
date:       2019-07-30 15:24:10 +0000
permalink:  machine_learning_pipelines
---


Below we will introduce the Pipeline class, a tool to chain together processing steps in a machine learning workflow. Real world applications of machine learning normally consists of sequential processing steps. Pipelines allows us to string together multiple steps into a single Python object that is attached to scikit-learn's fit, predict, and transform. When doing model evaluation using cross-validation and parameter tuning using grid search, the Pipeline class captures all the processing steps for proper evaluation, condenses the code, and reduces likelihood of making mistakes. Building a machine learning model can lead to confusion and disorganization due to its nature of multiple combinations and steps. Pipelines can help elimate overcomplication and simplify our workflow.  

```
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
%matplotlib inline 
import mglearn.plots

# load dataset 
cancer = load_breast_cancer()
```

```
# load and split data
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=2)

# scale using min/max scaling 
scaler = MinMaxScaler().fit(X_train)
``` 

```
#rescale training data and transforming it 
X_train_scaled = scaler.transform(X_train)

svm = SVC()
# learn an SVM on the scaled training data
svm.fit(X_train_scaled, y_train)
# scale the test data and score the scaled data
X_test_scaled = scaler.transform(X_test)
print("Test score: {:.2f}".format(svm.score(X_test_scaled, y_test)))

# output :
Test score: 0.96
```

## Parameter selection 

Let's say we want to find the best parameter for our Support Machine Vector (SVM). An initial approach might look like this:
```
from sklearn.model_selection import GridSearchCV
# for illustration purposes only, don't use this code! 
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_)) 
print("Best set score: {:.2f}".format(grid.score(X_test_scaled, y_test))) 
print("Best parameters: ", grid.best_params_)

# output: 
Best cross-validation accuracy: 0.98
Best set score: 0.98
Best parameters:  {'C': 10, 'gamma': 0.1}
```


## Building Pipelines


Let's build a pipeline workflow for training an SVM after scaling the data with `MinMaxScaler` and without Gridsearch yet. 

Here we created two steps: the first called "scaler", is an instance of MinMaxScaler and the second "svm", an instance of SVC. And we can fit the pipeline as usual after.

``` 
from sklearn.pipeline import Pipeline
pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])

pipe.fit(X_train, y_train)
```

```
# evaluate the test data using pipe.score
print("Test score: {:.2f}".format(pipe.score(X_test, y_test))) 

# output: 
Test score: 0.96
```

Using the pipeline, we reduced the code needed for our “preprocessing + classification” process. The main benefit of using the pipeline, however, is that we can now use this single estimator in cross_val_score or GridSearchCV. 

Pipeline can be used the same way with anyother estimator. 
- Define a parameter grid to search over 
- Construct a `GridSearchCV` from pipeline 
- Specify for each parameter which step of the pipeline it belongs to

```
param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
									
									
								
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_)) 

# output: 
Best cross-validation accuracy: 0.98
Test set score: 0.98
Best parameters: {'svm__C': 10, 'svm__gamma': 0.1}
print("Test set score: {:.2f}".format(grid.score(X_test, y_test))) 
print("Best parameters: {}".format(grid.best_params_))
```

## The General Pipeline Interface

The Pipeline class is not restricted to preprocessing and classification, but can in fact join any number of estimators together. For example, you could build a pipeline containing feature extraction, feature selection, scaling, and classification, for a total of four steps. Similarly, the last step could be regression or clustering instead of classification.
The only **requirement** for estimators in a pipeline is that all but the last step **need to have a transform method**, so they can produce a new representation of the data that can be used in the next step.

Internally, during the call to Pipeline.fit, the pipeline calls fit and then transform on each step in turn,2 with the input given by the output of the transform method of the previous step. For the last step in the pipeline, just fit is called.
Brushing over some finer details, this is implemented as follows. Remember that pipeline.steps is a list of tuples, so pipeline.steps[0][1] is the first estimator, pipe line.steps[1][1] is the second estimator, and so on:

```
def fit(self, X, y):
    X_transformed = X
    for name, estimator in self.steps[:-1]:
        # iterate over all but the final step
        # fit and transform the data
        X_transformed = estimator.fit_transform(X_transformed, y)
        # fit the last step 
    return self.steps[-1][1].fit(X_transformed, y) 

'''When predicting using pipeline, we transform the data using all BUT the last step, and the call predict on the last step'''


def predict(self, X):
    X_transformed = X
    for step in self.steps[:-1]:
        # iterate over all but the final step
        # transform the data
        X_transformed = step[1].transform(X_transformed)
        # fit the last step
    return self.steps[-1][1].predict(X_transformed)
```


## Convenient pipeline creation with make_pipeline 

```
from sklearn.pipeline import make_pipeline
# standard syntax
pipe_long = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC(C=100))]) # abbreviated syntax
pipe_short = make_pipeline(MinMaxScaler(), SVC(C=100))
```

`pipe_long` and `pipe_short` does the same thing, but `make_pipeline` will automatically name each step based on its class. 


```
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA

pipe = make_pipeline(StandardScaler(), PCA(n_components=2), StandardScaler()) 
print("Pipeline steps:\n{}".format(pipe.steps))


''' The easiest way to access the steps in a pipeline is via the named_steps attribute, 
which is a dictionary from the step names to the estimators'''

# fit the pipeline defined before to the cancer dataset
pipe.fit(cancer.data)
# extract the first two principal components from the "pca" step 
components = pipe.named_steps["pca"].components_ 
print("components.shape: {}".format(components.shape))

# output: 
components.shape: (2, 30)
```

## Grid-searching Preprocessing 

We can combine all the processing steps in our machine learning workflow in a single scikit-learn estimator. We can also adjust the parametes of the preprocessing using the result of a supervised machine learning like regression or classification. We're going to use the boston data to model the pipeline.

```
from sklearn.datasets import load_boston
boston = load_boston()

X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=42) 

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
pipe = make_pipeline(
    StandardScaler(), 
    PolynomialFeatures(),
    Ridge())
```

How do we know what inputs we should use for our pipeline? i.e. which degree of polynomials to choose? Ideally, we'd like to pick our parameter based on the outcome of our model. We'll use our pipeline and create a parameter grid for both. We'll define a param_grid that contains both appropriately prefixed by the step names:

```
param_grid = {'polynomialfeatures__degree': [1, 2, 3],
              'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}


grid = GridSearchCV(pipe, param_grid=param_grid, cv= 5, n_jobs=-1)
grid.fit(X_train, y_train)

# output: 
GridSearchCV(cv=5, error_score='raise-deprecating',
       estimator=Pipeline(memory=None,
     steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('polynomialfeatures', PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)), ('ridge', Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
   normalize=False, random_state=None, solver='auto', tol=0.001))]),
       fit_params=None, iid='warn', n_jobs=-1,
       param_grid={'polynomialfeatures__degree': [1, 2, 3], 'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring=None, verbose=0)
```


We cancheck our best parameters using `best_params_` to check our best parameters. Which resulted in  polynomials of `degree=2` and `alpha=10` being the best parameters. 

```
grid.best_params_

# output: 
{'polynomialfeatures__degree': 2, 'ridge__alpha': 10}
```

Let's run a grid search without polynomial features for comparison. 

```
param_grid = {'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
pipe = make_pipeline(StandardScaler(), Ridge())
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
print("Score without poly features: {:.2f}".format(grid.score(X_test, y_test)))

# output: 
Score without poly features: 0.68
```

Searching over pre-processing parameters together with model parameters is very powerful, but keep in mind the more parameters you add to GridSearchCV your grid exponentially increases the number of models is creates. Therefore increasing processing time.

## Using Grid Search to find best model 

We can use `GridSearchCV` and `Pipeline` to compare which models out performs the other.

```
pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])
```

```
from sklearn.ensemble import RandomForestClassifier

param_grid = [
      {'classifier': [SVC()], 'preprocessing': [StandardScaler(), None],
         'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
         'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
    
        {'classifier': [RandomForestClassifier(n_estimators=100)],
        'preprocessing': [None], 'classifier__max_features': [1, 2, 3]}]

# now we can instantiate and run the grid search

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best params:\n{}\n".format(grid.best_params_))
print("Best cross-validation score: {:.2f}".format(grid.best_score_)) 
print("Test-set score: {:.2f}".format(grid.score(X_test, y_test)))

# output: 
Best params:
{'classifier': SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False), 'classifier__C': 10, 'classifier__gamma': 0.01, 'preprocessing': StandardScaler(copy=True, with_mean=True, with_std=True)}

Best cross-validation score: 0.99
Test-set score: 0.98
```

### Observation: 

The result shows that Support Machine Vector (SVM) classifier or `SVC` with `C=10` , `gamma=0.01` gave the best results. 
