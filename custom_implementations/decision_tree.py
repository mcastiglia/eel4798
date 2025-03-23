### Acquire the dataset

import time

import numpy
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

features, classes = fetch_openml(data_id=44, as_frame=False, cache=True, return_X_y=True)
classes = classes.astype(numpy.int64)

x_train, x_test, y_train, y_test = train_test_split(
    features,
    classes,
    test_size=0.15,
    random_state=42,
)



### Find Best Hyperparameters with Cross-Validation Tool

from sklearn.model_selection import GridSearchCV

from concrete.ml.sklearn import DecisionTreeClassifier as ConcreteDecisionTreeClassifier

# List of hyper parameters to tune
param_grid = {
    "max_features": [None, "sqrt", "log2"],
    "min_samples_leaf": [1, 10, 100],
    "min_samples_split": [2, 10, 100],
    "max_depth": [None, 2, 4, 6, 8],
}

grid_search = GridSearchCV(
    ConcreteDecisionTreeClassifier(),
    param_grid,
    cv=10,
    scoring="average_precision",
    error_score="raise",
    n_jobs=1,
)

print("before grid search")

gs_results = grid_search.fit(x_train, y_train)
print("Best hyper parameters:", gs_results.best_params_)
print("Best score:", gs_results.best_score_)

# Build the model with best hyper parameters
model = ConcreteDecisionTreeClassifier(
    max_features=gs_results.best_params_["max_features"],
    min_samples_leaf=gs_results.best_params_["min_samples_leaf"],
    min_samples_split=gs_results.best_params_["min_samples_split"],
    max_depth=gs_results.best_params_["max_depth"],
    n_bits=6,
)



### Compute Test Set Metrics

model, sklearn_model = model.fit_benchmark(x_train, y_train)

# Compute average precision on test
from sklearn.metrics import average_precision_score

# pylint: disable=no-member
y_pred_concrete = model.predict_proba(x_test)[:, 1]
y_pred_sklearn = sklearn_model.predict_proba(x_test)[:, 1]
concrete_average_precision = average_precision_score(y_test, y_pred_concrete)
sklearn_average_precision = average_precision_score(y_test, y_pred_sklearn)
print(f"Sklearn average precision score: {sklearn_average_precision:0.2f}")
print(f"Concrete average precision score: {concrete_average_precision:0.2f}")

# Show the confusion matrix on x_test
from sklearn.metrics import confusion_matrix

y_pred = model.predict(x_test)
true_negative, false_positive, false_negative, true_positive = confusion_matrix(
    y_test, y_pred, normalize="true"
).ravel()

num_samples = len(y_test)
num_spam = sum(y_test)

print(f"Number of test samples: {num_samples}")
print(f"Number of spams in test samples: {num_spam}")

print(f"True Negative (legit mail well classified) rate: {true_negative}")
print(f"False Positive (legit mail classified as spam) rate: {false_positive}")
print(f"False Negative (spam mail classified as legit) rate: {false_negative}")
print(f"True Positive (spam well classified) rate: {true_positive}")



### "Now we are ready to go in the FHE domain"
# FHE stuff