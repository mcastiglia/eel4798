### Acquire the dataset

import time

import numpy
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from profiler import profile_block

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

# print("before grid search")

# gs_results = grid_search.fit(x_train, y_train)
# print("Best hyper parameters:", gs_results.best_params_)
# print("Best score:", gs_results.best_score_)

# Build the model with best hyper parameters
# model = ConcreteDecisionTreeClassifier(
#     max_features=gs_results.best_params_["max_features"],
#     min_samples_leaf=gs_results.best_params_["min_samples_leaf"],
#     min_samples_split=gs_results.best_params_["min_samples_split"],
#     max_depth=gs_results.best_params_["max_depth"],
#     n_bits=6,
# )

model = ConcreteDecisionTreeClassifier(
    max_features=None,
    min_samples_leaf=10,
    min_samples_split=100,
    max_depth=None,
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
# print(f"Sklearn average precision score: {sklearn_average_precision:0.2f}")
# print(f"Concrete average precision score: {concrete_average_precision:0.2f}")

# Show the confusion matrix on x_test
from sklearn.metrics import confusion_matrix

y_pred = profile_block(model.predict, x_test, label="SKLearn Decision Tree")
true_negative, false_positive, false_negative, true_positive = confusion_matrix(
    y_test, y_pred, normalize="true"
).ravel()

num_samples = len(y_test)
num_spam = sum(y_test)

# print(f"Number of test samples: {num_samples}")
# print(f"Number of spams in test samples: {num_spam}")

# print(f"True Negative (legit mail well classified) rate: {true_negative}")
# print(f"False Positive (legit mail classified as spam) rate: {false_positive}")
# print(f"False Negative (spam mail classified as legit) rate: {false_negative}")
# print(f"True Positive (spam well classified) rate: {true_positive}")

from concrete.compiler import check_gpu_available

use_gpu_if_available = False
device = "cuda" if use_gpu_if_available and check_gpu_available() else "cpu"

# We first compile the model with some data, here the training set
circuit = model.compile(x_train, device=device)

# Generate a key for the circuit
# print(f"Generating a key for an {circuit.graph.maximum_integer_bit_width()}-bit circuit")

time_begin = time.time()
circuit.client.keygen(force=False)
# print(f"Key generation time: {time.time() - time_begin:.2f} seconds")

# Reduce the sample size for a faster total execution time
FHE_SAMPLES = 10
x_test = x_test[:FHE_SAMPLES]
y_pred = y_pred[:FHE_SAMPLES]
y_reference = y_test[:FHE_SAMPLES]

# Predict in FHE for a few examples
time_begin = time.time()
y_pred_fhe = profile_block(model.predict, x_test, fhe="execute", label="Concrete ML FHE Decision Tree")
# print(f"Execution time: {(time.time() - time_begin) / len(x_test):.2f} seconds per sample")

y_pred_fhe = profile_block(model.predict, x_test, label="Concrete ML Non-FHE Decision Tree")

# Check prediction FHE vs sklearn
# print(f"Ground truth:       {y_reference}")
# print(f"Prediction sklearn: {y_pred}")
# print(f"Prediction FHE:     {y_pred_fhe}")

# print(
#     f"{numpy.sum(y_pred_fhe == y_pred)}/"
#     "10 predictions are similar between the FHE model and the clear sklearn model."
# )