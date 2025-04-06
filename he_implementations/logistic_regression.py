import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression

import matplotlib.pyplot as plt
from IPython.display import display

from profiler import profile_block



### Generate Dataset

X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=2,
    n_clusters_per_class=1,
)

rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)

b_min = np.min(X, axis=0)
b_max = np.max(X, axis=0)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

x_test_grid, y_test_grid = np.meshgrid(
    np.linspace(b_min[0], b_max[0], 30), np.linspace(b_min[1], b_max[1], 30)
)
x_grid_test = np.vstack([x_test_grid.ravel(), y_test_grid.ravel()]).transpose()

sklearn_logr = SklearnLogisticRegression()
sklearn_logr.fit(x_train, y_train)

concrete_logr = ConcreteLogisticRegression(n_bits=8)
concrete_logr.fit(x_train, y_train)

_ = sklearn_logr.predict_proba(x_test)[:, 1]
_ = profile_block(sklearn_logr.predict, x_test, label="SKLearn Log Regression")

# Predict on the test set
y_proba_q = concrete_logr.predict_proba(x_test)[:, 1]
y_pred_q = profile_block(concrete_logr.predict, x_test, label="Non-FHE Log Regression")

# Compute the probabilities on the whole domain in order to be able to plot the contours
y_proba_q_grid = concrete_logr.predict_proba(x_grid_test)[:, 1]
y_pred_q_grid = concrete_logr.predict(x_grid_test)

fhe_circuit = concrete_logr.compile(x_train)

print(f"Generating a key for an {fhe_circuit.graph.maximum_integer_bit_width()}-bit circuit")

time_begin = time.time()
fhe_circuit.client.keygen(force=False)
print(f"Key generation time: {time.time() - time_begin:.4f} seconds")

time_begin = time.time()
y_pred_fhe = profile_block(concrete_logr.predict, x_test, fhe="execute", label="FHE Log Regression")
print(f"Execution time: {(time.time() - time_begin) / len(x_test):.4f} seconds per sample")