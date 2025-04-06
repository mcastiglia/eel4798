### Import Libraries ###
import time
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from concrete.ml.sklearn import LinearRegression as ConcreteLinearRegression

from profiler import profile_block

#%matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import display

### Dataset Generation ###
# pylint: disable=unbalanced-tuple-unpacking
X, y = make_regression(
    n_samples=200, n_features=1, n_targets=1, bias=5.0, noise=30.0, random_state=42
)
# pylint: enable=unbalanced-tuple-unpacking

# We split the data-set into a training and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# We sort the test set for a better visualization
sorted_indexes = np.argsort(np.squeeze(X_test))
X_test = X_test[sorted_indexes, :]
y_test = y_test[sorted_indexes]

### Linear Regression Model from SkLearn ###

sklearn_lr = SklearnLinearRegression()
sklearn_lr.fit(X_train, y_train)
y_pred = profile_block(sklearn_lr.predict, X_test, label="SKLearn Linear Regression")

# Compute the R2 scores
sklearn_r2_score = r2_score(y_test, y_pred)

### "Linear Regression Model with Conrete ML"

# We quantize the inputs using 8-bits
concrete_lr = ConcreteLinearRegression(n_bits=8)

# We train the concrete linear regression model on clear data
concrete_lr.fit(X_train, y_train)

# We densify the space representation of the original X,
# to better visualize the resulting step function in the following figure
x_space = np.linspace(X_test.min(), X_test.max(), num=300)
x_space = x_space[:, np.newaxis]
y_pred_q_space = concrete_lr.predict(x_space)

# Now, we can test our Concrete ML model on the clear test data
y_pred_q = profile_block(concrete_lr.predict, X_test, label="Concrete ML Non-FHE Linear Regression")

# Compute the R2 scores
quantized_r2_score = r2_score(y_test, y_pred_q)

fhe_circuit = concrete_lr.compile(X_train)

# print(f"Generating a key for a {fhe_circuit.graph.maximum_integer_bit_width()}-bit circuit")

time_begin = time.time()
fhe_circuit.client.keygen(force=False)
# print(f"Key generation time: {time.time() - time_begin:.4f} seconds")

time_begin = time.time()
y_pred_fhe = profile_block(concrete_lr.predict, X_test, fhe="execute", label="Concrete ML FHE Linear Regression")
# print(f"Execution time: {(time.time() - time_begin) / len(X_test):.4f} seconds per sample")

# Measure the FHE R2 score
fhe_r2_score = r2_score(y_test, y_pred_fhe)

# print("R^2 scores:")
# print(f"scikit-learn (clear): {sklearn_r2_score:.4f}")
# print(f"Concrete ML (quantized): {quantized_r2_score:.4f}")
# print(f"Concrete ML (FHE): {fhe_r2_score:.4f}")

# Measure the error of the FHE quantized model with respect to the clear scikit-learn float model
concrete_score_difference = abs(fhe_r2_score - quantized_r2_score) * 100 / quantized_r2_score
# print(
#     "\nRelative score difference for Concrete ML (quantized clear) vs. Concrete ML (FHE):",
#     f"{concrete_score_difference:.2f}%",
# )

# Measure the error of the FHE quantized model with respect to the clear float model
score_difference = abs(fhe_r2_score - sklearn_r2_score) * 100 / sklearn_r2_score
# print(
#     "Relative score difference for scikit-learn (clear) vs. Concrete ML (FHE) scores:",
#     f"{score_difference:.2f}%",
# )