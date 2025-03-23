### Import Libraries ###
import time
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from concrete.ml.sklearn import LinearRegression as ConcreteLinearRegression

#%matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import display



### Visualization Tooling ###

train_plot_config = {"c": "black", "marker": "D", "s": 15, "label": "Train data"}
test_plot_config = {"c": "red", "marker": "x", "s": 15, "label": "Test data"}


def get_sklearn_plot_config(r2_score=None):
    label = "Scikit-Learn"
    if r2_score is not None:
        label += f", {'$R^2$'}={r2_score:.4f}"
    return {"c": "blue", "linewidth": 2.5, "label": label}


def get_concrete_plot_config(r2_score=None):
    label = "Concrete ML"
    if r2_score is not None:
        label += f", {'$R^2$'}={r2_score:.4f}"
    return {"c": "orange", "linewidth": 2.5, "label": label}



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
y_pred = sklearn_lr.predict(X_test)

# Compute the R2 scores
sklearn_r2_score = r2_score(y_test, y_pred)

# Visualize Outputs

plt.ioff()  # Turn off interactive mode
plt.clf()  # Clear any existing figures

fig, ax = plt.subplots(1, figsize=(10, 5))
fig.patch.set_facecolor("white")
ax.scatter(X_train, y_train, **train_plot_config)
ax.scatter(X_test, y_test, **test_plot_config)
ax.plot(X_test, y_pred, **get_sklearn_plot_config(sklearn_r2_score))
ax.legend()
plt.show()



### "Linear Regression Model with Conrete ML"
# FHE stuff

