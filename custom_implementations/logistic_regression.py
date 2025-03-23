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



### Train and Predict with Scikit-learn

sklearn_logr = SklearnLogisticRegression()
sklearn_logr.fit(x_train, y_train)
y_pred_test = sklearn_logr.predict(x_test)



### Visualize Output

y_score_grid = sklearn_logr.predict_proba(x_grid_test)[:, 1]

plt.ioff()
plt.clf()
fig, ax = plt.subplots(1, figsize=(12, 8))
fig.patch.set_facecolor("white")
ax.contourf(x_test_grid, y_test_grid, y_score_grid.reshape(x_test_grid.shape), cmap="coolwarm")
CS1 = ax.contour(
    x_test_grid,
    y_test_grid,
    y_score_grid.reshape(x_test_grid.shape),
    levels=[0.5],
    linewidths=2,
)
# CS1.layers[0].set_label("Sklearn decision boundary")
ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, marker="D", cmap="jet", label="Train data")
ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, marker="x", cmap="jet", label="Test data")
ax.legend(loc="upper right")
plt.show()



### "Train using Concrete ML"
# FHE Stuff