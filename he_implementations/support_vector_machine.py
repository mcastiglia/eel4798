# %matplotlib inline

# import numpy and matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC as SklearnLinearSVC

# import the concrete-ml LinearSVC implementation
from concrete.ml.sklearn.svm import LinearSVC as ConcreteLinearSVC

# Visualization function
def plot_decision_boundary(
    clf,
    X,
    y,
    title="LinearSVC Decision Boundary",
    xlabel="First Principal Component",
    ylabel="Second Principal Component",
):
    # Perform PCA to reduce the dimensionality to 2
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Create the mesh grid
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    # Transform the mesh grid points back to the original feature space
    mesh_points = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])

    # Make predictions using the classifier
    Z = clf.predict(mesh_points)
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    _, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.8)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors="k", marker="o", s=50)

    # Calculate the accuracy
    accuracy = accuracy_score(y, clf.predict(X))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title} (Accuracy: {accuracy:.4f})")
    plt.show()



### Import and Preprocess Dataset

df = pd.read_csv(
    "https://gist.githubusercontent.com/robinstraub/72f1cb27829dba85f49f68210979f561/"
    "raw/b9982ae654967028f6f4010bd235d850d38fe25b/pulsar-star-dataset.csv"
)
df.head()

# Extract the features and labels
X = df.drop(columns=["target_class"])
y = df["target_class"]

# Replace N/A values with the mean of the respective feature
X.fillna(X.mean(), inplace=True)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert the floating labels to integer labels for both train and test sets
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Perform the same steps with the Concrete ML LinearSVC implementation
svm_concrete = ConcreteLinearSVC(max_iter=100, n_bits=8)
svm_concrete.fit(X_train, y_train)
# plot the boundary
plot_decision_boundary(svm_concrete, X_test, y_test)

# A circuit needs to be compiled to enable FHE execution
circuit = svm_concrete.compile(X_train)
# Now that a circuit is compiled, the svm_concrete can predict value with FHE
y_pred = svm_concrete.predict(X_test, fhe="execute")
accuracy = accuracy_score(y_test, y_pred)
# print the accuracy
print(f"FHE Accuracy: {accuracy:.4f} (bit-width: {circuit.graph.maximum_integer_bit_width()})")