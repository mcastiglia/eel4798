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

from profiler import profile_block


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

svm_sklearn = SklearnLinearSVC(max_iter=100)
svm_sklearn.fit(X_train, y_train)

# Perform the same steps with the Concrete ML LinearSVC implementation
svm_concrete = ConcreteLinearSVC(max_iter=100, n_bits=8)
svm_concrete.fit(X_train, y_train)

profile_block(svm_sklearn.predict, X_test, label="SKLearn SVM")

# A circuit needs to be compiled to enable FHE execution
circuit = svm_concrete.compile(X_train)
# Now that a circuit is compiled, the svm_concrete can predict value with FHE
y_pred_fhe = profile_block(svm_concrete.predict, X_test, label="Concrete Non-FHE SVM")
y_pred_fhe = profile_block(svm_concrete.predict, X_test, fhe="execute", label="Concrete FHE SVM")
accuracy = accuracy_score(y_test, y_pred_fhe)
# print the accuracy
print(f"FHE Accuracy: {accuracy:.4f} (bit-width: {circuit.graph.maximum_integer_bit_width()})")