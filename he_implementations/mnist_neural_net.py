import time

import matplotlib.pyplot as plt
import numpy as np
from concrete.compiler import check_gpu_available
from joblib import Memory
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn

from concrete.ml.sklearn import NeuralNetClassifier

from profiler import profile_block

use_gpu_if_available = False
device = "cuda" if use_gpu_if_available and check_gpu_available() else "cpu"



### Load Data
# scikit-learn's fetch_openml method doesn't handle local cache:
# https://github.com/scikit-learn/scikit-learn/issues/18783#issuecomment-723471498
# This is a workaround that prevents downloading the data every time the notebook is ran
memory = Memory("./data/MNIST")
fetch_openml_cached = memory.cache(fetch_openml)

# Fetch the MNIST data-set, with inputs already flattened
mnist_dataset = fetch_openml_cached("mnist_784")

# Define max, mean and std values for the MNIST data-set
max_value = 255
mean = 0.1307
std = 0.3081

# Normalize the training data
data = (mnist_dataset.data) / max_value
data = ((data - mean) / std).round(decimals=4)

# Concrete ML's NNs do not support: category, str, object types
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2990
target = mnist_dataset.target.astype("int")

test_size = 10000
x_train, x_test, y_train, y_test = train_test_split(
    data, target, test_size=test_size, random_state=0
)

def plot_samples(data, targets, n_samples=5, title="Train target"):
    # MNIST images are originally of shape 28x28 with grayscale values
    samples_to_plot = np.array(data)[:n_samples].reshape((n_samples, 28, 28))

    fig = plt.figure(figsize=(30, 30))

    for i in range(n_samples):
        subplot = fig.add_subplot(1, n_samples, i + 1)
        subplot.set_title(f"{title}: {np.array(targets)[i]}", fontsize=15)
        subplot.imshow(samples_to_plot[i], cmap="gray", interpolation="nearest")

plot_samples(x_train, y_train)

params = {
    "module__n_layers": 2,
    "module__n_w_bits": 4,
    "module__n_a_bits": 4,
    "module__n_hidden_neurons_multiplier": 0.5,
    "module__activation_function": nn.ReLU,
    "max_epochs": 7,
}

model = NeuralNetClassifier(**params)

model.fit(X=x_train, y=y_train)

# Reduce the input-set's length to make the compilation time faster
# The input-set should be large enough to be representative of the input data
inputset = x_train.head(1000)
simulated_fhe_circuit = model.compile(inputset, device=device)

# Print the circuit's maximum bit-width reached during compilation
print(f"Circuit of {simulated_fhe_circuit.graph.maximum_integer_bit_width()}-bits (FHE simulation)")

# Evaluate the model using simulation
y_preds_simulated = profile_block(model.predict, x_test, label="Concrete Non-FHE Neural Net")

print(
    "The test accuracy (with FHE simulation) of the FHE model is "
    f"{accuracy_score(y_test, y_preds_simulated):.2f}"
)

# Print the circuit's maximum bit-width reached during compilation
print(f"FHE circuit of {model.fhe_circuit.graph.maximum_integer_bit_width()}-bits")

time_begin = time.time()
model.fhe_circuit.client.keygen(force=True)
print(f"Key generation time: {time.time() - time_begin:.2f} seconds")

# Reduce the test set
n_samples = 3
x_test_sample = x_test.head(n_samples)
y_test_sample = y_test.head(n_samples)

# Execute the predictions using FHE simulation on a few samples
simulated_fhe_predictions = model.predict(x_test_sample, fhe="simulate")

time_begin = time.time()
fhe_predictions = profile_block(model.predict, x_test_sample, fhe="execute", label="Concrete FHE Neural Net")
seconds_per_sample = (time.time() - time_begin) / len(x_test_sample)
print(f"Execution time in FHE: {seconds_per_sample:.2f} seconds per sample\n")

print("Expected values:", y_test_sample.tolist())
print("Simulated prediction values:", simulated_fhe_predictions)
print("FHE prediction values:", fhe_predictions)