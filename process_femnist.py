""" Script to process FederatedEMNIST dataset. """

import os
import json
import glob

import numpy as np
from scipy.io import savemat


DIGITS_ONLY = True
SMALL_MAT = False
NAME = "FederatedEMNIST" if DIGITS_ONLY else "FederatedEMNIST62"
DATA_PATH = os.path.join("data", NAME)
MAT_NAME = "emnist.mat" if DIGITS_ONLY else "emnist62.mat"
if SMALL_MAT:
    MAT_NAME = "small_" + MAT_NAME
MAX_CLASS = 9 if DIGITS_ONLY else 61
MIN_SAMPLES_PER_USER = 10
SMALL_CLIENTS = 36

total_train_inputs = []
total_train_labels = []
total_test_inputs = []
total_test_labels = []

# Process each split (train and test).
train_filenames = glob.glob(os.path.join(DATA_PATH, "train", "*.json"))
current_idx = 0
for train_filename in train_filenames:

    test_name = os.path.basename(train_filename).replace("train", "test")
    test_filename = os.path.join(DATA_PATH, "test", test_name)

    # Read in train and test data for group of clients.
    with open(train_filename, "r") as f:
        train_file_data = json.load(f)
    with open(test_filename, "r") as f:
        test_file_data = json.load(f)
    assert set(train_file_data["users"]) == set(train_file_data["user_data"].keys())
    assert set(test_file_data["users"]) == set(test_file_data["user_data"].keys())
    assert set(train_file_data["users"]) == set(test_file_data["users"])
    clients = list(train_file_data["users"])

    # Save each client's data in a separate file.
    for client in clients:

        # Collect client data.
        train_inputs = np.array(train_file_data["user_data"][client]["x"])
        train_labels = np.array(train_file_data["user_data"][client]["y"])
        test_inputs = np.array(test_file_data["user_data"][client]["x"])
        test_labels = np.array(test_file_data["user_data"][client]["y"])
        assert len(train_inputs.shape) == 2
        assert len(train_labels.shape) == 1
        assert len(test_inputs.shape) == 2
        assert len(test_labels.shape) == 1
        assert train_inputs.shape[0] == train_labels.shape[0]
        assert test_inputs.shape[0] == test_labels.shape[0]

        # Filter out alphabetical characters.
        train_valid = (train_labels <= MAX_CLASS).nonzero()[0]
        test_valid = (test_labels <= MAX_CLASS).nonzero()[0]
        train_inputs = train_inputs[train_valid]
        train_labels = train_labels[train_valid]
        test_inputs = test_inputs[test_valid]
        test_labels = test_labels[test_valid]
        if (
            len(train_valid) < MIN_SAMPLES_PER_USER
            or len(test_valid) < MIN_SAMPLES_PER_USER / 10
        ):
            continue

        # Save client data.
        train_x_name = os.path.join(DATA_PATH, "train", f"{current_idx}_train_x.npy")
        train_y_name = os.path.join(DATA_PATH, "train", f"{current_idx}_train_y.npy")
        test_x_name = os.path.join(DATA_PATH, "test", f"{current_idx}_test_x.npy")
        test_y_name = os.path.join(DATA_PATH, "test", f"{current_idx}_test_y.npy")
        np.save(train_x_name, train_inputs)
        np.save(train_y_name, train_labels)
        np.save(test_x_name, test_inputs)
        np.save(test_y_name, test_labels)

        if not SMALL_MAT or current_idx < SMALL_CLIENTS:
            total_train_inputs.append(train_inputs)
            total_train_labels.append(train_labels)
            total_test_inputs.append(test_inputs)
            total_test_labels.append(test_labels)

        current_idx += 1

train_sample_sizes = np.array([len(inputs) for inputs in total_train_inputs])
test_sample_sizes = np.array([len(inputs) for inputs in total_test_inputs])
total_train_inputs = np.concatenate(total_train_inputs, axis=0)
total_train_labels = np.concatenate(total_train_labels, axis=0)
total_test_inputs = np.concatenate(total_test_inputs, axis=0)
total_test_labels = np.concatenate(total_test_labels, axis=0)
total_train_inputs = np.transpose(total_train_inputs)
total_train_labels = np.expand_dims(total_train_labels, axis=0)
total_test_inputs = np.transpose(total_test_inputs)
total_test_labels = np.expand_dims(total_test_labels, axis=0)
mat_dict = {
    "X": total_train_inputs,
    "Y": total_train_labels,
    "testX": total_test_inputs,
    "testY": total_test_labels,
    "client_samples": train_sample_sizes,
    "test_client_samples": test_sample_sizes,
}
mat_path = os.path.join(DATA_PATH, MAT_NAME)
savemat(mat_path, mat_dict)

print(f"Total clients: {current_idx}")
