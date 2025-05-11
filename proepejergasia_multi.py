import numpy as np

# Load the preprocessed datasets from proepeksergasia_dl.py
train_ul = np.load("train_ul.npz")
test_ul = np.load("test_ul.npz")
train_dl = np.load("train_dl.npz")
test_dl = np.load("test_dl.npz")

# Extract sequences and labels for UL and DL
train_sequences_ul = train_ul["sequences"]  # Shape: (num_train_samples, sequence_length, 1)
train_labels_ul = train_ul["labels"]        # Shape: (num_train_samples,)
test_sequences_ul = test_ul["sequences"]    # Shape: (num_test_samples, sequence_length, 1)
test_labels_ul = test_ul["labels"]          # Shape: (num_test_samples,)

train_sequences_dl = train_dl["sequences"]  # Shape: (num_train_samples, sequence_length, 1)
train_labels_dl = train_dl["labels"]        # Shape: (num_train_samples,)
test_sequences_dl = test_dl["sequences"]    # Shape: (num_test_samples, sequence_length, 1)
test_labels_dl = test_dl["labels"]          # Shape: (num_test_samples,)

# Ensure the number of samples match by truncating the larger array
min_train_samples = min(train_sequences_ul.shape[0], train_sequences_dl.shape[0])
min_test_samples = min(test_sequences_ul.shape[0], test_sequences_dl.shape[0])

train_sequences_ul = train_sequences_ul[:min_train_samples]
train_labels_ul = train_labels_ul[:min_train_samples]
train_sequences_dl = train_sequences_dl[:min_train_samples]
train_labels_dl = train_labels_dl[:min_train_samples]

test_sequences_ul = test_sequences_ul[:min_test_samples]
test_labels_ul = test_labels_ul[:min_test_samples]
test_sequences_dl = test_sequences_dl[:min_test_samples]
test_labels_dl = test_labels_dl[:min_test_samples]

# Ensure the sequence lengths match by truncating the longer sequence
min_train_sequence_length = min(train_sequences_ul.shape[1], train_sequences_dl.shape[1])
min_test_sequence_length = min(test_sequences_ul.shape[1], test_sequences_dl.shape[1])

train_sequences_ul = train_sequences_ul[:, :min_train_sequence_length]
train_sequences_dl = train_sequences_dl[:, :min_train_sequence_length]

test_sequences_ul = test_sequences_ul[:, :min_test_sequence_length]
test_sequences_dl = test_sequences_dl[:, :min_test_sequence_length]

# Combine UL and DL sequences into a single multivariate dataset
train_sequences_multivariate = np.concatenate([train_sequences_ul, train_sequences_dl], axis=-1)  # Shape: (num_train_samples, sequence_length, 2)
test_sequences_multivariate = np.concatenate([test_sequences_ul, test_sequences_dl], axis=-1)    # Shape: (num_test_samples, sequence_length, 2)

# Combine UL and DL labels into a single multivariate label set
train_labels_multivariate = np.stack([train_labels_ul, train_labels_dl], axis=-1)  # Shape: (num_train_samples, 2)
test_labels_multivariate = np.stack([test_labels_ul, test_labels_dl], axis=-1)    # Shape: (num_test_samples, 2)

# Save the multivariate datasets
np.savez("train_multivariate.npz", sequences=train_sequences_multivariate, labels=train_labels_multivariate)
np.savez("test_multivariate.npz", sequences=test_sequences_multivariate, labels=test_labels_multivariate)

print("Multivariate datasets created and saved successfully!")
