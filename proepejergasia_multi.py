import numpy as np

# Load the preprocessed datasets from proepeksergasia_dl.py
train_ul = np.load("train_ul.npz")
val_ul = np.load("val_ul.npz")
test_ul = np.load("test_ul.npz")
train_dl = np.load("train_dl.npz")
val_dl = np.load("val_dl.npz")
test_dl = np.load("test_dl.npz")

# Extract sequences and labels for UL and DL
train_sequences_ul = train_ul["sequences"]  # (num_train_samples, seq_len, 1)
train_labels_ul = train_ul["labels"]        # (num_train_samples,)
val_sequences_ul = val_ul["sequences"]
val_labels_ul = val_ul["labels"]
test_sequences_ul = test_ul["sequences"]
test_labels_ul = test_ul["labels"]

train_sequences_dl = train_dl["sequences"]
train_labels_dl = train_dl["labels"]
val_sequences_dl = val_dl["sequences"]
val_labels_dl = val_dl["labels"]
test_sequences_dl = test_dl["sequences"]
test_labels_dl = test_dl["labels"]

# Ensure the number of samples match by truncating the larger array for each split
min_train_samples = min(train_sequences_ul.shape[0], train_sequences_dl.shape[0])
min_val_samples = min(val_sequences_ul.shape[0], val_sequences_dl.shape[0])
min_test_samples = min(test_sequences_ul.shape[0], test_sequences_dl.shape[0])

train_sequences_ul = train_sequences_ul[:min_train_samples]
train_labels_ul = train_labels_ul[:min_train_samples]
train_sequences_dl = train_sequences_dl[:min_train_samples]
train_labels_dl = train_labels_dl[:min_train_samples]

val_sequences_ul = val_sequences_ul[:min_val_samples]
val_labels_ul = val_labels_ul[:min_val_samples]
val_sequences_dl = val_sequences_dl[:min_val_samples]
val_labels_dl = val_labels_dl[:min_val_samples]

test_sequences_ul = test_sequences_ul[:min_test_samples]
test_labels_ul = test_labels_ul[:min_test_samples]
test_sequences_dl = test_sequences_dl[:min_test_samples]
test_labels_dl = test_labels_dl[:min_test_samples]

# Ensure the sequence lengths match by truncating the longer sequence for each split
min_train_seq_len = min(train_sequences_ul.shape[1], train_sequences_dl.shape[1])
min_val_seq_len = min(val_sequences_ul.shape[1], val_sequences_dl.shape[1])
min_test_seq_len = min(test_sequences_ul.shape[1], test_sequences_dl.shape[1])

train_sequences_ul = train_sequences_ul[:, :min_train_seq_len]
train_sequences_dl = train_sequences_dl[:, :min_train_seq_len]
val_sequences_ul = val_sequences_ul[:, :min_val_seq_len]
val_sequences_dl = val_sequences_dl[:, :min_val_seq_len]
test_sequences_ul = test_sequences_ul[:, :min_test_seq_len]
test_sequences_dl = test_sequences_dl[:, :min_test_seq_len]

# Combine UL and DL sequences into a single multivariate dataset for each split
train_sequences_multi = np.concatenate([train_sequences_ul, train_sequences_dl], axis=-1)  # (samples, seq_len, 2)
val_sequences_multi = np.concatenate([val_sequences_ul, val_sequences_dl], axis=-1)
test_sequences_multi = np.concatenate([test_sequences_ul, test_sequences_dl], axis=-1)

# Combine UL and DL labels into a single multivariate label set for each split
train_labels_multi = np.stack([train_labels_ul, train_labels_dl], axis=-1)  # (samples, 2)
val_labels_multi = np.stack([val_labels_ul, val_labels_dl], axis=-1)
test_labels_multi = np.stack([test_labels_ul, test_labels_dl], axis=-1)

# Save the multivariate datasets for each split
np.savez("train_multivariate.npz", sequences=train_sequences_multi, labels=train_labels_multi)
np.savez("val_multivariate.npz", sequences=val_sequences_multi, labels=val_labels_multi)
np.savez("test_multivariate.npz", sequences=test_sequences_multi, labels=test_labels_multi)

print("Multivariate train, validation, and test datasets created and saved successfully!")