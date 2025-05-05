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

# Combine UL and DL sequences into a single multivariate dataset
train_sequences_multivariate = np.concatenate([train_sequences_ul, train_sequences_dl], axis=-1)  # Shape: (num_train_samples, sequence_length, 2)
test_sequences_multivariate = np.concatenate([test_sequences_ul, test_sequences_dl], axis=-1)    # Shape: (num_test_samples, sequence_length, 2)

# Combine UL and DL labels into a single multivariate label set
train_labels_multivariate = np.stack([train_labels_ul, train_labels_dl], axis=-1)  # Shape: (num_train_samples, 2)
test_labels_multivariate = np.stack([test_labels_ul, test_labels_dl], axis=-1)    # Shape: (num_test_samples, 2)

#test _labels_multivariate = np.stack([test_labels_ul, test_labels_dl], axis=-1)    # Shape: (num_test_samples, 2)
# Example: Add lagged features and moving averages as additional features
def add_features(sequences):
    lag_1 = np.roll(sequences, shift=1, axis=1)  # Lagged by 1 time step
    lag_1[:, 0] = 0  # Replace the first value with 0 (no lag available)
    moving_avg = np.convolve(sequences.flatten(), np.ones(3)/3, mode='same')  # 3-step moving average
    moving_avg = moving_avg.reshape(sequences.shape)
    return np.stack([sequences, lag_1, moving_avg], axis=-1)  # Shape: (num_samples, sequence_length, num_features)

# Save the multivariate datasets
np.savez("train_multivariate.npz", sequences=train_sequences_multivariate, labels=train_labels_multivariate)
np.savez("test_multivariate.npz", sequences=test_sequences_multivariate, labels=test_labels_multivariate)

print("Multivariate datasets created and saved successfully!")
# import numpy as np

# # Load the original dataset
# data_ul = np.load("train_ul.npz")
# data_dl = np.load("train_dl.npz")

# sequences_ul = data_ul["sequences"]  # Shape: (num_samples, sequence_length)
# labels_ul = data_ul["labels"]        # Shape: (num_samples,)

# # sequences_dl = data_dl["sequences"]  # Shape: (num_samples, seqence_length)
# labels_dl = data_dl["labels"]        # Shape: (num_samples,)

# # Example: Add lagged features and moving averages as additional features
# def add_features(sequences):
#     lag_1 = np.roll(sequences, shift=1, axis=1)  # Lagged by 1 time step
#     lag_1[:, 0] = 0  # Replace the first value with 0 (no lag available)

#     moving_avg = np.convolve(sequences.flatten(), np.ones(3)/3, mode='same')  # 3-step moving average
#     moving_avg = moving_avg.reshape(sequences.shape)

#     # Stack the original sequence, lagged feature, and moving average
#     return np.stack([sequences, lag_1, moving_avg], axis=-1)  # Shape: (num_samples, sequence_length, num_features)

# # Add features to UL and DL sequences
# sequences_ul_multivariate = add_features(sequences_ul)  # Shape: (num_samples, sequence_length, num_features)
# sequences_dl_multivariate = add_features(sequences_dl)  # Shape: (num_samples, sequence_length, num_features)

# # Reshape the sequences to ensure they are 3D (batch_size, sequence_length, feature_size)
# sequences_ul_multivariate = sequences_ul_multivariate.reshape(
#     sequences_ul_multivariate.shape[0], sequences_ul_multivariate.shape[1], -1
# )
# sequences_dl_multivariate = sequences_dl_multivariate.reshape(
#     sequences_dl_multivariate.shape[0], sequences_dl_multivariate.shape[1], -1
# )

# # Example: Expand labels to include multiple targets (if applicable)
# output_dim = 8  # Number of target variables
# labels_ul_multivariate = np.tile(labels_ul[:, np.newaxis], (1, output_dim))  # Shape: (num_samples, output_dim)
# labels_dl_multivariate = np.tile(labels_dl[:, np.newaxis], (1, output_dim))  # Shape: (num_samples, output_dim)

# # Save the modified dataset
# np.savez("train_ul_multivariate.npz", sequences=sequences_ul_multivariate, labels=labels_ul_multivariate)
# np.savez("train_dl_multivariate.npz", sequences=sequences_dl_multivariate, labels=labels_dl_multivariate)