#Debugging and Loading Data
#Με numpy για να φορτώνει τα αρχεία train_dl.npz και test_dl.npz, τα οποία περιέχουν τα προεπεξεργασμένα δεδομένα για εκπαίδευση και δοκιμή αντίστοιχα.
#Από το train_dl.npz εξάγει τα train_sequences (σειρές εισόδου) και train_labels (τιμές-στόχοι).
#Από το test_dl.npz εξάγει τα test_sequences (σειρές εισόδου) και test_labels (τιμές-στόχοι).
#Εμφανίζει στην οθόνη τις πρώτες 5 σειρές από τα δεδομένα εκπαίδευσης και δοκιμών, δίνοντας μια εικόνα για το πώς είναι διαμορφωμένα.
#τυπώνει τις διαστάσεις των δεδομένων (shape) για να επαληθευτεί ότι έχουν το σωστό μέγεθος και σχήμα πριν χρησιμοποιηθούν

import numpy as np  # Importing numpy for numerical operations

# Load the preprocessed training and testing datasets
train_data = np.load("train_dl.npz")  # Load training dataset
test_data = np.load("test_dl.npz")  # Load testing dataset

# Extract sequences and labels from the training dataset
train_sequences = train_data["sequences"]  # Input sequences for training
train_labels = train_data["labels"]  # Corresponding labels for training

# Extract sequences and labels from the testing dataset
test_sequences = test_data["sequences"]  # Input sequences for testing
test_labels = test_data["labels"]  # Corresponding labels for testing

# Print a preview of the first 5 training sequences
print("Train Sequences (First 5 samples):")
print(train_sequences[:5])

# Print a preview of the first 5 testing sequences
print("\nTest Sequences (First 5 samples):")
print(test_sequences[:5])

# Print the shapes of the datasets to verify their structure
# print("\nShapes:")
# print(f"Train Sequences Shape: {train_sequences.shape}")  # Shape of training sequences
# print(f"Train Labels Shape: {train_labels.shape}")  # Shape of training labels
# print(f"Test Sequences Shape: {test_sequences.shape}")  # Shape of testing sequences
# print(f"Test Labels Shape: {test_labels.shape}")  # Shape of testing labels\

# Print the shapes of the datasets to verify their structure
print("\nShapes:")
print(f"Train Sequences Shape: {train_sequences.shape}")  # Should be (samples, 10, 1)
print(f"Train Labels Shape: {train_labels.shape}")
print(f"Test Sequences Shape: {test_sequences.shape}")  # Should be (samples, 10, 1)
print(f"Test Labels Shape: {test_labels.shape}")