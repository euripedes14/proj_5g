import numpy as np

# Φόρτωση των προεπεξεργασμένων δεδομένων
train_data = np.load("train_ul.npz")
test_data = np.load("test_ul.npz")

train_sequences = train_data["sequences"]
train_labels = train_data["labels"]
test_sequences = test_data["sequences"]
test_labels = test_data["labels"]

# Εκτύπωση προεπισκόπησης (χωρίς επιπλέον κανονικοποίηση)
print("Train Sequences (First 5 samples):")
print(train_sequences[:5])

print("\nTest Sequences (First 5 samples):")
print(test_sequences[:5])

print("\nShapes:")
print(f"Train Sequences Shape: {train_sequences.shape}")
print(f"Train Labels Shape: {train_labels.shape}")
print(f"Test Sequences Shape: {test_sequences.shape}")
print(f"Test Labels Shape: {test_labels.shape}")
