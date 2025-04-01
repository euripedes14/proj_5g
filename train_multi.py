import numpy as np

# Φόρτωση δεδομένων από τα 2 αρχεία
train_dl = np.load("train_dl.npz")
train_ul = np.load("train_ul.npz")

# Συνδυασμός των χρονικών ακολουθιών ως features (πολυμεταβλητή ανάλυση)
train_sequences = np.concatenate([train_dl["sequences"], train_ul["sequences"]], axis=-1)  # Προσθέτουμε ένα feature axis
train_labels = train_dl["labels"]  # Υποθέτουμε ότι οι ετικέτες είναι ίδιες

# Αποθήκευση του νέου συνδυασμένου dataset
np.savez("train_multivariate.npz", sequences=train_sequences, labels=train_labels)

print("Multivariate dataset saved!")
