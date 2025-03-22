import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Φόρτωση CSV
file_path = "youtube_dataset.csv"
columns = ["DL_bitrate", "UL_bitrate"]
df = pd.read_csv(file_path, names=columns, delimiter=",", skiprows=1)

# Ensure all values are numeric
df["DL_bitrate"] = pd.to_numeric(df["DL_bitrate"], errors='coerce')
df["UL_bitrate"] = pd.to_numeric(df["UL_bitrate"], errors='coerce')

# Drop rows with NaN values
df = df.dropna()

# Δημιουργία χρονικής στήλης (κάθε row = 1 δευτερόλεπτο)
df["timestamp"] = pd.date_range(start="2024-01-01", periods=len(df), freq="s")
df.set_index("timestamp", inplace=True)

# Συγκέντρωση ανά λεπτό (άθροισμα των δειγμάτων μέσα στο λεπτό)
df_minute = df.resample("min").sum()

# Αφαίρεση τυχόν NaN μετά τη συγκέντρωση
df_minute = df_minute.dropna()

# Διαχωρισμός σε train (80%) και test (20%)
train_size = int(len(df_minute) * 0.8)
train, test = df_minute.iloc[:train_size], df_minute.iloc[train_size:]

# Κανονικοποίηση δεδομένων στο [0,1]
scaler_dl = MinMaxScaler()
scaler_ul = MinMaxScaler()

train["DL_bitrate"] = scaler_dl.fit_transform(train[["DL_bitrate"]]).astype(np.float64)
train["UL_bitrate"] = scaler_ul.fit_transform(train[["UL_bitrate"]]).astype(np.float64)

test["DL_bitrate"] = scaler_dl.transform(test[["DL_bitrate"]]).astype(np.float64)
test["UL_bitrate"] = scaler_ul.transform(test[["UL_bitrate"]]).astype(np.float64)

# Αποθήκευση scalers για μελλοντική ανακανονικοποίηση
np.savez("scalers.npz", dl_min=scaler_dl.data_min_, dl_max=scaler_dl.data_max_,
         ul_min=scaler_ul.data_min_, ul_max=scaler_ul.data_max_)

# Συνάρτηση για δημιουργία ακολουθιών rolling window
def create_sequences(data, window_size):
    sequences, labels = [], []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i + window_size])
        labels.append(data[i + window_size])
    return np.array(sequences), np.array(labels)

# Παράμετρος rolling window
window_size = 10

# Δημιουργία ακολουθιών για Downlink (DL) και Uplink (UL)
train_sequences_dl, train_labels_dl = create_sequences(train["DL_bitrate"].values, window_size)
test_sequences_dl, test_labels_dl = create_sequences(test["DL_bitrate"].values, window_size)
train_sequences_ul, train_labels_ul = create_sequences(train["UL_bitrate"].values, window_size)
test_sequences_ul, test_labels_ul = create_sequences(test["UL_bitrate"].values, window_size)

# Μετατροπή στο κατάλληλο σχήμα για LSTM (samples, timesteps, features)
train_sequences_dl = train_sequences_dl.reshape(-1, window_size, 1)
test_sequences_dl = test_sequences_dl.reshape(-1, window_size, 1)
train_sequences_ul = train_sequences_ul.reshape(-1, window_size, 1)
test_sequences_ul = test_sequences_ul.reshape(-1, window_size, 1)

# Αποθήκευση των προεπεξεργασμένων δεδομένων
np.savez("train_dl.npz", sequences=train_sequences_dl, labels=train_labels_dl)
np.savez("test_dl.npz", sequences=test_sequences_dl, labels=test_labels_dl)
np.savez("train_ul.npz", sequences=train_sequences_ul, labels=train_labels_ul)
np.savez("test_ul.npz", sequences=test_sequences_ul, labels=test_labels_ul)
