import pandas as pd  # Importing pandas for data manipulation
import numpy as np  # Importing numpy for numerical operations
from sklearn.preprocessing import MinMaxScaler  # Importing MinMaxScaler for data normalization

# Define the file path for the dataset
file_path = "youtube_dataset.csv"

# Define column names for the dataset (downlink and uplink bitrates)
columns = ["DL_bitrate", "UL_bitrate"]  

# Read the CSV file into a pandas DataFrame
# Skip the first row (assuming it contains headers) and use a comma as the delimiter
df = pd.read_csv(file_path, names=columns, delimiter=",", skiprows=1)

# Convert DL_bitrate and UL_bitrate columns to numeric values to avoid processing errors
df["DL_bitrate"] = pd.to_numeric(df["DL_bitrate"], errors='coerce')
df["UL_bitrate"] = pd.to_numeric(df["UL_bitrate"], errors='coerce')

# Drop rows containing NaN values to maintain data integrity
df = df.dropna()

#test if the values are 0 and drop them
# Drop rows where both DL_bitrate and UL_bitrate are 0
#df = df[(df["DL_bitrate"] != 0) | (df["UL_bitrate"] != 0)]
# Create a timestamp column assuming each row represents one second
df["timestamp"] = pd.date_range(start="2024-01-01", periods=len(df), freq="s")

# Set the timestamp as the index for time series analysis
df.set_index("timestamp", inplace=True)

# Aggregate the data per minute by summing the values within each minute interval

#df_minute = df.resample("10s").sum()
#df_minute = df.resample("30s").sum()
#df_minute = df.resample("1min").sum()
df_minute = df.resample("5min").sum()

# Drop any potential NaN values that might have arisen due to resampling
df_minute = df_minute.dropna()

# Define the train-test split ratio (60% training, 40% testing)
train_size = int(len(df_minute) * 0.6)
train, test = df_minute.iloc[:train_size], df_minute.iloc[train_size:]

# Initialize Min-Max Scalers for normalizing both DL and UL bitrates
scaler_dl = MinMaxScaler()
scaler_ul = MinMaxScaler()

# Fit and transform the training data using Min-Max scaling
# train["DL_bitrate"] = scaler_dl.fit_transform(train[["DL_bitrate"]]).astype(np.float64)
# train["UL_bitrate"] = scaler_ul.fit_transform(train[["UL_bitrate"]]).astype(np.float64)

# # Transform the test data using the same scaling parameters
# test["DL_bitrate"] = scaler_dl.transform(test[["DL_bitrate"]]).astype(np.float64)
# test["UL_bitrate"] = scaler_ul.transform(test[["UL_bitrate"]]).astype(np.float64)
train = df_minute.iloc[:train_size].copy()
test = df_minute.iloc[train_size:].copy()

train["DL_bitrate"] = scaler_dl.fit_transform(train[["DL_bitrate"]]).astype(np.float64)
train["UL_bitrate"] = scaler_ul.fit_transform(train[["UL_bitrate"]]).astype(np.float64)
test["DL_bitrate"] = scaler_dl.transform(test[["DL_bitrate"]]).astype(np.float64)
test["UL_bitrate"] = scaler_ul.transform(test[["UL_bitrate"]]).astype(np.float64)


# Save the Min-Max scalers for future use (e.g., de-normalization)
np.savez("scalers.npz", dl_min=scaler_dl.data_min_, dl_max=scaler_dl.data_max_,
         ul_min=scaler_ul.data_min_, ul_max=scaler_ul.data_max_)

# Function to create sequences using a rolling window approach
def create_sequences(data, window_size):
    sequences, labels = [], []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i + window_size])  # Extract a sequence of length `window_size`
        labels.append(data[i + window_size])  # Assign the next value as the label
    return np.array(sequences), np.array(labels)

# Define the window size for the rolling sequences
window_size = 5

# Generate rolling window sequences for Downlink (DL) and Uplink (UL)
train_sequences_dl, train_labels_dl = create_sequences(train["DL_bitrate"].values, window_size)
test_sequences_dl, test_labels_dl = create_sequences(test["DL_bitrate"].values, window_size)
train_sequences_ul, train_labels_ul = create_sequences(train["UL_bitrate"].values, window_size)
test_sequences_ul, test_labels_ul = create_sequences(test["UL_bitrate"].values, window_size)

# Reshape data into the required format for LSTM models: (samples, timesteps, features)
train_sequences_dl = train_sequences_dl.reshape(-1, window_size, 1)
test_sequences_dl = test_sequences_dl.reshape(-1, window_size, 1)
train_sequences_ul = train_sequences_ul.reshape(-1, window_size, 1)
test_sequences_ul = test_sequences_ul.reshape(-1, window_size, 1)

# Save the preprocessed data into separate files for training and testing
np.savez("train_dl.npz", sequences=train_sequences_dl, labels=train_labels_dl)
np.savez("test_dl.npz", sequences=test_sequences_dl, labels=test_labels_dl)
np.savez("train_ul.npz", sequences=train_sequences_ul, labels=train_labels_ul)
np.savez("test_ul.npz", sequences=test_sequences_ul, labels=test_labels_ul)
