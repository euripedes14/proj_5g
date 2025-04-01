import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from train_lstm_ul import LSTMModel  # Import the LSTMModel class from the training script

# Check for GPU
device = torch.device("cpu")
print(f"Using device: {device}")

# Load the trained model
model = LSTMModel(input_size=1, hidden_size=50, output_size=1).to(device)
model.load_state_dict(torch.load("lstm_ul_model.pth"))
model.eval()

# Load test data
test_data = np.load("test_ul.npz")
test_sequences = torch.tensor(test_data["sequences"], dtype=torch.float32).to(device)
test_labels = torch.tensor(test_data["labels"], dtype=torch.float32).to(device)

# Ensure the input shape is correct
test_sequences = test_sequences.reshape(test_sequences.shape[0], test_sequences.shape[1], 1)

# Get predictions
with torch.no_grad():
    predictions = model(test_sequences).cpu().numpy()

test_labels = test_labels.cpu().numpy()

# Μετατόπιση των προβλέψεων προς τα πίσω κατά 1 βήμα
predictions = np.roll(predictions, shift=-1)

# Αφαίρεση του τελευταίου στοιχείου (άκυρο λόγω μετατόπισης)
predictions = predictions[:-1]
test_labels = test_labels[:-1]
# Compute evaluation metrics
rmse = np.sqrt(mean_squared_error(test_labels, predictions))

#MAPE calculation to avoid division by zero
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Avoid division by zero - Ignore zero values
    non_zero_mask = y_true != 0
    y_true_filtered = y_true[non_zero_mask]
    y_pred_filtered = y_pred[non_zero_mask]
    
    if len(y_true_filtered) == 0:
        return np.nan  # No valid MAPE calculation possible
    
    return np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100

# Calculate MAPE
mape = mean_absolute_percentage_error(test_labels, predictions)

print(f"RMSE: {rmse:.4f}, MAPE: {mape:.4f}")

# Plot predictions vs actual values
plt.figure(figsize=(12, 6))
plt.plot(test_labels, label='Actual', linestyle='dashed', alpha=0.7)
plt.plot(predictions, label='Predicted', alpha=0.7)
plt.legend()
plt.xlabel("Time")
plt.ylabel("UL Bitrate")
plt.title("LSTM Predictions vs Actual Values")
plt.grid(True)
plt.show()
