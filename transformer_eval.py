import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from train_transformer import TransformerTimeSeries  # Import the Transformer model class

# Ρύθμιση συσκευής
device = torch.device("cpu")
print(f"Using device: {device}")

# Φόρτωση του εκπαιδευμένου μοντέλου
input_dim = 2  # UL + DL Bitrate ως input
model = TransformerTimeSeries(input_dim=input_dim, model_dim=64, num_heads=8, num_layers=4).to(device)
model.load_state_dict(torch.load("best_transformer_model.pth"))
model.eval()

# Φόρτωση των test δεδομένων
test_ul = np.load("test_ul.npz")
test_dl = np.load("test_dl.npz")

# Μετατροπή σε PyTorch tensors
test_ul_sequences = torch.tensor(test_ul["sequences"], dtype=torch.float32).to(device)
test_dl_sequences = torch.tensor(test_dl["sequences"], dtype=torch.float32).to(device)
test_labels = torch.tensor(test_ul["labels"], dtype=torch.float32).to(device)

# Συνένωση UL & DL inputs
test_sequences = torch.cat((test_ul_sequences, test_dl_sequences), dim=-1)

# Προβλέψεις
with torch.no_grad():
    predictions = model(test_sequences).cpu().numpy()

test_labels = test_labels.cpu().numpy()

# Μετατόπιση των προβλέψεων προς τα πίσω κατά 1 βήμα
predictions = np.roll(predictions, shift=-1)

# Αφαίρεση του τελευταίου στοιχείου (άκυρο λόγω μετατόπισης)
predictions = predictions[:-1]
test_labels = test_labels[:-1]

# Υπολογισμός RMSE
rmse = np.sqrt(mean_squared_error(test_labels, predictions))

# Υπολογισμός MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    y_true_filtered = y_true[non_zero_mask]
    y_pred_filtered = y_pred[non_zero_mask]
    
    if len(y_true_filtered) == 0:
        return np.nan  # No valid MAPE calculation possible
    
    return np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100

mape = mean_absolute_percentage_error(test_labels, predictions)

print(f"RMSE: {rmse:.4f}, MAPE: {mape:.4f}")

# Διάγραμμα προβλέψεων vs πραγματικών τιμών
plt.figure(figsize=(12, 6))
plt.plot(test_labels, label='Actual UL Bitrate', linestyle='dashed', alpha=0.7)
plt.plot(predictions, label='Predicted UL Bitrate', alpha=0.7)
plt.plot(test_dl["labels"][:-1], label='Actual DL Bitrate', linestyle='dotted', alpha=0.7)  # Προσθήκη DL bitrate
plt.legend()
plt.xlabel("Time")
plt.ylabel("Bitrate")
plt.title("Transformer Predictions vs Actual Values")
plt.grid(True)
plt.show()
