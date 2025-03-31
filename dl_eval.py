import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from train_lstm_dl import LSTMModel  # Import the LSTMModel class from the training script

# Check for GPU
device = torch.device("cpu")
print(f"Using device: {device}")

# Load the trained model
model = LSTMModel(input_size=1, hidden_size=50, output_size=1).to(device)
model.load_state_dict(torch.load("lstm_dl_model.pth"))
model.eval()

# Load test data
test_data = np.load("test_dl.npz")
test_sequences = torch.tensor(test_data["sequences"], dtype=torch.float32).to(device)
test_labels = torch.tensor(test_data["labels"], dtype=torch.float32).to(device)

# Ensure the input shape is correct
test_sequences = test_sequences.reshape(test_sequences.shape[0], test_sequences.shape[1], 1)

# Get predictions
with torch.no_grad():
    predictions = model(test_sequences).cpu().numpy()

test_labels = test_labels.cpu().numpy()

# Compute evaluation metrics
rmse = np.sqrt(mean_squared_error(test_labels, predictions))

# Corrected MAPE calculation to avoid division by zero
def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = 1e-8  # Προσθήκη ενός μικρού αριθμού για αποφυγή διαίρεσης με μηδέν
    return np.mean(np.abs((y_true - y_pred) / (np.maximum(np.abs(y_true), epsilon)))) * 100

# Calculate MAPE
mape = mean_absolute_percentage_error(test_labels, predictions)

print(f"RMSE: {rmse:.4f}, MAPE: {mape:.4f}")

# Plot predictions vs actual values
plt.figure(figsize=(12, 6))
plt.plot(test_labels, label='Actual', linestyle='dashed', alpha=0.7)
plt.plot(predictions, label='Predicted', alpha=0.7)
plt.legend()
plt.xlabel("Time")
plt.ylabel("DL Bitrate")
plt.title("LSTM Predictions vs Actual Values")
plt.grid(True)  # Προσθήκη πλέγματος για καλύτερη ανάγνωση
plt.show()
