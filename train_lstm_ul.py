# input_size:
# The number of features in the input at each time step (for univariate time series, this is 1).

# hidden_size:
# The number of hidden units (neurons) in each LSTM layer. Higher values allow the model to learn more complex patterns.

# output_size:
# The number of outputs the model predicts. For regression of a single value (like UL bitrate), this is 1.

# num_layers:
# The number of stacked LSTM layers in the network. More layers can capture more complex temporal relationships.

# dropout:
# The dropout rate (between 0 and 1) applied between LSTM layers to help prevent overfitting.

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# Check for GPU availability and set device accordingly (CUDA if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the preprocessed uplink (UL) training and testing data
train_data = np.load("train_ul.npz")  # Load training data for UL
test_data = np.load("test_ul.npz")    # Load testing data for UL
val_data = np.load("val_ul.npz")      # Load validation data for UL

# Convert NumPy arrays to PyTorch tensors and move them to the selected device
train_sequences = torch.tensor(train_data["sequences"], dtype=torch.float32).to(device)
train_labels = torch.tensor(train_data["labels"], dtype=torch.float32).to(device)
test_sequences = torch.tensor(test_data["sequences"], dtype=torch.float32).to(device)
test_labels = torch.tensor(test_data["labels"], dtype=torch.float32).to(device)
val_sequences = torch.tensor(val_data["sequences"], dtype=torch.float32).to(device)
val_labels = torch.tensor(val_data["labels"], dtype=torch.float32).to(device)
# Create DataLoader objects for batching and shuffling the data during training/testing
batch_size = 16  # Number of samples per batch (smaller batch = slower but more stable training)
train_dataset = TensorDataset(train_sequences, train_labels)
test_dataset = TensorDataset(test_sequences, test_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   # Shuffle for training
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    # No shuffle for testing
val_dataset = TensorDataset(val_sequences, val_labels)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # No shuffle for validation
# Define the LSTM model class for time series prediction
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=100, output_size=1, num_layers=3, dropout=0.2):
        super().__init__()
        # LSTM layer: bidirectional, multiple layers, with dropout
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True
        )
        # Fully connected layer to map LSTM output to final prediction
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional

    def forward(self, x):
        out, _ = self.lstm(x)  # Pass input through LSTM layers
        out = self.fc(out[:, -1, :])  # Use only the last time step's output
        return out.squeeze(-1)  # Remove last dimension for compatibility

# Initialize model, loss function, and optimizer
input_size = 1      # One feature per timestep (UL bitrate)
hidden_size = 50    # Number of hidden units in LSTM
output_size = 1     # Predict a single value (UL bitrate)
model = LSTMModel(input_size, hidden_size, output_size).to(device)

criterion = nn.MSELoss()  # Mean Squared Error loss for regression
# optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Alternative optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.0005)  # AdamW optimizer for better regularization
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Reduce LR every 5 epochs

# Early stopping parameters
patience = 3              # Stop if no improvement for 3 consecutive epochs
best_loss = float('inf')  # Initialize best loss as infinity
epochs_no_improve = 0     # Counter for epochs without improvement

# Training loop
num_epochs = 50
total_train_loss_sum = 0  # Accumulate total training loss for reporting
total_test_loss_sum = 0   # Accumulate total test loss for reporting

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for sequences, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(sequences)
        labels = labels.squeeze(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item()
    scheduler.step()

    # Evaluate on validation set
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for sequences, labels in val_loader:
            outputs = model(sequences)
            labels = labels.squeeze(-1)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
    total_val_loss /= len(val_loader)
    total_train_loss /= len(train_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_train_loss:.4f}, Val Loss: {total_val_loss:.4f}')

    # Early stopping logic (uses validation loss)
    if total_val_loss < best_loss:
        best_loss = total_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_lstm_dl_model.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# After training, evaluate on the test set
model.load_state_dict(torch.load("best_lstm_dl_model.pth"))
model.eval()
total_test_loss = 0
with torch.no_grad():
    for sequences, labels in test_loader:
        outputs = model(sequences)
        labels = labels.squeeze(-1)
        loss = criterion(outputs, labels)
        total_test_loss += loss.item()
total_test_loss /= len(test_loader)
print(f'Final Test Loss: {total_test_loss:.4f}')
torch.save(model.state_dict(), "lstm_ul_model.pth")
print("Model saved successfully!")