import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the preprocessed data
train_data = np.load("train_dl.npz")  
test_data = np.load("test_dl.npz")
val_data = np.load("val_dl.npz")    


# Convert the loaded NumPy arrays into PyTorch tensors and move them to the selected device
train_sequences = torch.tensor(train_data["sequences"], dtype=torch.float32).to(device) 
train_labels = torch.tensor(train_data["labels"], dtype=torch.float32).to(device)
val_sequences= torch.tensor(train_data["sequences"], dtype=torch.float32).to(device) 
val_labels = torch.tensor(train_data["labels"], dtype=torch.float32).to(device)
test_sequences = torch.tensor(test_data["sequences"], dtype=torch.float32).to(device) 
test_labels = torch.tensor(test_data["labels"], dtype=torch.float32).to(device)

batch_size = 16 # small batches takes some time but better training
# ara ypologizontai 16 batches taytoxrona ptin enimerothoun ta bari toy diktyou

# Create DataLoader for training and testing datasets
train_dataset = TensorDataset(train_sequences, train_labels)
val_dataset = TensorDataset(val_sequences, val_labels)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # Validation set should not be shuffled
test_dataset = TensorDataset(test_sequences, test_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#DEFINE MODEL
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=100, output_size=1, num_layers=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Adjust for bidirectional LSTM (hidden_size * 2)

    def forward(self, x):
        out, _ = self.lstm(x) #input through LSTM layers
        out = self.fc(out[:, -1, :])  #only the last output of the LSTM
        return out.squeeze(-1)  # Reshape output for compatibility with labels

# Initialize model, loss function, optimizer
input_size = 1 # One feature per timestep
hidden_size = 50 # Number of hidden units in LSTM layers
output_size = 1 # Single output prediction

model = LSTMModel(input_size, hidden_size, output_size).to(device)

# Define the loss function (Mean Squared Error for regression tasks)
criterion = nn.MSELoss()
#optimizer = optim.Adam(model.parameters(), lr=0.0005) #or rmsprop
optimizer = optim.AdamW(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Μείωση learning rate κάθε 5 epochs
#scheduler gia prosarmogi lr. Apotropi prooris siglisis kai kaliteri evresi t min

# Initialize early stopping variables
patience = 3  #an to loss den veltionetai gia 3 diadoxika epochs early stop
best_loss = float('inf')  # Initialize the best loss as infinity
epochs_no_improve = 0  # Counter for epochs with no improvement

# Train the model
num_epochs = 50
total_train_loss_sum = 0  # Initialize total sum of training loss
total_test_loss_sum = 0  # Initialize total sum of test loss

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

    # Early stopping logic (now uses validation loss)
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
# After training, load the best model and evaluate on the test set
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

# Optionally save the best model under a generic name
torch.save(model.state_dict(), "lstm_dl_model.pth")
print("Model saved successfully!")