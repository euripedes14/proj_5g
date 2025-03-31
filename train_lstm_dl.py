import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the preprocessed data
train_data = np.load("train_dl.npz")  # Χρησιμοποιούμε το σωστό αρχείο
test_data = np.load("test_dl.npz")

train_sequences = torch.tensor(train_data["sequences"], dtype=torch.float32).to(device)
train_labels = torch.tensor(train_data["labels"], dtype=torch.float32).to(device)
test_sequences = torch.tensor(test_data["sequences"], dtype=torch.float32).to(device)
test_labels = torch.tensor(test_data["labels"], dtype=torch.float32).to(device)

# Create DataLoader
batch_size = 32
train_dataset = TensorDataset(train_sequences, train_labels)
test_dataset = TensorDataset(test_sequences, test_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Πάρε μόνο την τελευταία έξοδο του LSTM
        return out.squeeze(-1)  # Διόρθωση για να ταιριάζει με τα labels

# Initialize model, loss function, optimizer
input_size = 1
hidden_size = 50
output_size = 1
model = LSTMModel(input_size, hidden_size, output_size).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Μείωση learning rate κάθε 10 epochs


# Train the model
num_epochs = 50
total_train_loss_sum = 0  # Initialize total sum of training loss
total_test_loss_sum = 0  # Initialize total sum of test loss

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    total_train_loss = 0  # Initialize total training loss for the epoch
    for sequences, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(sequences)
        labels = labels.squeeze(-1)  # Διόρθωση για συμβατότητα με την έξοδο
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent exploding gradients
        optimizer.step()
        epoch_loss += loss.item()
        total_train_loss += loss.item()  # Accumulate training loss
    
    scheduler.step()  # Adjust learning rate

    # Evaluate on test set
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for sequences, labels in test_loader:
            outputs = model(sequences)
            labels = labels.squeeze(-1)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()
    total_test_loss /= len(test_loader)
    total_train_loss /= len(train_loader)  # Calculate average training loss

    total_train_loss_sum += total_train_loss  # Accumulate total training loss
    total_test_loss_sum += total_test_loss  # Accumulate total test loss

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_train_loss:.4f}, Test Loss: {total_test_loss:.4f}')

total_loss_sum = total_train_loss_sum + total_test_loss_sum  # Calculate total sum of losses
print(f'Total Train Loss Sum: {total_train_loss_sum:.4f}, Total Test Loss Sum: {total_test_loss_sum:.4f}, Total Loss Sum: {total_loss_sum:.4f}')

# Save the model
torch.save(model.state_dict(), "lstm_dl_model.pth")
print("Model saved successfully!")