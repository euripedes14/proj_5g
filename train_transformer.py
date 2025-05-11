import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load train datasets
train_data = np.load("train_multivariate.npz")
train_sequences = torch.tensor(train_data["sequences"], dtype=torch.float32)
train_labels = torch.tensor(train_data["labels"], dtype=torch.float32)

# Load test datasets
test_data = np.load("test_multivariate.npz")
test_sequences = torch.tensor(test_data["sequences"], dtype=torch.float32)
test_labels = torch.tensor(test_data["labels"], dtype=torch.float32)

# DataLoader
batch_size = 16

train_dataset = TensorDataset(train_sequences, train_labels)
test_dataset = TensorDataset(test_sequences, test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Transformer Model
class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, output_dim)  # Adjusted for multivariate output

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.fc_out(x[:, -1, :])
        return x

# Define model
input_dim = train_sequences.shape[-1]  # Input feature size (2: UL and DL)
output_dim = train_labels.shape[-1]  # Output feature size (2: UL and DL)

# Adjusted for multivariate output
# model = TransformerTimeSeries(input_dim=input_dim, model_dim=64, num_heads=8, num_layers=4, output_dim=output_dim).to(device)
model = TransformerTimeSeries(input_dim=input_dim, model_dim=64, num_heads=8, num_layers=4, output_dim=output_dim).to(device)

criterion = nn.MSELoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

# Early Stopping
patience = 12
best_loss = float("inf")
patience_counter = 0

# Training
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for batch_sequences, batch_labels in train_dataloader:
        batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        outputs = model(batch_sequences)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

    # Evaluate on test data
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for batch_sequences, batch_labels in test_dataloader:
            batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)
            outputs = model(batch_sequences)
            loss = criterion(outputs, batch_labels)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {avg_test_loss:.4f}")

    # Early stopping
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_transformer_model.pth")
        print("Model improved and saved!")
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter}/{patience} epochs.")

    if patience_counter >= patience:
        print("Early stopping triggered. Training stopped.")
        break

total_loss_sum = total_train_loss + total_test_loss
print(f"Overall Train Loss: {total_train_loss:.4f}")
print(f"Overall Test Loss: {total_test_loss:.4f}")
print(f"Total Loss: {total_loss_sum:.4f}")

print("Training finished!")


