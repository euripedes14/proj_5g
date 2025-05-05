import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Datasets
data_ul = np.load("train_ul_multivariate.npz")
data_dl = np.load("train_dl_multivariate.npz")

sequences_ul = torch.tensor(data_ul["sequences"], dtype=torch.float32)
labels_ul = torch.tensor(data_ul["labels"], dtype=torch.float32)  # Multivariate labels

sequences_dl = torch.tensor(data_dl["sequences"], dtype=torch.float32)
labels_dl = torch.tensor(data_dl["labels"], dtype=torch.float32)  # Multivariate labels

# DataLoader
batch_size = 16

dataset_ul = TensorDataset(sequences_ul, labels_ul)
dataset_dl = TensorDataset(sequences_dl, labels_dl)

dataloader_ul = DataLoader(dataset_ul, batch_size=batch_size, shuffle=True)
dataloader_dl = DataLoader(dataset_dl, batch_size=batch_size, shuffle=True)

# Transformer Model
class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(TransformerTimeSeries, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, output_dim)  # Adjusted for multivariate output

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.fc_out(x[:, -1, :])
        return x

# Define models
input_dim = max(sequences_ul.shape[-1], sequences_dl.shape[-1])  # Input feature size
output_dim = labels_ul.shape[-1]  # Output feature size (multivariate)

model_ul = TransformerTimeSeries(input_dim=input_dim, model_dim=64, num_heads=8, num_layers=4, output_dim=output_dim).to(device)
model_dl = TransformerTimeSeries(input_dim=input_dim, model_dim=64, num_heads=8, num_layers=4, output_dim=output_dim).to(device)

criterion = nn.MSELoss()
optimizer_ul = torch.optim.Adam(model_ul.parameters(), lr=0.0005)
optimizer_dl = torch.optim.Adam(model_dl.parameters(), lr=0.0005)

# Early Stopping
patience = 3
best_loss = float("inf")
patience_counter = 0

# Training
num_epochs = 30
for epoch in range(num_epochs):
    model_ul.train()
    model_dl.train()
    total_loss_ul = 0
    total_loss_dl = 0

    # UL data
    for batch_sequences, batch_labels in dataloader_ul:
        batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)
        optimizer_ul.zero_grad()
        outputs = model_ul(batch_sequences)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer_ul.step()
        total_loss_ul += loss.item()

    # DL data
    for batch_sequences, batch_labels in dataloader_dl:
        batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)
        optimizer_dl.zero_grad()
        outputs = model_dl(batch_sequences)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer_dl.step()
        total_loss_dl += loss.item()

    avg_loss = (total_loss_ul + total_loss_dl) / 2  
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss UL: {total_loss_ul:.4f}, Loss DL: {total_loss_dl:.4f}, Avg Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save({
            'model_ul_state_dict': model_ul.state_dict(),
            'model_dl_state_dict': model_dl.state_dict()
        }, 'best_transformer_model.pth')  
        print("Model improved!")
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter}/{patience} epochs.")

    if patience_counter >= patience:
        print("Early stopping triggered. Training stopped.")
        break 

print("Training finished!")


