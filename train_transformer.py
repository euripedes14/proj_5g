import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Ρύθμιση της συσκευής
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Φόρτωση του πολυμεταβλητού dataset
data = np.load("train_multivariate.npz")
sequences = torch.tensor(data["sequences"], dtype=torch.float32).to(device)
labels = torch.tensor(data["labels"], dtype=torch.float32).to(device).unsqueeze(1)  

# Μετατροπή σε DataLoader
batch_size = 16
dataset = TensorDataset(sequences, labels)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Positional Encoding για την αναπαράσταση του χρόνου
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# Transformer Model
class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerTimeSeries, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.fc_out(x[:, -1, :])  # Παίρνουμε την τελευταία χρονική στιγμή
        return x

# Ορισμός του μοντέλου και υπερπαραμέτρων
input_dim = sequences.shape[-1]  # Αριθμός features
model = TransformerTimeSeries(input_dim=input_dim, model_dim=64, num_heads=8, num_layers=4).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# Εκπαίδευση του μοντέλου
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_sequences, batch_labels in data_loader:
        optimizer.zero_grad()
        outputs = model(batch_sequences)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Αποθήκευση του μοντέλου
torch.save(model.state_dict(), "transformer_model.pth")
print("Model saved successfully!")
