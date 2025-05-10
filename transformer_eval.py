import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from train_transformer import TransformerTimeSeries  # Import the Transformer model class

device = torch.device("cpu")
print(f"Using device: {device}")

# Model parameters (MUST match training)
input_dim = 2  # UL and DL as input features
model_dim = 64
num_heads = 8
num_layers = 4
output_dim = 2  # UL and DL as output features

# Load the trained model
model = TransformerTimeSeries(input_dim=input_dim, model_dim=model_dim, num_heads=num_heads, num_layers=num_layers, output_dim=output_dim).to(device)

# Load the checkpoint
checkpoint = torch.load("best_transformer_model.pth", map_location=device)
model.load_state_dict(checkpoint)
model.eval()



def evaluate_transformer(model, test_file, title, ylabel):
    # Load test data
    test_data = np.load(test_file)
    test_sequences = torch.tensor(test_data["sequences"], dtype=torch.float32).to(device)
    test_labels = torch.tensor(test_data["labels"], dtype=torch.float32).to(device)
    test_sequences = torch.tensor(test_data["sequences"], dtype=torch.float32)
    test_labels = torch.tensor(test_data["labels"], dtype=torch.float32)

    # Ensure the model is in evaluation mode
    model.eval()

    # Get predictions
    with torch.no_grad():
        predictions = model(test_sequences).cpu().numpy()

    test_labels = test_labels.cpu().numpy()
    test_labels = test_labels.numpy()
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    y_true_filtered = y_true[non_zero_mask]
    y_pred_filtered = y_pred[non_zero_mask]
    if len(y_true_filtered) == 0:
        return np.nan
    return np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
    # Compute RMSE
    rmse_ul = np.sqrt(np.mean((test_labels[:, 0] - predictions[:, 0]) ** 2))
    rmse_dl = np.sqrt(np.mean((test_labels[:, 1] - predictions[:, 1]) ** 2))
    # Compute RMSE and MAPE for UL and DL
    rmse_ul = np.sqrt(mean_squared_error(test_labels[:, 0], predictions[:, 0]))
    rmse_dl = np.sqrt(mean_squared_error(test_labels[:, 1], predictions[:, 1]))
    mape_ul = mean_absolute_percentage_error(test_labels[:, 0], predictions[:, 0])
    mape_dl = mean_absolute_percentage_error(test_labels[:, 1], predictions[:, 1])

    # Compute MAPE
    def mean_absolute_percentage_error(y_true, y_pred):
        non_zero_mask = y_true != 0
        y_true_filtered = y_true[non_zero_mask]
        y_pred_filtered = y_pred[non_zero_mask]
        if len(y_true_filtered) == 0:
            return np.nan
        return np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
    print(f"{title} - RMSE UL: {rmse_ul:.4f}, RMSE DL: {rmse_dl:.4f}")
    print(f"{title} - MAPE UL: {mape_ul:.4f}, MAPE DL: {mape_dl:.4f}")

    return mape_ul, mape_dl, rmse_ul, rmse_dl
    mape_ul = mean_absolute_percentage_error(test_labels[:, 0], predictions[:, 0])
    mape_dl = mean_absolute_percentage_error(test_labels[:, 1], predictions[:, 1])

    print(f"{title} - RMSE UL: {rmse_ul:.4f}, RMSE DL: {rmse_dl:.4f}")
    print(f"{title} - MAPE UL: {mape_ul:.4f}, MAPE DL: {mape_dl:.4f}")

    # Plot predictions vs actual values for UL
    plt.figure(figsize=(12, 6))
    plt.plot(test_labels[:, 0], label='Actual UL', linestyle='dashed', alpha=0.7, color='blue')
    plt.plot(predictions[:, 0], label='Predicted UL', alpha=0.7, color='cyan')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(f"{title} - UL Predictions vs Actual Values")
    plt.grid(True)
    plt.show()

    # Plot predictions vs actual values for DL
    plt.figure(figsize=(12, 6))
    plt.plot(test_labels[:, 1], label='Actual DL', linestyle='dashed', alpha=0.7, color='red')
    plt.plot(predictions[:, 1], label='Predicted DL', alpha=0.7, color='orange')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(f"{title} - DL Predictions vs Actual Values")
    plt.grid(True)
    plt.show()

# Evaluate on test data
#evaluate_transformer(model, "test_multivariate.npz", title="Transformer Evaluation", ylabel="Bitrate")
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error

# from train_transformer import TransformerTimeSeries  # Import the Transformer model class

# device = torch.device("cpu")
# print(f"Using device: {device}")

# # Model parameters (MUST match training)
# input_dim = 3  # Corrected to match the training input_dim
# model_dim = 64