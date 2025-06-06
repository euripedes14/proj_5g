import tkinter as tk
from tkinter import messagebox
import numpy as np
import torch
import importlib
from train_transformer import TransformerTimeSeries
from ul_eval import evaluate_lstm_ul
from dl_eval import evaluate_lstm_dl
from transformer_eval import evaluate_transformer

def run_lstm(model_type, num_runs, eval_func, train_module_name):
    metrics = {"MAPE": [], "RMSE": []}
    train_module = importlib.import_module(train_module_name)
    for run in range(num_runs):
        print(f"Training {model_type} for run {run + 1}/{num_runs}...")
        importlib.reload(train_module)
        print(f"{model_type} training completed!")
        mape, rmse = eval_func()
        metrics["MAPE"].append(mape)
        metrics["RMSE"].append(rmse)
    return metrics

def run_transformer(num_runs):
    metrics = {"MAPE": [], "RMSE": []}
    import train_transformer
    for run in range(num_runs):
        print(f"Training Transformer for run {run + 1}/{num_runs}...")
        importlib.reload(train_transformer)
        print("Transformer training completed!")
        model = TransformerTimeSeries(
            input_dim=2, model_dim=64, num_heads=8, num_layers=4, output_dim=2
        ).to(torch.device("cpu"))
        checkpoint = torch.load("best_transformer_model.pth", map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint)
        model.eval()
        try:
            mape_ul, mape_dl, rmse_ul, rmse_dl = evaluate_transformer(
                model=model,
                test_file="test_multivariate.npz",
                title=f"Transformer Evaluation Run {run + 1}",
                ylabel="Bitrate"
            )
            metrics["MAPE"].append((mape_ul, mape_dl))
            metrics["RMSE"].append((rmse_ul, rmse_dl))
        except Exception as e:
            print(f"Error during Transformer evaluation: {e}")
            messagebox.showerror("Error", f"Transformer evaluation failed: {e}")
            return None
    return metrics

def show_metrics(model_type, metrics):
    if metrics is None:
        return
    if model_type == "Transformer":
        metrics_mape = np.array(metrics["MAPE"])  # shape: (num_runs, 2)
        metrics_rmse = np.array(metrics["RMSE"])  # shape: (num_runs, 2)
        avg_mape = np.mean(metrics_mape, axis=0)
        std_mape = np.std(metrics_mape, axis=0)
        avg_rmse = np.mean(metrics_rmse, axis=0)
        std_rmse = np.std(metrics_rmse, axis=0)
        messagebox.showinfo(
            "Results",
            f"Transformer Results:\n"
            f"MAPE UL: {avg_mape[0]:.4f} ± {std_mape[0]:.4f}\n"
            f"MAPE DL: {avg_mape[1]:.4f} ± {std_mape[1]:.4f}\n"
            f"RMSE UL: {avg_rmse[0]:.4f} ± {std_rmse[0]:.4f}\n"
            f"RMSE DL: {avg_rmse[1]:.4f} ± {std_rmse[1]:.4f}"
        )
    else:
        metrics_mape = np.array(metrics["MAPE"])
        metrics_rmse = np.array(metrics["RMSE"])
        avg_mape = np.mean(metrics_mape)
        std_mape = np.std(metrics_mape)
        avg_rmse = np.mean(metrics_rmse)
        std_rmse = np.std(metrics_rmse)
        messagebox.showinfo(
            "Results",
            f"{model_type} Results:\n"
            f"MAPE: {avg_mape:.4f} ± {std_mape:.4f}\n"
            f"RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}"
        )

def evaluate_model(model_type, num_runs):
    if model_type == "LSTM_DL":
        metrics = run_lstm("LSTM_DL", num_runs, evaluate_lstm_dl, "train_lstm_dl")
    elif model_type == "LSTM_UL":
        metrics = run_lstm("LSTM_UL", num_runs, evaluate_lstm_ul, "train_lstm_ul")
    elif model_type == "Transformer":
        metrics = run_transformer(num_runs)
    else:
        messagebox.showerror("Error", "Unknown model type selected.")
        return
    show_metrics(model_type, metrics)

# GUI setup
def create_gui():
    def on_evaluate():
        model_type = model_var.get()
        try:
            num_runs = int(num_runs_entry.get())
            if num_runs <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of runs (positive integer).")
            return

        evaluate_model(model_type, num_runs)

    root = tk.Tk()
    root.title("Model Comparison")

    # Model selection
    tk.Label(root, text="Select Model:").grid(row=0, column=0, padx=10, pady=10)
    model_var = tk.StringVar(value="LSTM_DL")
    tk.Radiobutton(root, text="LSTM (DL)", variable=model_var, value="LSTM_DL").grid(row=0, column=1, padx=10, pady=10)
    tk.Radiobutton(root, text="LSTM (UL)", variable=model_var, value="LSTM_UL").grid(row=0, column=2, padx=10, pady=10)
    tk.Radiobutton(root, text="Transformer", variable=model_var, value="Transformer").grid(row=0, column=3, padx=10, pady=10)

    # Number of runs
    tk.Label(root, text="Number of Runs:").grid(row=1, column=0, padx=10, pady=10)
    num_runs_entry = tk.Entry(root)
    num_runs_entry.grid(row=1, column=1, padx=10, pady=10)

    # Evaluate button
    evaluate_button = tk.Button(root, text="Evaluate", command=on_evaluate)
    evaluate_button.grid(row=2, column=0, columnspan=4, pady=10)

    root.mainloop()

# Run the GUI
if __name__ == "__main__":
    create_gui()
# import tkinter as tk
# from tkinter import messagebox
# import numpy as np
# import torch
# import importlib
# from train_transformer import TransformerTimeSeries
# from ul_eval import evaluate_lstm_ul
# from dl_eval import evaluate_lstm_dl
# from transformer_eval import evaluate_transformer

# # Function to evaluate the selected model
# def evaluate_model(model_type, num_runs):
#     metrics = {"MAPE": [], "RMSE": []}

#     if model_type == "LSTM_DL":
#         import train_lstm_dl
#         for run in range(num_runs):
#             print(f"Training LSTM_DL for run {run + 1}/{num_runs}...")
#             importlib.reload(train_lstm_dl)
#             print("LSTM (DL) training completed!")
#             mape, rmse = evaluate_lstm_dl()
#             metrics["MAPE"].append(mape)
#             metrics["RMSE"].append(rmse)

#     elif model_type == "LSTM_UL":
#         import train_lstm_ul
#         for run in range(num_runs):
#             print(f"Training LSTM_UL for run {run + 1}/{num_runs}...")
#             importlib.reload(train_lstm_ul)
#             print("LSTM (UL) training completed!")
#             mape, rmse = evaluate_lstm_ul()
#             metrics["MAPE"].append(mape)
#             metrics["RMSE"].append(rmse)

#     elif model_type == "Transformer":
#         import train_transformer
#         for run in range(num_runs):
#             print(f"Training Transformer for run {run + 1}/{num_runs}...")
#             importlib.reload(train_transformer)
#             print("Transformer training completed!")

#             # Load the Transformer model
#             model = TransformerTimeSeries(
#                 input_dim=2,
#                 model_dim=64,
#                 num_heads=8,
#                 num_layers=4,
#                 output_dim=2
#             ).to(torch.device("cpu"))

#             # Load the checkpoint
#             checkpoint = torch.load("best_transformer_model.pth", map_location=torch.device("cpu"))
#             model.load_state_dict(checkpoint)
#             model.eval()

#             # Evaluate the Transformer model
#             try:
#                 mape_ul, mape_dl, rmse_ul, rmse_dl = evaluate_transformer(
#                     model=model,
#                     test_file="test_multivariate.npz",
#                     title=f"Transformer Evaluation Run {run + 1}",
#                     ylabel="Bitrate"
#                 )
#                 metrics["MAPE"].append((mape_ul, mape_dl))
#                 metrics["RMSE"].append((rmse_ul, rmse_dl))
#             except Exception as e:
#                 print(f"Error during Transformer evaluation: {e}")
#                 messagebox.showerror("Error", f"Transformer evaluation failed: {e}")
#                 return

#     # Compute average and standard deviation
#     if model_type == "Transformer":
#         metrics_mape = np.array(metrics["MAPE"])  # shape: (num_runs, 2)
#         metrics_rmse = np.array(metrics["RMSE"])  # shape: (num_runs, 2)
#         avg_mape = np.mean(metrics_mape, axis=0)
#         std_mape = np.std(metrics_mape, axis=0)
#         avg_rmse = np.mean(metrics_rmse, axis=0)
#         std_rmse = np.std(metrics_rmse, axis=0)
#         messagebox.showinfo(
#             "Results",
#             f"Transformer Results:\n"
#             f"MAPE UL: {avg_mape[0]:.4f} ± {std_mape[0]:.4f}\n"
#             f"MAPE DL: {avg_mape[1]:.4f} ± {std_mape[1]:.4f}\n"
#             f"RMSE UL: {avg_rmse[0]:.4f} ± {std_rmse[0]:.4f}\n"
#             f"RMSE DL: {avg_rmse[1]:.4f} ± {std_rmse[1]:.4f}"
#         )
#     else:
#         metrics_mape = np.array(metrics["MAPE"])  # shape: (num_runs,)
#         metrics_rmse = np.array(metrics["RMSE"])  # shape: (num_runs,)
#         avg_mape = np.mean(metrics_mape)
#         std_mape = np.std(metrics_mape)
#         avg_rmse = np.mean(metrics_rmse)
#         std_rmse = np.std(metrics_rmse)
#         messagebox.showinfo(
#             "Results",
#             f"{model_type} Results:\n"
#             f"MAPE: {avg_mape:.4f} ± {std_mape:.4f}\n"
#             f"RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}"
#         )

# # GUI setup
# def create_gui():
#     def on_evaluate():
#         model_type = model_var.get()
#         try:
#             num_runs = int(num_runs_entry.get())
#             if num_runs <= 0:
#                 raise ValueError
#         except ValueError:
#             messagebox.showerror("Error", "Please enter a valid number of runs (positive integer).")
#             return

#         evaluate_model(model_type, num_runs)

#     root = tk.Tk()
#     root.title("Model Comparison")

#     # Model selection
#     tk.Label(root, text="Select Model:").grid(row=0, column=0, padx=10, pady=10)
#     model_var = tk.StringVar(value="LSTM_DL")
#     tk.Radiobutton(root, text="LSTM (DL)", variable=model_var, value="LSTM_DL").grid(row=0, column=1, padx=10, pady=10)
#     tk.Radiobutton(root, text="LSTM (UL)", variable=model_var, value="LSTM_UL").grid(row=0, column=2, padx=10, pady=10)
#     tk.Radiobutton(root, text="Transformer", variable=model_var, value="Transformer").grid(row=0, column=3, padx=10, pady=10)

#     # Number of runs
#     tk.Label(root, text="Number of Runs:").grid(row=1, column=0, padx=10, pady=10)
#     num_runs_entry = tk.Entry(root)
#     num_runs_entry.grid(row=1, column=1, padx=10, pady=10)

#     # Evaluate button
#     evaluate_button = tk.Button(root, text="Evaluate", command=on_evaluate)
#     evaluate_button.grid(row=2, column=0, columnspan=4, pady=10)

#     root.mainloop()

# # Run the GUI
# if __name__ == "__main__":
#     create_gui()