import os
import torch
import numpy as np
from dataloader import get_lishen_test_loaders
from model import MPINN_Evaluator

def run_evaluation():
    # Paths configured as requested
    model_path = '../Models/best_model.pth'
    data_path = '../data/LISHEN/Feature_Extraction_Output'
    
    print("Initializing Evaluation Pipeline...")
    
    # Load Data
    test_loaders, client_temps, num_clients = get_lishen_test_loaders(data_path)
    if num_clients == 0:
        print(f"Error: No clients found in {data_path}. Please check the data path.")
        return

    # Initialize Model and inject weights
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MPINN_Evaluator(num_clients=num_clients).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights successfully loaded.")
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        return

    model.set_client_temperatures(client_temps)
    
    # Evaluation Loop
    all_metrics = []
    
    print(f"\n--- Evaluation Results (LISHEN Dataset) ---")
    print(f"{'Client ID':<12} | {'Temp (C)':<10} | {'MAE':<8} | {'MAPE':<8} | {'MSE':<10} | {'RMSE':<8}")
    print("-" * 68)
    
    for client_id, loader in test_loaders.items():
        _, _, metrics = model.evaluate(loader, client_id)
        all_metrics.append(metrics)
        
        print(f"Client_{client_id:<5} | {client_temps[client_id]:<10.1f} | "
              f"{metrics[0]:.4f}   | {metrics[1]:.4f}   | {metrics[2]:.2e} | {metrics[3]:.4f}")
        
    # Aggregate Metrics
    if all_metrics:
        avg_metrics = np.mean(np.array(all_metrics), axis=0)
        print("-" * 68)
        print(f"{'AVERAGE':<12} | {'-':<10} | "
              f"{avg_metrics[0]:.4f}   | {avg_metrics[1]:.4f}   | {avg_metrics[2]:.2e} | {avg_metrics[3]:.4f}")
        print("-" * 68)

if __name__ == '__main__':
    run_evaluation()
