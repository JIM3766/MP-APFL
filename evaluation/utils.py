import numpy as np

def eval_metrics(y_pred, y_true):
    """
    Calculates standard evaluation metrics for SOH estimation.
    Returns: [MAE, MAPE, MSE, RMSE]
    """
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean(np.square(y_true - y_pred))
    rmse = np.sqrt(mse)
    
    # Avoid division by zero
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)))
    
    return [mae, mape, mse, rmse]
