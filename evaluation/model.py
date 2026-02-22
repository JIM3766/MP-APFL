import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad
from utils import eval_metrics

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, layers_num, hidden_dim, dropout=0.2):
        super(MLP, self).__init__()
        layers = []
        for i in range(layers_num):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(Sin())
            elif i == layers_num - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(Sin())
                layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MPINN_Evaluator(nn.Module):
    """
    Evaluation-only Multiphysics-Informed Neural Network.
    Variables are annotated with their corresponding paper PDE parameters.
    """
    def __init__(self, num_clients, F_layers=3, F_hidden=60):
        super().__init__()
        self.num_clients = num_clients
        self.client_temperatures = {}

        # Shared Encoder
        self.encoder = MLP(input_dim=17, output_dim=32, layers_num=3, hidden_dim=60)
        
        # Personalized Predictors
        self.predictors = nn.ModuleList([
            MLP(input_dim=32, output_dim=1, layers_num=2, hidden_dim=32)
            for _ in range(num_clients)
        ])
        
        # Unmodeled Dynamics Net (F)
        self.dynamical_F = nn.ModuleList([
            MLP(input_dim=35, output_dim=1, layers_num=F_layers, hidden_dim=F_hidden)
            for _ in range(num_clients)
        ])
        
        # --- Physics Parameters ---
        # p_C maps to SEI initial capacity loss (\delta_SEI)
        # p_K maps to maximum capacity loss (q_max)
        # alpha2 maps to knee-point intensity (\alpha)
        # beta2 maps to knee-point variance (\beta)
        # u2 maps to knee-point onset threshold (u_knee)
        # E_a maps to Activation Energy (E_a)
        
        self.log_p_r = nn.ParameterList([nn.Parameter(torch.empty(())) for _ in range(num_clients)])
        self.log_p_K = nn.ParameterList([nn.Parameter(torch.empty(())) for _ in range(num_clients)])
        self.log_p_C = nn.ParameterList([nn.Parameter(torch.empty(())) for _ in range(num_clients)])
        self.log_E_a = nn.ParameterList([nn.Parameter(torch.empty(())) for _ in range(num_clients)])
        self.alpha1 = nn.ParameterList([nn.Parameter(torch.empty(())) for _ in range(num_clients)])
        self.log_gamma = nn.ParameterList([nn.Parameter(torch.empty(())) for _ in range(num_clients)])
        self.logit_u1 = nn.ParameterList([nn.Parameter(torch.empty(())) for _ in range(num_clients)])
        self.alpha2 = nn.ParameterList([nn.Parameter(torch.empty(())) for _ in range(num_clients)])
        self.log_beta2 = nn.ParameterList([nn.Parameter(torch.empty(())) for _ in range(num_clients)])
        self.logit_u2 = nn.ParameterList([nn.Parameter(torch.empty(())) for _ in range(num_clients)])

    def set_client_temperatures(self, temp_dict):
        self.client_temperatures = temp_dict

    def evaluate(self, test_loader, client_id):
        """
        Runs the forward pass without gradient tracking and returns metrics.
        """
        self.eval()
        true_labels, pred_labels = [], []
        
        if not test_loader:
             return np.array([]), np.array([]), [0.0, 0.0, 0.0, 0.0]
             
        with torch.no_grad():
            for x1, _, y1, _ in test_loader:
                x1, y1 = x1.to(device), y1.to(device)
                embedding = self.encoder(x1)
                u1 = self.predictors[client_id](embedding)
                
                true_labels.append(y1.cpu().numpy())
                pred_labels.append(u1.cpu().numpy())
                
        true_labels = np.concatenate(true_labels, axis=0)
        pred_labels = np.concatenate(pred_labels, axis=0)
        
        metrics = eval_metrics(pred_labels, true_labels)
        return true_labels, pred_labels, metrics
