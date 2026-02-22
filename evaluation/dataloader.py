import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

class EvalDataLoader:
    def __init__(self, normalization_method='min-max'):
        self.normalization_method = normalization_method

    def _3_sigma(self, series):
        rule = (series.mean() - 3 * series.std() > series) | (series.mean() + 3 * series.std() < series)
        return np.arange(series.shape[0])[rule]

    def delete_3_sigma(self, df):
        df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        out_index = []
        for col in df.columns:
            out_index.extend(self._3_sigma(df[col]))
        out_index = list(set(out_index))
        return df.drop(out_index, axis=0).reset_index(drop=True)

    def read_one_csv(self, file_name, nominal_capacity=None):
        df = pd.read_csv(file_name)
        df.insert(df.shape[1] - 1, 'cycle index', np.arange(df.shape[0]))
        df = self.delete_3_sigma(df)
        
        if nominal_capacity:
            df['capacity'] /= nominal_capacity
            f_df = df.iloc[:, :-1]
            if self.normalization_method == 'min-max':
                f_df = 2 * (f_df - f_df.min()) / (f_df.max() - f_df.min()) - 1
            elif self.normalization_method == 'z-score':
                f_df = (f_df - f_df.mean()) / f_df.std()
            df.iloc[:, :-1] = f_df
        return df

    def load_one_battery(self, path, nominal_capacity=None):
        df = self.read_one_csv(path, nominal_capacity)
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        x1, x2 = x[:-1], x[1:]
        y1, y2 = y[:-1], y[1:]
        return (x1, x2, y1.reshape(-1, 1), y2.reshape(-1, 1))

def get_lishen_test_loaders(data_root, batch_size=512, nominal_capacity=1.5):
    """
    Parses the LISHEN directory, groups by temperature, 
    and returns DataLoaders specifically for the test set.
    """
    loader_utils = EvalDataLoader()
    lishen_conditions = {}
    
    for dir_name in sorted(os.listdir(data_root)):
        full_subdir_path = os.path.join(data_root, dir_name)
        match = re.match(r'完成_力神-(\d+\.?\d*)-(\d+\.?\d*)C-#(\d+)', dir_name)
        
        if match and os.path.isdir(full_subdir_path):
            temp, c_rate, num = match.groups()
            target_file = os.path.join(full_subdir_path, 'feature_summary_trimmed_filtered.csv')
            if os.path.exists(target_file):
                condition_key = temp 
                if condition_key not in lishen_conditions: 
                    lishen_conditions[condition_key] = []
                lishen_conditions[condition_key].append((int(num), target_file))
                
    test_loaders = {}
    client_idx = 0
    client_temps = {}
    
    for temp, file_group in sorted(lishen_conditions.items()):
        file_group.sort()
        file_paths = [path for _, path in file_group]
        
        # The last file in the sorted group is treated as the test set
        test_list = [file_paths[-1]] if len(file_paths) > 1 else []
        
        if test_list:
            test_parts = [loader_utils.load_one_battery(p, nominal_capacity) for p in test_list]
            x1 = np.concatenate([d[0] for d in test_parts])
            x2 = np.concatenate([d[1] for d in test_parts])
            y1 = np.concatenate([d[2] for d in test_parts])
            y2 = np.concatenate([d[3] for d in test_parts])
            
            tensors = [torch.from_numpy(d).float() for d in (x1, x2, y1, y2)]
            test_loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=False)
            
            test_loaders[client_idx] = test_loader
            client_temps[client_idx] = float(temp)
            client_idx += 1
            
    return test_loaders, client_temps, client_idx
