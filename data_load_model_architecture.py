import torch
from model import GlucoseRCNN # <--- Import from model.py
from sklearn.preprocessing import StandardScaler
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

# --- 1. THE DATASET CLASS ---

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

class GlucoseDataset(Dataset):
    def __init__(self, ppg_signals, tabular_data, targets):
        # ppg_signals: (N, 6, 1920)
        # tabular_data: (N, 6, 4)
        # targets: (N,)
        self.ppg = torch.FloatTensor(ppg_signals)
        self.tab = torch.FloatTensor(tabular_data)
        self.y = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.ppg[idx], self.tab[idx], self.y[idx]

class GlucoseRCNN(nn.Module):
    def __init__(self, tab_dim):
        super(GlucoseRCNN, self).__init__()
        # 1D CNN for PPG morphology
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16) 
        )
        
        # LSTM for temporal metabolic trend
        self.lstm = nn.LSTM(input_size=512 + tab_dim, hidden_size=128, 
                            num_layers=2, batch_first=True, dropout=0.2)
        
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, ppg_seq, tab_seq):
        batch_size, seq_len, sig_len = ppg_seq.size()
        ppg_reshaped = ppg_seq.view(batch_size * seq_len, 1, sig_len)
        cnn_feats = self.cnn(ppg_reshaped) 
        cnn_feats = cnn_feats.view(batch_size, seq_len, -1)
        
        combined = torch.cat((cnn_feats, tab_seq), dim=2)
        lstm_out, _ = self.lstm(combined)
        
        # Output is (batch_size, 1)
        return self.regressor(lstm_out[:, -1, :])
