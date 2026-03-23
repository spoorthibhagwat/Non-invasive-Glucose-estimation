import torch
import torch.nn as nn

class GlucoseRCNN(nn.Module):
    def __init__(self, tab_dim=4):
        super(GlucoseRCNN, self).__init__()
        # 1D-CNN for PPG Morphology (Arterial Stiffness features)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.AdaptiveAvgPool1d(64)
        )
        # LSTM for Temporal Metabolic Momentum
        self.lstm = nn.LSTM(input_size=64 + tab_dim, hidden_size=128, 
                            num_layers=2, batch_first=True)
        self.regressor = nn.Linear(128, 1)

    def forward(self, ppg, tab):
        batch_size, seq_len, signal_len = ppg.size()
        # Process each PPG window in the sequence through CNN
        ppg = ppg.view(batch_size * seq_len, 1, signal_len)
        features = self.cnn(ppg).view(batch_size, seq_len, -1)
        # Concatenate PPG features with Tabular (COB, HbA1c, etc.)
        combined = torch.cat((features, tab), dim=2)
        out, _ = self.lstm(combined)
        return self.regressor(out[:, -1, :]) # Predict only the last step