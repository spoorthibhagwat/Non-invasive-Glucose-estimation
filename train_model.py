import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from constants import *
# --- 1. THE DATASET CLASS ---

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from preprocess import GlucosePreprocessor
from data_load_model_architecture import GlucoseDataset
from data_load_model_architecture import GlucoseRCNN


# --- 1. CONFIGURATION ---
DATA_ROOT = r"D:\PhysioNet\big-ideas-glycemic-wearable"
SUBJECT_IDS = [f"{i:03d}" for i in range(1, 17)]
FS_BVP = 64 # BVP Frequency
SEQ_LEN = 6 # 30 minutes of history (6 * 5-min intervals)
BVP_WINDOW_SEC = 30 # Use 30s of PPG for morphology at each step
demographics = pd.read_csv(r"D:\PhysioNet\big-ideas-glycemic-wearable\Demographics.csv").set_index("ID")
Glc_processor=GlucosePreprocessor()
data_load=GlucoseDataset()




all_ppg, all_tab, all_y = [], [], []
for sid in SUBJECT_IDS:
    # Get HbA1c from your demographics df
    #h = 5.7 # Placeholder - use demographics.loc[int(sid), "HbA1c"]
    h = demographics.loc[int(sid), "HbA1c"] 
    p, t, y = GlucosePreprocessor.process_subject_for_dl(DATA_ROOT, sid, h)
    if p is not None:
        all_ppg.append(p); all_tab.append(t); all_y.append(y)

X_ppg = np.concatenate(all_ppg)
X_tab = np.concatenate(all_tab)
Y = np.concatenate(all_y)

# 2. Scale features (Critical for NNs)
# Note: You should scale tabular features per column
scaler = StandardScaler()
X_tab_flat = X_tab.reshape(-1, X_tab.shape[-1])
#X_tab_scaled = scaler.fit_transform(X_tab_flat).reshape(X_tab.shape)

X_tab_scaled = scaler.fit_transform(X_tab_flat).reshape(X_tab.shape)
dataset = GlucoseDataset(X_ppg, X_tab_scaled, Y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

def train_and_evaluate(X_ppg, X_tab, Y, epochs=50, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 80/20 Split
    indices = np.arange(len(Y))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    full_dataset = GlucoseDataset(X_ppg, X_tab, Y)
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=batch_size, shuffle=False)

    model = GlucoseRCNN(tab_dim=X_tab.shape[2]).to(device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_r2 = -np.inf

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for ppg, tab, targets in train_loader:
            ppg, tab, targets = ppg.to(device), tab.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(ppg, tab)
            
            # Use .view(-1) to ensure shape matches targets [batch_size]
            loss = criterion(outputs.view(-1), targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for ppg, tab, targets in val_loader:
                ppg, tab, targets = ppg.to(device), tab.to(device), targets.to(device)
                outputs = model(ppg, tab)
                
                # FIX: .view(-1) prevents 0-d array errors on small batches
                val_preds.extend(outputs.view(-1).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        current_r2 = r2_score(val_targets, val_preds)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d} | Loss: {np.mean(train_losses):.4f} | Val R2: {current_r2:.4f}")

        if current_r2 > best_r2:
            best_r2 = current_r2
            torch.save(model.state_dict(), 'best_model.pth')

    print(f"\nBest R2 achieved: {best_r2:.4f}")
    return model
scaler = StandardScaler()
N, S, F = X_tab.shape
X_tab_scaled = scaler.fit_transform(X_tab.reshape(-1, F)).reshape(N, S, F)

# 2. Run
model = train_and_evaluate(
    X_ppg, 
    X_tab_scaled, 
    Y, 
    epochs=100, 
    batch_size=64, 
    lr=0.0005
)