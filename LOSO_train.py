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

def run_loso_validation(X_ppg, X_tab, Y, subjects_array):
    unique_subs = np.unique(subjects_array)
    all_r2, all_mard = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for test_sub in unique_subs:
        # Create masks - ensure both are treated as strings to avoid match errors
        train_mask = (subjects_array != test_sub)
        test_mask = (subjects_array == test_sub)
        
        if np.sum(test_mask) == 0:
            print(f"  Skipping {test_sub}: Mask returned zero samples.")
            continue

        # Prepare Fold Data
        train_ds = GlucoseDataset(X_ppg[train_mask], X_tab[train_mask], Y[train_mask])
        test_ds = GlucoseDataset(X_ppg[test_mask], X_tab[test_mask], Y[test_mask])
        
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

        # Initialize fresh model for zero-bias testing
        model = GlucoseRCNN(tab_dim=X_tab.shape[2]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.HuberLoss()

        # Fold Training (20 epochs is usually enough to see subject-level trend)
        for epoch in range(20):
            model.train()
            for p, t, y_b in train_loader:
                p, t, y_b = p.to(device), t.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss = criterion(model(p, t).view(-1), y_b)
                loss.backward()
                optimizer.step()

        # Fold Evaluation
        model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for p, t, y_b in test_loader:
                out = model(p.to(device), t.to(device))
                preds.extend(out.view(-1).cpu().numpy())
                actuals.extend(y_b.numpy())

        r2 = r2_score(actuals, preds)
        mard = np.mean(np.abs(np.array(actuals) - np.array(preds)) / np.array(actuals)) * 100
        
        all_r2.append(r2)
        all_mard.append(mard)
        print(f"Subject {test_sub} Results -> R2: {r2:.4f}, MARD: {mard:.2f}%")

    print(f"\nFINAL LOSO MEAN R2: {np.mean(all_r2):.4f} (+/- {np.std(all_r2):.4f})")
    print(f"FINAL LOSO MEAN MARD: {np.mean(all_mard):.2f}%")

# Execute
#run_loso_validation(X_ppg, X_tab, Y, S)