import numpy as np

# Your exact counts from the logs
subject_counts = {
    '001': 1996, '002': 1816, '003': 1345, '004': 1280,
    '005': 2286, '006': 1589, '007': 1995, '008': 2038,
    '009': 2028, '010': 1949, '011': 2084, '012': 1872,
    '013': 1811, '014': 1511, '015': 464,  '016': 1778
}

# Construct the array in order
mapping_list = []
for sub_id, count in subject_counts.items():
    mapping_list.extend([sub_id] * count)

true_subject_mapping = np.array(mapping_list)

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


def run_loso_with_ablation(X_ppg, X_tab, Y, subjects_array):
    # Ensure subjects_array is a numpy array of strings to match your data
    subjects_array = np.array(subjects_array).astype(str)
    unique_subs = np.unique(subjects_array)
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Starting LOSO with Ablation for {len(unique_subs)} subjects...")

    for test_sub in unique_subs:
        # Create masks with string-safe comparison
        test_mask = (subjects_array == str(test_sub))
        train_mask = (subjects_array != str(test_sub))
        
        num_test = np.sum(test_mask)
        if num_test == 0:
            print(f"  - Subject {test_sub}: No samples found, skipping.")
            continue

        # Prepare Fold Data
        train_ds = GlucoseDataset(X_ppg[train_mask], X_tab[train_mask], Y[train_mask])
        test_ds = GlucoseDataset(X_ppg[test_mask], X_tab[test_mask], Y[test_mask])
        
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

        # Initialize and Train
        model = GlucoseRCNN(tab_dim=X_tab.shape[2]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.HuberLoss()

        # Fold Training
        model.train()
        for epoch in range(20):
            for p, t, y_b in train_loader:
                p, t, y_b = p.to(device), t.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss = criterion(model(p, t).view(-1), y_b)
                loss.backward()
                optimizer.step()

        # Fold Evaluation
        model.eval()
        full_preds, ablated_preds, actuals = [], [], []
        
        with torch.no_grad():
            for p, t, y_b in test_loader:
                p, t = p.to(device), t.to(device)
                
                # 1. Full Context
                out_full = model(p, t)
                full_preds.extend(out_full.view(-1).cpu().numpy())
                
                # 2. Ablated (Zero out Carbs Index 1 and HbA1c Index 3)
                t_ablated = t.clone()
                t_ablated[:, :, [1, 3]] = 0 
                out_ablated = model(p, t_ablated)
                ablated_preds.extend(out_ablated.view(-1).cpu().numpy())
                
                actuals.extend(y_b.numpy())

        # Metrics
        actuals_np = np.array(actuals)
        m_full = np.mean(np.abs(actuals_np - np.array(full_preds)) / actuals_np) * 100
        m_ablated = np.mean(np.abs(actuals_np - np.array(ablated_preds)) / actuals_np) * 100
        r2_full = r2_score(actuals_np, full_preds)

        results.append({
            'Subject': test_sub,
            'Proposed_MARD': m_full,
            'Ablated_MARD': m_ablated,
            'Proposed_R2': r2_full
        })
        print(f"  - Sub {test_sub} | Prop MARD: {m_full:.2f}% | Ablated MARD: {m_ablated:.2f}%")

    if not results:
        print("CRITICAL ERROR: No subjects were processed. Check subjects_array content.")
        return None

    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print(f"FINAL MEAN PROPOSED MARD: {df['Proposed_MARD'].mean():.2f}%")
    print(f"FINAL MEAN ABLATED MARD:  {df['Ablated_MARD'].mean():.2f}%")
    print("="*50)
    return df