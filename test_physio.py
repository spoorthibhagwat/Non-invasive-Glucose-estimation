import torch
import joblib
import pandas as pd
from model import GlucoseRCNN # <--- Reusing the same brain
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

def run_external_test(subject_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load System
    scaler = joblib.load('glucose_scaler.joblib')
    model = GlucoseRCNN(tab_dim=4).to(device)
    model.load_state_dict(torch.load('glucose_rcnn_v1.pth', map_location=device))
    model.eval()
    
    # 2. Load & Pre-process PhysioCGM Data
    # [Insert the PhysioCGM processing logic we wrote earlier]
    # ...
    
    print(f"Results for {subject_path}: R2={r2_score_val:.4f}, MARD={mard_val:.2f}%")

if __name__ == "__main__":
    run_external_test('physiocgm/c1s01')