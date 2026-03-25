import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_rel

# --- 1. MODEL TESTING LOOP ---
def test_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for signals, meta_data, targets in dataloader:
            signals, meta_data = signals.to(device), meta_data.to(device)
            outputs = model(signals, meta_data)
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.numpy())
            
    return np.array(all_preds).flatten(), np.array(all_targets).flatten()

# --- 2. CLARKE ERROR GRID VISUALIZATION ---
def plot_clarke_error_grid(ref, pred, title="Clarke Error Grid Analysis"):
    """
    Standard IEEE JBHI clinical safety visualization.
    Zones: A (Accurate), B (Acceptable), C (Overcorrecting), D (Failure to Detect), E (Erroneous)
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(ref, pred, marker='o', color='blue', s=8, alpha=0.5)
    plt.title(title, fontsize=14)
    plt.xlabel("Reference Glucose (mg/dL)", fontsize=12)
    plt.ylabel("Estimated Glucose (mg/dL)", fontsize=12)
    
    # Grid Boundaries (Standard Equations)
    plt.plot([0, 400], [0, 400], 'k-', alpha=0.3) # Diagonal
    # Zone A boundaries
    plt.plot([0, 175/1.2], [0, 175], 'k--', alpha=0.3)
    plt.plot([175/1.2, 400], [175, 400], 'k--', alpha=0.3)
    # Add Zone Labels (Simplified for plot logic)
    plt.text(350, 380, "A", fontsize=15, fontweight='bold')
    plt.text(350, 250, "B", fontsize=15, fontweight='bold')
    
    plt.xlim(0, 400)
    plt.ylim(0, 400)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

# --- 3. CUMULATIVE DISTRIBUTION FUNCTION (CDF) ---
def plot_error_cdf(ref, pred):
    """
    Plots the CDF of Absolute Error to confirm the 9.10 mg/dL 90th percentile.
    """
    abs_error = np.abs(ref - pred)
    sorted_error = np.sort(abs_error)
    cdf = np.arange(len(sorted_error)) / float(len(sorted_error))
    
    # Calculate 90th percentile
    p90_error = np.percentile(abs_error, 90)
    
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_error, cdf, color='green', lw=2, label="Estimation Error CDF")
    plt.axhline(0.9, color='red', linestyle='--', alpha=0.7, label="90th Percentile")
    plt.axvline(p90_error, color='red', linestyle='--', alpha=0.7)
    
    plt.title("Cumulative Distribution of Absolute Errors", fontsize=14)
    plt.xlabel("Absolute Error (mg/dL)", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.text(p90_error + 1, 0.4, f"P90 = {p90_error:.2 f} mg/dL", color='red')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# --- 4. MARD CALCULATION ---
def calculate_metrics(ref, pred):
    mard = np.mean(np.abs((ref - pred) / ref)) * 100
    print(f"--- Final Evaluation Results ---")
    print(f"Mean Absolute Relative Difference (MARD): {mard:.2 f}%")
    print(f"Total Samples Tested: {len(ref)}")
def plot_bland_altman(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mean = (y_true + y_pred) / 2
    diff = y_pred - y_true  # Prediction Error
    md = np.mean(diff)      # Mean Bias
    sd = np.std(diff, axis=0) # Standard Deviation of Error

    plt.figure(figsize=(10, 6))
    plt.scatter(mean, diff, alpha=0.5, color='royalblue')
    plt.axhline(md, color='red', linestyle='--', label=f'Mean Bias: {md:.2f}')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--', label=f'+1.96 SD: {md + 1.96*sd:.2f}')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--', label=f'-1.96 SD: {md - 1.96*sd:.2f}')
    
    plt.title("Bland-Altman Plot: Agreement Analysis")
    plt.xlabel("Mean of Actual and Predicted (mg/dL)")
    plt.ylabel("Difference (Predicted - Actual) (mg/dL)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.show()

# Run the plot
#plot_bland_altman(Y_true, preds)
# --- EXECUTION ---
# y_pred, y_ref = test_model(model, test_loader, device)
# calculate_metrics(y_ref, y_pred)
# plot_clarke_error_grid(y_ref, y_pred)
# plot_error_cdf(y_ref, y_pred)