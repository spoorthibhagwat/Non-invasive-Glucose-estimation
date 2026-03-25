# --- 1. CONFIGURATION ---
DATA_ROOT = r"D:\PhysioNet\big-ideas-glycemic-wearable"
SUBJECT_IDS = [f"{i:03d}" for i in range(1, 17)]
FS_BVP = 64 # BVP Frequency
SEQ_LEN = 6 # 30 minutes of history (6 * 5-min intervals)
BVP_WINDOW_SEC = 30 # Use 30s of PPG for morphology at each step