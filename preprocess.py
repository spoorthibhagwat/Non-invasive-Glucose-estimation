import numpy as np
import pandas as pd
import pywt
from datetime import timedelta

class GlucosePreprocessor:
    def __init__(self, sampling_rate=64, window_seconds=30):
        self.fs = sampling_rate
        self.win_len = sampling_rate * window_seconds # 1920 samples

    def denoise_signal(self, data):
        """Applies Discrete Wavelet Transform (DWT) denoising."""
        # Using Daubechies 4 wavelet as it mimics the PPG pulse shape
        coeffs = pywt.wavedec(data, 'db4', level=4)
        # Soft thresholding of high-frequency coefficients
        threshold = 0.02 * np.max(np.abs(coeffs[-1]))
        coeffs[1:] = [pywt.threshold(i, value=threshold, mode='soft') for i in coeffs[1:]]
        clean_sig = pywt.waverec(coeffs, 'db4')
        return clean_sig[:self.win_len]

    def calculate_cob(self, food_df, current_time, decay_constant=0.015):
        """Calculates Carbs-on-Board using exponential decay."""
        if food_df is None or food_df.empty:
            return 0.0
        
        # Ensure timestamps are datetime objects
        food_df['timestamp'] = pd.to_datetime(food_df['timestamp'])
        current_time = pd.to_datetime(current_time)
        
        past_meals = food_df[food_df['timestamp'] <= current_time]
        cob = 0.0
        for _, meal in past_meals.iterrows():
            minutes_ago = (current_time - meal['timestamp']).total_seconds() / 60
            cob += meal['carbs'] * np.exp(-decay_constant * minutes_ago)
        return cob

    def create_sequences(self, bvp_df, cgm_df, food_df, hba1c, seq_steps=6):
        """
        Creates (PPG, Tabular) pairs.
        seq_steps=6 creates a 30-minute lookback (6 steps * 5 mins).
        """
        X_ppg, X_tab, Y = [], [], []
        
        # Ensure sorted by time
        cgm_df = cgm_df.sort_values('timestamp')
        
        for i in range(seq_steps, len(cgm_df)):
            current_row = cgm_df.iloc[i]
            t_now = pd.to_datetime(current_row['timestamp'])
            
            seq_p, seq_t = [], []
            valid_seq = True
            
            for step in range(seq_steps-1, -1, -1):
                t_step = t_now - timedelta(minutes=step * 5)
                
                # 1. Extract 30s PPG Window
                mask = (bvp_df['timestamp'] > t_step - timedelta(seconds=30)) & \
                       (bvp_df['timestamp'] <= t_step)
                ppg_win = bvp_df.loc[mask, 'value'].values
                
                if len(ppg_win) < self.win_len * 0.95: # Allow 5% missingness
                    valid_seq = False; break
                
                # Resample or pad to exact length
                if len(ppg_win) != self.win_len:
                    ppg_win = np.interp(np.linspace(0, 1, self.win_len), 
                                        np.linspace(0, 1, len(ppg_win)), ppg_win)
                
                seq_p.append(self.denoise_signal(ppg_win))
                
                # 2. Extract Tabular Features
                cob = self.calculate_cob(food_df, t_step)
                h_sin = np.sin(2 * np.pi * t_step.hour / 24)
                # Note: We use the CGM value at t_step as an input feature (lag)
                # and the CGM value at t_now as the target Y.
                glc_lag = cgm_df.iloc[i - step]['glucose']
                
                seq_t.append([glc_lag, cob, h_sin, hba1c])
                
            if valid_seq:
                X_ppg.append(seq_p)
                X_tab.append(seq_t)
                Y.append(current_row['glucose'])
                
        return np.array(X_ppg), np.array(X_tab), np.array(Y)