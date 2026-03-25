import numpy as np
import pandas as pd
import pywt
from datetime import timedelta
import os



class GlucosePreprocessor:
    def __init__(self, sampling_rate=64, window_seconds=30):
        self.fs = sampling_rate
        self.win_len = sampling_rate * window_seconds # 1920 samples

    def denoise_signal(self, data):
        """Applies Discrete Wavelet Transform (DWT) denoising."""
        # Using Daubechies 4 wavelet as it mimics the PPG pulse shape
        wav_coeffs = pywt.wavedec(data, 'db4', level=4)
        # Soft thresholding of high-frequency coefficients
        threshold = 0.02 * np.max(np.abs(wav_coeffs[-1]))
        wav_coeffs[1:] = [pywt.threshold(i, value=threshold, mode='soft') for i in wav_coeffs[1:]]
        clean_sig = pywt.waverec(wav_coeffs, 'db4')
        return clean_sig[:self.win_len]

    def cob_window(self, food_data, current_time, decay_constant=0.015):
        """Calculates Carbs-on-Board using exponential decay."""
        if food_data is None or food_data.empty:
            return 0.0
        
        # Ensure timestamps are datetime objects
        food_data['timestamp'] = pd.to_datetime(food_data['timestamp'])
        current_time = pd.to_datetime(current_time)
        
        past_meals = food_data[food_data['timestamp'] <= current_time]
        cob = 0.0
        for _, meal in past_meals.iterrows():
            minutes_ago = (current_time - meal['timestamp']).total_seconds() / 60
            cob += meal['carbs'] * np.exp(-decay_constant * minutes_ago)
        return cob
# --- 3. THE FULL PROCESSING FUNCTION ---

    def process_subject_for_dl(self,base_path, sid, hba1c):
        print(f"Deep Processing Subject {sid}...")
        sub_path = os.path.join(base_path, sid)
        
        try:
            cgm = pd.read_csv(os.path.join(sub_path, f"Dexcom_{sid}.csv"))
            # In some versions of this dataset, files are in the root or have slightly different names
            bvp = pd.read_csv(os.path.join(sub_path, f"BVP_{sid}.csv"))
            food = pd.read_csv(os.path.join(sub_path, f"Food_Log_{sid}.csv"))
        except FileNotFoundError:
            print(f"  Files missing for {sid}, skipping...")
            return None, None, None

        # Clean CGM
        cgm = cgm[cgm['Event Type'] == 'EGV'].copy()
        cgm['dt'] = pd.to_datetime(cgm['Timestamp (YYYY-MM-DDThh:mm:ss)'], format='mixed')
        cgm = cgm.dropna(subset=['Glucose Value (mg/dL)']).sort_values('dt')
        
        # Clean BVP
        bvp['dt'] = pd.to_datetime(bvp['datetime'], format='mixed')
        bvp_sig = bvp.iloc[:, 1].values 
        bvp_times = bvp['dt'].values.astype('datetime64[ns]')
        
        # Clean Food
        food['dt_str'] = food['date'].astype(str) + ' ' + food['time'].astype(str)
        food['datetime'] = pd.to_datetime(food['dt_str'], format='mixed')
        food['total_carb'] = pd.to_numeric(food['total_carb'], errors='coerce').fillna(0)
        S_BVP = 64 # BVP Frequency
        SEQ_LEN = 6 # 30 minutes of history (6 * 5-min intervals)
        BVP_WINDOW_SEC = 30 # Use 30s of PPG for morphology at each step
        ppg_sequences, tab_sequences, targets = [], [], []

        for i in range(SEQ_LEN, len(cgm)):
            t_now = cgm.iloc[i]['dt']
            
            # --- FIXED LINE BELOW: Changed m=2 to minutes=2 ---
            t_fut = t_now + pd.Timedelta(minutes=30)
            fut_val = cgm[(cgm['dt'] >= t_fut - pd.Timedelta(minutes=2)) & 
                        (cgm['dt'] <= t_fut + pd.Timedelta(minutes=2))]
            
            if fut_val.empty: continue
            target_val = fut_val.iloc[0]['Glucose Value (mg/dL)']

            current_ppg_seq, current_tab_seq = [], []
            valid_seq = True
            
            for j in range(SEQ_LEN - 1, -1, -1): 
                t_step = t_now - pd.Timedelta(minutes=j*5)
                
                # Find the closest CGM reading for this time step
                step_data = cgm[(cgm['dt'] >= t_step - pd.Timedelta(minutes=2)) & 
                                (cgm['dt'] <= t_step + pd.Timedelta(minutes=2))]
                
                if step_data.empty:
                    valid_seq = False
                    break
                    
                glc_step = step_data.iloc[0]['Glucose Value (mg/dL)']
                cob_step = self.cob_window(food, t_step)
                hr_sin = np.sin(2 * np.pi * t_step.hour / 24)
                
                current_tab_seq.append([glc_step, cob_step, hr_sin, hba1c])

                # PPG Segment extraction
                t_start_bvp = t_step - pd.Timedelta(seconds=BVP_WINDOW_SEC)
                idx_start = np.searchsorted(bvp_times, t_start_bvp.to_datetime64())
                idx_end = np.searchsorted(bvp_times, t_step.to_datetime64())
                
                seg = bvp_sig[idx_start:idx_end]
                if len(seg) < (FS_BVP * BVP_WINDOW_SEC * 0.8): # Tolerance for slight gaps
                    valid_seq = False
                    break
                
                # Standardization/Padding
                if len(seg) != 1920:
                    seg = np.interp(np.linspace(0, 1, 1920), np.linspace(0, 1, len(seg)), seg)
                
                current_ppg_seq.append(self.denoise_signal(seg))

            if valid_seq:
                ppg_sequences.append(current_ppg_seq)
                tab_sequences.append(current_tab_seq)
                targets.append(target_val)

        return np.array(ppg_sequences), np.array(tab_sequences), np.array(targets)
