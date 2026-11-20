import os
import json
import pandas as pd
import numpy as np
import zipfile
import inflate64

class Deflate64Decompressor:
    def __init__(self): self._inflater = inflate64.Inflater()
    @property
    def eof(self): return self._inflater.eof
    def decompress(self, data): return self._inflater.inflate(data)

orig_g = zipfile._get_decompressor
orig_c = zipfile._check_compression
zipfile._get_decompressor = lambda c: Deflate64Decompressor() if c == 9 else orig_g(c)
zipfile._check_compression = lambda c: None if c == 9 else orig_c(c)

def extract_dataset(meta_csv, zip_path, speed_col, dv_key, out_csv):
    print(f"Loading metadata from {meta_csv}...")
    df_meta = pd.read_csv(meta_csv)
    
    for col in ['SpecInjury', 'InjSide', 'Gender']:
        if col in df_meta.columns:
            df_meta[col] = df_meta[col].astype(str).str.lower().str.strip()

    df_meta = df_meta.dropna(subset=['Weight', 'Height', speed_col, 'Gender', 'age'])
    
    inj_masks = {
        0: df_meta['SpecInjury'].str.contains('patellofemoral|pfps', na=False),
        1: df_meta['SpecInjury'].str.contains('itb syndrome', na=False),
        2: df_meta['SpecInjury'].str.contains('achilles tendonitis', na=False),
        3: df_meta['SpecInjury'].str.contains('plantar fasciitis', na=False),
        4: df_meta['SpecInjury'].str.contains('medial tibial stress syndrome|shin splints', na=False)
    }
    
    df_meta['Target'] = -1
    for t_val, mask in inj_masks.items():
        df_meta.loc[mask, 'Target'] = t_val
        
    df_filtered = df_meta[df_meta['Target'] != -1].copy()
    print(f"Found {len(df_filtered)} valid records.")
    
    valid_filenames = set(df_filtered['filename'].dropna())
    filename_to_meta = df_filtered.set_index('filename')

    features = []
    
    print(f"Extracting discrete variables from {zip_path} ({dv_key})...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        for item in z.infolist():
            if not item.filename.endswith('.json'): continue
            basename = os.path.basename(item.filename)
            if basename not in valid_filenames: continue
            
            try:
                data = json.loads(z.read(item.filename))
            except Exception as e:
                continue
                
            meta = filename_to_meta.loc[basename]
            if isinstance(meta, pd.DataFrame): meta = meta.iloc[0]
            
            row = {
                'sub_id': meta['sub_id'],
                'Target': meta['Target'],
                'Group': {0: 'PFPS', 1: 'ITBS', 2: 'Achilles', 3: 'Plantar', 4: 'MTSS'}[meta['Target']],
                'Age': meta['age'],
                'Mass': meta['Weight'],
                'Height': meta['Height'],
                'Sex': 1 if meta['Gender'] == 'f' else 0,
                'Speed': meta[speed_col]
            }
            
            side = meta['InjSide']
            if side == 'right': sides_to_process = ['right']
            elif side == 'left': sides_to_process = ['left']
            else: sides_to_process = ['right', 'left']
            
            if dv_key not in data: continue
            
            extracted_kin = {}
            valid = True
            for s in sides_to_process:
                if s not in data[dv_key]: 
                    valid = False; break
                for k, v in data[dv_key][s].items():
                    if v is None or np.isnan(v):
                        valid = False
                        break
                    extracted_kin.setdefault(k, []).append(v)
            
            if not valid: continue
            
            for k, v_list in extracted_kin.items():
                row[k] = np.mean(v_list)
                
            features.append(row)
            
    out_df = pd.DataFrame(features).dropna()
    os.makedirs('Supplementary_Data', exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved {len(out_df)} records to {out_csv}\n")

if __name__ == '__main__':
    #extract_dataset('Dataset/run_data_meta.csv', 'Dataset/ric_data.zip', 'speed_r', 'dv_r', 'Supplementary_Data/ml_run_dataset.csv')
    extract_dataset('walk_data_meta.csv', 'Dataset/ric_data.zip', 'speed_w', 'dv_w', 'Supplementary_Data/ml_walk_dataset.csv')
