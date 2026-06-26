import os
import glob
import time
import tqdm
import h5py
import shutil
import parmap
import fnmatch
import argparse
import matplotlib
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.figure_factory as ff
from rpnet import *
from hyperparams import *
from datetime import datetime
from obspy import UTCDateTime
from obspy import Stream, Trace
from keras_self_attention import SeqSelfAttention
from sklearn.model_selection import train_test_split
from mpl_toolkits.axes_grid1 import make_axes_locatable
from obspy import read

# -----------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------
def get_add(args):
    z, pha_set, cat_ids, join_col, fptime, fstime = args
    id = z.split('/')[-2]
    sta = z.split('/')[-1].split('.')[0]
    if (id, sta) in pha_set:
        return pd.DataFrame()
    if id not in cat_ids:
        return pd.DataFrame()
    return pd.DataFrame({join_col: [id], 'sta': [sta], fptime: [np.nan], fstime: [np.nan]})

def est_taup_2(vals):
    idx, val, cat, ftime, pha, model, keep_initial_phase, surface_offset_km = vals

    model = TauPyModel(model=model)

    if pha == 'P':
        if keep_initial_phase and pd.notnull(val['ptime0']):
            return UTCDateTime(val['ptime0'])
        else:
            try:
                arrivals = model.get_travel_times_geo(
                    cat.dep, #+ surface_offset_km,
                    cat.lat, cat.lon,
                    val.lat, val.lon,
                    phase_list=['P', 'Pg', 'Pn', 'p']
                )
                if not arrivals:
                    # Fallback: devolver el ptime original si existe
                    return UTCDateTime(val['ptime0']) if pd.notnull(val['ptime0']) else None
                est = min(arrivals, key=lambda a: a.time).time
                return UTCDateTime(cat[ftime]) + est
            except Exception as e:
                print(f"  est_taup_2 P error (dep={cat.dep:.1f}, lat={cat.lat:.3f}, lon={cat.lon:.3f}): {e}")
                return UTCDateTime(val['ptime0']) if pd.notnull(val['ptime0']) else None

    elif pha == 'S':
        if keep_initial_phase and pd.notnull(val['stime0']):
            return UTCDateTime(val['stime0'])
        else:
            try:
                arrivals = model.get_travel_times_geo(
                    cat.dep, #+ surface_offset_km,
                    cat.lat, cat.lon,
                    val.lat, val.lon,
                    phase_list=['S', 'Sg', 'Sn', 's']
                )
                if not arrivals:
                    return UTCDateTime(val['stime0']) if pd.notnull(val['stime0']) else None
                est = min(arrivals, key=lambda a: a.time).time
                return UTCDateTime(cat[ftime]) + est
            except Exception as e:
                print(f"  est_taup_2 S error (dep={cat.dep:.1f}, lat={cat.lat:.3f}, lon={cat.lon:.3f}): {e}")
                return UTCDateTime(val['stime0']) if pd.notnull(val['stime0']) else None

def preprocess_2(p_time, s_time, stream_path, sp_win, low_freq=1.0, high_freq=20.0, taper_pct=0.01):
    """
    Loading and preprocessing the stream data.
    """
    st = read(stream_path)
    st.trim(starttime=p_time + sp_win[0] - 5, endtime=s_time + sp_win[-1] + 10)
    st.filter('bandpass', freqmin=low_freq, freqmax=high_freq)
    if st[0].stats.sampling_rate != 100:
        st.resample(100)
    st.normalize(global_max=True)
    st.detrend('demean')
    st.detrend('linear')
    st.taper(taper_pct)
    st.sort()
    for tr in st:
        tr.data *= 1e3
    return st

def calc_amplitude_2(p_time, s_time, station, st, sp_win):
    if len(st) < 2:
        return "None"
    try:
        n_vals, p_vals, s_vals = [], [], []
        for trace in st:
            n_win = trace.slice(p_time + sp_win[0], p_time + sp_win[1]).copy()
            p_win = trace.slice(p_time + sp_win[2], p_time + sp_win[3]).copy()
            s_win = trace.slice(s_time + sp_win[4], s_time + sp_win[5]).copy()
            
            # Skip empty windows
            if len(n_win) == 0 or len(p_win) == 0 or len(s_win) == 0:
                continue
                
            n_vals.append(max(n_win.data) - min(n_win.data))
            p_vals.append(max(p_win.data) - min(p_win.data))
            s_vals.append(max(s_win.data) - min(s_win.data))
        
        if not p_vals:  # No valid windows
            return "None"
            
        N = np.sqrt(sum(val ** 2 for val in n_vals))
        P = np.sqrt(sum(val ** 2 for val in p_vals))
        S = np.sqrt(sum(val ** 2 for val in s_vals))
        sp_ratio = S / P if P != 0 else float('inf')
        
        return f"{station.sta:4s} {station.chan:3s} {station.net:2s} {0.0:4.1f} {0.0:4.1f} {N:18.3f} {N:10.3f} {P:10.3f} {S:10.3f}"
    
    except Exception as e:
        return "None"

def prepare_amplitudes_2(params):
    station_df, pick_id, event_info, data_dir, sp_freq, sp_win = params
    station_df = station_df.drop_duplicates(subset=['sta0'])
    station_df = station_df[station_df['ptime'].notnull() & station_df['stime'].notnull()].reset_index(drop=True)
    
    amp_list = []
    for _, row in station_df.iterrows():
        p_time = UTCDateTime(row.ptime)
        s_time = UTCDateTime(row.stime)
        stream_path = f"{data_dir}/{pick_id}/{row.sta0}.*"
        
        try:
            st = preprocess_2(p_time, s_time, stream_path, sp_win, sp_freq[0], sp_freq[1])
        except Exception as e:
            # File not found or stream error — skip this station silently
            continue
        
        amp_str = calc_amplitude_2(p_time, s_time, row, st, sp_win)
        if amp_str != "None":
            amp_list.append(amp_str)
    
    amp_list.sort()
    header = f"{event_info} {len(amp_list)}"
    amp_list.insert(0, header)
    
    return amp_list

def prep_skhash_2(cat_df, pol_df, amp, sta_df, out_dir, ftime, fwfid, ctrl0, hash_version='hash2'):

    if os.path.exists(out_dir + '/' + hash_version):
        shutil.rmtree(out_dir + '/' + hash_version)
    os.makedirs(out_dir + '/' + hash_version + '/IN')
    os.makedirs(out_dir + '/' + hash_version + '/OUT')

    cat_df = cat_df.sort_values([fwfid]).reset_index(drop=True)
    sta_df = sta_df.sort_values(['sta']).reset_index(drop=True)

    pol_df.to_csv(out_dir + '/' + hash_version + '/uniq_pol.csv', index=False)

    # --- Station file (SKHASH CSV format) ---
    with open(out_dir + '/' + hash_version + '/IN/station.csv', 'w') as f:
        f.write('station,network,location,channel,latitude,longitude,elevation,start_time,end_time\n')
        for _, val in sta_df.iterrows():
            f.write(f"{val.sta},{val.net},,{val.chan},{val.lat:.5f},{val.lon:.5f},{int(val.elv)},1900-01-01,3000-01-01\n")

    # --- Polarity file (SKHASH CSV format) ---
    with open(out_dir + '/' + hash_version + '/IN/phase.csv', 'w') as f:
        f.write('event_id,station,network,location,channel,p_polarity,origin_latitude,origin_longitude,origin_depth_km\n')
        for _, val in cat_df.iterrows():
            s_df = pol_df[pol_df[fwfid] == val[fwfid]].drop_duplicates(['sta']).sort_values(['sta']).reset_index(drop=True)
            for _, val2 in s_df.iterrows():
                # Convert polarity: U -> 1.0, D -> -1.0, K -> 0.0
                pol_map = {'U': 1.0, 'D': -1.0, 'K': 0.0}
                p_pol = pol_map.get(str(val2.predict)[0], 0.0)
                sta_row = sta_df[sta_df.sta0 == val2.sta].iloc[0]
                f.write(f"{val[fwfid]},{sta_row.sta},{sta_row.net},--,{sta_row.chan},{p_pol},"
                        f"{val.lat:.5f},{val.lon:.5f},{val.dep:.3f}\n")

    # --- Amplitude file (SKHASH CSV format) ---
    if hash_version == 'hash3' and amp is not None:
        with open(out_dir + '/' + hash_version + '/IN/amp.csv', 'w') as f:
            f.write('event_id,station,network,location,channel,noise_p,noise_s,amp_p,amp_s\n')
            current_event = None
            for line in amp:
                parts = line.strip().split()
                # Header line: event_id n_stations
                if len(parts) == 2 and not parts[1][0].isalpha():
                    current_event = parts[0]
                # Data line: STA chan net 0.0 0.0 N N P S
                elif len(parts) == 9 and current_event is not None:
                    sta, chan, net = parts[0], parts[1], parts[2]
                    N, P, S = float(parts[5]), float(parts[7]), float(parts[8])
                    f.write(f"{current_event},{sta},{net},--,{chan},{N},{N},{P},{S}\n")

    # --- Control file ---
    with open(out_dir + '/' + hash_version + '/control_file.txt', 'w') as f:
        f.write('## Control file for SKHASH (SKHASH format)\n\n')
        f.write('$input_format\nskhash\n\n')
        f.write('$stfile\n' + out_dir + '/' + hash_version + '/IN/station.csv\n\n')
        f.write('$fpfile\n' + out_dir + '/' + hash_version + '/IN/phase.csv\n\n')
        if hash_version == 'hash3':
            f.write('$ampfile\n' + out_dir + '/' + hash_version + '/IN/amp.csv\n\n')
        f.write('$outfile1\n' + out_dir + '/' + hash_version + '/OUT/out.csv\n\n')
        f.write('$outfile2\n' + out_dir + '/' + hash_version + '/OUT/out2.csv\n\n')
        f.write('$outfolder_plots\n' + out_dir + '/' + hash_version + '/OUT/figure\n\n')
        with open(ctrl0, 'r') as f4:
            for l in f4:
                f.write(l)

    return

def process_single_day(event_catalog, phase_metadata, sta_metadata, year=None, jday=None):
    """
    Process a single day of data for RPNet polarity picking.
    Uses original rpnet.wf2matrix with modified wf_dir structure.
    
    Parameters:
    -----------
    event_catalog : str
        Path to event catalog CSV
    phase_metadata : str
        Path to phase metadata CSV
    sta_metadata : str
        Path to station metadata CSV
    year : int, optional
        Year to filter events (e.g., 2024)
    jday : int, optional
        Day of year to filter events (1-366; e.g., 100 = April 10)
    """
    global out_dir
    
    # set gpu number
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num

    stime=time.time()

    # Create unique output directory based on date (YYYY_JDD) if year and jday are provided
    # This allows multiple days to be processed in parallel without directory conflicts
    if year is not None and jday is not None:
        out_dir = os.path.join(out_dir, f"{year}_{jday:03d}")
    
    # make output directory / if exist remove it
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)

    # load raw data (catalog, phase, station files)
    print('# Loading catalogs and metadata')
    cat_df=pd.read_csv(event_catalog)
    pha_df=pd.read_csv(phase_metadata)
    sta_df=pd.read_csv(sta_metadata).sort_values(['sta']).reset_index(drop=True)
    
    # Data type validation and cleaning
    print('# Validating and cleaning datetime columns')
    
    # Ensure fptime column in phase dataframe is string type
    if fptime in pha_df.columns:
        # Convert to string if it's numeric or datetime
        if pha_df[fptime].dtype in ['float64', 'int64']:
            print(f"  Warning: {fptime} in phase catalog is numeric type, converting to string")
            pha_df[fptime] = pha_df[fptime].astype(str)
        elif pha_df[fptime].dtype == 'object':
            # Already string/object, try to convert any NaN to string
            pha_df[fptime] = pha_df[fptime].fillna('NaT').astype(str)
        
        # Remove any 'nan' or 'NaT' string values
        invalid_mask = pha_df[fptime].isin(['nan', 'NaT', 'None', ''])
        if invalid_mask.any():
            print(f"  Dropping {invalid_mask.sum()} phase records with invalid {fptime} values")
            pha_df = pha_df[~invalid_mask].reset_index(drop=True)
    
    # Ensure ftime column in catalog is proper datetime
    if ftime in cat_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(cat_df[ftime]):
            cat_df[ftime] = pd.to_datetime(cat_df[ftime], errors='coerce')
        # Drop any rows with invalid datetime
        invalid_cat = cat_df[ftime].isna()
        if invalid_cat.any():
            print(f"  Dropping {invalid_cat.sum()} events with invalid {ftime} values")
            cat_df = cat_df[~invalid_cat].reset_index(drop=True)
    
    print('- Done')
    
    # Rename station metadata columns
    if 'ntw' in sta_df.columns:
        sta_df = sta_df.rename(columns={'ntw': 'net'})
    if 'chn' in sta_df.columns:
        sta_df = sta_df.rename(columns={'chn': 'chan'})
    
    sta_df['sta0']=sta_df['sta']
    pha_df['source']='original'

    _join_col = 'event_id' if 'event_id' in pha_df.columns else fwfid
    pha_set = set(zip(pha_df[_join_col], pha_df['sta']))
    cat_ids = set(cat_df[_join_col].to_list())

    # Construir wf_pattern desde event times (necesario para matching con directorios)
    if not pd.api.types.is_datetime64_any_dtype(cat_df[ftime]):
        cat_df[ftime] = pd.to_datetime(cat_df[ftime], errors='coerce')

    cat_df['wf_pattern'] = cat_df[ftime].apply(
        lambda t: f"{t.year}_{pd.Timestamp(t).day_of_year:03d}_{t.hour:02d}{t.minute:02d}{t.second:02d}"
    )

    # Mapeo bidireccional: wf_pattern <-> event_id
    pattern_to_id = cat_df.set_index('wf_pattern')[_join_col].to_dict()
    cat_patterns  = set(cat_df['wf_pattern'])

    # --- Bloque add_sta actualizado ---
    if add_sta:
        z_files = sorted(glob.glob(wf_dir + '/*/*.mseed'))
        print('# get list of additional stations')

        patterns = [z.split('/')[-2] for z in z_files]
        stas     = [z.split('/')[-1].split('.')[0] for z in z_files]

        all_df = pd.DataFrame({'wf_pattern': patterns, 'sta': stas})
        all_df = all_df[all_df['wf_pattern'].isin(cat_patterns)]
        all_df[_join_col] = all_df['wf_pattern'].map(pattern_to_id)
        already_picked = all_df.set_index([_join_col, 'sta']).index.isin(pha_set)
        all_df = all_df[~already_picked]
        all_df[fptime]   = np.nan
        all_df[fstime]   = np.nan
        all_df['source'] = 'add'

        print(f'  wf_pattern duplicados: {cat_df["wf_pattern"].duplicated().sum()}')
        print(f'  Added {len(all_df)} empty picks from {len(z_files)} files')

        pha_df = pd.concat([pha_df, all_df], ignore_index=True)  # <- faltaba este concat!
        sta_df = sta_df[sta_df['sta'].isin(pha_df['sta'])].reset_index(drop=True)

    # Add station metadata to phase df  (net/chan se añaden aquí)
    print('\n# Arrange metadata')
    pha_df = pha_df[pha_df['sta'].isin(sta_df['sta'])].reset_index(drop=True)
    pha_df['lat']  = [sta_df[sta_df.sta==i]['lat'].iloc[0]  for i in pha_df['sta']]
    pha_df['lon']  = [sta_df[sta_df.sta==i]['lon'].iloc[0]  for i in pha_df['sta']]
    pha_df['elv']  = [sta_df[sta_df.sta==i]['elv'].iloc[0]  for i in pha_df['sta']]
    pha_df['net']  = [sta_df[sta_df.sta==i]['net'].iloc[0]  for i in pha_df['sta']]
    pha_df['chan'] = [sta_df[sta_df.sta==i]['chan'].iloc[0]  for i in pha_df['sta']]

    # Ahora sí existen net y chan
    pha_df = pha_df.drop_duplicates([_join_col, 'net', 'sta', 'chan', fptime, fstime])
    pha_df = pha_df[pha_df[_join_col].isin(cat_df[_join_col])].reset_index(drop=True)

    # make UTCDateTime objects
    cat_df[ftime]=[UTCDateTime(i) for i in cat_df[ftime].to_list()]

    # Add data_id to phase_df
    if 'event_id' in pha_df.columns and 'data_id' not in pha_df.columns:
        print(f"# Adding data_id to phase catalog")
        event_id_to_data_id = cat_df[['event_id', 'data_id']].drop_duplicates()
        pha_df = pha_df.merge(event_id_to_data_id, on='event_id', how='left')
        dropped = pha_df['data_id'].isnull().sum()
        if dropped > 0:
            print(f"  Dropping {dropped} phases with no matching data_id")
            pha_df = pha_df.dropna(subset=['data_id']).reset_index(drop=True)
        print(f"  Phases after merge: {len(pha_df)}")

    # # Change to TauP P arrival times (OPTION; considering pick uncertainty)
    # if change2taup:
    #     print('\n\n# change to TauP arrival')
    #     pha_df['ptime0']=pha_df[fptime]
    #     results=parmap.map(est_taup,[[idx,val,cat_df[cat_df[fwfid]==val[fwfid]].iloc[0],ftime,'P',taup_model,keep_initial_phase] for idx,val in pha_df.iterrows()]
    #                     , pm_pbar=True, pm_processes=cores,pm_chunksize=1)
    #     pha_df[fptime]=results
    #     print('- TauP (P) Done')

    #     print('# change to TauP S arrival')
    #     pha_df['stime0']=pha_df[fstime]
    #     results=parmap.map(est_taup,[[idx,val,cat_df[cat_df[fwfid]==val[fwfid]].iloc[0],ftime,'S',taup_model,keep_initial_phase] for idx,val in pha_df.iterrows()]
    #                     , pm_pbar=True, pm_processes=cores,pm_chunksize=1)
    #     pha_df[fstime]=results
    #     print('- TauP (S) Done')
    if change2taup:
        print('\n\n# change to TauP arrival')
        pha_df['ptime0'] = pha_df[fptime]
        results = parmap.map(
            est_taup_2,  # <-- nueva función con offset
            [[idx, val, cat_df[cat_df[fwfid]==val[fwfid]].iloc[0], ftime, 'P', taup_model, keep_initial_phase, 15.0]  # <-- 15.0
             for idx, val in pha_df.iterrows()],
            pm_pbar=True, pm_processes=cores, pm_chunksize=1
        )
        pha_df[fptime] = results
        print('- TauP (P) Done')

        print('# change to TauP S arrival')
        pha_df['stime0'] = pha_df[fstime]
        results = parmap.map(
            est_taup_2,  # <-- nueva función con offset
            [[idx, val, cat_df[cat_df[fwfid]==val[fwfid]].iloc[0], ftime, 'S', taup_model, keep_initial_phase, 15.0]  # <-- 15.0
             for idx, val in pha_df.iterrows()],
            pm_pbar=True, pm_processes=cores, pm_chunksize=1
        )
        pha_df[fstime] = results
        print('- TauP (S) Done')

    pha_df=pha_df.sort_values([_join_col,'net','sta']).reset_index(drop=True)
    pha_df
    
    # Filter to only P and S phases (exclude amplitude picks like AML)
    print(f"\n# Filtering to only P and S phases")
    initial_count = len(pha_df)
    pha_df = pha_df[pha_df['phase'].isin(['P', 'S'])].reset_index(drop=True)
    print(f"  Phases before filtering: {initial_count}")
    print(f"  Phases after filtering (P and S only): {len(pha_df)}")
    
    # Filter by year and day of year if specified
    if year is not None and jday is not None:
        from datetime import datetime, timedelta
        print(f"\n# Filtering events for year={year}, jday={jday}")
        
        try:
            start_date = datetime(year, 1, 1) + timedelta(days=jday-1)
            end_date = start_date + timedelta(days=1)
            print(f"  Date range: {start_date} to {end_date}")
            
            if ftime in cat_df.columns:
                if isinstance(cat_df[ftime].iloc[0], str):
                    cat_df[ftime] = pd.to_datetime(cat_df[ftime])
                
                cat_df_filtered = cat_df[(cat_df[ftime] >= start_date) & (cat_df[ftime] < end_date)]
                print(f"  Events before filtering: {len(cat_df)}")
                print(f"  Events after filtering: {len(cat_df_filtered)}")
                
                cat_df = cat_df_filtered.reset_index(drop=True)
                
                # Filter phases to only those events
                if fwfid in pha_df.columns and fwfid in cat_df.columns:
                    event_ids = cat_df[fwfid].unique()
                    pha_df = pha_df[pha_df[fwfid].isin(event_ids)].reset_index(drop=True)
                    print(f"  Phases for filtered events: {len(pha_df)}")
        except ValueError as e:
            print(f"  Error converting to date: {e}")

    # Create waveform pattern in data_id from event times
    print('\n# Preparing data')
    cat_df[ftime]=[UTCDateTime(i) for i in cat_df[ftime].to_list()]
    
    # Map data_id to wf_pattern (YYYY_JDD_HHMMSS)
    cat_df['wf_pattern'] = [
        f"{ut.year}_{ut.julday:03d}_{ut.hour:02d}{ut.minute:02d}{ut.second:02d}"
        for ut in cat_df[ftime]
    ]
    
    # Filter to only events that have waveforms available
    print('# Filtering to only events with available waveforms')
    import glob as glob_module
    available_events = set()
    for event_dir in glob_module.glob(os.path.join(wf_dir, '*/')):
        dir_name = os.path.basename(event_dir.rstrip('/'))
        available_events.add(dir_name)
    
    print(f"  Available waveform directories: {len(available_events)}")
    
    cat_df_filtered = cat_df[cat_df['wf_pattern'].isin(available_events)].reset_index(drop=True)
    print(f"  Events before filtering: {len(cat_df)}")
    print(f"  Events after filtering: {len(cat_df_filtered)}")
    cat_df = cat_df_filtered
    
    # Filter phases to only those events
    if fwfid in pha_df.columns and fwfid in cat_df.columns:
        event_ids = cat_df[fwfid].unique()
        pha_df = pha_df[pha_df[fwfid].isin(event_ids)].reset_index(drop=True)
        print(f"  Phases for filtered events: {len(pha_df)}")
    
    # Merge pattern into phases
    pattern_map = cat_df[[fwfid, 'wf_pattern']].drop_duplicates()
    pha_df = pha_df.merge(pattern_map, on=fwfid, how='left')
    
    # Add station info
    pha_df['net']=[sta_df[sta_df.sta==i]['net'].iloc[0] for i in pha_df['sta'].to_list()]
    pha_df['chan']=[sta_df[sta_df.sta==i]['chan'].iloc[0] for i in pha_df['sta'].to_list()]
    
    pha_df=pha_df.sort_values([fwfid,'sta']).reset_index(drop=True)
    print('- Done')

    # Make input data matrix (following original run_RPNet.py)
    # wf_dir already has structure: WAVEFORMS/data_id/STATION.mseed
    print('\n\n# Make input matrix from waveform data')
    print(f'  Using wf_dir: {wf_dir}')
    print(f'  Structure: data_id/STATION.mseed')
    results=parmap.map(wf2matrix,[[idx,val,fwfid,fptime,wf_dir,out_dir] for idx, val in pha_df.iterrows()], pm_pbar=True, pm_processes=cores,pm_chunksize=1)
    results=[i for i in results if i is not None]
    if not results:
        print('ERROR: No valid waveforms processed!')
        return False
    
    a,b=zip(*results)
    pha_df=pha_df.iloc[list(a)].reset_index(drop=True)
    
    # Normalize all matrices to 500 samples (pad or truncate as needed)
    print(f'  Normalizing {len(b)} matrices to 500 samples')
    b_normalized = []
    for matrix in b:
        if matrix.shape[1] < 500:
            # Pad with zeros
            padded = np.zeros((1, 500))
            padded[0, :matrix.shape[1]] = matrix[0, :matrix.shape[1]]
            b_normalized.append(padded)
        elif matrix.shape[1] > 500:
            # Truncate
            b_normalized.append(matrix[:, :500])
        else:
            b_normalized.append(matrix)
    
    in_mat=np.vstack(b_normalized)
    print('% calculation time (min): ','%.2f'%((time.time()-stime)/60))
    print('- Done')

    # RPNet prediction
    print('\n\n# Predict polarity (RPNet)')
    r_df=pred_rpnet(model,in_mat,pha_df,batch_size=batch_size,iteration=iteration,gpu_num=gpu_num,time_shift=0.0,mid_point=250)
    print('% calculation time (min): ','%.2f'%((time.time()-stime)/60))
    r_df.to_csv(out_dir+'/pol_result.csv',index=False)
    print('- Done')

    # let's make amplitude file for hash3
    r_df['sta0'] = r_df['sta']

    print(r_df[['sta', 'ptime', 'stime']].head(20))
    print(f"stime nulls: {r_df['stime'].isnull().sum()} / {len(r_df)}")

    if hash_version=='hash3':
        print('# Prep for amplitude ratio')
        picks=r_df.drop_duplicates([fwfid]).sort_values([fwfid])[fwfid].to_list()
        # amps=parmap.map(prepare_amplitudes,[[r_df[r_df[fwfid]==p].reset_index(drop=True),p,p,wf_dir] for p in picks], pm_pbar=True, pm_processes=cores,pm_chunksize=1)
        amps=parmap.map(prepare_amplitudes_2,[[r_df[r_df[fwfid]==p].reset_index(drop=True),p,p,wf_dir,sp_freq,sp_win] for p in picks]
                        , pm_pbar=True, pm_processes=cores,pm_chunksize=1)
        amp=sum(amps,[])
    else:
        amp=None

    # Make SKHASH input setting
    if iteration!=0:
        r_df.loc[r_df['std'] > std_threshold, 'predict'] = 'K'
    # make threshold for mean
    if iteration!=0 and mean_threshold!=0:
        r_df.loc[r_df['prob'] < mean_threshold, 'predict'] = 'K'
        # r_df=r_df[r_df['prob']>=mean_thresuld].reset_index(drop=True)
    if rm_unknwon:
        r_df=r_df[r_df['predict']!='K'].reset_index(drop=True)

    # print('\n\n# Final result:')
    # print(r_df)

    r_df=r_df.drop_duplicates(['sta',fwfid]).reset_index(drop=True)
    prep_skhash_2(cat_df=cat_df,pol_df=r_df,amp=amp,sta_df=sta_df,ftime=ftime,fwfid=fwfid,ctrl0=ctrl0,out_dir=out_dir,hash_version=hash_version)
    print('% calculation time (min): ','%.2f'%((time.time()-stime)/60))
    print('\n\n@ ALL DONE!')

    # # Final processing
    # if iteration!=0:
    #     r_df.loc[r_df['std'] > std_threshold, 'predict'] = 'K'
    # if rm_unknwon:
    #     r_df=r_df[r_df['predict']!='K'].reset_index(drop=True)
    # if iteration!=0 and mean_threshold!=0:
    #     r_df=r_df[r_df['prob']>=mean_thresuld].reset_index(drop=True)
    
    # print('\n\n# Final result:')
    # print(r_df.to_string())

    # # Filter to ensure consistent sets of events
    # cat_df = cat_df.drop_duplicates([fwfid]).reset_index(drop=True)
    # pha_df = pha_df[pha_df[fwfid].isin(cat_df[fwfid].to_list())].reset_index(drop=True)
    # r_df = r_df[r_df[fwfid].isin(cat_df[fwfid].to_list())].reset_index(drop=True)
    
    # # Filter catalog to only events with results
    # cat_df = cat_df[cat_df[fwfid].isin(r_df[fwfid].to_list())].reset_index(drop=True)
    
    # # Filter stations to only those in the results
    # sta_df = sta_df[sta_df['sta'].isin(r_df['sta'].to_list())].reset_index(drop=True)
    
    # r_df=r_df.drop_duplicates(['sta',fwfid]).reset_index(drop=True)
    # prep_skhash(cat_df=cat_df,pol_df=r_df,amp=[],sta_df=sta_df,ftime=ftime,fwfid=fwfid,ctrl0=ctrl0,out_dir=out_dir,hash_version=hash_version)
    # print('% calculation time (min): ','%.2f'%((time.time()-stime)/60))
    
    # print('\n\n@ ALL DONE!')

# -----------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------

def main():
    """ Main function """
    starttime = datetime.now()
    parser = argparse.ArgumentParser(description="Run RPNet polarity picker for a single day")
    parser.add_argument('--event_catalog', type=str, default=event_catalog, 
                       help='Path to event catalog CSV file')
    parser.add_argument('--phase_metadata', type=str, default=phase_metadata, 
                       help='Path to phase metadata CSV file')
    parser.add_argument('--sta_metadata', type=str, default=sta_metadata, 
                       help='Path to station metadata CSV file')
    parser.add_argument('--year', type=int, default=None,
                       help='Year to process (e.g., 2024)')
    parser.add_argument('--jday', type=int, default=None,
                       help='Day of year to process (1-366, e.g., 100 for ~April 10)')
    args = parser.parse_args()
    
    # Validate year/jday combination
    if (args.year is None) != (args.jday is None):
        print("Error: Both --year and --jday must be specified together, or neither")
        return False
    
    # Run processing
    process_single_day(args.event_catalog, args.phase_metadata, args.sta_metadata, 
                      year=args.year, jday=args.jday)
    
    endtime = datetime.now()
    print(f"\nTotal processing time: {endtime - starttime}")
    # print("@ SCRIPT COMPLETED")


if __name__ == '__main__':
    main()
