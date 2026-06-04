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
# MAIN SCRIPT
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# MAIN SCRIPT
# -----------------------------------------------------------------------
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

    # Final processing
    if iteration!=0:
        r_df.loc[r_df['std'] > std_threshold, 'predict'] = 'K'
    if rm_unknwon:
        r_df=r_df[r_df['predict']!='K'].reset_index(drop=True)
    if iteration!=0 and mean_threshold!=0:
        r_df=r_df[r_df['prob']>=mean_thresuld].reset_index(drop=True)
    
    print('\n\n# Final result:')
    print(r_df.to_string())

    # Filter to ensure consistent sets of events
    cat_df = cat_df.drop_duplicates([fwfid]).reset_index(drop=True)
    pha_df = pha_df[pha_df[fwfid].isin(cat_df[fwfid].to_list())].reset_index(drop=True)
    r_df = r_df[r_df[fwfid].isin(cat_df[fwfid].to_list())].reset_index(drop=True)
    
    # Filter catalog to only events with results
    cat_df = cat_df[cat_df[fwfid].isin(r_df[fwfid].to_list())].reset_index(drop=True)
    
    # Filter stations to only those in the results
    sta_df = sta_df[sta_df['sta'].isin(r_df['sta'].to_list())].reset_index(drop=True)
    
    r_df=r_df.drop_duplicates(['sta',fwfid]).reset_index(drop=True)
    prep_skhash(cat_df=cat_df,pol_df=r_df,amp=[],sta_df=sta_df,ftime=ftime,fwfid=fwfid,ctrl0=ctrl0,out_dir=out_dir,hash_version=hash_version)
    print('% calculation time (min): ','%.2f'%((time.time()-stime)/60))
    
    print('\n\n@ ALL DONE!')



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
