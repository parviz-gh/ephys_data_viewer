"""
NWB Data Viewer
Designed by: Parviz Ghaderi
Data related to: "Contextual gating of whisker-evoked responses by frontal cortex supports flexible decision making"
2025-07-01
"""

import os
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
from pynwb import NWBHDF5IO
import h5py
import pandas as pd
from scipy.signal import savgol_filter
from functools import lru_cache




# import allensdk
# from allensdk.core.reference_space_cache import ReferenceSpaceCache

# MOVE AllenSDK imports inside try-catch block:
try:
    import allensdk
    from allensdk.core.reference_space_cache import ReferenceSpaceCache
    ALLENSDK_AVAILABLE = True
    print("AllenSDK successfully imported")
except ImportError as e:
    print(f"AllenSDK not available: {e}")
    ReferenceSpaceCache = None
    ALLENSDK_AVAILABLE = False

# REMOVE this duplicate try-catch block:
# # Add AllenSDK import for mesh
# try:
#     from allensdk.core.reference_space_cache import ReferenceSpaceCache
# except ImportError:
#     ReferenceSpaceCache = None  # fallback if not installed

# Path to NWB files
NWB_DIR = r'G:\nwb-behaviour-processed'
import numpy as np
import plotly.graph_objs as go

def add_brain_wire_mesh(fig, grid_path='brainGridData.npy', color='rgba(0,0,0,0.3)'):
    # Load the grid data (N x 3 array)
    bp = np.load(grid_path)
    bp = bp.astype(float)
    bp[np.sum(bp, axis=1) == 0, :] = np.nan  # Remove zero rows (MATLAB NaN logic)

    # Plot as a single 3D line (or use mode='markers' for dots)
    fig.add_trace(go.Scatter3d(
        x=bp[:, 0], y=bp[:, 1], z=bp[:, 2],
        mode='lines',
        line=dict(color=color, width=2),
        name='Brain Wire Mesh',
        hoverinfo='skip',
        showlegend=False
    ))
    return fig
# Helper to list NWB files
def list_nwb_files():
    try:
        return [f for f in os.listdir(NWB_DIR) if f.endswith('.nwb')]
    except:
        return []

# Helper to get file structure
def get_nwb_structure(file_path):
    def visit(name, node):
        items.append((name, type(node).__name__))
    items = []
    with h5py.File(file_path, 'r') as f:
        f.visititems(visit)
    return items
def sc_scale2color(value, low_range, high_range, zero=0):
    """Python version of your MATLAB SC_Scale2Color colormap."""
    if value < low_range:
        color = [0.8, 0.8, 1.0]
    elif value >= low_range and value < zero:
        color_ind = 1.8 * (zero - value) / abs(low_range)
        if color_ind <= 1:
            color = [0, 0, color_ind]
        else:
            color = [color_ind - 1, color_ind - 1, 1]
    elif value >= zero and value <= high_range:
        color_ind = 2 * (value - zero) / high_range
        if color_ind <= 1:
            color = [color_ind, 0, 0]
        else:
            color = [1, color_ind - 1, 0]
    elif value > high_range:
        color = [1, 1, 0]
    return color
# Helper to get metadata
def get_nwb_metadata(file_path):
    with NWBHDF5IO(file_path, 'r') as io:
        nwbfile = io.read()
        meta = {
            'Session Description': nwbfile.session_description,
            'Identifier': nwbfile.identifier,
            'Session Start Time': str(nwbfile.session_start_time),
            'Subject': str(nwbfile.subject),
        }
    return meta

# Helper to preview a dataset
def preview_dataset(file_path, dataset_path, max_items=100):
    with h5py.File(file_path, 'r') as f:
        data = f[dataset_path][()]
        if hasattr(data, 'shape') and data.shape:
            preview = data[:max_items]
        else:
            preview = data
    return preview

# Helper to find time series or spike data
def find_timeseries_datasets(file_path):
    ts_paths = []
    with h5py.File(file_path, 'r') as f:
        def visit(name, node):
            if isinstance(node, h5py.Dataset) and ('data' in name or 'spike_times' in name):
                ts_paths.append(name)
        f.visititems(visit)
    return ts_paths

# Helper to get event/time fields from trials table
def get_trial_event_fields(file_path):
    with NWBHDF5IO(file_path, 'r') as io:
        nwbfile = io.read()
        trial_table = nwbfile.trials
        # Get all columns that are float or int (likely time fields)
        event_fields = []
        for col in trial_table.colnames:
            data = trial_table[col].data[:]
            if np.issubdtype(data.dtype, np.floating) or np.issubdtype(data.dtype, np.integer):
                event_fields.append(col)
        return event_fields

# Helper to get all unit ids
def get_unit_ids(file_path):
    with NWBHDF5IO(file_path, 'r') as io:
        nwbfile = io.read()
        units = nwbfile.units
        if 'id' in units.colnames:
            return list(units['id'].data[:])
        else:
            return list(range(len(units)))

# Updated helper function to get unit ids with brain regions
def get_unit_ids_with_regions(file_path):
    """Get unit IDs along with their brain regions"""
    with NWBHDF5IO(file_path, 'r') as io:
        nwbfile = io.read()
        units = nwbfile.units
        units_df = units.to_dataframe()
        
        unit_info = []
        for idx, row in units_df.iterrows():
            # Get unit ID
            if 'id' in units_df.columns:
                unit_id = row['id']
            else:
                unit_id = idx
            
            # Get brain region
            if 'location' in units_df.columns and pd.notna(row['location']):
                region = row['location']
                unit_info.append({
                    'id': str(unit_id),
                    'region': region,
                    'label': f"{region} - Unit {unit_id}",
                    'value': str(unit_id)
                })
            else:
                unit_info.append({
                    'id': str(unit_id),
                    'region': 'Unknown',
                    'label': f"Unknown - Unit {unit_id}",
                    'value': str(unit_id)
                })
        
        return unit_info

# Helper to get spike times for a unit
def get_unit_spike_times(file_path, unit_index):
    with NWBHDF5IO(file_path, 'r') as io:
        nwbfile = io.read()
        units = nwbfile.units
        return units['spike_times'].data[unit_index]

# Helper to get trial event times
def get_trial_event_times(file_path, event_field):
    with NWBHDF5IO(file_path, 'r') as io:
        nwbfile = io.read()
        trial_table = nwbfile.trials
        return trial_table[event_field].data[:]

# Enhanced function to get behavioral data (following MATLAB approach exactly)
def get_behavioral_data(file_path):
    """Extract behavioral data exactly like MATLAB code"""
    try:
        with NWBHDF5IO(file_path, 'r') as io:
            nwbfile = io.read()
            behavior_strc = {}
            
            print(f"Extracting behavioral data from: {file_path}")
            
            # Get behavioral time series from processing module (following MATLAB exactly)
            if 'behavior' in nwbfile.processing:
                behavior_module = nwbfile.processing['behavior']
                
                # Extract behavior time series (like MATLAB)
                if 'BehavioralTimeSeries' in behavior_module.data_interfaces:
                    beh_ts = behavior_module.data_interfaces['BehavioralTimeSeries']
                    
                    # Get all time series names (like beh_timeseriesName in MATLAB)
                    beh_timeseries_names = list(beh_ts.time_series.keys())
                    print(f"Found BehavioralTimeSeries signals: {beh_timeseries_names}")
                    
                    # Extract each time series data (like the MATLAB loop)
                    for ts_name in beh_timeseries_names:
                        ts_obj = beh_ts.time_series[ts_name]
                        behavior_strc[ts_name] = {
                            'data': ts_obj.data[:],
                            'timestamps': ts_obj.timestamps[:]
                        }
                        print(f"  Extracted {ts_name}: data shape {behavior_strc[ts_name]['data'].shape}")
                    
                    # Get specific timestamps (like MATLAB)
                    if 'C2Whisker_Angle' in beh_timeseries_names:
                        behavior_strc['video_timestamp'] = beh_ts.time_series['C2Whisker_Angle'].timestamps[:]
                        print("  Extracted video_timestamp from C2Whisker_Angle")
                    
                    if 'Piezo_lick_trace' in beh_timeseries_names:
                        behavior_strc['piezo_timestamp'] = beh_ts.time_series['Piezo_lick_trace'].timestamps[:]
                        print("  Extracted piezo_timestamp from Piezo_lick_trace")
                
                # Extract behavioral events (like MATLAB)
                if 'BehavioralEvents' in behavior_module.data_interfaces:
                    beh_events = behavior_module.data_interfaces['BehavioralEvents']
                    beh_events_names = list(beh_events.time_series.keys())
                    print(f"Found BehavioralEvents: {beh_events_names}")
                    
                    for event_name in beh_events_names:
                        event_obj = beh_events.time_series[event_name]
                        behavior_strc[event_name] = {
                            'timestamps': event_obj.timestamps[:]
                        }
                        print(f"  Extracted event {event_name}")
            
            # Standardize fields (like MATLAB standardization)
            for field_name in behavior_strc.keys():
                if 'data' in behavior_strc[field_name]:
                    data = behavior_strc[field_name]['data']
                    if hasattr(data, 'shape') and len(data.shape) > 1:
                        if data.shape[0] < data.shape[1]:  # Transpose if needed
                            behavior_strc[field_name]['data'] = data.T
                            print(f"  Transposed {field_name} data")
                
                if 'timestamps' in behavior_strc[field_name]:
                    timestamps = behavior_strc[field_name]['timestamps']
                    if hasattr(timestamps, 'shape') and len(timestamps.shape) > 1:
                        if timestamps.shape[0] < timestamps.shape[1]:  # Transpose if needed
                            behavior_strc[field_name]['timestamps'] = timestamps.T
                            print(f"  Transposed {field_name} timestamps")
            
            print(f"Successfully extracted {len(behavior_strc)} behavioral signals/timestamps")
            return behavior_strc
    
    except Exception as e:
        print(f"Error in get_behavioral_data: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

# Enhanced function to get units data
def get_units_data(file_path):
    """Extract units data similar to MATLAB code"""
    with NWBHDF5IO(file_path, 'r') as io:
        nwbfile = io.read()
        units_data = {}
        
        if nwbfile.units is not None:
            units_table = nwbfile.units.to_dataframe()
            units_data = units_table.to_dict('list')
    
    return units_data

# Enhanced function to get electrodes data
def get_electrodes_data(file_path):
    """Extract electrodes data similar to MATLAB code"""
    with NWBHDF5IO(file_path, 'r') as io:
        nwbfile = io.read()
        electrodes_data = {}
        
        if nwbfile.electrodes is not None:
            electrodes_table = nwbfile.electrodes.to_dataframe()
            electrodes_data = electrodes_table.to_dict('list')
    
    return electrodes_data

# Enhanced function to get trials data
def get_trials_data(file_path):
    """Extract trials data similar to MATLAB code"""
    with NWBHDF5IO(file_path, 'r') as io:
        nwbfile = io.read()
        trials_data = {}
        
        if nwbfile.trials is not None:
            trials_table = nwbfile.trials.to_dataframe()
            trials_data = trials_table.to_dict('list')
    
    return trials_data

# Enhanced function to process movement signals (following MATLAB exactly)
def process_movement_signals(behavior_data):
    """Process movement signals exactly like MATLAB code"""
    try:
        movement_signals = {}
        
        # Extract basic signals first (like MATLAB)
        jaw_coordinate = behavior_data.get('Jaw_Coordinate', {}).get('data')
        tongue_coordinate = behavior_data.get('Tongue_Coordinate', {}).get('data')
        snout_angle = behavior_data.get('Snout_Angle', {}).get('data')
        whisker_angle = behavior_data.get('C2Whisker_Angle', {}).get('data')
        piezo_lick_trace = behavior_data.get('Piezo_lick_trace', {}).get('data')
        
        # Process whisker signal (exactly like MATLAB)
        if whisker_angle is not None:
            whisker_angle = np.array(whisker_angle)
            # whisker_speed=[abs(diff(whisker_angle,1,1))];
            whisker_speed = np.abs(np.diff(whisker_angle, axis=0))
            # whisker_speed=[whisker_speed(1);whisker_speed];
            whisker_speed = np.concatenate([[whisker_speed[0]], whisker_speed])
            
            movement_signals['whisker_angle'] = whisker_angle
            movement_signals['whisker_speed'] = whisker_speed
            print(f"Processed whisker signals: angle shape {whisker_angle.shape}, speed shape {whisker_speed.shape}")
        
        # Process jaw signal (exactly like MATLAB)
        if jaw_coordinate is not None:
            jaw_coordinate = np.array(jaw_coordinate)
            if jaw_coordinate.ndim > 1 and jaw_coordinate.shape[1] >= 2:
                # jawcordX=Jaw_Coordinate(:,1); jawcordY=Jaw_Coordinate(:,2);
                jaw_cord_x = jaw_coordinate[:, 0]
                jaw_cord_y = jaw_coordinate[:, 1]
                # jaw_movement = sqrt( (jawcordY-mode(jawcordY)).^2 + (jawcordX-mode(jawcordX)).^2 );
                # Using median instead of mode (more robust in Python)
                jaw_movement = np.sqrt((jaw_cord_y - np.median(jaw_cord_y))**2 + 
                                     (jaw_cord_x - np.median(jaw_cord_x))**2)
                movement_signals['jaw_movement'] = jaw_movement
                print(f"Processed jaw movement: shape {jaw_movement.shape}")
        
        # Process tongue signal (exactly like MATLAB)
        if tongue_coordinate is not None and jaw_coordinate is not None:
            tongue_coordinate = np.array(tongue_coordinate)
            if tongue_coordinate.ndim > 1 and tongue_coordinate.shape[1] >= 2:
                # TonguecordY=Tongue_Coordinate(:,2); TonguecordX=Tongue_Coordinate(:,1);
                tongue_cord_x = tongue_coordinate[:, 0]
                tongue_cord_y = tongue_coordinate[:, 1]
                # tongue_movement = sqrt( (TonguecordY-mode(jawcordY)).^2 + (TonguecordX-mode(jawcordX)).^2 );
                jaw_cord_y_ref = jaw_coordinate[:, 1] if jaw_coordinate.ndim > 1 else jaw_coordinate
                jaw_cord_x_ref = jaw_coordinate[:, 0] if jaw_coordinate.ndim > 1 else jaw_coordinate
                tongue_movement = np.sqrt((tongue_cord_y - np.median(jaw_cord_y_ref))**2 + 
                                        (tongue_cord_x - np.median(jaw_cord_x_ref))**2)
                # tongue_movement(20*mean(tongue_movement)<tongue_movement)=nan;
                mean_tongue = np.nanmean(tongue_movement)
                tongue_movement[tongue_movement > 20 * mean_tongue] = np.nan
                movement_signals['tongue_movement'] = tongue_movement
                print(f"Processed tongue movement: shape {tongue_movement.shape}")
        
        # Process snout angle
        if snout_angle is not None:
            movement_signals['snout_angle'] = np.array(snout_angle)
            print(f"Added snout angle: shape {movement_signals['snout_angle'].shape}")
        
        # Process piezo signal (exactly like MATLAB)
        if piezo_lick_trace is not None:
            # piezo_lick_trace = sgolayfilt(double(piezo_lick_trace),5,11);
            piezo_signal = np.array(piezo_lick_trace, dtype=float)
            if len(piezo_signal) > 11:
                filtered_piezo = savgol_filter(piezo_signal, 11, 5)
                movement_signals['piezo_lick_trace'] = filtered_piezo
                print(f"Processed piezo signal: shape {filtered_piezo.shape}")
        
        print(f"Total processed movement signals: {list(movement_signals.keys())}")
        return movement_signals
    
    except Exception as e:
        print(f"Error processing movement signals: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

# Batch processing function similar to MATLAB
def process_all_nwb_files():
    """Process all NWB files and create a comprehensive dataset"""
    nwb_files = list_nwb_files()
    processed_data = []
    
    for file_name in nwb_files:
        file_path = os.path.join(NWB_DIR, file_name)
        
        try:
            # Get all data components
            behavior_data = get_behavioral_data(file_path)
            units_data = get_units_data(file_path)
            electrodes_data = get_electrodes_data(file_path)
            trials_data = get_trials_data(file_path)
            
            # Process movement signals
            movement_signals = process_movement_signals(behavior_data)
            
            # Get unique brain areas
            if 'location' in units_data:
                unique_locations = list(set(units_data['location']))
                
                for location in unique_locations:
                    location_indices = [i for i, loc in enumerate(units_data['location']) if loc == location]
                    
                    session_data = {
                        'session_id': file_name,
                        'probe_location': location,
                        'behavior_data': behavior_data,
                        'movement_signals': movement_signals,
                        'trials_data': trials_data,
                        'units_data': {key: [val[i] for i in location_indices] for key, val in units_data.items()},
                        'electrodes_data': electrodes_data
                    }
                    
                    processed_data.append(session_data)
        
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
    
    return processed_data

# Enhanced PSTH calculation
def calculate_psth_enhanced(file_path, event_field, unit_id, window=[-1, 2], bin_size=0.005):
    """Enhanced PSTH calculation similar to MATLAB code"""
    try:
        with NWBHDF5IO(file_path, 'r') as io:
            nwbfile = io.read()
            
            # Get event times from trials
            trials_df = nwbfile.trials.to_dataframe()
            event_times = trials_df[event_field].values
            
            # Get units data
            units_df = nwbfile.units.to_dataframe()
            
            if unit_id == 'all':
                unit_indices = list(range(len(units_df)))
            else:
                unit_indices = [int(unit_id)]
            
            all_spike_counts = []
            
            for unit_idx in unit_indices:
                spike_times = units_df.iloc[unit_idx]['spike_times']
                
                for event_time in event_times:
                    # Align spikes to event
                    aligned_spikes = spike_times - event_time
                    
                    # Filter spikes within window
                    mask = (aligned_spikes >= window[0]) & (aligned_spikes <= window[1])
                    windowed_spikes = aligned_spikes[mask]
                    
                    # Create bins
                    bins = np.arange(window[0], window[1] + bin_size, bin_size)
                    counts, _ = np.histogram(windowed_spikes, bins=bins)
                    all_spike_counts.append(counts)
            
            if not all_spike_counts:
                return None, None
            
            all_spike_counts = np.array(all_spike_counts)
            mean_psth = np.mean(all_spike_counts, axis=0)
            sem_psth = np.std(all_spike_counts, axis=0) / np.sqrt(all_spike_counts.shape[0])
            
            bins = np.arange(window[0], window[1] + bin_size, bin_size)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            return bin_centers, mean_psth, sem_psth
    
    except Exception as e:
        print(f"Error calculating PSTH: {str(e)}")
        return None, None, None

# Enhanced function to get trial grouping options
def get_trial_grouping_options(file_path):
    """Get available trial grouping columns"""
    with NWBHDF5IO(file_path, 'r') as io:
        nwbfile = io.read()
        trials_df = nwbfile.trials.to_dataframe()
        
        # Look for common grouping columns
        grouping_columns = []
        for col in trials_df.columns:
            # Check if column contains binary data (0/1) or categorical data
            unique_vals = trials_df[col].dropna().unique()
            if len(unique_vals) <= 10:  # Categorical with few categories
                grouping_columns.append(col)
        
        return grouping_columns, trials_df

# Enhanced PSTH calculation with trial selection
def calculate_psth_with_grouping(file_path, event_field, unit_id, trial_filters=None, window=[-1, 2], bin_size=0.005):
    """Enhanced PSTH calculation with trial grouping and filtering"""
    try:
        with NWBHDF5IO(file_path, 'r') as io:
            nwbfile = io.read()
            
            # Get trials data
            trials_df = nwbfile.trials.to_dataframe()
            
            # Apply trial filters if provided
            if trial_filters:
                trial_mask = np.ones(len(trials_df), dtype=bool)
                for filter_col, filter_val in trial_filters.items():
                    if filter_col in trials_df.columns and filter_val is not None:
                        trial_mask &= (trials_df[filter_col] == filter_val)
                
                # Filter trials
                trials_df = trials_df[trial_mask]
            
            if len(trials_df) == 0:
                return None, None, None, 0
            
            event_times = trials_df[event_field].values
            
            # Get units data
            units_df = nwbfile.units.to_dataframe()
            
            if unit_id == 'all':
                unit_indices = list(range(len(units_df)))
            else:
                unit_indices = [int(unit_id)]
            
            all_spike_counts = []
            
            for unit_idx in unit_indices:
                spike_times = units_df.iloc[unit_idx]['spike_times']
                
                for event_time in event_times:
                    # Align spikes to event
                    aligned_spikes = spike_times - event_time
                    
                    # Filter spikes within window
                    mask = (aligned_spikes >= window[0]) & (aligned_spikes <= window[1])
                    windowed_spikes = aligned_spikes[mask]
                    
                    # Create bins
                    bins = np.arange(window[0], window[1] + bin_size, bin_size)
                    counts, _ = np.histogram(windowed_spikes, bins=bins)
                    all_spike_counts.append(counts)
            
            if not all_spike_counts:
                return None, None, None, 0
            
            all_spike_counts = np.array(all_spike_counts)
            mean_psth = np.mean(all_spike_counts, axis=0)
            sem_psth = np.std(all_spike_counts, axis=0) / np.sqrt(all_spike_counts.shape[0])
            
            bins = np.arange(window[0], window[1] + bin_size, bin_size)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            return bin_centers, mean_psth, sem_psth, len(trials_df)
    
    except Exception as e:
        print(f"Error calculating PSTH with grouping: {str(e)}")
        return None, None, None, 0

# Function to calculate behavioral PSTH
def calculate_behavioral_psth(file_path, event_field, behavioral_signal, trial_filters=None, window=[-1, 2], sampling_rate=200):
    """Calculate PSTH for behavioral signals aligned to events"""
    try:
        with NWBHDF5IO(file_path, 'r') as io:
            nwbfile = io.read()
            
            # Get trials data
            trials_df = nwbfile.trials.to_dataframe()
            
            # Apply trial filters if provided
            if trial_filters:
                trial_mask = np.ones(len(trials_df), dtype=bool)
                for filter_col, filter_val in trial_filters.items():
                    if filter_col in trials_df.columns and filter_val is not None:
                        trial_mask &= (trials_df[filter_col] == filter_val)
                trials_df = trials_df[trial_mask]
            
            if len(trials_df) == 0:
                return None, None, None, 0
            
            event_times = trials_df[event_field].values
            
            # Get behavioral data
            behavior_data = get_behavioral_data(file_path)
            processed_signals = process_movement_signals(behavior_data)
            
            # Determine if this is a raw behavioral signal or processed signal
            if behavioral_signal in behavior_data:
                signal_data = behavior_data[behavioral_signal]['data']
                signal_timestamps = behavior_data[behavioral_signal]['timestamps']
            elif behavioral_signal in processed_signals:
                signal_data = processed_signals[behavioral_signal]
                # Use timestamps from the original signal (typically from whisker or jaw data)
                if 'C2Whisker_Angle' in behavior_data:
                    signal_timestamps = behavior_data['C2Whisker_Angle']['timestamps']
                elif 'Jaw_Coordinate' in behavior_data:
                    signal_timestamps = behavior_data['Jaw_Coordinate']['timestamps']
                else:
                    # Use first available signal's timestamps
                    first_signal = list(behavior_data.keys())[0]
                    signal_timestamps = behavior_data[first_signal]['timestamps']
            else:
                print(f"Signal {behavioral_signal} not found in behavioral data")
                return None, None, None, 0
            
            # Ensure signal_data is 1D
            if hasattr(signal_data, 'ndim') and signal_data.ndim > 1:
                # For multi-dimensional data (like coordinates), calculate magnitude
                if signal_data.shape[1] == 2:  # X,Y coordinates
                    signal_data = np.sqrt(signal_data[:, 0]**2 + signal_data[:, 1]**2)
                else:
                    signal_data = signal_data[:, 0]  # Take first dimension
            
            all_behavioral_traces = []
            
            # Create time vector for the window
            dt = 1.0 / sampling_rate
            time_vector = np.arange(window[0], window[1], dt)
            
            for event_time in event_times:
                # Find indices for the time window around the event
                start_time = event_time + window[0]
                end_time = event_time + window[1]
                
                # Find closest timestamps
                start_idx = np.argmin(np.abs(signal_timestamps - start_time))
                end_idx = np.argmin(np.abs(signal_timestamps - end_time))
                
                if end_idx > start_idx and len(signal_data) > end_idx:
                    # Extract behavioral trace for this trial
                    trial_trace = signal_data[start_idx:end_idx]
                    trial_timestamps = signal_timestamps[start_idx:end_idx] - event_time
                    
                    # Interpolate to common time vector
                    if len(trial_trace) > 1 and len(trial_timestamps) > 1:
                        interpolated_trace = np.interp(time_vector, trial_timestamps, trial_trace)
                        all_behavioral_traces.append(interpolated_trace)
            
            if not all_behavioral_traces:
                return None, None, None, 0
            
            # Convert to array and calculate statistics
            all_behavioral_traces = np.array(all_behavioral_traces)
            mean_trace = np.nanmean(all_behavioral_traces, axis=0)
            sem_trace = np.nanstd(all_behavioral_traces, axis=0) / np.sqrt(all_behavioral_traces.shape[0])
            
            return time_vector, mean_trace, sem_trace, len(all_behavioral_traces)
    
    except Exception as e:
        print(f"Error calculating behavioral PSTH: {str(e)}")
        return None, None, None, 0

# Enhanced function to get available behavioral signals (following MATLAB approach)
def get_behavioral_signals(file_path):
    """Get list of available behavioral signals following MATLAB approach"""
    try:
        with NWBHDF5IO(file_path, 'r') as io:
            nwbfile = io.read()
            behavioral_signals = []
            
            print(f"Processing file: {file_path}")
            
            # Get behavioral time series from processing module
            if 'behavior' in nwbfile.processing:
                behavior_module = nwbfile.processing['behavior']
                print(f"Found behavior module with interfaces: {list(behavior_module.data_interfaces.keys())}")
                
                # Get BehavioralTimeSeries (like MATLAB beh_timeseriesName)
                if 'BehavioralTimeSeries' in behavior_module.data_interfaces:
                    beh_ts = behavior_module.data_interfaces['BehavioralTimeSeries']
                    ts_names = list(beh_ts.time_series.keys())
                    print(f"Found BehavioralTimeSeries signals: {ts_names}")
                    behavioral_signals.extend(ts_names)
                else:
                    print("No BehavioralTimeSeries found in behavior module")
                
                # Get BehavioralEvents (like MATLAB beh_eventsName)
                if 'BehavioralEvents' in behavior_module.data_interfaces:
                    beh_events = behavior_module.data_interfaces['BehavioralEvents']
                    event_names = list(beh_events.time_series.keys())
                    print(f"Found BehavioralEvents: {event_names}")
                    for event_name in event_names:
                        behavioral_signals.append(f"{event_name}_events")
                else:
                    print("No BehavioralEvents found in behavior module")
            else:
                print("No 'behavior' module found in processing")
            
            # Add processed movement signals (like MATLAB MovementSignal)
            try:
                behavior_data = get_behavioral_data(file_path)
                processed_signals = process_movement_signals(behavior_data)
                
                # Add the movement signal names (like MATLAB MovementSignal fields)
                movement_signal_names = ['whisker_speed', 'jaw_movement', 'tongue_movement']
                for signal_name in movement_signal_names:
                    if signal_name in processed_signals and signal_name not in behavioral_signals:
                        behavioral_signals.append(signal_name)
                
                print(f"Added processed movement signals: {[s for s in movement_signal_names if s in processed_signals]}")
                
            except Exception as e:
                print(f"Error processing movement signals: {str(e)}")
                import traceback
                traceback.print_exc()
            
            print(f"Final behavioral signals list: {behavioral_signals}")
            return sorted(behavioral_signals)
    
    except Exception as e:
        print(f"Error getting behavioral signals: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

# Enhanced PSTH calculation with brain region grouping
def calculate_psth_by_brain_region(file_path, event_field, trial_filters=None, window=[-1, 2], bin_size=0.005):
    """Calculate PSTH grouped by brain region (unit_location)"""
    try:
        with NWBHDF5IO(file_path, 'r') as io:
            nwbfile = io.read()
            
            # Get trials data
            trials_df = nwbfile.trials.to_dataframe()
            
            # Apply trial filters if provided
            if trial_filters:
                trial_mask = np.ones(len(trials_df), dtype=bool)
                for filter_col, filter_val in trial_filters.items():
                    if filter_col in trials_df.columns and filter_val is not None:
                        trial_mask &= (trials_df[filter_col] == filter_val)
                trials_df = trials_df[trial_mask]
            
            if len(trials_df) == 0:
                return {}, None, 0
            
            event_times = trials_df[event_field].values
            
            # Get units data
            units_df = nwbfile.units.to_dataframe()
            
            # Check if unit_location column exists
            if 'location' not in units_df.columns:
                print("No 'location' column found in units table")
                return {}, None, 0
            
            # Group units by brain region
            brain_regions = units_df['location'].unique()
            brain_regions = [region for region in brain_regions if pd.notna(region)]
            
            print(f"Found brain regions: {brain_regions}")
            
            region_psths = {}
            bins = np.arange(window[0], window[1] + bin_size, bin_size)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            for region in brain_regions:
                # Get units in this brain region
                region_units = units_df[units_df['location'] == region]
                
                all_spike_counts = []
                
                for unit_idx in region_units.index:
                    spike_times = units_df.iloc[unit_idx]['spike_times']
                    
                    for event_time in event_times:
                        # Align spikes to event
                        aligned_spikes = spike_times - event_time
                        
                        # Filter spikes within window
                        mask = (aligned_spikes >= window[0]) & (aligned_spikes <= window[1])
                        windowed_spikes = aligned_spikes[mask]
                        
                        # Create histogram
                        counts, _ = np.histogram(windowed_spikes, bins=bins)
                        all_spike_counts.append(counts)
                
                if all_spike_counts:
                    all_spike_counts = np.array(all_spike_counts)
                    mean_psth = np.mean(all_spike_counts, axis=0)
                    sem_psth = np.std(all_spike_counts, axis=0) / np.sqrt(all_spike_counts.shape[0])
                    
                    region_psths[region] = {
                        'mean': mean_psth,
                        'sem': sem_psth,
                        'n_units': len(region_units),
                        'n_trials': len(all_spike_counts)
                    }
                    
                    print(f"Region {region}: {len(region_units)} units, {len(all_spike_counts)} trials")
            
            return region_psths, bin_centers, len(trials_df)
    
    except Exception as e:
        print(f"Error calculating PSTH by brain region: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}, None, 0

# Fixed color definitions - all in hex format
BRAIN_REGION_COLORS = {
    'wS1': '#0008FF',   # Blue
    'wS2': '#00FF00',   # Green  
    'ALM': '#ff0000',   # Red
    'wM2': '#000000',   # Black
    'A1': '#FF00FF'     # Magenta
}

# Fixed default colors - all in hex format
DEFAULT_COLORS = ['#FFA500', '#A52A2A', '#FFC0CB', '#808080', '#808000', '#00FFFF', '#FFFF00', '#800080']

def get_region_color(region, region_index):
    """Get color for a brain region based on predefined mapping"""
    if region in BRAIN_REGION_COLORS:
        return BRAIN_REGION_COLORS[region]
    else:
        # Use default colors for unknown regions
        return DEFAULT_COLORS[region_index % len(DEFAULT_COLORS)]

def hex_to_rgba(hex_color, alpha=0.3):
    """Convert hex color to rgba format for fill"""
    # Handle both hex colors and named colors
    if not hex_color.startswith('#'):
        # Convert named colors to hex if needed
        named_colors = {
            'orange': '#FFA500', 'brown': '#A52A2A', 'pink': '#FFC0CB', 
            'gray': '#808080', 'olive': '#808000', 'cyan': '#00FFFF', 
            'yellow': '#FFFF00', 'purple': '#800080', 'red': '#FF0000',
            'blue': '#0000FF', 'green': '#00FF00', 'black': '#000000',
            'white': '#FFFFFF', 'magenta': '#FF00FF'
        }
        hex_color = named_colors.get(hex_color.lower(), '#808080')  # Default to gray
    
    hex_color = hex_color.lstrip('#')
    
    # Ensure we have a valid 6-character hex string
    if len(hex_color) != 6:
        hex_color = '808080'  # Default to gray if invalid
    
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f'rgba({r}, {g}, {b}, {alpha})'
    except ValueError:
        # If conversion fails, return a default gray color
        return f'rgba(128, 128, 128, {alpha})'

# Fixed function to extract CCF coordinates using correct PyNWB methods
def get_unit_ccf_coordinates_fixed(file_path):
    """Extract CCF coordinates from the 'ccf_xyz' column in the units table."""
    try:
        with NWBHDF5IO(file_path, 'r') as io:
            nwbfile = io.read()
            unit_coordinates = {}
            if nwbfile.units is not None:
                units_df = nwbfile.units.to_dataframe()
                print("Available columns in units table:", list(units_df.columns))
                if 'ccf_xyz' in units_df.columns:
                    for idx, row in units_df.iterrows():
                        region = row['location'] if 'location' in row else 'Unknown'
                        ccf = row['ccf_xyz']
                        if isinstance(ccf, (list, np.ndarray)) and len(ccf) == 3:
                            ap, ml, dv = ccf
                            x_mm = ap / 1
                            y_mm = ml / 1
                            z_mm = dv / 1
                            if region not in unit_coordinates:
                                unit_coordinates[region] = []
                            unit_coordinates[region].append({
                                'unit_id': row['id'] if 'id' in row else idx,
                                'x': x_mm,
                                'y': y_mm,
                                'z': z_mm,
                                'ccf_original': ccf
                            })
                    # Debug print
                    for region, units in unit_coordinates.items():
                        print(f"{region}: {len(units)} units")
                        for u in units[:3]:
                            print(f"  Unit {u['unit_id']} at (x={u['x']:.2f}, y={u['y']:.2f}, z={u['z']:.2f}) mm, original: {u['ccf_original']}")
                    return unit_coordinates
                else:
                    print("Could not find 'ccf_xyz' column in units table.")
                    return {}
            else:
                print("No units table found in NWB file")
                return {}
    except Exception as e:
        print(f"Error extracting CCF coordinates: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

# Enhanced H5 method with more thorough column search
def get_unit_ccf_coordinates_h5_enhanced(file_path):
    """Enhanced H5 method to find coordinate columns"""
    try:
        with h5py.File(file_path, 'r') as f:
            unit_coordinates = {}
            
            # Navigate to units table
            if 'units' in f:
                units_group = f['units']
                print(f"Found units group with keys: {list(units_group.keys())}")
                
                # Look for coordinate columns (various possible names)
                coord_candidates = [
                    'unit_ccf_xyz', 'ccf_xyz', 'coordinates', 'xyz', 
                    'ccf_coordinates', 'electrode_ccf_xyz', 'probe_ccf_xyz'
                ]
                
                coord_column = None
                for candidate in coord_candidates:
                    if candidate in units_group:
                        coord_column = candidate
                        print(f"Found coordinate column: {coord_column}")
                        break
                
                if coord_column is None:
                    print(f"No coordinate columns found. Available: {list(units_group.keys())}")
                    return {}
                
                # Look for location columns
                location_candidates = ['location', 'brain_area', 'region', 'area', 'structure']
                
                location_column = None
                for candidate in location_candidates:
                    if candidate in units_group:
                        location_column = candidate
                        print(f"Found location column: {location_column}")
                        break
                
                if location_column is None:
                    print(f"No location columns found. Available: {list(units_group.keys())}")
                    return {}
                
                # Extract data
                locations = units_group[location_column][()]
                if isinstance(locations[0], bytes):
                    locations = [loc.decode('utf-8') for loc in locations]
                
                ccf_coords = units_group[coord_column][()]
                print(f"Found {len(locations)} units with coordinates shape: {ccf_coords.shape}")
                
                # Extract unit IDs if available
                if 'id' in units_group:
                    unit_ids = units_group['id'][()]
                else:
                    unit_ids = list(range(len(locations)))
                
                # Process each unit
                for i, (location, ccf_coord, unit_id) in enumerate(zip(locations, ccf_coords, unit_ids)):
                    if pd.notna(location) and hasattr(ccf_coord, '__len__') and len(ccf_coord) >= 3:
                        region = location
                        
                        if region not in unit_coordinates:
                            unit_coordinates[region] = []
                        
                        # Convert to mm (assuming input is in microns)
                        x_mm = ccf_coord[0] / 1
                        y_mm = ccf_coord[2] / 1
                        z_mm = ccf_coord[1] / 1
                        
                        unit_coordinates[region].append({
                            'unit_id': unit_id,
                            'x': x_mm,
                            'y': y_mm,
                            'z': z_mm,
                            'ccf_original': ccf_coord
                        })
                
                total_units = sum(len(units) for units in unit_coordinates.values())
                print(f"H5 enhanced method: Extracted CCF coordinates for {total_units} units across {len(unit_coordinates)} regions")
                
                return unit_coordinates
        
    except Exception as e:
        print(f"Error with H5 enhanced method: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

# Updated robust method with fixed approaches
def get_unit_ccf_coordinates_robust_fixed(file_path):
    """Fixed robust method to extract CCF coordinates"""
    try:
        # First try the fixed NWB method
        print("Trying fixed NWB method...")
        unit_coordinates = get_unit_ccf_coordinates_fixed(file_path)
        
        if unit_coordinates:
            return unit_coordinates
        
        # If that fails, try enhanced H5 access
        print("NWB method failed, trying enhanced H5 method...")
        unit_coordinates = get_unit_ccf_coordinates_h5_enhanced(file_path)
        
        if unit_coordinates:
            return unit_coordinates
        
        # If both fail, show what's available
        print("Both methods failed, showing available data...")
        try:
            with NWBHDF5IO(file_path, 'r') as io:
                nwbfile = io.read()
                if nwbfile.units is not None:
                    units_df = nwbfile.units.to_dataframe()
                    print(f"Available columns in units table: {list(units_df.columns)}")
                    
                    # Show first few rows of data
                    print("First 3 rows of units table:")
                    print(units_df.head(3))
                    
        except Exception as e:
            print(f"Error inspecting file: {str(e)}")
        
        return {}
    
    except Exception as e:
        print(f"Error in robust CCF extraction: {str(e)}")
        return {}

# Updated function to calculate unit spike activity with custom window
def calculate_unit_spike_activity(file_path, event_field, window=[-1, 2]):
    """Calculate spike activity for each unit to color-code 3D points with custom window"""
    try:
        with NWBHDF5IO(file_path, 'r') as io:
            nwbfile = io.read()
            
            # Get trials and event times
            trials_df = nwbfile.trials.to_dataframe()
            event_times = trials_df[event_field].values
            
            # Get units data
            units_df = nwbfile.units.to_dataframe()
            
            spike_activity = {}
            
            if 'location' in units_df.columns:
                for region in units_df['location'].unique():
                    if pd.notna(region):
                        region_units = units_df[units_df['location'] == region]
                        region_activities = []
                        
                        for idx, unit_row in region_units.iterrows():
                            spike_times = unit_row['spike_times']
                            
                            # Calculate average firing rate in custom window around events
                            total_spikes = 0
                            for event_time in event_times:
                                spikes_in_window = np.sum(
                                    (spike_times >= event_time + window[0]) & 
                                    (spike_times <= event_time + window[1])
                                )
                                total_spikes += spikes_in_window
                            
                            # Average spikes per trial in the window
                            window_duration = window[1] - window[0]
                            avg_firing_rate = (total_spikes / len(event_times)) / window_duration if len(event_times) > 0 else 0
                            region_activities.append(avg_firing_rate)
                        
                        spike_activity[region] = region_activities
            
            return spike_activity
    
    except Exception as e:
        print(f"Error calculating spike activity: {str(e)}")
        return {}

# Add brain boundary coordinates (Allen CCF approximate boundaries in mm)
def get_allen_ccf_brain_boundaries():
    """Get approximate Allen CCF brain boundaries in mm for mouse brain"""
    # These are approximate boundaries for the adult mouse brain in CCF space
    # Converted from 10um voxels to mm coordinates
    boundaries = {
        'anterior_posterior': [0, 1000],  # ~0 to 1320 in 10um units
        'medial_lateral': [50, 600],     # ~0 to 1140 in 10um units  
        'dorsal_ventral': [0, 700]       # ~0 to 800 in 10um units
    }
    return boundaries

def create_brain_boundary_wireframe():
    """Create wireframe traces for brain boundaries"""
    bounds = get_allen_ccf_brain_boundaries()
    
    # Define corners of the brain bounding box
    x_min, x_max = bounds['anterior_posterior']
    y_min, y_max = bounds['medial_lateral']
    z_min, z_max = bounds['dorsal_ventral']
    
    wireframe_traces = []
    
    # Create wireframe edges (12 edges of a rectangular box)
    edges = [
        # Bottom face (z=z_min)
        ([x_min, x_max], [y_min, y_min], [z_min, z_min]),
        ([x_max, x_max], [y_min, y_max], [z_min, z_min]),
        ([x_max, x_min], [y_max, y_max], [z_min, z_min]),
        ([x_min, x_min], [y_max, y_min], [z_min, z_min]),
        
        # Top face (z=z_max)
        ([x_min, x_max], [y_min, y_min], [z_max, z_max]),
        ([x_max, x_max], [y_min, y_max], [z_max, z_max]),
        ([x_max, x_min], [y_max, y_max], [z_max, z_max]),
        ([x_min, x_min], [y_max, y_min], [z_max, z_max]),
        
        # Vertical edges
        ([x_min, x_min], [y_min, y_min], [z_min, z_max]),
        ([x_max, x_max], [y_min, y_min], [z_min, z_max]),
        ([x_max, x_max], [y_max, y_max], [z_min, z_max]),
        ([x_min, x_min], [y_max, y_max], [z_min, z_max])
    ]
    
    for i, (x_coords, y_coords, z_coords) in enumerate(edges):
        wireframe_traces.append(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='lines',
            line=dict(color='gray', width=2, dash='dash'),
            name='Brain Boundary' if i == 0 else None,
            showlegend=True if i == 0 else False,
            hoverinfo='skip'
        ))
    
    return wireframe_traces

def get_allen_whole_brain_mesh_plotly(resolution=10):
    """Load the Allen Mouse Brain CCF whole brain mesh using AllenSDK and return Plotly mesh3d trace."""
    if ReferenceSpaceCache is None:
        print("AllenSDK not installed. Cannot load brain mesh.")
        return None
    import os
    # Download directory for CCF data
    ccf_dir = os.path.join(os.path.expanduser("~"), ".allensdk", "ccf")
    rspc = ReferenceSpaceCache(
        manifest=os.path.join(ccf_dir, f"manifest_{resolution}.json"),
        resolution=resolution,
        reference_space_key="annotation/ccf_2017"
    )
    # 997 is the structure id for whole brain
    mesh_data = rspc.get_structure_mesh(997)
    # Support tuple or dict return types, and handle extra tuple elements
    if isinstance(mesh_data, tuple):
        verts, faces = mesh_data[:2]  # Only take the first two elements
    elif isinstance(mesh_data, dict):
        verts = mesh_data['vertices']
        faces = mesh_data['triangles']
    else:
        print("Unknown mesh_data format:", type(mesh_data))
        return None
    x, y, z = verts[:, 0] / 1, verts[:, 1] / 1, verts[:, 2] / 1  # convert um to mm
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color='lightgray',
        opacity=0.5,
        name='Whole Brain Mesh',
        showscale=False
    )
    return mesh

# Updated 3D brain function with boundaries and reversed Z-axis
def create_3d_brain_with_ccf_coordinates(file_path, spike_activity=None):
    """Create 3D brain plot using Allen CCF mesh as background and overlay unit positions."""
    try:
        unit_coordinates = get_unit_ccf_coordinates_robust_fixed(file_path)
        if not unit_coordinates:
            print("No CCF coordinates found, using fallback")
            return create_simple_3d_brain_fallback()

        fig = go.Figure()
        # Add Allen whole brain mesh as background
        add_brain_wire_mesh(fig, grid_path='brainGridData.npy')
        mesh = get_allen_whole_brain_mesh_plotly()
        if mesh is not None:
            fig.add_trace(mesh)
        else:
            print("Could not add Allen mesh, falling back to wireframe.")
            wireframe_traces = create_brain_boundary_wireframe()
            for trace in wireframe_traces:
                fig.add_trace(trace)

        # If activity, calculate min/max for colorbar
        min_activity, max_activity = None, None
        if spike_activity:
            all_activities = []
            for acts in spike_activity.values():
                all_activities.extend(acts)
            if all_activities:
                min_activity = float(np.min(all_activities))
                max_activity = float(np.max(all_activities))
            else:
                min_activity = 0
                max_activity = 1

        for region, units in unit_coordinates.items():
            color = BRAIN_REGION_COLORS.get(region, '#888888')
            x_coords = [unit['x'] for unit in units]
            y_coords = [unit['z'] for unit in units]
            z_coords = [unit['y'] for unit in units]
            unit_ids = [unit['unit_id'] for unit in units]

            # Assign color to each unit using SC_Scale2Color if activity is present
            if spike_activity and region in spike_activity and min_activity is not None and max_activity is not None:
                marker_colors = [
                    f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
                    for r, g, b in [sc_scale2color(val, min_activity, max_activity, 0) for val in spike_activity[region]]
                ]
                colorscale = None
                colorbar = dict(
                    title=f'{region} Activity (Hz)',
                    tickvals=[min_activity, 0, max_activity],
                    ticktext=[f'{min_activity:.2f}', '0', f'{max_activity:.2f}']
                )
            else:
                marker_colors = color
                colorscale = None
                colorbar = None

            fig.add_trace(go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='markers',
                marker=dict(
                    size=4,
                    color=marker_colors,
                    colorscale=colorscale,
                    colorbar=colorbar,
                    opacity=0.8,
                    line=dict(width=1, color='white'),
                    symbol='circle'
                ),
                name=f'{region} (n={len(units)})',
                text=[f'{region}: Unit {uid}<br>CCF: ({u["x"]:.1f}, {u["y"]:.1f}, {u["z"]:.1f}) mm<br>Click to plot PSTH'
                      for uid, u in zip(unit_ids, units)],
                hovertemplate='<b>%{text}</b><extra></extra>',
                customdata=unit_ids
            ))

        # Add brain reference points (keep Bregma reference larger for visibility)
        bregma_x, bregma_y, bregma_z = 540, 570, 44
        fig.add_trace(go.Scatter3d(
            x=[bregma_x],
            y=[bregma_y],
            z=[bregma_z],
            mode='markers',
            marker=dict(
                size=10,
                color='red',
                symbol='cross',
                line=dict(width=2, color='black')
            ),
            name='Bregma Reference',
            text='Bregma (0, 0, 0)',
            hovertemplate='<b>%{text}</b><extra></extra>',
            customdata=['bregma']
        ))

        bounds = get_allen_ccf_brain_boundaries()
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title='Anterior-Posterior (mm)',
                    range=[bounds['anterior_posterior'][0] - 0.5, bounds['anterior_posterior'][1] + 0.5],
                    showgrid=True,
                    gridcolor='lightgray',
                    showline=True,
                    linecolor='black'
                ),
                yaxis=dict(
                    title='Medial-Lateral (mm)',
                    range=[bounds['medial_lateral'][0] - 0.5, bounds['medial_lateral'][1] + 0.5],
                    showgrid=True,
                    gridcolor='lightgray',
                    showline=True,
                    linecolor='black'
                ),
                zaxis=dict(
                    title='Dorsal-Ventral (mm)',
                    range=[bounds['dorsal_ventral'][1] + 0.5, bounds['dorsal_ventral'][0] - 0.5],
                    showgrid=True,
                    gridcolor='lightgray',
                    showline=True,
                    linecolor='black'
                ),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                aspectmode='manual',
                aspectratio=dict(
                    x=bounds['anterior_posterior'][1] / max(bounds['anterior_posterior'][1], bounds['medial_lateral'][1], bounds['dorsal_ventral'][1]),
                    y=bounds['medial_lateral'][1] / max(bounds['anterior_posterior'][1], bounds['medial_lateral'][1], bounds['dorsal_ventral'][1]),
                    z=bounds['dorsal_ventral'][1] / max(bounds['anterior_posterior'][1], bounds['medial_lateral'][1], bounds['dorsal_ventral'][1])
                ),
                bgcolor='rgba(240, 240, 240, 0.1)'
            ),
            title='3D Brain with Allen CCF Boundaries<br>Units plotted at real recording locations<br>Click on any unit to plot its PSTH',
            showlegend=True,
            height=600
        )

        return fig

    except Exception as e:
        print(f"Error creating 3D brain with CCF coordinates: {str(e)}")
        return create_simple_3d_brain_fallback()
# Updated fallback function with boundaries and reversed Z-axis
def create_simple_3d_brain_fallback():
    """Fallback 3D brain with Allen CCF mesh and boundaries if CCF coordinates aren't available"""
    fig = go.Figure()
    
    # Add Allen CCF mesh if available
    mesh = get_allen_whole_brain_mesh_plotly()
    if mesh is not None:
        fig.add_trace(mesh)
    else:
        print("Could not add Allen mesh, falling back to wireframe only.")

    # Optionally, add wireframe boundaries for clarity
    wireframe_traces = create_brain_boundary_wireframe()
    for trace in wireframe_traces:
        fig.add_trace(trace)

    bounds = get_allen_ccf_brain_boundaries()
    fig.update_layout(
        title='3D Brain View - No CCF coordinates found<br>Showing Allen CCF mesh and boundaries',
        scene=dict(
            xaxis=dict(
                title='Anterior-Posterior (mm)',
                range=[bounds['anterior_posterior'][0] - 0.5, bounds['anterior_posterior'][1] + 0.5],
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title='Medial-Lateral (mm)',
                range=[bounds['medial_lateral'][0] - 0.5, bounds['medial_lateral'][1] + 0.5],
                showgrid=True,
                gridcolor='lightgray'
            ),
            zaxis=dict(
                title='Dorsal-Ventral (mm)',
                # REVERSED Z-axis in fallback too
                range=[bounds['dorsal_ventral'][1] + 0.5, bounds['dorsal_ventral'][0] - 0.5],
                showgrid=True,
                gridcolor='lightgray'
            ),
            aspectmode='manual',
            aspectratio=dict(
                x=bounds['anterior_posterior'][1] / max(bounds['anterior_posterior'][1], bounds['medial_lateral'][1], bounds['dorsal_ventral'][1]),
                y=bounds['medial_lateral'][1] / max(bounds['anterior_posterior'][1], bounds['medial_lateral'][1], bounds['dorsal_ventral'][1]),
                z=bounds['dorsal_ventral'][1] / max(bounds['anterior_posterior'][1], bounds['medial_lateral'][1], bounds['dorsal_ventral'][1])
            )
        )
    )
    
    return fig

# Updated highlight function with reversed Z-axis
def highlight_selected_unit_in_3d(selected_unit, selected_file, color_mode, analysis_mode, event_field, activity_start, activity_end):
    """Highlight the selected unit in the 3D brain plot with Allen mesh and wire mesh, no bounding box."""
    if not selected_file or analysis_mode != 'single' or not selected_unit or selected_unit == 'all':
        raise dash.exceptions.PreventUpdate

    file_path = os.path.join(NWB_DIR, selected_file)

    try:
        # Calculate spike activity if in activity mode and event is selected
        spike_activity = None
        if color_mode == 'activity' and event_field:
            if activity_start is not None and activity_end is not None:
                custom_window = [activity_start, activity_end]
            else:
                custom_window = [-0.5, 1.0]
            spike_activity = calculate_unit_spike_activity(file_path, event_field, window=custom_window)

        # Get actual CCF coordinates
        unit_coordinates = get_unit_ccf_coordinates_robust_fixed(file_path)

        if not unit_coordinates:
            raise dash.exceptions.PreventUpdate

        fig = go.Figure()

        # --- Add Allen mesh as background ---
        mesh = get_allen_whole_brain_mesh_plotly()
        if mesh is not None:
            fig.add_trace(mesh)

        # --- Add wire mesh (from brainGridData.npy) ---
        add_brain_wire_mesh(fig, grid_path='brainGridData.npy')

        # Add brain regions with actual unit locations (same as before...)
        for region, units in unit_coordinates.items():
            color = BRAIN_REGION_COLORS.get(region, '#888888')

            # Separate selected unit from others
            selected_units = []
            other_units = []

            for unit in units:
                if str(unit['unit_id']) == str(selected_unit):
                    selected_units.append(unit)
                else:
                    other_units.append(unit)

            # Plot other units (normal size)
            if other_units:
                x_coords = [unit['x'] for unit in other_units]
                y_coords = [unit['y'] for unit in other_units]
                z_coords = [unit['z'] for unit in other_units]
                unit_ids = [unit['unit_id'] for unit in other_units]
                if spike_activity and region in spike_activity:
                    all_activities = []
                    for acts in spike_activity.values():
                        all_activities.extend(acts)
                    min_activity = float(np.min(all_activities)) if all_activities else 0
                    max_activity = float(np.max(all_activities)) if all_activities else 1

                    all_unit_ids = [u['unit_id'] for u in units]
                    marker_colors = [
                        f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
                        for unit_id in unit_ids
                        for r, g, b in [sc_scale2color(
                            spike_activity[region][next(i for i, uid in enumerate(all_unit_ids) if uid == unit_id)],
                            min_activity, max_activity, 0)]
                    ]
                    colorscale = None
                    colorbar = None
                else:
                    marker_colors = color
                    colorscale = None
                    colorbar = None
                fig.add_trace(go.Scatter3d(
                    x=x_coords, y=y_coords, z=z_coords,
                    mode='markers',
                    marker=dict(
                        size=4, color=marker_colors, colorscale=colorscale, colorbar=colorbar,
                        opacity=0.6, line=dict(width=1, color='white'), symbol='circle'
                    ),
                    name=f'{region} (other units)',
                    text=[f'{region}: Unit {uid}<br>CCF: ({u["x"]:.1f}, {u["y"]:.1f}, {u["z"]:.1f}) mm<br>Click to plot PSTH'
                          for uid, u in zip(unit_ids, other_units)],
                    hovertemplate='<b>%{text}</b><extra></extra>',
                    customdata=unit_ids, showlegend=False
                ))

            # Plot selected unit (highlighted)
            if selected_units:
                unit = selected_units[0]
                if spike_activity and region in spike_activity:
                    all_activities = []
                    for acts in spike_activity.values():
                        all_activities.extend(acts)
                    min_activity = float(np.min(all_activities)) if all_activities else 0
                    max_activity = float(np.max(all_activities)) if all_activities else 1

                    all_unit_ids = [u['unit_id'] for u in units]
                    unit_idx = next(i for i, uid in enumerate(all_unit_ids) if str(uid) == str(selected_unit))
                    selected_activity = spike_activity[region][unit_idx]
                    r, g, b = sc_scale2color(selected_activity, min_activity, max_activity, 0)
                    marker_color = f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
                    colorbar = dict(
                        title=f'Activity (Hz)',
                        tickvals=[min_activity, 0, max_activity],
                        ticktext=[f'{min_activity:.2f}', '0', f'{max_activity:.2f}']
                    )
                else:
                    marker_color = 'yellow'
                    colorbar = None

                fig.add_trace(go.Scatter3d(
                    x=[unit['x']], y=[unit['y']], z=[unit['z']],
                    mode='markers',
                    marker=dict(
                        size=8, color=marker_color,
                        colorscale=None,
                        colorbar=colorbar, opacity=1.0,
                        line=dict(width=3, color='black'), symbol='circle'
                    ),
                    name=f'Selected: {region} Unit {selected_unit}',
                    text=f'{region}: Unit {selected_unit} (SELECTED)<br>CCF: ({unit["x"]:.1f}, {unit["y"]:.1f}, {unit["z"]:.1f}) mm',
                    hovertemplate='<b>%{text}</b><extra></extra>',
                    customdata=[selected_unit]
                ))

        # Add brain reference points
        bregma_x, bregma_y, bregma_z = 540, 570, 44
        fig.add_trace(go.Scatter3d(
            x=[bregma_x], y=[bregma_y], z=[bregma_z],
            mode='markers',
            marker=dict(size=10, color='red', symbol='cross', line=dict(width=2, color='black')),
            name='Bregma Reference',
            text='Bregma (0, 0, 0)',
            hovertemplate='<b>%{text}</b><extra></extra>',
            customdata=['bregma']
        ))

        # Get boundaries for axis limits
        bounds = get_allen_ccf_brain_boundaries()

        # Update layout with brain boundaries and REVERSED Z-axis
        title_text = f'3D Brain - Unit {selected_unit} Selected<br>Click on any unit to plot its PSTH'
        if color_mode == 'activity' and event_field:
            window_start = activity_start if activity_start is not None else -0.5
            window_end = activity_end if activity_end is not None else 1.0
            title_text += f'<br>Activity calculated from {window_start}s to {window_end}s relative to {event_field}'

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title='Anterior-Posterior (mm)',
                    range=[bounds['anterior_posterior'][0] - 0.5, bounds['anterior_posterior'][1] + 0.5],
                    showgrid=True, gridcolor='lightgray', showline=True, linecolor='black'
                ),
                yaxis=dict(
                    title='Medial-Lateral (mm)',
                    range=[bounds['medial_lateral'][0] - 0.5, bounds['medial_lateral'][1] + 0.5],
                    showgrid=True, gridcolor='lightgray', showline=True, linecolor='black'
                ),
                zaxis=dict(
                    title='Dorsal-Ventral (mm)',
                    # REVERSED Z-axis in highlight function too
                    range=[bounds['dorsal_ventral'][1] + 0.5, bounds['dorsal_ventral'][0] - 0.5],
                    showgrid=True, gridcolor='lightgray', showline=True, linecolor='black'
                ),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                aspectmode='manual',
                aspectratio=dict(
                    x=bounds['anterior_posterior'][1] / max(bounds['anterior_posterior'][1], bounds['medial_lateral'][1], bounds['dorsal_ventral'][1]),
                    y=bounds['medial_lateral'][1] / max(bounds['anterior_posterior'][1], bounds['medial_lateral'][1], bounds['dorsal_ventral'][1]),
                    z=bounds['dorsal_ventral'][1] / max(bounds['anterior_posterior'][1], bounds['medial_lateral'][1], bounds['dorsal_ventral'][1])
                ),
                bgcolor='rgba(240, 240, 240, 0.1)'
            ),
            title=title_text,
            showlegend=True,
            height=600
        )

        return fig

    except Exception as e:
        print(f"Error highlighting selected unit: {str(e)}")
        raise dash.exceptions.PreventUpdate

@lru_cache(maxsize=4)
def get_nwbfile_cached(file_path):
    with NWBHDF5IO(file_path, 'r') as io:
        return io.read()

# Dash app initialization
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Updated layout with PSTH plots on left and 3D brain on right
app.layout = dbc.Container([
    # Compact Header
    dbc.Card([
        dbc.CardBody([
            html.H4('NWB Data Viewer', className='text-center mb-1'),
            html.H6('Designed by: Parviz Ghaderi', className='text-center text-muted mb-1'),
            html.P([
                html.Small([
                    'Data related to: Contextual gating of whisker-evoked responses by frontal cortex supports flexible decision making',
                    html.Br(),
                    'Parviz Ghaderi, Sylvain Crochet, and Carl Petersen, 2025'
                ])
            ], className='text-center text-muted mb-0')
        ])
    ], color='light', className='mb-2'),
    
    # Main controls row
    dbc.Row([
        dbc.Col([
            html.Label('NWB File:', className='mb-1'),
            dcc.Dropdown(
                id='nwb-file-dropdown',
                options=[{'label': f, 'value': f} for f in list_nwb_files()],
                placeholder='Choose session...',
                className='mb-2'
            ),
        ], width=3),
        
        dbc.Col([
            html.Label('Event for PSTH:', className='mb-1'),
            dcc.Dropdown(id='event-dropdown', placeholder='Select event...', className='mb-2'),
        ], width=2),
        
        dbc.Col([
            html.Label('Analysis Mode:', className='mb-1'),
            dcc.RadioItems(
                id='analysis-mode',
                options=[
                    {'label': 'Single Unit', 'value': 'single'},
                    {'label': 'By Brain Region', 'value': 'region'}
                ],
                value='region',
                className='mb-2'
            ),
        ], width=2),
        
        dbc.Col([
            html.Label('Unit:', className='mb-1'),
            dcc.Dropdown(id='unit-dropdown', placeholder='Select unit...', className='mb-2'),
        ], width=2),
        
        dbc.Col([
            html.Label('Behavioral Signal:', className='mb-1'),
            dcc.Dropdown(id='behavioral-signal-dropdown', placeholder='Select signal...', className='mb-2'),
        ], width=3),
    ], className='mb-2'),
    
    # Secondary controls row
    dbc.Row([
        dbc.Col([
            html.Label('3D Brain Color Mode:', className='mb-1'),
            dcc.RadioItems(
                id='brain-color-mode',
                options=[
                    {'label': 'Brain Region Colors', 'value': 'region'},
                    {'label': 'Unit Activity', 'value': 'activity'}
                ],
                value='region',
                className='mb-2'
            ),
        ], width=3),
        
        # Activity interval controls (only shown when Unit Activity is selected)
        dbc.Col([
            html.Div(id='activity-interval-controls', children=[
                html.Label('Activity Interval (s):', className='mb-1'),
                dbc.Row([
                    dbc.Col([
                        html.Label('Start:', style={'fontSize': '11px'}),
                        dcc.Input(
                            id='activity-start-input',
                            type='number',
                            value=-0.5,
                            step=0.01,
                            size='sm',
                            style={'width': '100%', 'fontSize': '11px'}
                        )
                    ], width=6),
                    dbc.Col([
                        html.Label('End:', style={'fontSize': '11px'}),
                        dcc.Input(
                            id='activity-end-input',
                            type='number',
                            value=1.0,
                            step=0.01,
                            size='sm',
                            style={'width': '100%', 'fontSize': '11px'}
                        )
                    ], width=6)
                ])
            ])
        ], width=3),
        
        # Trial grouping controls
        dbc.Col([
            html.Label('Trial Filters:', className='mb-1'),
            dbc.Row([
                dbc.Col([
                    html.Label('lick_flag:', style={'fontSize': '10px'}),
                    dcc.Dropdown(id='lick-flag-dropdown', placeholder='Select...', style={'fontSize': '11px'}),
                ], width=4),
                dbc.Col([
                    html.Label('trial_type:', style={'fontSize': '10px'}),
                    dcc.Dropdown(id='trial-type-dropdown', placeholder='Select...', style={'fontSize': '11px'}),
                ], width=4),
                dbc.Col([
                    html.Label('early_lick:', style={'fontSize': '10px'}),
                    dcc.Dropdown(id='early-lick-dropdown', placeholder='Select...', style={'fontSize': '11px'}),
                ], width=4),
            ])
        ], width=4),
        
        dbc.Col([
            html.Br(),  # Space for alignment
            dbc.Button('Plot PSTH', id='plot-grouped-btn', color='primary', size='sm', className='mb-2'),
        ], width=2),
    ], className='mb-2'),
    
    # Compact metadata and trial info
    dbc.Row([
        dbc.Col([
            html.Div(id='file-metadata', className='mb-1', style={'fontSize': '10px', 'maxHeight': '40px', 'overflowY': 'auto'}),
        ], width=8),
        dbc.Col([
            html.Div(id='trial-count-info', className='text-info', style={'fontSize': '10px'}),
        ], width=4),
    ], className='mb-2'),
    
    # Main content area - Split between PSTH plots (left) and 3D brain (right)
    dbc.Row([
        # Left column - PSTH plots
        dbc.Col([
            # Neural PSTH plot
            dbc.Row([
                dbc.Col([
                    html.Label('Neural PSTH:', className='mb-1'),
                    dcc.Graph(id='grouped-psth-plot', style={'height': '400px'})
                ], width=12)
            ], className='mb-2'),
            
            # Behavioral PSTH plot
            dbc.Row([
                dbc.Col([
                    html.Label('Behavioral PSTH:', className='mb-1'),
                    dcc.Graph(id='behavioral-psth-plot', style={'height': '400px'})
                ], width=12)
            ])
        ], width=6),  # Left half for PSTH plots
        
        # Right column - 3D Brain plot
        dbc.Col([
            html.Label('3D Brain View (Actual CCF Coordinates):', className='mb-1'),
            dcc.Graph(id='brain-3d-plot', style={'height': '820px'})  # Taller to match combined height of PSTH plots
        ], width=6)  # Right half for 3D brain
    ], style={'height': '85vh'}),
    
    # Compact Footer
    html.Hr(className='my-1'),
    html.P([
        html.Small('© 2025 Parviz Ghaderi | Interactive neural data viewer', className='text-muted')
    ], className='text-center mb-0')
], fluid=True, style={'height': '100vh', 'padding': '10px'})

# Callback to update file metadata
@app.callback(
    Output('file-metadata', 'children'),
    [Input('nwb-file-dropdown', 'value')]
)
def update_file_metadata(selected_file):
    if not selected_file:
        return "Select a file to view metadata"
    
    file_path = os.path.join(NWB_DIR, selected_file)
    try:
        metadata = get_nwb_metadata(file_path)
        metadata_content = []
        for key, value in metadata.items():
            metadata_content.append(html.Div(f"{key}: {value}"))
        return metadata_content
    except Exception as e:
        return f"Error loading metadata: {str(e)}"

# Fixed callback for event and unit dropdowns with brain region prefix (COMBINED)
@app.callback(
    [Output('event-dropdown', 'options'), Output('event-dropdown', 'value'),
     Output('unit-dropdown', 'options'), Output('unit-dropdown', 'value'),
     Output('unit-dropdown', 'style')],
    [Input('nwb-file-dropdown', 'value'),
     Input('analysis-mode', 'value')],
    [State('unit-dropdown', 'value')]
)
def update_event_unit_dropdowns_combined(selected_file, analysis_mode, current_unit_value):
    ctx = dash.callback_context
    
    # Determine which input triggered the callback
    if not ctx.triggered:
        trigger_id = None
    else:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle unit dropdown style based on analysis mode
    if analysis_mode == 'region':
        unit_style = {'display': 'none'}  # Hide unit dropdown in region mode
    else:
        unit_style = {'display': 'block'}  # Show unit dropdown in single unit mode
    
    # Handle file selection change
    if not selected_file:
        return [], None, [], None, unit_style
    
    file_path = os.path.join(NWB_DIR, selected_file)
    try:
        # Get event fields
        event_fields = get_trial_event_fields(file_path)
        event_options = [{'label': f, 'value': f} for f in event_fields]
        
        # Get units with brain regions
        unit_info = get_unit_ids_with_regions(file_path)
        
        # Sort units by brain region first, then by unit ID
        unit_info_sorted = sorted(unit_info, key=lambda x: (x['region'], int(x['id']) if x['id'].isdigit() else 0))
        
        # Create unit options with brain region prefix
        unit_options = []
        
        # Add "All Units" option first
        unit_options.append({'label': 'All Units', 'value': 'all'})
        
        # Add individual units with region prefix (format: "RegionName: Unit ID")
        for unit in unit_info_sorted:
            unit_options.append({
                'label': f"{unit['region']}: Unit {unit['id']}",
                'value': unit['value']
            })
        
        # Determine unit value
        if trigger_id == 'nwb-file-dropdown':
            # File changed, set default value
            unit_value = 'all'
        elif trigger_id == 'analysis-mode':
            # Analysis mode changed, preserve current value if valid in single mode, reset in region mode
            if analysis_mode == 'region':
                unit_value = 'all'  # Reset to all in region mode
            else:
                # Keep current value if it exists in the options, otherwise default to 'all'
                valid_values = [opt['value'] for opt in unit_options]
                unit_value = current_unit_value if current_unit_value in valid_values else 'all'
        else:
            # No specific trigger, keep current value if valid
            valid_values = [opt['value'] for opt in unit_options]
            unit_value = current_unit_value if current_unit_value in valid_values else 'all'
        
        return (event_options, 
                event_options[0]['value'] if event_options else None, 
                unit_options, 
                unit_value,
                unit_style)
    
    except Exception as e:
        print(f"Error updating dropdowns: {str(e)}")
        return [], None, [], None, unit_style

# Callback to update behavioral signal dropdown
@app.callback(
    [Output('behavioral-signal-dropdown', 'options'), Output('behavioral-signal-dropdown', 'value')],
    [Input('nwb-file-dropdown', 'value')]
)
def update_behavioral_signal_dropdown(selected_file):
    if not selected_file:
        return [], None
    
    file_path = os.path.join(NWB_DIR, selected_file)
    try:
        behavioral_signals = get_behavioral_signals(file_path)
        options = [{'label': signal, 'value': signal} for signal in behavioral_signals]
        return options, options[0]['value'] if options else None
    except Exception as e:
        print(f"Error getting behavioral signals: {str(e)}")
        return [], None

# Callback to update trial grouping dropdowns
@app.callback(
    [Output('lick-flag-dropdown', 'options'), Output('lick-flag-dropdown', 'value'),
     Output('trial-type-dropdown', 'options'), Output('trial-type-dropdown', 'value'),
     Output('early-lick-dropdown', 'options'), Output('early-lick-dropdown', 'value')],
    [Input('nwb-file-dropdown', 'value')]
)
def update_trial_grouping_dropdowns(selected_file):
    if not selected_file:
        return [], None, [], None, [], None
    
    file_path = os.path.join(NWB_DIR, selected_file)
    try:
        grouping_columns, trials_df = get_trial_grouping_options(file_path)
        
        # Initialize options
        lick_flag_options = []
        trial_type_options = []
        early_lick_options = []
        
        # Check for lick_flag column
        if 'lick_flag' in trials_df.columns:
            unique_vals = sorted(trials_df['lick_flag'].dropna().unique())
            lick_flag_options = [{'label': str(val), 'value': val} for val in unique_vals]
        
        # Check for trial_type column
        if 'trial_type' in trials_df.columns:
            unique_vals = sorted(trials_df['trial_type'].dropna().unique())
            trial_type_options = [{'label': str(val), 'value': val} for val in unique_vals]
        
        # Check for early_lick column
        if 'early_lick' in trials_df.columns:
            unique_vals = sorted(trials_df['early_lick'].dropna().unique())
            early_lick_options = [{'label': str(val), 'value': val} for val in unique_vals]
        
        return (lick_flag_options, None,
                trial_type_options, None,
                early_lick_options, None)
    
    except Exception as e:
        print(f"Error updating trial grouping dropdowns: {str(e)}")
        return [], None, [], None, [], None

# Updated PSTH callback with brain region support and specific colors
@app.callback(
    [Output('grouped-psth-plot', 'figure'),
     Output('behavioral-psth-plot', 'figure'),
     Output('trial-count-info', 'children')],
    [Input('plot-grouped-btn', 'n_clicks')],
    [State('nwb-file-dropdown', 'value'),
     State('event-dropdown', 'value'),
     State('analysis-mode', 'value'),
     State('unit-dropdown', 'value'),
     State('behavioral-signal-dropdown', 'value'),
     State('lick-flag-dropdown', 'value'),
     State('trial-type-dropdown', 'value'),
     State('early-lick-dropdown', 'value')]
)
def update_both_psth_plots(n_clicks, selected_file, event_field, analysis_mode, unit_id, behavioral_signal,
                          lick_flag, trial_type, early_lick):
    if not n_clicks or not selected_file or not event_field or not behavioral_signal:
        return go.Figure(), go.Figure(), ""
    
    file_path = os.path.join(NWB_DIR, selected_file)
    
    # Create trial filters dictionary
    trial_filters = {}
    filter_labels = []
    
    if lick_flag is not None:
        trial_filters['lick_flag'] = lick_flag
        filter_labels.append(f'lick_flag={lick_flag}')
    
    if trial_type is not None:
        trial_filters['trial_type'] = trial_type
        filter_labels.append(f'trial_type={trial_type}')
    
    if early_lick is not None:
        trial_filters['early_lick'] = early_lick
        filter_labels.append(f'early_lick={early_lick}')
    
    # Calculate Neural PSTH based on analysis mode
    if analysis_mode == 'region':
        # Calculate PSTH by brain region
        region_psths, bin_centers, trial_count = calculate_psth_by_brain_region(file_path, event_field, trial_filters)
        neural_result = (region_psths, bin_centers, trial_count)
    else:
        # Calculate PSTH for single unit (original method)
        if not unit_id:
            return go.Figure(), go.Figure(), "Please select a unit for single unit analysis."
        neural_result = calculate_psth_with_grouping(file_path, event_field, unit_id, trial_filters)
    
    # Calculate Behavioral PSTH
    behavioral_result = calculate_behavioral_psth(file_path, event_field, behavioral_signal, trial_filters)
    
    # Determine common x-axis range
    x_min, x_max = -1, 2  # Default range
    
    # Create Neural PSTH figure
    neural_fig = go.Figure()
    
    if analysis_mode == 'region' and neural_result[0]:  # Brain region mode
        region_psths, bin_centers, trial_count = neural_result
        
        if bin_centers is not None:
            x_min = min(x_min, min(bin_centers))
            x_max = max(x_max, max(bin_centers))
        
        region_info = []
        
        # Sort regions to ensure consistent order (put known regions first)
        known_regions = [r for r in region_psths.keys() if r in BRAIN_REGION_COLORS]
        unknown_regions = [r for r in region_psths.keys() if r not in BRAIN_REGION_COLORS]
        sorted_regions = known_regions + unknown_regions
        
        for i, region in enumerate(sorted_regions):  # FIXED: Added enumerate() properly
            data = region_psths[region]  # Now 'region' is the actual region name
            color = get_region_color(region, i)  # 'i' is the index for color selection
            
            # Add shaded area for SEM with appropriate alpha
            neural_fig.add_trace(go.Scatter(
                x=np.concatenate([bin_centers, bin_centers[::-1]]),
                y=np.concatenate([data['mean'] + data['sem'], (data['mean'] - data['sem'])[::-1]]),
                fill='toself',
                fillcolor=hex_to_rgba(color, 0.3),
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
                name=f'{region} SEM'
            ))
            
            # Add the main PSTH line
            neural_fig.add_trace(go.Scatter(
                x=bin_centers, 
                y=data['mean'], 
                mode='lines', 
                name=f'{region} (n={data["n_units"]} units)', 
                line=dict(color=color, width=2)
            ))
            
            region_info.append(f"{region}: {data['n_units']} units")
        
        # Create title with filter information
        filter_str = ' & '.join(filter_labels) if filter_labels else 'All trials'
        neural_title = f'Neural PSTH by Brain Region<br>Filters: {filter_str}<br>Event: {event_field}'
        
    else:  # Single unit mode
        # ...existing code...
        if neural_result[0] is not None:
            bin_centers, mean_psth, sem_psth, trial_count = neural_result
            x_min = min(x_min, min(bin_centers))
            x_max = max(x_max, max(bin_centers))
            bin_size = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 0.005  # Default to 5 ms
        
            # Convert to Hz
            mean_psth_hz = mean_psth / bin_size
            sem_psth_hz = sem_psth / bin_size
        
            # Add shaded area for SEM
            neural_fig.add_trace(go.Scatter(
                x=np.concatenate([bin_centers, bin_centers[::-1]]),
                y=np.concatenate([mean_psth_hz + sem_psth_hz, (mean_psth_hz - sem_psth_hz)[::-1]]),
                fill='toself',
                fillcolor='rgba(255, 100, 80, 0.3)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
                name='SEM'
            ))
        
            # Add the main PSTH line
            neural_fig.add_trace(go.Scatter(
                x=bin_centers, 
                y=mean_psth_hz, 
                mode='lines', 
                name='Neural PSTH (Hz)', 
                line=dict(color='red', width=2)
            ))
        # ...existing code...        
        # Create title with filter information
        filter_str = ' & '.join(filter_labels) if filter_labels else 'All trials'
        neural_title = f'Neural PSTH: {filter_str}<br>Event: {event_field}, Unit: {unit_id}'
    
    # Update x-axis range based on behavioral data
    if behavioral_result[0] is not None:
        time_vector, mean_trace, sem_trace, _ = behavioral_result
        x_min = min(x_min, min(time_vector))
        x_max = max(x_max, max(time_vector))
    
    # Add vertical line at event time (t=0) for neural plot
    neural_fig.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="Event")
    
    neural_fig.update_layout(
        title=neural_title,
        xaxis_title='Time (s) from event',
        yaxis_title='Firing rate (Hz)',
        template='plotly_white',
        hovermode='x unified',
        uirevision='neural-constant',
        xaxis=dict(
            range=[x_min, x_max],
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            showline=True,
            showgrid=True
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)',  # Semi-transparent background
            bordercolor='rgba(0,0,0,0.3)',
            borderwidth=1
        )
    )
    
    # Create Behavioral PSTH figure (unchanged)
    behavioral_fig = go.Figure()
    
    if behavioral_result[0] is not None:
        time_vector, mean_trace, sem_trace, _ = behavioral_result
        
        # Add shaded area for SEM
        behavioral_fig.add_trace(go.Scatter(
            x=np.concatenate([time_vector, time_vector[::-1]]),
            y=np.concatenate([mean_trace + sem_trace, (mean_trace - sem_trace)[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 100, 255, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            name='SEM'
        ))
        
        # Add the main behavioral trace
        behavioral_fig.add_trace(go.Scatter(
            x=time_vector, 
            y=mean_trace, 
            mode='lines', 
            name='Behavioral PSTH', 
            line=dict(color='blue', width=2)
        ))
    
    # Add vertical line at event time (t=0) for behavioral plot
    behavioral_fig.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="Event")
    
    behavioral_title = f'Behavioral PSTH: {behavioral_signal}<br>Filters: {filter_str}<br>Event: {event_field}'
    
    behavioral_fig.update_layout(
        title=behavioral_title,
        xaxis_title='Time (s) from event',
        yaxis_title=f'{behavioral_signal} (averaged)',
        template='plotly_white',
        hovermode='x unified',
        uirevision='behavioral-constant',
        xaxis=dict(
            range=[x_min, x_max],  # Same range as neural plot
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            showline=True,
            showgrid=True
        )
    )
    
    # Create trial info
    if analysis_mode == 'region':
        if region_psths:
            region_info_str = ', '.join([f"{region}: {data['n_units']} units" for region, data in region_psths.items()])
            trial_info = f"Brain regions: {region_info_str}<br>Trials: {trial_count}"
        else:
            trial_info = "No brain region data available"
    else:
        trial_info = f"Number of trials matching criteria: {trial_count if neural_result[0] is not None else 0}"
    
    if filter_labels:
        trial_info += f"<br>Filters: {', '.join(filter_labels)}"
    
    return neural_fig, behavioral_fig, trial_info

# New callback to show/hide activity interval controls
@app.callback(
    Output('activity-interval-controls', 'style'),
    [Input('brain-color-mode', 'value')]
)
def toggle_activity_interval_controls(color_mode):
    if color_mode == 'activity':
        return {'display': 'block'}
    else:
        return {'display': 'none'}

# Updated callback for 3D brain plot with debounced activity interval inputs
@app.callback(
    Output('brain-3d-plot', 'figure'),
    [Input('nwb-file-dropdown', 'value'),
     Input('brain-color-mode', 'value'),
     Input('plot-grouped-btn', 'n_clicks')],
    [State('event-dropdown', 'value'),
     State('activity-start-input', 'value'),
     State('activity-end-input', 'value')]
)
def update_3d_brain_plot_ccf_with_interval(selected_file, color_mode, n_clicks, event_field, activity_start, activity_end):
    if not selected_file:
        return go.Figure()
    
    file_path = os.path.join(NWB_DIR, selected_file)
    
    try:
        # Calculate spike activity if in activity mode and event is selected
        spike_activity = None
        if color_mode == 'activity' and event_field:
            # Use custom interval if provided, otherwise use defaults
            if activity_start is not None and activity_end is not None:
                custom_window = [activity_start, activity_end]
                print(f"Using custom activity window: [{activity_start}, {activity_end}]")
            else:
                custom_window = [-0.5, 1.0]  # Default window
                print(f"Using default activity window: {custom_window}")
            
            print(f"Calculating spike activity for event: {event_field}")
            spike_activity = calculate_unit_spike_activity(file_path, event_field, window=custom_window)
            
            if spike_activity:
                print(f"Successfully calculated spike activity for {len(spike_activity)} regions")
            else:
                print("No spike activity calculated")
        
        # Create 3D brain plot
        print("Creating 3D brain plot...")
        fig = create_3d_brain_with_ccf_coordinates(file_path, spike_activity)
        
        # Update title to show activity calculation window if in activity mode
        if color_mode == 'activity' and event_field:
            window_start = activity_start if activity_start is not None else -0.5
            window_end = activity_end if activity_end is not None else 1.0
            current_title = fig.layout.title.text
            fig.update_layout(
                title=f"{current_title}<br>Activity calculated from {window_start}s to {window_end}s relative to {event_field}"
            )
        
        print("3D brain plot created successfully")
        return fig
    
    except Exception as e:
        print(f"Error creating 3D brain plot: {str(e)}")
        import traceback
        traceback.print_exc()
        return create_simple_3d_brain_fallback()

# Add a separate callback to update 3D plot when activity interval changes (with button trigger)
@app.callback(
    Output('brain-3d-plot', 'figure', allow_duplicate=True),
    [Input('activity-start-input', 'value'),
     Input('activity-end-input', 'value')],
    [State('nwb-file-dropdown', 'value'),
     State('brain-color-mode', 'value'),
     State('event-dropdown', 'value')],
    prevent_initial_call=True
)
def update_3d_brain_activity_interval(activity_start, activity_end, selected_file, color_mode, event_field):
    # Only update if we're in activity mode and have all required inputs
    if not selected_file or color_mode != 'activity' or not event_field:
        raise dash.exceptions.PreventUpdate
    
    # Only proceed if both values are valid numbers
    if activity_start is None or activity_end is None:
        raise dash.exceptions.PreventUpdate
    
    # Validate that end > start
    if activity_end <= activity_start:
        print(f"Invalid interval: end ({activity_end}) must be greater than start ({activity_start})")
        raise dash.exceptions.PreventUpdate
    
    file_path = os.path.join(NWB_DIR, selected_file)
    
    try:
        custom_window = [activity_start, activity_end]
        print(f"Updating 3D plot with new activity window: [{activity_start}, {activity_end}]")
        
        # Calculate spike activity with new window
        spike_activity = calculate_unit_spike_activity(file_path, event_field, window=custom_window)
        
        if spike_activity:
            print(f"Recalculated spike activity for {len(spike_activity)} regions")
        else:
            print("No spike activity calculated with new window")
        
        # Create 3D brain plot
        fig = create_3d_brain_with_ccf_coordinates(file_path, spike_activity)
        
        # Update title to show new calculation window
        current_title = fig.layout.title.text
        fig.update_layout(
            title=f"{current_title}<br>Activity calculated from {activity_start}s to {activity_end}s relative to {event_field}"
        )
        
        print("3D brain plot updated successfully with new interval")
        return fig
    
    except Exception as e:
        print(f"Error updating 3D brain plot with new interval: {str(e)}")
        import traceback
        traceback.print_exc()
        raise dash.exceptions.PreventUpdate

# Add new callback to handle 3D brain plot clicks
@app.callback(
    [Output('unit-dropdown', 'value', allow_duplicate=True),
     Output('analysis-mode', 'value', allow_duplicate=True)],
    [Input('brain-3d-plot', 'clickData')],
    [State('nwb-file-dropdown', 'value')],
    prevent_initial_call=True
)
def handle_3d_brain_click(click_data, selected_file):
    """Handle clicks on 3D brain plot to select units"""
    if not click_data or not selected_file:
        raise dash.exceptions.PreventUpdate
    
    try:
        # Extract clicked point data
        point_data = click_data['points'][0]
        
        # Check if customdata exists and contains unit ID
        if 'customdata' in point_data and point_data['customdata'] != 'bregma':
            clicked_unit_id = str(point_data['customdata'])
            print(f"3D Brain click detected: Unit {clicked_unit_id}")
            
            # Switch to single unit mode and select the clicked unit
            return clicked_unit_id, 'single'
        else:
            # Clicked on reference point or invalid data
            print("Clicked on reference point or invalid unit")
            raise dash.exceptions.PreventUpdate
    
    except Exception as e:
        print(f"Error handling 3D brain click: {str(e)}")
        raise dash.exceptions.PreventUpdate

# Add callback to automatically plot PSTH when unit is selected via 3D brain click
@app.callback(
    Output('plot-grouped-btn', 'n_clicks', allow_duplicate=True),
    [Input('unit-dropdown', 'value')],
    [State('analysis-mode', 'value'),
     State('event-dropdown', 'value'),
     State('behavioral-signal-dropdown', 'value')],
    prevent_initial_call=True
)
def auto_plot_on_unit_selection(selected_unit, analysis_mode, event_field, behavioral_signal):
    """Automatically trigger PSTH plot when a unit is selected via 3D brain click"""
    ctx = dash.callback_context
    
    # Only trigger if unit was changed and we have the required inputs
    if (ctx.triggered and 
        selected_unit and 
        selected_unit != 'all' and 
        analysis_mode == 'single' and 
        event_field and 
        behavioral_signal):
        
        print(f"Auto-triggering PSTH plot for unit {selected_unit}")
        # Increment the button click count to trigger the PSTH plot
        return 1
    
    raise dash.exceptions.PreventUpdate

# Add visual feedback callback to highlight selected unit in 3D brain
@app.callback(
    Output('brain-3d-plot', 'figure', allow_duplicate=True),
    [Input('unit-dropdown', 'value')],
    [State('nwb-file-dropdown', 'value'),
     State('brain-color-mode', 'value'),
     State('analysis-mode', 'value'),
     State('event-dropdown', 'value'),
     State('activity-start-input', 'value'),
     State('activity-end-input', 'value')],
    prevent_initial_call=True
)
def highlight_selected_unit_in_3d(selected_unit, selected_file, color_mode, analysis_mode, event_field, activity_start, activity_end):
    """Highlight the selected unit in the 3D brain plot with Allen mesh and wire mesh, no bounding box."""
    if not selected_file or analysis_mode != 'single' or not selected_unit or selected_unit == 'all':
        raise dash.exceptions.PreventUpdate

    file_path = os.path.join(NWB_DIR, selected_file)

    try:
        # Calculate spike activity if in activity mode and event is selected
        spike_activity = None
        if color_mode == 'activity' and event_field:
            if activity_start is not None and activity_end is not None:
                custom_window = [activity_start, activity_end]
            else:
                custom_window = [-0.5, 1.0]
            spike_activity = calculate_unit_spike_activity(file_path, event_field, window=custom_window)

        # Get actual CCF coordinates
        unit_coordinates = get_unit_ccf_coordinates_robust_fixed(file_path)

        if not unit_coordinates:
            raise dash.exceptions.PreventUpdate

        fig = go.Figure()

        # --- Add Allen mesh as background ---
        mesh = get_allen_whole_brain_mesh_plotly()
        if mesh is not None:
            fig.add_trace(mesh)

        # --- Add wire mesh (from brainGridData.npy) ---
        add_brain_wire_mesh(fig, grid_path='brainGridData.npy')

        # Add brain regions with actual unit locations (same as before...)
        for region, units in unit_coordinates.items():
            color = BRAIN_REGION_COLORS.get(region, '#888888')

            # Separate selected unit from others
            selected_units = []
            other_units = []

            for unit in units:
                if str(unit['unit_id']) == str(selected_unit):
                    selected_units.append(unit)
                else:
                    other_units.append(unit)

            # Plot other units (normal size)
            if other_units:
                x_coords = [unit['x'] for unit in other_units]
                y_coords = [unit['y'] for unit in other_units]
                z_coords = [unit['z'] for unit in other_units]
                unit_ids = [unit['unit_id'] for unit in other_units]
                if spike_activity and region in spike_activity:
                    all_activities = []
                    for acts in spike_activity.values():
                        all_activities.extend(acts)
                    min_activity = float(np.min(all_activities)) if all_activities else 0
                    max_activity = float(np.max(all_activities)) if all_activities else 1

                    all_unit_ids = [u['unit_id'] for u in units]
                    marker_colors = [
                        f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
                        for unit_id in unit_ids
                        for r, g, b in [sc_scale2color(
                            spike_activity[region][next(i for i, uid in enumerate(all_unit_ids) if uid == unit_id)],
                            min_activity, max_activity, 0)]
                    ]
                    colorscale = None
                    colorbar = None
                else:
                    marker_colors = color
                    colorscale = None
                    colorbar = None
                fig.add_trace(go.Scatter3d(
                    x=x_coords, y=y_coords, z=z_coords,
                    mode='markers',
                    marker=dict(
                        size=4, color=marker_colors, colorscale=colorscale, colorbar=colorbar,
                        opacity=0.6, line=dict(width=1, color='white'), symbol='circle'
                    ),
                    name=f'{region} (other units)',
                    text=[f'{region}: Unit {uid}<br>CCF: ({u["x"]:.1f}, {u["y"]:.1f}, {u["z"]:.1f}) mm<br>Click to plot PSTH'
                          for uid, u in zip(unit_ids, other_units)],
                    hovertemplate='<b>%{text}</b><extra></extra>',
                    customdata=unit_ids, showlegend=False
                ))

            # Plot selected unit (highlighted)
            if selected_units:
                unit = selected_units[0]
                if spike_activity and region in spike_activity:
                    all_activities = []
                    for acts in spike_activity.values():
                        all_activities.extend(acts)
                    min_activity = float(np.min(all_activities)) if all_activities else 0
                    max_activity = float(np.max(all_activities)) if all_activities else 1

                    all_unit_ids = [u['unit_id'] for u in units]
                    unit_idx = next(i for i, uid in enumerate(all_unit_ids) if str(uid) == str(selected_unit))
                    selected_activity = spike_activity[region][unit_idx]
                    r, g, b = sc_scale2color(selected_activity, min_activity, max_activity, 0)
                    marker_color = f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
                    colorbar = dict(
                        title=f'Activity (Hz)',
                        tickvals=[min_activity, 0, max_activity],
                        ticktext=[f'{min_activity:.2f}', '0', f'{max_activity:.2f}']
                    )
                else:
                    marker_color = 'yellow'
                    colorbar = None

                fig.add_trace(go.Scatter3d(
                    x=[unit['x']], y=[unit['y']], z=[unit['z']],
                    mode='markers',
                    marker=dict(
                        size=8, color=marker_color,
                        colorscale=None,
                        colorbar=colorbar, opacity=1.0,
                        line=dict(width=3, color='black'), symbol='circle'
                    ),
                    name=f'Selected: {region} Unit {selected_unit}',
                    text=f'{region}: Unit {selected_unit} (SELECTED)<br>CCF: ({unit["x"]:.1f}, {unit["y"]:.1f}, {unit["z"]:.1f}) mm',
                    hovertemplate='<b>%{text}</b><extra></extra>',
                    customdata=[selected_unit]
                ))

        # Add brain reference points
        bregma_x, bregma_y, bregma_z = 540, 570, 44
        fig.add_trace(go.Scatter3d(
            x=[bregma_x], y=[bregma_y], z=[bregma_z],
            mode='markers',
            marker=dict(size=10, color='red', symbol='cross', line=dict(width=2, color='black')),
            name='Bregma Reference',
            text='Bregma (0, 0, 0)',
            hovertemplate='<b>%{text}</b><extra></extra>',
            customdata=['bregma']
        ))

        # Get boundaries for axis limits
        bounds = get_allen_ccf_brain_boundaries()

        # Update layout with brain boundaries and REVERSED Z-axis
        title_text = f'3D Brain - Unit {selected_unit} Selected<br>Click on any unit to plot its PSTH'
        if color_mode == 'activity' and event_field:
            window_start = activity_start if activity_start is not None else -0.5
            window_end = activity_end if activity_end is not None else 1.0
            title_text += f'<br>Activity calculated from {window_start}s to {window_end}s relative to {event_field}'

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title='Anterior-Posterior (mm)',
                    range=[bounds['anterior_posterior'][0] - 0.5, bounds['anterior_posterior'][1] + 0.5],
                    showgrid=True, gridcolor='lightgray', showline=True, linecolor='black'
                ),
                yaxis=dict(
                    title='Medial-Lateral (mm)',
                    range=[bounds['medial_lateral'][0] - 0.5, bounds['medial_lateral'][1] + 0.5],
                    showgrid=True, gridcolor='lightgray', showline=True, linecolor='black'
                ),
                zaxis=dict(
                    title='Dorsal-Ventral (mm)',
                    # REVERSED Z-axis in highlight function too
                    range=[bounds['dorsal_ventral'][1] + 0.5, bounds['dorsal_ventral'][0] - 0.5],
                    showgrid=True, gridcolor='lightgray', showline=True, linecolor='black'
                ),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                aspectmode='manual',
                aspectratio=dict(
                    x=bounds['anterior_posterior'][1] / max(bounds['anterior_posterior'][1], bounds['medial_lateral'][1], bounds['dorsal_ventral'][1]),
                    y=bounds['medial_lateral'][1] / max(bounds['anterior_posterior'][1], bounds['medial_lateral'][1], bounds['dorsal_ventral'][1]),
                    z=bounds['dorsal_ventral'][1] / max(bounds['anterior_posterior'][1], bounds['medial_lateral'][1], bounds['dorsal_ventral'][1])
                ),
                bgcolor='rgba(240, 240, 240, 0.1)'
            ),
            title=title_text,
            showlegend=True,
            height=600
        )

        return fig

    except Exception as e:
        print(f"Error highlighting selected unit: {str(e)}")
        raise dash.exceptions.PreventUpdate

if __name__ == '__main__':
    app.run(debug=True)
