#!/usr/bin/env python3
# Phosphorylation Site Prediction: Data Processing Script
# Memory-optimized version using datatable library

import os
import gc
import random
import time
import math
import numpy as np
from tqdm import tqdm
import datatable as dt
from datatable import f, by, join
import pandas as pd
import psutil
from sklearn.model_selection import train_test_split

# Enable garbage collection
gc.enable()

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Function to calculate memory usage
def get_memory_usage():
    """Get the current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def print_memory_usage(label=""):
    """Print current memory usage with an optional label"""
    print(f"{label} Memory usage: {get_memory_usage():.2f} MB")

def load_presplit_data():
    """Load pre-split data to match transformer model"""
    print("Loading pre-split data...")
    
    # Check if the matched data directory exists
    if not os.path.exists("matched_data/split_data"):
        print("Error: matched_data/split_data directory not found.")
        print("Please run the data matching notebook first to create the splits.")
        return None, None, None
    
    # Load the pre-split data
    try:
        train_df = pd.read_csv("matched_data/split_data/train_data.csv")
        val_df = pd.read_csv("matched_data/split_data/val_data.csv")
        test_df = pd.read_csv("matched_data/split_data/test_data.csv")
        
        print(f"Successfully loaded pre-split data with:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples")
        
        # Convert to datatable Frames
        train_dt = dt.Frame(train_df)
        val_dt = dt.Frame(val_df)
        test_dt = dt.Frame(test_df)
        
        # Clean up pandas DataFrames
        del train_df, val_df, test_df
        gc.collect()
        
        return train_dt, val_dt, test_dt
    except Exception as e:
        print(f"Error loading pre-split data: {e}")
        return None, None, None

def estimate_memory_requirements(window_size):
    """
    Estimate memory requirements for processing with the given window size
    
    Returns:
    - estimated_memory: Estimated peak memory in GB
    - will_fit: Boolean indicating if processing will fit in available memory
    """
    # Get available system memory
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    # Sample data to estimate sizes
    sample_size = 1000  # Number of rows to sample for estimation
    
    # Estimate sequence data size
    try:
        with open("Sequence_data.txt", "r") as file:
            first_lines = []
            line_count = 0
            for _ in range(sample_size * 2):  # Multiply by 2 to account for header lines
                line = file.readline()
                if not line:
                    break
                first_lines.append(line)
                line_count += 1
        
        avg_line_size = sum(len(line) for line in first_lines) / line_count if line_count > 0 else 0
        total_lines = sum(1 for _ in open("Sequence_data.txt", "r"))
        sequence_data_size_gb = (avg_line_size * total_lines) / (1024**3)
        
        # Read a small subset of data to estimate feature memory requirements
        protein_count = 0
        seq_count = 0
        with open("Sequence_data.txt", "r") as file:
            for line in file:
                if line.startswith(">"):
                    protein_count += 1
                else:
                    seq_count += 1
                if protein_count >= 50:  # Sample 50 proteins
                    break
        
        # Estimate label data size
        try:
            labels_dt = dt.fread("labels.xlsx", nthreads=4)
            labels_size_gb = labels_dt.nbytes / (1024**3)
            phospho_sites_count = labels_dt.nrows
        except:
            # Fallback if Excel file can't be read
            labels_size_gb = 0.01  # Assume 10 MB for labels
            phospho_sites_count = 30000  # Approximate from documentation
        
        # Estimate feature sizes
        # AAC: 20 amino acids
        aac_size_per_row = 8 * 20  # 8 bytes per float * 20 features
        
        # DPC: 20*20 = 400 dipeptide combinations
        dpc_size_per_row = 8 * 400
        
        # TPC: 20*20*20 = 8000 tripeptide combinations
        tpc_size_per_row = 8 * 8000
        
        # Binary encoding: depends on window size
        be_size_per_row = 8 * 20 * (2 * window_size + 1)
        
        # Physicochemical: ~16 properties * window size positions
        pc_size_per_row = 8 * 16 * (2 * window_size + 1)
        
        # Estimated number of rows after processing (phospho sites + negative samples)
        est_total_rows = phospho_sites_count * 2  # positive + equal number of negative samples
        
        # Total memory required for features
        feature_size_gb = (est_total_rows * (aac_size_per_row + dpc_size_per_row + 
                                           tpc_size_per_row + be_size_per_row + 
                                           pc_size_per_row)) / (1024**3)
        
        # Add overhead for processing (temporary variables, etc.)
        processing_overhead_gb = 2.0
        
        # Estimated peak memory usage
        estimated_memory = sequence_data_size_gb + labels_size_gb + feature_size_gb + processing_overhead_gb
        
        # Add extra 20% for safety margin
        estimated_memory = estimated_memory * 1.2
        
        # Determine if it will fit in available memory (leaving 2GB for system)
        system_reserve_gb = 2.0
        will_fit = (available_memory_gb - system_reserve_gb) > estimated_memory
        
        return {
            'estimated_memory_gb': estimated_memory,
            'will_fit': will_fit,
            'available_memory_gb': available_memory_gb,
            'est_total_rows': est_total_rows,
            'sequence_data_size_gb': sequence_data_size_gb,
            'labels_size_gb': labels_size_gb,
            'feature_size_gb': feature_size_gb
        }
    
    except Exception as e:
        print(f"Error estimating memory requirements: {e}")
        return {
            'estimated_memory_gb': None,
            'will_fit': False,
            'error': str(e)
        }

def load_sequences(file_path="Sequence_data.txt"):
    """
    Load protein sequences from a FASTA file
    Using datatable for better memory efficiency
    """
    print("Loading protein sequences...")
    start_time = time.time()
    print_memory_usage("Before loading sequences")
    
    headers = []
    sequences = []
    current_header = None
    current_seq = ""
    
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                # If there is an existing sequence, save it
                if current_header:
                    headers.append(current_header)
                    sequences.append(current_seq)
                
                # Extract header ID (middle part)
                full_header = line[1:]
                parts = full_header.split("|")
                current_header = parts[1] if len(parts) > 1 else full_header
                current_seq = ""
            else:
                current_seq += line
        
        # Don't forget the last sequence
        if current_header:
            headers.append(current_header)
            sequences.append(current_seq)
    
    # Create datatable Frame
    df = dt.Frame({
        "Header": headers,
        "Sequence": sequences
    })
    
    print(f"Loaded {df.nrows} protein sequences in {time.time() - start_time:.2f} seconds")
    print_memory_usage("After loading sequences")
    
    return df

def load_labels(file_path="labels.xlsx"):
    """
    Load phosphorylation site labels
    """
    print("Loading phosphorylation site labels...")
    start_time = time.time()
    print_memory_usage("Before loading labels")
    
    try:
        # Try to read Excel directly with datatable
        df_labels = dt.fread(file_path, nthreads=4)
        print(f"Direct Excel read successful with {df_labels.nrows} rows")
    except:
        # Fallback: If datatable can't read Excel directly, try pandas
        try:
            import pandas as pd
            df_labels_pd = pd.read_excel(file_path)
            # Convert pandas DataFrame to datatable Frame
            df_labels = dt.Frame(df_labels_pd)
            del df_labels_pd
            gc.collect()
            print(f"Excel read via pandas successful with {df_labels.nrows} rows")
        except Exception as e:
            raise RuntimeError(f"Could not read labels file: {e}")
    
    # Ensure column names are as expected
    if 'UniProt ID' not in df_labels.names and 'UniProtID' in df_labels.names:
        df_labels.names = {'UniProtID': 'UniProt ID'}
    
    print(f"Loaded {df_labels.nrows} phosphorylation sites in {time.time() - start_time:.2f} seconds")
    print_memory_usage("After loading labels")
    
    return df_labels

def merge_sequence_and_labels(df_seq, df_labels):
    """
    Merge sequence data with labels data
    
    For datatable, joins are different from pandas merges
    """
    print("Merging sequences with labels...")
    start_time = time.time()
    print_memory_usage("Before merging")
    
    # First, ensure the key column names are consistent
    if 'UniProt ID' not in df_labels.names:
        raise ValueError("Label data must have 'UniProt ID' column")
    
    # Convert to pandas for the merge since datatable join is causing issues
    import pandas as pd
    
    # Convert datatable frames to pandas dataframes
    seq_pd = df_seq.to_pandas()
    labels_pd = df_labels.to_pandas()
    
    # Perform the merge using pandas
    merged_pd = pd.merge(
        seq_pd,
        labels_pd,
        left_on="Header",
        right_on="UniProt ID",
        how="inner"
    )
    
    # Add target column
    merged_pd["target"] = 1  # All these are positive examples
    
    # Convert back to datatable
    df_merged = dt.Frame(merged_pd)
    
    # Clean up pandas dataframes to save memory
    del seq_pd, labels_pd, merged_pd
    gc.collect()
    
    print(f"Merged data contains {df_merged.nrows} rows")
    print(f"Completed merge in {time.time() - start_time:.2f} seconds")
    print_memory_usage("After merging")
    
    return df_merged

def clean_data(df_merged, max_seq_length=5000):
    """
    Clean the merged data:
    - Remove sequences with length > max_seq_length
    """
    print(f"Cleaning data (removing sequences longer than {max_seq_length})...")
    start_time = time.time()
    print_memory_usage("Before cleaning")
    
    # Calculate sequence lengths and filter (using pandas for simplicity)
    import pandas as pd
    
    # Convert to pandas
    df_pd = df_merged.to_pandas()
    
    # Calculate sequence lengths
    df_pd['SeqLength'] = df_pd['Sequence'].str.len()
    
    # Drop sequences with length > max_seq_length
    df_pd = df_pd[df_pd['SeqLength'] <= max_seq_length]
    
    # Drop the sequence length column
    df_pd = df_pd.drop('SeqLength', axis=1)
    
    # Convert back to datatable
    df_merged = dt.Frame(df_pd)
    
    # Clean up
    del df_pd
    gc.collect()
    
    print(f"After cleaning, data contains {df_merged.nrows} rows")
    print(f"Completed cleaning in {time.time() - start_time:.2f} seconds")
    print_memory_usage("After cleaning")
    
    return df_merged

def generate_negative_samples(df_merged):
    """
    Generate negative samples for each protein sequence by randomly sampling
    from S/T/Y sites that are not known phosphorylation sites.
    Using same random seed as transformer code for reproducibility.
    """
    print("Generating negative samples...")
    start_time = time.time()
    print_memory_usage("Before generating negative samples")
    
    # Convert to pandas for easier processing
    df_pd = df_merged.to_pandas()
    
    # Initialize a list to store all processed rows
    all_rows = []
    
    # Group by Header (Protein ID)
    for header, group in tqdm(df_pd.groupby('Header'), desc="Processing proteins"):
        # Extract sequence from the first row (should be the same for all rows in the group)
        seq = group['Sequence'].iloc[0]
        
        # Get positive positions from the group
        positive_positions = group['Position'].astype(int).tolist()
        
        # Find all S/T/Y positions in the sequence
        sty_positions = [i+1 for i, aa in enumerate(seq) if aa in ["S", "T", "Y"]]
        
        # Exclude the positives to get negative candidates
        negative_candidates = [pos for pos in sty_positions if pos not in positive_positions]
        
        # Number of positives for this sequence
        n_pos = len(positive_positions)
        
        # Sample negative sites (same number as positives if possible, or all available if less)
        sample_size = min(n_pos, len(negative_candidates))
        if sample_size > 0:
            # Use deterministic seed for this protein (same as transformer approach)
            # but ensure different proteins get different samples
            random.seed(42 + hash(header) % 10000)
            sampled_negatives = random.sample(negative_candidates, sample_size)
            
            # Add original positive rows to the result
            all_rows.append(group)
            
            # Create new rows for negative sites
            for neg_pos in sampled_negatives:
                # Create a copy of the first row as a template
                new_row = group.iloc[0].copy()
                
                # Update the specific fields
                new_row['AA'] = seq[neg_pos - 1]
                new_row['Position'] = neg_pos
                new_row['target'] = 0
                
                # Add the new row
                all_rows.append(pd.DataFrame([new_row]))
    
    # Combine all rows
    df_final_pd = pd.concat(all_rows, ignore_index=True)
    
    # Convert back to datatable
    df_final = dt.Frame(df_final_pd)
    
    # Clean up
    del df_pd, df_final_pd, all_rows
    gc.collect()
    
    print(f"Generated data with {df_final.nrows} rows (includes both positive and negative samples)")
    print(f"Completed negative sample generation in {time.time() - start_time:.2f} seconds")
    print_memory_usage("After generating negative samples")
    
    return df_final

def split_dataset_like_transformer(df):
    """
    Split dataset to match transformer model's approach (70/15/15 train/val/test)
    Using protein-level splitting to avoid data leakage
    """
    print("Splitting data to match transformer approach...")
    
    # Convert to pandas for easier handling
    df_pd = df.to_pandas()
    
    # Get all unique protein headers
    headers = df_pd['Header'].unique()
    n_headers = len(headers)
    
    # Shuffle headers using same random seed as transformer
    np.random.seed(42)
    np.random.shuffle(headers)
    
    # Split points
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    train_split = int(n_headers * train_ratio)
    val_split = int(n_headers * (train_ratio + val_ratio))
    
    # Split headers
    train_headers = headers[:train_split]
    val_headers = headers[train_split:val_split]
    test_headers = headers[val_split:]
    
    # Create dataframes
    train_df = df_pd[df_pd['Header'].isin(train_headers)]
    val_df = df_pd[df_pd['Header'].isin(val_headers)]
    test_df = df_pd[df_pd['Header'].isin(test_headers)]
    
    print(f"Train set: {len(train_df)} samples from {len(train_headers)} proteins")
    print(f"Validation set: {len(val_df)} samples from {len(val_headers)} proteins")
    print(f"Test set: {len(test_df)} samples from {len(test_headers)} proteins")
    
    # Convert back to datatable
    train_dt = dt.Frame(train_df)
    val_dt = dt.Frame(val_df)
    test_dt = dt.Frame(test_df)
    
    # Clean up
    del df_pd, train_df, val_df, test_df
    gc.collect()
    
    return train_dt, val_dt, test_dt

def extract_window(sequence, position, window_size=5):
    """
    Extract a window of amino acids around a position
    
    Parameters:
    - sequence: Protein sequence
    - position: Position to center the window on (1-based)
    - window_size: Half size of the window on each side
    
    Returns:
    - Window sequence
    """
    pos_idx = position - 1  # Convert to 0-based index
    
    start = max(0, pos_idx - window_size)
    end = min(len(sequence), pos_idx + window_size + 1)
    
    window = sequence[start:end]
    return window

# Feature extraction functions
def extract_aac(sequence):
    """
    Extract Amino Acid Composition (AAC) features
    """
    # List of 20 standard amino acids
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    # Initialize dictionary with zeros
    aac = {aa: 0 for aa in amino_acids}
    
    # Count amino acids
    seq_length = len(sequence)
    for aa in sequence:
        if aa in aac:
            aac[aa] += 1
    
    # Convert counts to frequencies
    for aa in aac:
        aac[aa] = aac[aa] / seq_length if seq_length > 0 else 0
        
    return aac

def extract_dpc(sequence):
    """
    Extract Dipeptide Composition (DPC) features
    """
    # List of 20 standard amino acids
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    # Initialize dictionary with all possible dipeptides
    dpc = {}
    for aa1 in amino_acids:
        for aa2 in amino_acids:
            dpc[aa1 + aa2] = 0
    
    # Count dipeptides
    if len(sequence) < 2:
        return dpc
    
    for i in range(len(sequence) - 1):
        dipeptide = sequence[i:i+2]
        if dipeptide in dpc:
            dpc[dipeptide] += 1
    
    # Convert counts to frequencies
    total_dipeptides = len(sequence) - 1
    for dipeptide in dpc:
        dpc[dipeptide] = dpc[dipeptide] / total_dipeptides if total_dipeptides > 0 else 0
        
    return dpc

def extract_tpc(sequence):
    """
    Extract Tripeptide Composition (TPC) features
    """
    # List of 20 standard amino acids
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    # Initialize dictionary with a subset of all possible tripeptides
    # This uses a heuristic approach to save memory by only tracking observed tripeptides
    tpc = {}
    
    # Count tripeptides
    if len(sequence) < 3:
        return tpc
    
    for i in range(len(sequence) - 2):
        tripeptide = sequence[i:i+3]
        if all(aa in amino_acids for aa in tripeptide):  # Only count valid tripeptides
            if tripeptide not in tpc:
                tpc[tripeptide] = 0
            tpc[tripeptide] += 1
    
    # Convert counts to frequencies
    total_tripeptides = len(sequence) - 2
    for tripeptide in tpc:
        tpc[tripeptide] = tpc[tripeptide] / total_tripeptides if total_tripeptides > 0 else 0
    
    # Convert to standard format where missing tripeptides have value 0
    standard_tpc = {}
    for aa1 in amino_acids:
        for aa2 in amino_acids:
            for aa3 in amino_acids:
                tri = aa1 + aa2 + aa3
                standard_tpc[tri] = tpc.get(tri, 0)
    
    return standard_tpc

def binary_encode_amino_acid(aa):
    """
    Binary encode a single amino acid into a 20-dimensional vector
    """
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    # Initialize vector with zeros
    encoding = [0] * 20
    
    # Set the corresponding position to 1
    if aa in amino_acids:
        idx = amino_acids.index(aa)
        encoding[idx] = 1
    
    return encoding

def extract_binary_encoding(sequence, position, window_size=5):
    """
    Extract binary encoding features for a window around the phosphorylation site
    """
    # Convert position to 0-based index
    pos_idx = position - 1
    
    # Define window boundaries
    start = max(0, pos_idx - window_size)
    end = min(len(sequence), pos_idx + window_size + 1)
    
    # Extract window sequence
    window = sequence[start:end]
    
    # Pad with 'X' if necessary to reach the desired window length
    left_pad = "X" * max(0, window_size - (pos_idx - start))
    right_pad = "X" * max(0, window_size - (end - pos_idx - 1))
    padded_window = left_pad + window + right_pad
    
    # Binary encode each amino acid in the window
    binary_features = {}
    for i, aa in enumerate(padded_window):
        encoding = binary_encode_amino_acid(aa)
        for j, value in enumerate(encoding):
            binary_features[f"BE_{i*20 + j + 1}"] = value
    
    return binary_features

def load_physicochemical_properties(file_path="physiochemical_property.csv"):
    """
    Load physicochemical properties from CSV file
    """
    props_dt = dt.fread(file_path)
    
    # Convert to dictionary format for easier access
    properties = {}
    for i in range(props_dt.nrows):
        row = props_dt[i, :].to_list()[0]
        aa = row[0]  # First column is amino acid
        properties[aa] = row[1:]  # Rest are properties
    
    return properties

def extract_physicochemical_features(sequence, position, window_size=5, properties=None):
    """
    Extract physicochemical features for a window around the phosphorylation site
    """
    if properties is None:
        properties = load_physicochemical_properties()
    
    # Convert position to 0-based index
    pos_idx = position - 1
    
    # Define window boundaries
    start = max(0, pos_idx - window_size)
    end = min(len(sequence), pos_idx + window_size + 1)
    
    # Extract window sequence
    window = sequence[start:end]
    
    # Pad with 'X' if necessary to reach the desired window length
    left_pad = "X" * max(0, window_size - (pos_idx - start))
    right_pad = "X" * max(0, window_size - (end - pos_idx - 1))
    padded_window = left_pad + window + right_pad
    
    # Get properties for each amino acid in the window
    physico_features = {}
    for i, aa in enumerate(padded_window):
        if aa in properties:
            for j, value in enumerate(properties[aa]):
                physico_features[f"PC_{i*len(properties[aa]) + j + 1}"] = value
        else:
            # Default values for unknown amino acids
            prop_len = len(next(iter(properties.values())))
            for j in range(prop_len):
                physico_features[f"PC_{i*prop_len + j + 1}"] = 0
    
    return physico_features

def process_feature_batch(batch_data, feature_type, window_size=5, properties=None):
    """
    Process a batch of data for a specific feature type
    
    Parameters:
    - batch_data: List of (sequence, position, header, target) tuples
    - feature_type: 'aac', 'dpc', 'tpc', 'binary', or 'physicochemical'
    - window_size: Window size for extraction
    - properties: Physicochemical properties dictionary (for 'physicochemical' type)
    
    Returns:
    - List of dictionaries with feature values
    """
    batch_features = []
    
    for seq, pos, header, target in batch_data:
        # Extract window
        window = extract_window(seq, pos, window_size)
        
        # Extract features based on type
        if feature_type == 'aac':
            features = extract_aac(window)
        elif feature_type == 'dpc':
            features = extract_dpc(window)
        elif feature_type == 'tpc':
            features = extract_tpc(window)
        elif feature_type == 'binary':
            features = extract_binary_encoding(seq, pos, window_size)
        elif feature_type == 'physicochemical':
            features = extract_physicochemical_features(seq, pos, window_size, properties)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        # Add identifier columns
        features['Header'] = header
        features['Position'] = pos
        features['target'] = target
        
        batch_features.append(features)
    
    return batch_features

def extract_features_for_dataset(df_data, window_size=5, batch_size=1000, output_dir="./",
                       extract_aac_feat=True, extract_dpc_feat=True, 
                       extract_tpc_feat=True, extract_binary_feat=True, 
                       extract_physico_feat=True, dataset_name=""):
    """
    Extract features for a specific dataset (train, val, or test)
    
    Parameters:
    - df_data: DataFrame for the specific dataset
    - window_size: Window size for feature extraction
    - batch_size: Batch size for processing
    - output_dir: Directory to save feature files
    - extract_*: Boolean flags for each feature type
    - dataset_name: Name to include in output files (train, val, test)
    
    Returns:
    - Dictionary of DataFrames for each feature type
    """
    print(f"Extracting features for {dataset_name} set with window size {window_size}...")
    start_time = time.time()
    print_memory_usage(f"Before feature extraction ({dataset_name})")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to pandas for easier access to data
    df_pd = df_data.to_pandas()
    
    # Convert data to Python lists for batch processing
    data_list = []
    for _, row in df_pd.iterrows():
        seq = row['Sequence']
        pos = int(row['Position'])
        header = row['Header']
        target = int(row['target'])
        data_list.append((seq, pos, header, target))
    
    # Clean up pandas dataframe to save memory
    del df_pd
    gc.collect()
    
    feature_frames = {}
    
    # Load physicochemical properties once if needed
    properties = None
    if extract_physico_feat:
        properties = load_physicochemical_properties()
    
    # Process each feature type
    if extract_aac_feat:
        print(f"Extracting AAC features for {dataset_name}...")
        aac_features = []
        for i in tqdm(range(0, len(data_list), batch_size), desc=f"AAC ({dataset_name})"):
            batch = data_list[i:i+batch_size]
            batch_features = process_feature_batch(batch, 'aac', window_size)
            aac_features.extend(batch_features)
        feature_frames['aac'] = dt.Frame(aac_features)
        feature_frames['aac'].to_csv(f"{output_dir}/phosphorylation_aac_features_{dataset_name}_window{window_size}.csv")
        print(f"AAC features saved to {output_dir}/phosphorylation_aac_features_{dataset_name}_window{window_size}.csv")
        print_memory_usage(f"After AAC extraction ({dataset_name})")
    
    if extract_dpc_feat:
        print(f"Extracting DPC features for {dataset_name}...")
        dpc_features = []
        for i in tqdm(range(0, len(data_list), batch_size), desc=f"DPC ({dataset_name})"):
            batch = data_list[i:i+batch_size]
            batch_features = process_feature_batch(batch, 'dpc', window_size)
            dpc_features.extend(batch_features)
        feature_frames['dpc'] = dt.Frame(dpc_features)
        feature_frames['dpc'].to_csv(f"{output_dir}/phosphorylation_dpc_features_{dataset_name}_window{window_size}.csv")
        print(f"DPC features saved to {output_dir}/phosphorylation_dpc_features_{dataset_name}_window{window_size}.csv")
        # Clean up to save memory
        del dpc_features
        gc.collect()
        print_memory_usage(f"After DPC extraction ({dataset_name})")
    
    if extract_tpc_feat:
        print(f"Extracting TPC features for {dataset_name}...")
        # For TPC, process in smaller batches to manage memory
        tpc_batch_size = min(batch_size, 500)
        tpc_batch_dir = f"{output_dir}/tpc_batches_{dataset_name}"
        os.makedirs(tpc_batch_dir, exist_ok=True)
        
        tpc_batch_files = []
        
        for batch_idx, i in enumerate(tqdm(range(0, len(data_list), tpc_batch_size), desc=f"TPC ({dataset_name})")):
            batch = data_list[i:i+tpc_batch_size]
            batch_features = process_feature_batch(batch, 'tpc', window_size)
            
            # Save batch to file
            batch_file = f"{tpc_batch_dir}/tpc_features_batch_{batch_idx+1}.csv"
            dt.Frame(batch_features).to_csv(batch_file)
            tpc_batch_files.append(batch_file)
            
            # Clean up this batch's data
            del batch_features
            gc.collect()
        
        # Combine TPC batches (note: this could still be memory-intensive)
        print(f"Combining TPC batches for {dataset_name}...")
        combined_file = f"{output_dir}/phosphorylation_tpc_features_{dataset_name}_window{window_size}.csv"
        
        # Read and combine in chunks
        combined_tpc = None
        for batch_file in tqdm(tpc_batch_files, desc=f"Combining TPC ({dataset_name})"):
            batch_tpc = dt.fread(batch_file)
            if combined_tpc is None:
                combined_tpc = batch_tpc
            else:
                combined_tpc = dt.rbind(combined_tpc, batch_tpc)
            
            # Clean up
            del batch_tpc
            gc.collect()
        
        if combined_tpc is not None:
            combined_tpc.to_csv(combined_file)
            feature_frames['tpc'] = combined_tpc
            print(f"TPC features saved to {combined_file}")
        else:
            print(f"Warning: No TPC features were generated for {dataset_name}")
        
        print_memory_usage(f"After TPC extraction ({dataset_name})")
    
    if extract_binary_feat:
        print(f"Extracting Binary Encoding features for {dataset_name}...")
        binary_features = []
        for i in tqdm(range(0, len(data_list), batch_size), desc=f"Binary Encoding ({dataset_name})"):
            batch = data_list[i:i+batch_size]
            batch_features = process_feature_batch(batch, 'binary', window_size)
            binary_features.extend(batch_features)
        feature_frames['binary'] = dt.Frame(binary_features)
        feature_frames['binary'].to_csv(f"{output_dir}/phosphorylation_binary_encoding_{dataset_name}_window{window_size}.csv")
        print(f"Binary encoding features saved to {output_dir}/phosphorylation_binary_encoding_{dataset_name}_window{window_size}.csv")
        # Clean up to save memory
        del binary_features
        gc.collect()
        print_memory_usage(f"After Binary Encoding extraction ({dataset_name})")
    
    if extract_physico_feat:
        print(f"Extracting Physicochemical features for {dataset_name}...")
        physico_features = []
        for i in tqdm(range(0, len(data_list), batch_size), desc=f"Physicochemical ({dataset_name})"):
            batch = data_list[i:i+batch_size]
            batch_features = process_feature_batch(batch, 'physicochemical', window_size, properties)
            physico_features.extend(batch_features)
        feature_frames['physicochemical'] = dt.Frame(physico_features)
        feature_frames['physicochemical'].to_csv(f"{output_dir}/phosphorylation_physicochemical_{dataset_name}_window{window_size}.csv")
        print(f"Physicochemical features saved to {output_dir}/phosphorylation_physicochemical_{dataset_name}_window{window_size}.csv")
        # Clean up to save memory
        del physico_features
        gc.collect()
        print_memory_usage(f"After Physicochemical extraction ({dataset_name})")
    
    print(f"Feature extraction for {dataset_name} completed in {time.time() - start_time:.2f} seconds")
    
    return feature_frames

def merge_features_for_dataset(window_size=5, output_dir="./", dataset_name=""):
    """
    Merge all extracted features for a specific dataset (train, val, or test)
    
    Parameters:
    - window_size: Window size used for feature extraction
    - output_dir: Directory containing feature files
    - dataset_name: Name of the dataset (train, val, test)
    
    Returns:
    - Merged features DataFrame
    """
    print(f"Merging features for {dataset_name}...")
    start_time = time.time()
    print_memory_usage(f"Before merging features ({dataset_name})")
    
    # Load individual feature files
    feature_files = {
        'aac': f"{output_dir}/phosphorylation_aac_features_{dataset_name}_window{window_size}.csv",
        'dpc': f"{output_dir}/phosphorylation_dpc_features_{dataset_name}_window{window_size}.csv",
        'tpc': f"{output_dir}/phosphorylation_tpc_features_{dataset_name}_window{window_size}.csv",
        'binary': f"{output_dir}/phosphorylation_binary_encoding_{dataset_name}_window{window_size}.csv",
        'physicochemical': f"{output_dir}/phosphorylation_physicochemical_{dataset_name}_window{window_size}.csv"
    }
    
    # Use pandas for merging to avoid datatable join issues
    import pandas as pd
    merged_df = None
    
    # Load and merge in stages to control memory usage
    for feature_type, file_path in feature_files.items():
        print(f"Loading {feature_type} features for {dataset_name} from {file_path}...")
        try:
            # Read using pandas
            feature_df = pd.read_csv(file_path)
            
            if merged_df is None:
                # First file, just use it as the base
                merged_df = feature_df
            else:
                # For subsequent files, perform a merge on key columns
                merged_df = pd.merge(
                    merged_df, 
                    feature_df,
                    on=["Header", "Position", "target"],
                    how="inner"
                )
            
            # Force garbage collection after each merge
            del feature_df
            gc.collect()
            
            print(f"Merged {feature_type} features for {dataset_name}, current shape: {merged_df.shape}")
            print_memory_usage(f"After merging {feature_type} ({dataset_name})")
            
        except Exception as e:
            print(f"Error loading {feature_type} features for {dataset_name}: {e}")
    
    # Save merged features
    if merged_df is not None:
        output_file = f"{output_dir}/phosphorylation_all_features_{dataset_name}_window{window_size}.csv"
        merged_df.to_csv(output_file, index=False)
        
        # Convert to datatable
        merged_dt = dt.Frame(merged_df)
        
        # Clean up pandas dataframe
        del merged_df
        gc.collect()
        
        print(f"All features for {dataset_name} merged and saved to {output_file}")
        print(f"Merged features shape: {merged_dt.shape}")
        print(f"Merging for {dataset_name} completed in {time.time() - start_time:.2f} seconds")
    else:
        print(f"No features were successfully merged for {dataset_name}")
        merged_dt = None
    
    print_memory_usage(f"After merging all features ({dataset_name})")
    
    return merged_dt

# def main(window_size=5, use_presplit=True):
#     """
#     Main function to execute the entire pipeline
    
#     Parameters:
#     - window_size: Window size for feature extraction
#     - use_presplit: Whether to use pre-split data that matches transformer model
#     """
#     start_time = time.time()
#     print(f"Starting phosphorylation site prediction pipeline with window size {window_size}")
#     print_memory_usage("Start")
    
#     # Step 1: Estimate memory requirements and check if processing is feasible
#     mem_estimate = estimate_memory_requirements(window_size)
    
#     print("\n--- Memory Estimation ---")
#     print(f"Estimated peak memory usage: {mem_estimate['estimated_memory_gb']:.2f} GB")
#     print(f"Available memory: {mem_estimate['available_memory_gb']:.2f} GB")
    
#     if mem_estimate['will_fit']:
#         print("Sufficient memory available. Proceeding with processing.\n")
#     else:
#         print("WARNING: Estimated memory usage exceeds available memory.")
#         print("The process may use excessive swap space or crash.")
#         print("Consider reducing window size or processing on a system with more memory.\n")
        
#         # Ask for confirmation to continue
#         response = input("Do you want to continue anyway? (y/n): ")
#         if response.lower() != 'y':
#             print("Exiting as requested.")
#             return
    
#     # Ensure split_data directory exists
#     os.makedirs("./split_data", exist_ok=True)
    
#     # Create feature output directory
#     feature_output_dir = "./features"
#     os.makedirs(feature_output_dir, exist_ok=True)
    
#     if use_presplit:
#         # Use pre-split data that matches transformer model
#         print("Using pre-split data to match transformer model...")
#         train_dt, val_dt, test_dt = load_presplit_data()
        
#         if train_dt is None or val_dt is None or test_dt is None:
#             print("Error loading pre-split data. Exiting.")
#             return
#     else:
#         # Regular processing path (not using pre-split)
#         print("Using standard processing path...")
#         # Step 2: Load and prepare data
#         df_seq = load_sequences()
#         df_labels = load_labels()
#         df_merged = merge_sequence_and_labels(df_seq, df_labels)
        
#         # Clean up original dataframes to save memory
#         del df_seq, df_labels
#         gc.collect()
        
#         df_merged = clean_data(df_merged)
#         df_final = generate_negative_samples(df_merged)
        
#         # Clean up
#         del df_merged
#         gc.collect()
        
#         # Split data using transformer's approach
#         train_dt, val_dt, test_dt = split_dataset_like_transformer(df_final)
        
#         # Clean up
#         del df_final
#         gc.collect()
        
#         # Save splits for reference
#         train_dt.to_csv("./split_data/train_data_original.csv")
#         val_dt.to_csv("./split_data/val_data_original.csv")
#         test_dt.to_csv("./split_data/test_data_original.csv")
    
#     # Step 3: Extract features for each split separately
#     print("\nExtracting features for each split...")
    
#     # Process train set
#     train_features = extract_features_for_dataset(
#         train_dt, window_size=window_size, 
#         output_dir=feature_output_dir, dataset_name="train"
#     )
    
#     # Process validation set
#     val_features = extract_features_for_dataset(
#         val_dt, window_size=window_size, 
#         output_dir=feature_output_dir, dataset_name="val"
#     )
    
#     # Process test set
#     test_features = extract_features_for_dataset(
#         test_dt, window_size=window_size, 
#         output_dir=feature_output_dir, dataset_name="test"
#     )
    
#     # Clean up to save memory
#     del train_features, val_features, test_features
#     gc.collect()
    
#     # Step 4: Merge features for each split
#     train_merged = merge_features_for_dataset(
#         window_size=window_size, output_dir=feature_output_dir, dataset_name="train"
#     )
    
#     val_merged = merge_features_for_dataset(
#         window_size=window_size, output_dir=feature_output_dir, dataset_name="val"
#     )
    
#     test_merged = merge_features_for_dataset(
#         window_size=window_size, output_dir=feature_output_dir, dataset_name="test"
#     )
    
#     # Step 5: Save merged features to split_data directory for XGBoost training
#     if train_merged is not None:
#         print("Saving merged train features to split_data directory...")
#         train_merged.to_csv("./split_data/train_data.csv", index=False)
    
#     if val_merged is not None:
#         print("Saving merged validation features to split_data directory...")
#         val_merged.to_csv("./split_data/val_data.csv", index=False)
    
#     if test_merged is not None:
#         print("Saving merged test features to split_data directory...")
#         test_merged.to_csv("./split_data/test_data.csv", index=False)
    
#     # Clean up
#     del train_merged, val_merged, test_merged
#     gc.collect()
    
#     print(f"Pipeline completed in {(time.time() - start_time) / 60:.2f} minutes")
#     print_memory_usage("End")
    
#     print("\nNext steps:")
#     print("1. Train the XGBoost model using phosphorylation_prediction.py")
#     print("2. Run the ensemble notebook to combine XGBoost and Transformer predictions")

# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Phosphorylation Site Prediction Data Processor")
#     parser.add_argument("--window-size", type=int, default=21, 
#                         help="Window size for feature extraction (default: 21)")
#     parser.add_argument("--use-presplit", action="store_true",
#                         help="Use pre-split data that matches transformer model")
#     args = parser.parse_args()
    
#     main(window_size=args.window_size, use_presplit=args.use_presplit)

def main_continue_from_step5(window_size=5):
    """
    Continue the pipeline from Step 5 (saving the merged features to split_data)
    This is useful when the pipeline crashed during the final steps.
    
    Parameters:
    - window_size: Window size that was used for feature extraction
    """
    start_time = time.time()
    print("Continuing pipeline from Step 5 (saving merged features)...")
    
    # Ensure split_data directory exists
    os.makedirs("./split_data", exist_ok=True)
    
    # Feature output directory where merged files should be
    feature_output_dir = "./features"
    
    # Step 5: Load the merged features from the features directory
    print("Loading merged features from features directory...")
    
    try:
        # Load train features
        train_file = f"{feature_output_dir}/phosphorylation_all_features_train_window{window_size}.csv"
        print(f"Loading train features from {train_file}...")
        train_merged = pd.read_csv(train_file)
        print(f"Train features loaded, shape: {train_merged.shape}")
        
        # Load validation features
        val_file = f"{feature_output_dir}/phosphorylation_all_features_val_window{window_size}.csv"
        print(f"Loading validation features from {val_file}...")
        val_merged = pd.read_csv(val_file)
        print(f"Validation features loaded, shape: {val_merged.shape}")
        
        # Load test features
        test_file = f"{feature_output_dir}/phosphorylation_all_features_test_window{window_size}.csv"
        print(f"Loading test features from {test_file}...")
        test_merged = pd.read_csv(test_file)
        print(f"Test features loaded, shape: {test_merged.shape}")
        
        # Save merged features to split_data directory for XGBoost training
        print("Saving merged train features to split_data directory...")
        train_merged.to_csv("./split_data/train_data.csv")
        
        print("Saving merged validation features to split_data directory...")
        val_merged.to_csv("./split_data/val_data.csv")
        
        print("Saving merged test features to split_data directory...")
        test_merged.to_csv("./split_data/test_data.csv")
        
        # Clean up
        del train_merged, val_merged, test_merged
        gc.collect()
        
        print(f"Step 5 completed in {(time.time() - start_time) / 60:.2f} minutes")
        
        print("\nNext steps:")
        print("1. Train the XGBoost model using phosphorylation_prediction.py")
        print("2. Run the ensemble notebook to combine XGBoost and Transformer predictions")
        
    except Exception as e:
        print(f"Error during Step 5: {e}")
        print("Please check if the merged feature files exist in the features directory.")
        return

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Continue Phosphorylation Site Prediction from Step 5")
    parser.add_argument("--window-size", type=int, default=21, 
                        help="Window size used for feature extraction (default: 21)")
    args = parser.parse_args()
    
    main_continue_from_step5(window_size=args.window_size)