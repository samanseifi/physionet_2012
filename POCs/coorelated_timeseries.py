import os
import numpy as np
import orjson  # Faster than json
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy.signal import resample
from scipy.stats import pearsonr, linregress
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Directory containing patient JSON files
json_dir = "data/patient_jsons"

# Convert "HH:MM" timestamps into total minutes
def convert_timestamp_to_minutes(timestamp):
    try:
        hours, minutes = map(int, timestamp.split(":"))
        return hours * 60 + minutes  # Convert to total minutes
    except ValueError:
        return None  # Invalid timestamp

# Load JSON files in parallel
def load_json(filename):
    file_path = os.path.join(json_dir, filename)
    with open(file_path, "rb") as file:
        return orjson.loads(file.read())

# Load patient data efficiently
def load_patient_data(json_dir):
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    with mp.Pool(mp.cpu_count()) as pool:
        patient_list = pool.map(load_json, json_files)
    return {int(p["patient_id"]): p for p in patient_list}

# Extract and resample a patient's time-series to 100 points
def extract_resample_timeseries(patient_data, param, num_points=100):
    time_values, values = [], []
    for entry in patient_data.get("time_series", {}).get(param, []):
        t = convert_timestamp_to_minutes(entry["timestamp"])
        v = float(entry["value"])
        if t is not None:
            time_values.append(t)
            values.append(v)

    if len(time_values) > 1:
        time_values, values = np.array(time_values), np.array(values)
        common_grid = np.linspace(min(time_values), max(time_values), num=num_points)
        resampled_values = resample(values, num_points)
        return common_grid, resampled_values
    return None, None

# Extract key statistical features from time-series
def extract_features(values):
    if values is None or len(values) < 2:  # Handle missing or insufficient data
        return [np.nan, np.nan, np.nan, np.nan]  # Default values for missing time-series

    mean_val = np.mean(values)
    first_val = values[0]
    last_val = values[-1]
    slope, _, _, _, _ = linregress(np.arange(len(values)), values)
    
    return [mean_val, first_val, last_val, slope]

# Compute similarity in parallel
def compute_similarity(args):
    ref_values, other_values, ref_features, other_features, patient_id, method = args
    if other_values is not None:
        if method == "pearson":
            corr, _ = pearsonr(ref_values, other_values)
            return patient_id, corr  # Higher is better
        elif method == "dtw":
            distance, _ = fastdtw(ref_values, other_values, dist=euclidean)
            return patient_id, -distance  # Lower is better (negate for sorting)
        elif method == "euclidean":
            distance = np.linalg.norm(ref_values - other_values)
            return patient_id, -distance  # Lower is better (negate for sorting)
        elif method == "feature":
            feature_distance = np.linalg.norm(np.array(ref_features) - np.array(other_features))
            return patient_id, -feature_distance  # Lower is better (negate for sorting)
    return patient_id, None

# Load all patients
patient_data = load_patient_data(json_dir)

# User input: Patient ID, parameter, and similarity method
patient_id_to_compare = input("Enter the Patient ID to compare: ").strip()
param_to_compare = input("Enter the time-series parameter to compare (e.g., 'hr', 'temp', 'map'): ").strip().lower()
similarity_method = input("Choose similarity method (pearson/dtw/euclidean/feature): ").strip().lower()

if similarity_method not in ["pearson", "dtw", "euclidean", "feature"]:
    print("Invalid method. Choose from 'pearson', 'dtw', 'euclidean', or 'feature'.")
    exit()

# Convert Patient ID to int
try:
    patient_id_to_compare = int(patient_id_to_compare)
except ValueError:
    print("Invalid Patient ID format. Please enter a numeric ID.")
    exit()

# Ensure patient exists
if patient_id_to_compare not in patient_data:
    print(f"Patient ID {patient_id_to_compare} not found in the dataset.")
else:
    # Extract reference patient's resampled time-series & features
    ref_grid, ref_values = extract_resample_timeseries(patient_data[patient_id_to_compare], param_to_compare)
    ref_features = extract_features(ref_values) if ref_values is not None else [np.nan] * 4

    if ref_values is None:
        print(f"No valid time-series data for '{param_to_compare}' in patient {patient_id_to_compare}.")
    else:
        # Prepare arguments for parallel similarity computation
        patient_ids = [pid for pid in patient_data if pid != patient_id_to_compare]
        patient_timeseries = [extract_resample_timeseries(patient_data[pid], param_to_compare) for pid in patient_ids]
        
        # Ensure no None values in timeseries before passing to extract_features
        patient_features = [extract_features(ts[1]) if ts[1] is not None else [np.nan] * 4 for ts in patient_timeseries]

        args = [(ref_values, ts[1], ref_features, fts, pid, similarity_method) 
                for ts, fts, pid in zip(patient_timeseries, patient_features, patient_ids) if ts[1] is not None]

        # Compute similarity in parallel
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(compute_similarity, args)

        # Filter valid scores
        similarity_scores = {pid: score for pid, score in results if score is not None}

        # Get top 5 most similar patients
        top_5_similar_patients = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:5]

        # Display results
        print(f"\nTop 5 Most Similar Patients ({similarity_method.upper()}) for:", param_to_compare.upper())
        for rank, (patient_id, score) in enumerate(top_5_similar_patients, 1):
            print(f"{rank}. Patient ID: {patient_id} | Score: {score:.3f}")

        # Plot reference patient + top 5 similar patients
        plt.figure(figsize=(12, 6), dpi=100)
        plt.plot(ref_grid, ref_values, label=f"Patient {patient_id_to_compare} (Reference)", linewidth=2, color="black")

        # Plot top 5 similar patients
        colors = ["blue", "green", "red", "purple", "orange"]
        for idx, (patient_id, _) in enumerate(top_5_similar_patients):
            other_grid, other_values = extract_resample_timeseries(patient_data[patient_id], param_to_compare)
            if other_values is not None:
                plt.plot(other_grid, other_values, label=f"Patient {patient_id}", color=colors[idx], alpha=0.6)

        # Plot styling
        plt.xlabel("Time (minutes)")
        plt.ylabel(param_to_compare.upper())
        plt.title(f"{param_to_compare.upper()} Time-Series: Reference vs. Top 5 Similar ({similarity_method.upper()})")
        plt.legend()
        plt.grid(True)
        plt.show()
