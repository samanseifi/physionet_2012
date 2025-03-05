import os
import numpy as np
import orjson
import logging
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import resample
from scipy.stats import pearsonr
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# **Logging Configuration**
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)

# **Directories**
json_dir = "data/patient_jsons"

# **Static Parameters (from "general" section in JSON)**
demographic_params = ["age", "gender", "height", "weight", "icutype"]

# **Convert "HH:MM" timestamps into total minutes**
def convert_timestamp_to_minutes(timestamp):
    try:
        hours, minutes = map(int, timestamp.split(":"))
        return hours * 60 + minutes
    except ValueError:
        return None  # Handle invalid timestamps

# **Load all JSON files**
def load_patient_data(json_dir):
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    patient_data = {}
    for filename in json_files:
        file_path = os.path.join(json_dir, filename)
        with open(file_path, "rb") as file:
            data = orjson.loads(file.read())
            patient_data[int(data["patient_id"])] = data
    return patient_data

# **Extract & resample time-series to fixed 100 points**
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

# **Compute similarity between two vectors (time-series or static values)**
def compute_similarity(ref_values, other_values, method="pearson"):
    if other_values is None or len(ref_values) != len(other_values):
        return None  # Skip invalid comparisons

    if method == "pearson":
        corr, _ = pearsonr(ref_values, other_values)
        return corr  # Higher is better
    elif method == "dtw":
        distance, _ = fastdtw(ref_values, other_values, dist=euclidean)
        return -distance  # Lower is better (negate for sorting)
    elif method == "euclidean":
        distance = np.linalg.norm(ref_values - other_values)
        return -distance  # Lower is better (negate for sorting)
    return None

# **Extract static attributes from the "general" section**
def extract_static_values(patient):
    general_info = patient.get("general", {})
    return np.array([general_info.get(param, np.nan) for param in demographic_params])

# **Normalize static values for fair comparison**
def normalize_static_values(patient_data):
    all_values = {param: [] for param in demographic_params}

    # Collect values across all patients
    for patient in patient_data.values():
        for param in demographic_params:
            if "general" in patient and param in patient["general"] and isinstance(patient["general"][param], (int, float)):
                all_values[param].append(patient["general"][param])

    # Compute min/max for normalization
    min_max = {param: (min(vals), max(vals)) for param, vals in all_values.items() if vals}

    # Normalize static values
    for patient in patient_data.values():
        for param in demographic_params:
            if "general" in patient and param in patient["general"] and isinstance(patient["general"][param], (int, float)):
                min_val, max_val = min_max.get(param, (0, 1))
                if max_val > min_val:  # Avoid division by zero
                    patient["general"][param] = (patient["general"][param] - min_val) / (max_val - min_val)

# **Find the most similar patient based on all parameters (time-series + static)**
def find_nearest_patient_all_params(patient_id, method="pearson"):
    logging.info(f"Loading patient data...")
    patient_data = load_patient_data(json_dir)
    normalize_static_values(patient_data)

    if patient_id not in patient_data:
        logging.error(f"Patient {patient_id} not found!")
        return None, None

    all_timeseries_params = list(next(iter(patient_data.values()))["time_series"].keys())
    logging.info(f"Comparing based on {len(all_timeseries_params)} time-series parameters and {len(demographic_params)} static parameters.")

    # **Extract reference patient time-series and static values**
    ref_vectors = []
    for param in all_timeseries_params:
        values = extract_resample_timeseries(patient_data[patient_id], param)[1]
        if values is not None:
            ref_vectors.append(values)

    ref_static_values = extract_static_values(patient_data[patient_id])

    if not ref_vectors:
        logging.warning(f"Patient {patient_id} has no valid time-series.")
        return None, None

    ref_matrix = np.vstack(ref_vectors)

    nearest_patient, best_score = None, -np.inf if method == "pearson" else np.inf

    for other_id, data in patient_data.items():
        if other_id == patient_id:
            continue  # Skip self

        other_vectors = []
        for param in all_timeseries_params:
            values = extract_resample_timeseries(data, param)[1]
            if values is not None:
                other_vectors.append(values)

        other_static_values = extract_static_values(data)

        if len(other_vectors) == len(ref_vectors):  # Ensure same number of time-series
            other_matrix = np.vstack(other_vectors)
            total_similarity = np.mean([
                compute_similarity(ref_matrix[i], other_matrix[i], method)
                for i in range(len(ref_vectors))
            ])

            # **Include static values similarity (Euclidean distance)**
            static_similarity = -np.linalg.norm(ref_static_values - other_static_values)

            combined_similarity = 0.7 * total_similarity + 0.3 * static_similarity  # Weighted

            if combined_similarity is not None:
                if method == "pearson" and combined_similarity > best_score:
                    best_score = combined_similarity
                    nearest_patient = other_id
                elif method in ["dtw", "euclidean"] and combined_similarity < best_score:
                    best_score = combined_similarity
                    nearest_patient = other_id

    if nearest_patient:
        logging.info(f"Nearest patient to {patient_id} ({method}): {nearest_patient} (Score: {best_score:.3f})")
    else:
        logging.warning(f"No similar patients found for {patient_id}.")

    return nearest_patient, patient_data
# **Plot time-series comparison between two patients**
def plot_patient_comparison(patient_id, nearest_patient, patient_data):
    if nearest_patient is None:
        logging.error("No nearest patient found. Cannot plot comparison.")
        return

    ref_patient = patient_data[patient_id]
    near_patient = patient_data[nearest_patient]
    all_params = list(ref_patient["time_series"].keys())

    num_plots = len(all_params)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, num_plots * 3), sharex=True)

    for i, param in enumerate(all_params):
        ax = axes[i] if num_plots > 1 else axes
        ref_t, ref_v = extract_resample_timeseries(ref_patient, param)
        near_t, near_v = extract_resample_timeseries(near_patient, param)

        if ref_v is not None and near_v is not None:
            ax.plot(ref_t, ref_v, label=f"Patient {patient_id}", color="blue", alpha=0.7)
            ax.plot(near_t, near_v, label=f"Nearest Patient {nearest_patient}", color="red", alpha=0.7)
            ax.set_ylabel(param.upper())
            ax.legend()
        else:
            ax.set_ylabel(param.upper())
            ax.text(0.5, 0.5, "No Data", fontsize=12, ha="center", va="center", transform=ax.transAxes)

    plt.xlabel("Time (minutes)")
    plt.suptitle(f"Time-Series Comparison: Patient {patient_id} vs {nearest_patient}")
    plt.tight_layout()
    plt.show()

# **User input for testing**
patient_id = int(input("Enter Patient ID: "))
method = input("Choose similarity method (pearson/dtw/euclidean): ").strip().lower()

# **Find the nearest patient using all parameters**
nearest_patient, patient_data = find_nearest_patient_all_params(patient_id, method)

# **Compare Static Values**
if nearest_patient:
    ref_attrs = patient_data[patient_id].get("general", {})
    nearest_attrs = patient_data[nearest_patient].get("general", {})

    df = pd.DataFrame([ref_attrs, nearest_attrs], index=[f"Patient {patient_id}", f"Nearest Patient {nearest_patient}"])
    print("\n🔹 **Static Attribute Comparison** 🔹")
    print(df)

# **Plot time-series comparison**
plot_patient_comparison(patient_id, nearest_patient, patient_data)


