import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Directory containing patient JSON files
json_dir = "data/patient_jsons"

# Function to convert "HH:MM" timestamps into total minutes
def convert_timestamp_to_minutes(timestamp):
    try:
        hours, minutes = map(int, timestamp.split(":"))
        return hours * 60 + minutes  # Convert to total minutes
    except ValueError:
        return None  # Invalid timestamp

# Function to load patient JSON files
def load_patient_data(json_dir):
    patient_data = {}
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(json_dir, filename)
            with open(file_path, "r") as file:
                data = json.load(file)
                patient_data[data["patient_id"]] = data
    return patient_data

# Load all patients
patient_data = load_patient_data(json_dir)

# User input for parameter selection
param_to_plot = input("Enter the time-series parameter to plot (e.g., 'hr', 'temp', 'map'): ").strip().lower()

# Filter patients with the selected parameter
patients_with_param = [pid for pid, data in patient_data.items() if "time_series" in data and param_to_plot in data["time_series"]]

# Check if there are valid patients for this parameter
if not patients_with_param:
    print(f"No patients found with time-series data for '{param_to_plot}'.")
else:
    # Select up to 400 patients with the parameter data
    selected_patients = np.random.choice(patients_with_param, min(400, len(patients_with_param)), replace=False)

    # Plot all selected time-series
    plt.figure(figsize=(12, 6), dpi=100)

    for patient_id in selected_patients:
        ts_data = patient_data[patient_id]["time_series"][param_to_plot]

        try:
            # Extract timestamps and values, ensuring they are numeric
            time_values = []
            param_values = []

            for entry in ts_data:
                t = convert_timestamp_to_minutes(entry["timestamp"])  # Convert to minutes
                v = float(entry["value"])  # Convert parameter value to float

                # Ensure valid timestamps and values
                if t is not None:
                    time_values.append(t)
                    param_values.append(v)

            # Ensure we have valid data before plotting
            if len(time_values) > 1:
                time_values = np.array(time_values)
                param_values = np.array(param_values)

                # Interpolate onto a common time grid (100 points)
                common_time_grid = np.linspace(min(time_values), max(time_values), num=100)
                interp_func = interp1d(time_values, param_values, kind="linear", fill_value="extrapolate")
                param_interp = interp_func(common_time_grid)
                plt.plot(common_time_grid, param_interp, alpha=0.3, lw=0.8)  # Semi-transparent for overlap
            else:
                print(f"Skipping patient {patient_id}: No valid data after filtering.")

        except Exception as e:
            print(f"Skipping patient {patient_id}: {e}")

    # Plot styling
    plt.xlabel("Time (minutes)")
    plt.ylabel(param_to_plot.upper())
    plt.title(f"{param_to_plot.upper()} Time-Series for 400 Random Patients")
    plt.grid(True)
    plt.show()
