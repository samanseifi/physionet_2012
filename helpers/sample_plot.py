import json
import os
import matplotlib.pyplot as plt

# Directory containing JSON files
json_dir = "data/patient_jsons"

# Select a sample patient file
sample_file = os.path.join(json_dir, "patient_142428.json")  # Replace with an actual patient ID

# Load patient data
with open(sample_file, "r") as f:
    patient_data = json.load(f)

# Choose a time-series parameter to plot
param_to_plot = "hr"  # Replace with any available parameter

if param_to_plot in patient_data["time_series"]:
    ts_data = patient_data["time_series"][param_to_plot]
    timestamps = [entry["timestamp"] for entry in ts_data]
    values = [entry["value"] for entry in ts_data]
    
    # Plot the time-series data
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, values, marker="o", linestyle="-")
    plt.xlabel("Time")
    plt.ylabel(param_to_plot.upper())
    plt.title(f"Time Series of {param_to_plot.upper()} for Patient {patient_data['patient_id']}")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()
else:
    print(f"No time-series data available for {param_to_plot}")