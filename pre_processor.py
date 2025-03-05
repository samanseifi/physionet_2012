import pandas as pd
import numpy as np
import json
import os
import logging
from scipy.stats import linregress, iqr
from multiprocessing import Pool, cpu_count

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class PatientDataProcessor:
    """
    A class to process ICU patient time-series data efficiently with logging and parallelization.
    """

    def __init__(self, dataset_path, outcome_path, output_dir="data/patient_jsons"):
        self.dataset_path = dataset_path
        self.outcome_path = outcome_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Load data
        logging.info("Loading dataset...")
        self.df = pd.read_csv(self.dataset_path)
        self.outcome_df = pd.read_csv(self.outcome_path)

        # Define parameter categories
        self.general_list = ["Age", "Gender", "Height", "Weight", "ICUType"]
        self.timeseries_params = self._get_timeseries_params()
        logging.info(f"Identified {len(self.timeseries_params)} time-series parameters.")

    def _get_timeseries_params(self):
        """Get unique time-series parameters excluding generals and 'RecordID'."""
        parameters = self.df["Parameter"].unique()
        return [param for param in parameters if param not in self.general_list and param != "RecordID"]

    @staticmethod
    def extract_features(param_str, timeseries):
        """Extract statistical, trend, and change-based features for ICU time-series data."""
        features = {}
        
        # first find outliers and remove them
        if len(timeseries) > 1:
            q1, q3 = np.percentile(timeseries, [25, 75])
            iqr_val = q3 - q1
            lower_bound = q1 - 1.5 * iqr_val
            upper_bound = q3 + 1.5 * iqr_val
            timeseries = timeseries[(timeseries >= lower_bound) & (timeseries <= upper_bound)]

        if len(timeseries) == 0:
            return {
                param_str+"_mean": np.nan,
                param_str+"_median": np.nan,
                param_str+"_slope": np.nan,
                param_str+"_min_value": np.nan,
                param_str+"_max_value": np.nan,
                param_str+"_first_value": np.nan,
                param_str+"_last_value": np.nan,
            }

        # Basic Statistical Features
        features[param_str+"_mean"] = np.mean(timeseries)
        features[param_str+"_median"] = np.median(timeseries)


        # Trend Features (Avoid errors with constant or very short series)
        if len(timeseries) > 1 and np.ptp(timeseries) > 1e-6:
            time = np.arange(len(timeseries))
            slope, _, _, _, _ = linregress(time, timeseries)
            features[param_str+"_slope"] = slope
        else:
            features[param_str+"_slope"] = np.nan

        features[param_str+"_min_value"] = timeseries.min()
        features[param_str+"_max_value"] = timeseries.max()
        features[param_str+"_first_value"] = timeseries[0]
        features[param_str+"_last_value"] = timeseries[-1]

        return features

    def process_patient(self, patient_data):
        """Process a single patient: extract general, time-series features, and save JSON."""
        patient_id = int(patient_data["PATIENT_ID"].iloc[0])  # Convert to standard Python int
        patient_info = {"patient_id": patient_id, "general": {}, "time_series": {}}  # <-- Added 'generals' section
        extracted_features = {"patient_id": patient_id}

        # Extract general Data
        for param in self.general_list:
            values = patient_data.loc[patient_data["Parameter"] == param, "Value"].values
            value = float(values[0]) if len(values) > 0 else np.nan
            extracted_features[param.lower()] = value
            patient_info["general"][param.lower()] = value  # <-- Save in JSON output

        # Extract Time-Series Data and Compute Features
        for param in self.timeseries_params:
            ts_data = patient_data.loc[patient_data["Parameter"] == param, ["Time", "Value"]].dropna()

            if not ts_data.empty:
                timeseries_values = ts_data["Value"].values
                patient_info["time_series"][param.lower()] = [
                    {"timestamp": row["Time"], "value": float(row["Value"])} for _, row in ts_data.iterrows()
                ]
            else:
                timeseries_values = []

            extracted_features.update(self.extract_features(param.lower(), timeseries_values))

        # **Convert NumPy types to native Python types before saving JSON**
        patient_info = json.loads(json.dumps(patient_info, default=lambda o: int(o) if isinstance(o, np.integer) else float(o)))

        # Save JSON
        json_file_path = os.path.join(self.output_dir, f"patient_{patient_id}.json")
        with open(json_file_path, "w") as json_file:
            json.dump(patient_info, json_file, indent=4)

        return extracted_features
    
    def print_summary(self):
            """Print a summary of time-series features and the percentage of patients having them."""
            logging.info("Generating summary of dataset...")
            total_patients = self.df["PATIENT_ID"].nunique()
            summary = {}
            
            for param in self.timeseries_params:
                num_patients_with_param = self.df[self.df["Parameter"] == param]["PATIENT_ID"].nunique()
                percentage = (num_patients_with_param / total_patients) * 100
                summary[param] = percentage
            
            logging.info("Summary of Time-Series Features:")
            for param, pct in summary.items():
                logging.info(f"{param}: Present in {pct:.2f}% of patients")
        


    def process_patients_parallel(self):
        """Process all patients in parallel to speed up feature extraction."""
        logging.info("Starting parallel processing of patient data...")
        patients_master_table = []

        with Pool(processes=cpu_count() - 1) as pool:  # Use all but one CPU
            results = pool.map(self.process_patient, [group for _, group in self.df.groupby("PATIENT_ID")])
            patients_master_table.extend(results)

        # Convert to DataFrame
        master_df = pd.DataFrame(patients_master_table)

        # Merge with Outcome Data
        final_df = self.merge_outcomes(master_df)

        # Save Final CSV
        final_csv_path = "data/patients_master_table.csv"
        final_df.to_csv(final_csv_path, index=False)
        logging.info(f"Master table saved in {final_csv_path}")

    def merge_outcomes(self, master_df):
        """Merge patient master table with outcome data."""
        self.outcome_df.rename(columns={"RecordID": "patient_id"}, inplace=True)
        merged_df = master_df.merge(self.outcome_df, on="patient_id", how="left")
        logging.info("Merged outcome data with master dataset.")
        return merged_df


# Run Processing with Optimized Code
if __name__ == "__main__":
    processor = PatientDataProcessor("data/seta_data.csv", "data/Outcomes-a.txt")
    processor.process_patients_parallel()
