import pandas as pd  # pandas is used for reading CSV files and handling tabular data
import numpy as np  # numpy provides efficient array operations and mathematical tools

# List of experimental runs to include in the analysis
# Each run is defined by: (CSV file name, string label, humidity value in %)
run_info = [
    # The format is (filename, label, humidity, V_gnd)
    # You can include or exclude runs by commenting/uncommenting lines

    # PRE RE-COATING
    # ("20241028 Run1.csv", "20241028", 39.23, 17),  # short data collection time
    # ("20241108.csv", "20241108", 14.51, 12),
    # ("20241112.csv", "20241112", 14.95, 11),
    # # # ("20241115.csv", "20241115", 14.52, 4), # Peculiar tail, flattens hard
    # ("20250219.csv", "20250219", 7.05, 31),
    # ("20250221.csv", "20250221", 7.17, 35),
    # ("20250407.csv", "20250407", 15.41, 4),
    # # #("20250428.csv", "20250428", 85.49),  # commented out due to uncertainty in V_inf (Human error on probe)
    # ("20250505.csv", "20250505", 95, 19),
    # ("20250527.csv", "20250527", 91, 20),
    # # # ("20250618.csv", "20250618", 81, 35) # Study on effect of multiple droplets

    # WIDE, RECENT COATING
    ("20250716_cleaned.csv", "20250716", 28.03, 43),
    ("20250718.csv", "20250718", 27.4, 42),
    ("20250721.csv", "20250721", 28.2, 37),
    ("20250723.csv", "20250723", 28.6, 35),

    # THIN, RECENT COATING
    ("20250726.csv", "20250726", 28.6, 23),
    ("20250728.csv", "20250728", 31.1, 23),
    ("20250729.csv", "22050729", 30.6, 25)

]

# This dictionary will store all the processed data for each run
# The keys will be run labels (e.g., "20241108"), and the values will be dicts
data_dict = {}
# === Data Processing Loop ===
for run in run_info:
    # Unpack tuple elements: file name, label, and humidity for current run
    file_name, label, humidity, v_inf = run

    # Load CSV data into a pandas DataFrame
    data = pd.read_csv(file_name)

    # Extract time and voltage columns as numpy arrays
    t = data["Time (s)"].to_numpy()  # time values in seconds
    v = data["Voltage (V)"].to_numpy()  # voltage values in volts


    # Invert voltage: original voltages are negative due to setup,
    # so flip sign to make data positive and easier to interpret
    v = -v

    # Identify the index of the initial voltage spike (maximum value after negation)
    spike_idx = np.argmax(v)

    # Mask time and voltage to only include data **after** the initial spike
    # This truncates any pre-spike noise or setup fluctuations
    t_mask = ((t >= t[spike_idx]) & (v < 280))
    t, v = t[t_mask], v[t_mask]

    # Normalize time: set time = 0 at the moment of the spike (now shifted to t = 1e-12, so log scale can be used)
    # Avoids issues with taking log(0) or dividing by zero in later analysis
    # t = t - t[0] + 1e-12
    t = t - t[0] + 0.1 # For display purposes, as 1e-12 makes it look weird
    # Normalize voltage by subtracting the post-spike plateau level
    # This sets the minimum value of the remaining voltage data to 0
    v = v - v_inf

    # Store processed data in the main dictionary
    # Data is stored under a label, with keys for time, voltage, and humidity
    data_dict[label] = {
        "Time": t,
        "Voltage": v,
        "Humidity": humidity
    }


# Function to return the full dictionary of processed datasets
def return_data():
    return data_dict
