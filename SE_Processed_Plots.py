# Import plotting and numerical libraries
import matplotlib.pyplot as plt  # for creating plots and visualizations
import numpy as np  # for numerical operations and array handling

# Import previously processed experimental data
# This assumes SE_File_Loading.py defines return_data(), which provides the data_dict
from SE_File_Loading import return_data

# Load the processed data dictionary
data_dict = return_data()

# === Function: Plot each run individually ===
def single_output(data_dict):
    # Loop through each dataset in the dictionary
    for label, data in data_dict.items():
        # Extract time, voltage, and humidity for the current run
        t = data["Time"]
        v = data["Voltage"]
        hum = data["Humidity"]

        # Plot the voltage vs time using dots (scatter-like style)
        # Label includes date (formatted) and humidity
        plt.plot(t, v, '.', label=f'{label[:4]}/{label[4:6]}/{label[6:]}\n{hum}%')

        # Plot a horizontal line at zero voltage for reference
        plt.plot(t, np.zeros(len(t)))

        # Set x-axis to logarithmic scale to better capture behavior across orders of magnitude
        plt.xscale('log')

        # Optional: Uncomment to use logarithmic scale on y-axis as well
        # plt.yscale('log')

        # Label axes and set plot title
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.title("Voltage vs Time (Log‑Log Scale)")

        # Add legend with dataset info
        plt.legend()

        # Optimize plot layout to prevent clipping
        plt.tight_layout()

        # Display the plot
        plt.show()


# === Function: Plot all runs together in one figure ===
def multi_output(data_dict):
    # Loop through all runs and plot them on the same axes
    for label, data in data_dict.items():
        t = data["Time"]
        v = data["Voltage"]
        hum = data["Humidity"]

        # Plot each run with its own label for date and humidity
        plt.plot(t, v, '.', label=f'{label[:4]}/{label[4:6]}/{label[6:]}\n{hum}%')

    # Add zero voltage baseline across the time domain of the last run
    plt.plot(t, np.zeros(len(t)))

    # Use log scale for x-axis to capture early-time behavior more clearly
    plt.xscale('log')

    # Optional: Uncomment if you want to view voltage decay on a log scale as well
    plt.yscale('log')

    # Label axes and title
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Voltage (V)", fontsize=12)
    plt.title("Voltage vs Time (Lin‑Log Scale)", fontsize=12)

    # Display all run labels in a legend
    # plt.legend()

    # Clean layout
    plt.tight_layout()

    # Show the composite plot
    plt.show()


# === Call the multi_output function to plot all runs together ===
multi_output(data_dict)
