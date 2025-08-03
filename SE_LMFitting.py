import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from lmfit import Model, Parameters, fit_report
from SE_File_Loading import return_data
from numpy.polynomial.polynomial import Polynomial

plt.rcParams['font.size'] = 12

# === Define Model: Stretched exponential + Exponential + Power-law + Offset ===
def voltage_model(t, A1, tau1, beta, A2, tau2, A3, lam, mu, C):
    t = np.clip(t, 1e-6, None)
    return (
        A1 * np.exp(-(t / tau1) ** beta) +       # Stretched exponential (electron)
        A2 * np.exp(-t / tau2) +                 # Standard exponential (second electron process)
        A3 * ((1 + t / lam) ** mu + C)             # Power-law (ion drift) with constant offset
    )

# === Initial Guesses ===
initial_params = {
    "A1": 165, "tau1": 280, "beta": 0.47,
    "A2": 15,  "tau2": 550,
    "A3": 50,  "lam": 4e5, "mu": -2,
    "C": -0.3
}

# === Bounds ===
bounds = {
    # "A1": (100, 300),
    # "tau1": (150, 350),
    # "beta": (0.43, 0.5),
    # "A2": (3, 30),
    # "tau2": (400, 800),
    # "A3": (0, None),
    # "lam": (1e4, 1e7),
    # "mu": (-3, -0.1),
    # "C": (-1, 1)

    # LETS NEW DATA FIT, LOW VOLT
    "A1": (0, 500),
    "tau1": (0, 1000),
    "beta": (0, 1),
    "A2": (-1, 500),
    "tau2": (0, 1000),
    "A3": (0, None),
    "lam": (0, 1e8),
    "mu": (-10, -0.1),
    "C": (-1, 1)
}


# === Load Data ===
data_dict = return_data()
print(f"Loaded datasets: {list(data_dict.keys())}")

model = Model(voltage_model)
fit_results = {}
residuals = {}

# === Fit Loop ===
for label, data in data_dict.items():
    t = data["Time"]
    v = data["Voltage"]

    if len(t) != len(v):
        print(f"Skipping {label}: mismatched lengths")
        continue

    print(f"Fitting {label}...")

    # Initialize Parameters
    params = Parameters()
    for key, val in initial_params.items():
        params.add(key, value=val)
    for key, (lo, hi) in bounds.items():
        params[key].min = lo
        if hi is not None:
            params[key].max = hi


    def progress_callback(params, iter, resid, *args, **kwargs):
        if iter % 50 == 0:
            print(f"    Iteration {iter}")


    try:
        print(f"Fitting {label}...")
        result = model.fit(
            v, params, t=t,
            weights=np.full_like(v, 1 / 0.22),
            max_nfev=3000,
            iter_cb=progress_callback,
            verbose=0  # Suppresses built-in verbose, use our callback
        )
        fit_results[label] = (t, v, result)
        residuals[label] = v - result.best_fit

        print(f"\n=== Fit Report for {label} ===")
        print(fit_report(result))

    except Exception as e:
        print(f"Fit failed for {label}: {e}")

# Get a list of unique colors from a continuous colormap
cmap = cm.get_cmap('tab20')  # 'tab10' if <10 datasets, 'Set3' also works
colors = [cmap(i) for i in range(len(fit_results))]
# === Plot: Fit Overlaid on Data ===
plt.figure(figsize=(10, 6))
for i, (label, (t, v, result)) in enumerate(fit_results.items()):
    color = colors[i % len(colors)]  # wrap if > colormap length
    model_t = np.logspace(np.log10(min(t)), np.log10(max(t)), 100000)

    plt.plot(t, v, 'o', alpha=0.4, markersize=4, label=f'{label} Data', color=color)
    plt.plot(model_t, voltage_model(model_t, **result.best_values), '-', linewidth=2, label=f'{label} Fit', color=color)

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Time (s)", fontsize=20)
plt.ylabel("Voltage (V)", fontsize=20)
plt.title("Voltage Fit: 1 Stretch + 1 Exp + Power Law + Offset", fontsize=20)
plt.grid(True, which='both', ls='--', linewidth=0.5)
plt.legend(fontsize=7)
plt.tight_layout()
plt.show()

# === Plot: Residuals ===
plt.figure(figsize=(10, 4))
for label in residuals:
    t = fit_results[label][0]
    resids = residuals[label]
    plt.plot(t, resids, '.', label=label)

plt.axhline(0, ls='--', color='black')
plt.xscale('log')
plt.xlabel("Time (s)")
plt.ylabel("Residual (V)")
plt.title("Residuals: Full Model")
plt.grid(True, which='both', ls='--', linewidth=0.5)
plt.legend(fontsize=7)
plt.tight_layout()
plt.show()

# === Plot Parameter Values vs Humidity ===
humidity = []
params_by_label = {
    "A1": [], "tau1": [], "beta": [],
    "A2": [], "tau2": [],
    "A3": [], "lam": [], "mu": [],
    "C": []
}

for label, (t, v, result) in fit_results.items():
    hum = data_dict[label]["Humidity"]
    humidity.append(hum)

    for param in params_by_label:
        params_by_label[param].append(result.params[param].value)

# Plot each parameter vs humidity
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
axs = axs.flatten()

for i, param in enumerate(params_by_label):
    axs[i].scatter(humidity, params_by_label[param], color='tab:blue')
    axs[i].set_xlabel("Humidity (%)")
    axs[i].set_ylabel(param)
    axs[i].set_title(f"{param} vs. Humidity")
    axs[i].grid(True)

plt.tight_layout()
plt.show()

# === Print Parameter Statistics ===
print("\n=== Parameter Statistics ===")
for param, values in params_by_label.items():
    values_np = np.array(values)
    print(f"Parameter: {param}")
    print(f"  Extrema: ({np.min(values_np):.4g}, {np.max(values_np):.4g})")
    print(f"  Mean ± Std: {np.mean(values_np):.4g} ± {np.std(values_np):.4g}")

ratios = []
humidity_for_ratios = []

for label, (t, v, result) in fit_results.items():
    t_min, t_max = np.min(t), np.max(t)
    t_eval = np.linspace(t_min, t_max, 100000)
    v_eval = voltage_model(t_eval, **result.best_values)
    log_t_eval = np.log(t_eval)

    # First and second derivatives w.r.t. log(t)
    df_dx = np.gradient(v_eval, log_t_eval)
    d2f_dx2 = np.gradient(df_dx, log_t_eval)

    # Find indices where slope is negative and becoming less negative, and log(t) < 10.4
    candidate_indices = np.where((df_dx < 0) & (d2f_dx2 > 0) & (log_t_eval < 10.4))[0]

    if len(candidate_indices) == 0:
        print(f"No candidate points found for {label}")
        continue

    # Take the middle candidate index
    cnt_idx = candidate_indices[len(candidate_indices) // 2]

    # Compute ratio of initial voltage to voltage at cnt_idx
    ratio = v_eval[0] / v_eval[cnt_idx]
    ratios.append(ratio)
    humidity_for_ratios.append(data_dict[label]["Humidity"])

    # Your existing plotting code here...

# Now plot ratios vs humidity
plt.figure(figsize=(8, 5))
plt.scatter(humidity_for_ratios, ratios, color='tab:purple')
plt.xlabel("Humidity (%)")
plt.ylabel("Voltage Ratio (V_0 / V_at_point)")
plt.title("Voltage Ratio vs. Humidity")
plt.grid(True)
plt.show()





