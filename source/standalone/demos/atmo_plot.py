import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Apply publication-style settings
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX for rendering
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2,
    "grid.linestyle": "--",
})

# Function to load data from an HDF5 file
def load_h5_data(h5_filepath):
    with h5py.File(h5_filepath, 'r') as file:
        times_log = np.array(file["times"])
        observations_log = np.array(file["observations"])
        actions_log = np.array(file["actions"])
        filtered_actions_log = np.array(file["filtered_actions"])
        x_ref_log = np.array(file["x_ref"])
    return times_log, observations_log, actions_log, filtered_actions_log, x_ref_log

# Function to plot data with publication-quality formatting
def plot_data(times_log, observations_log, actions_log, filtered_actions_log, x_ref_log, save_path=None):
    sns.set_theme(style="whitegrid")

    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(r"\textbf{ATMO Simulation Results}", fontsize=18)

    # Plot the actions
    axs[0].plot(times_log, actions_log[:, 0], label="Thrust 0")
    axs[0].plot(times_log, actions_log[:, 1], label="Thrust 1")
    axs[0].plot(times_log, actions_log[:, 2], label="Thrust 2")
    axs[0].plot(times_log, actions_log[:, 3], label="Thrust 3")
    axs[0].plot(times_log, actions_log[:, 4], label="Tilt", linestyle="--")
    axs[0].set_title("Motor Actions")
    axs[0].set_ylabel("Value")
    axs[0].legend()
    axs[0].grid(True)

    # Plot filtered actions
    axs[1].plot(times_log, filtered_actions_log[:, 0], label="Filtered Thrust 0")
    axs[1].plot(times_log, filtered_actions_log[:, 1], label="Filtered Thrust 1")
    axs[1].plot(times_log, filtered_actions_log[:, 2], label="Filtered Thrust 2")
    axs[1].plot(times_log, filtered_actions_log[:, 3], label="Filtered Thrust 3")
    axs[1].plot(times_log, filtered_actions_log[:, 4], label="Filtered Tilt", linestyle="--")
    axs[1].set_title("Filtered Actions")
    axs[1].set_ylabel("Value")
    axs[1].legend()
    axs[1].grid(True)

    # Plot observations
    axs[2].plot(times_log, observations_log[:, 0], label=r"$X$")
    axs[2].plot(times_log, observations_log[:, 1], label=r"$Y$")
    axs[2].plot(times_log, observations_log[:, 2], label=r"$Z$")
    axs[2].plot(times_log, x_ref_log[:, 0], label=r"$X_{ref}$", linestyle="--")
    axs[2].plot(times_log, x_ref_log[:, 1], label=r"$Y_{ref}$", linestyle="--")
    axs[2].plot(times_log, x_ref_log[:, 2], label=r"$Z_{ref}$", linestyle="--")
    axs[2].set_title("Position Observations and References")
    axs[2].set_ylabel("Position (m)")
    axs[2].legend()
    axs[2].grid(True)

    # Plot velocities
    axs[3].plot(times_log, observations_log[:, 12], label=r"$V_X$")
    axs[3].plot(times_log, observations_log[:, 13], label=r"$V_Y$")
    axs[3].plot(times_log, observations_log[:, 14], label=r"$V_Z$")
    axs[3].set_title("Velocity Observations")
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("Velocity (m/s)")
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    
    # Save the plots
    if save_path:
        # fig.savefig(os.path.join(save_path, "atmo_simulation_results.pdf"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(save_path, "atmo_simulation_results.png"), dpi=300, bbox_inches="tight")
    
    plt.show()

    # Plot tilt angle separately
    tilt_angle = np.rad2deg(observations_log[:, 18])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times_log, tilt_angle, color="tab:orange")
    ax.set_title("Tilt Angle")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tilt Angle (degrees)")
    ax.grid(True)

    if save_path:
        # fig.savefig(os.path.join(save_path, "tilt_angle.pdf"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(save_path, "tilt_angle.png"), dpi=300, bbox_inches="tight")

    plt.show()

# Main execution
if __name__ == "__main__":
    h5_filepath = "/home/m4pc/src/IsaacLab/source/standalone/demos/data/run_mpc_2025-02-17_17-04-11/data_mpc.h5"  # Change this to the actual file path
    output_folder = os.path.dirname(h5_filepath)  # Save in the same directory as HDF5 file

    times_log, observations_log, actions_log, filtered_actions_log, x_ref_log = load_h5_data(h5_filepath)
    plot_data(times_log.squeeze(), observations_log.squeeze(), actions_log.squeeze(), filtered_actions_log.squeeze(), x_ref_log.squeeze(), save_path=output_folder)
