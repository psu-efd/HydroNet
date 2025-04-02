#Plot the time history of the simulation results in "Output_MISC" folder
# - monitoring lines
# - monitoring points

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_time_history_data(data_dir):
    """
    Plot time history data from monitoring points and lines.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the monitoring data files
    """
    # Define colors for different variables
    colors = {
        'WSE': 'blue',
        'VEL': 'red',
        'DEP': 'green',
        'tau_b': 'orange',
        'Q': 'purple'
    }
    
    # Define labels for different variables
    labels = {
        'WSE': 'Water Surface Elevation (m)',
        'VEL': 'Velocity (m/s)',
        'DEP': 'Water Depth (m)',
        'Q': 'Discharge (m³/s)',
        'tau_b': 'Bed Shear Stress (N/m²)'
    }
    
    # Process monitoring points (PT files)
    pt_files = [f for f in os.listdir(data_dir) if f.startswith('SacromentoRiver_PT')]
    for pt_file in pt_files:
        # Read data
        # Data: Time(hours)    X_meter         Y_meter         Bed_Elev_meter  Water_Elev_m    Water_Depth_m   Vel_X_m_p_s     Vel_Y_m_p_s     Vel_Mag_m_p_s   Froude          B_Stress_n_p_m2 
        data = np.loadtxt(os.path.join(data_dir, pt_file), skiprows=1)
        time = data[:, 0]  # First column is time
        
        # Create figure with subplots for each variable
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Time History at {pt_file.replace(".dat", "")}', fontsize=16)
        
        # Plot each variable
        for idx, (var, col) in enumerate([('WSE', 4), ('VEL', 8), ('DEP', 5), ('tau_b', 10)]):
            ax = axs[idx//2, idx%2]

            #special treatment of WSE and DEP (-999 means dry, thus WSE is the same as Bed_Elev and DEP is 0)
            if var == 'WSE':
                # For dry cells (-999), WSE equals bed elevation
                mask = data[:, col] == -999
                data[mask, col] = data[mask, 3]  # Use bed elevation for dry cells
            elif var == 'DEP':
                # For dry cells (-999), depth is 0
                mask = data[:, col] == -999
                data[mask, col] = 0

            ax.plot(time, data[:, col], color=colors[var], linewidth=2)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(labels[var])
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_title(f'{var} vs Time')
        
        plt.tight_layout()
        plt.savefig(f'{pt_file.replace(".dat", "_time_history.png")}', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Process monitoring lines (LN files)
    ln_files = [f for f in os.listdir(data_dir) if f.startswith('SacromentoRiver_LN')]
    for ln_file in ln_files:
        # Read data
        data = np.loadtxt(os.path.join(data_dir, ln_file), skiprows=1)
        time = data[:, 0]  # First column is time
        
        # Create figure with subplots for Q and WSE
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'Time History at {ln_file.replace(".dat", "")}', fontsize=16)
        
        # Plot discharge
        ax1.plot(time, data[:, 1], color=colors['Q'], linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel(labels['Q'])
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_title('Discharge vs Time')
        
        # Plot water surface elevation
        ax2.plot(time, data[:, 2], color=colors['WSE'], linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel(labels['WSE'])
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_title('Water Surface Elevation vs Time')
        
        plt.tight_layout()
        plt.savefig(f'{ln_file.replace(".dat", "_time_history.png")}', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Process exit rating curve (EXITH_RC file)
    rc_file = 'SacromentoRiver_EXITH_RC1.dat'
    if os.path.exists(os.path.join(data_dir, rc_file)):
        # Read data
        data = np.loadtxt(os.path.join(data_dir, rc_file), skiprows=4)
        time = data[:, 0]  # First column is time
        
        # Create figure with subplots for each variable
        fig, axs = plt.subplots(2, 1, figsize=(15, 12))
        fig.suptitle('Time History at Exit Rating Curve', fontsize=16)
        
        # Plot each variable
        for idx, (var, col) in enumerate([('Q', 1), ('WSE', 2)]):
            ax = axs[idx]
            ax.plot(time, data[:, col], color=colors[var], linewidth=2)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(labels[var])
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_title(f'{var} vs Time')
        
        plt.tight_layout()
        plt.savefig('exit_rating_curve_time_history.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # Set the directory containing the monitoring data files
    data_dir = "Output_MISC"
    
    # Create plots
    plot_time_history_data(data_dir)
    print("Time history plots have been generated and saved.")


