"""

This script is used to plot the parameter samples.

"""

import numpy as np
from scipy import stats
from scipy.stats import truncnorm, lognorm, pearson3, t, chi2, norm, uniform
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

import os
import re
import json
from typing import Dict, Any
from itertools import combinations

# Set the random seed before generating any random numbers
np.random.seed(123456)  # You can use any integer as the seed

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"
 

def plot_inlet_discharge_pairs(Q_samples_train_val_test, Q_samples_application):
    """
    Plot pairwise relationships between inlet discharges from Q_samples array. The samples for both train/val/test and application sets are plotted together.
    
    Parameters:
    -----------
    Q_samples_train_val_test : numpy.ndarray
        2D array of shape (n_samples, 3) containing discharge samples for 3 inlets for the train, val, and test sets
    Q_samples_application : numpy.ndarray
        2D array of shape (n_samples, 3) containing discharge samples for 3 inlets for the application set
    """   
    
    # Create labels for inlets
    inlet_labels = [f'$Q_{i+1}$' for i in range(3)]
    
    # Create figure with subplots for each pair
    n_pairs = len(list(combinations(range(Q_samples_train_val_test.shape[1]), 2)))
    n_cols = 1  # Number of columns in the subplot grid
    n_rows = 3 # (n_pairs + n_cols - 1) // n_cols  # Calculate number of rows needed

    #plot a 2by2 subplot grid with the fixed selection of pairs: (1,2), (1,3), (1,4), (2,3)
    pairs = [(0,1), (0,2), (1,2)]

    #indics for the special points in the application set (with min, average, and max Wasserstein distances)
    application_special_indices = np.array([1, 70, 73]) - 1 #subtract 1 to get the indices in the array

    # Define colors for special cases (one color per point)
    n_special = 3  #len(special_case_indices); we have 3 special cases: average distance, minimum distance, and maximum distance    
    special_colors = ['red', 'green', 'blue']

    fig = plt.figure(figsize=(15, 5*n_rows))
    for idx, (i, j) in enumerate(pairs):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        #ax.scatter(Q_samples[:100, i], Q_samples[:100, j], c='blue', s=30, alpha=0.6)
        ax.scatter(Q_samples_train_val_test[:, i], Q_samples_train_val_test[:, j], c='k', s=30, alpha=0.6, label='Train/Val/Test parameters')
        ax.scatter(Q_samples_application[:, i], Q_samples_application[:, j], edgecolors='k', facecolors='none', s=40, label='Out-of-distribution parameters')

        # Mark the special cases with different colored dots
        ax.scatter(Q_samples_application[application_special_indices, i], Q_samples_application[application_special_indices, j], facecolors='none', edgecolors=special_colors, marker='o', linewidths=1.5, s=200)
        

        # Add labels
        ax.set_xlabel(f'$Q_{i+1}$ (m³/s)', fontsize=28)
        ax.set_ylabel(f'$Q_{j+1}$ (m³/s)', fontsize=28)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add correlation coefficient
        #correlation = np.corrcoef(Q_samples_train_val_test[:, i], Q_samples_train_val_test[:, j])[0,1]
        #correlation_application = np.corrcoef(Q_samples_application[:, i], Q_samples_application[:, j])[0,1]
        #ax.text(0.05, 0.9, f'Correlation coefficient: {correlation:.3f} (train/val/test)\nCorrelation coefficient: {correlation_application:.3f} (application)', 
        #        transform=ax.transAxes, fontsize=24,
        #        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Format the x and y axis labels
        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.tick_params(axis='both', which='minor', labelsize=24)

        # Add legend
        ax.legend(fontsize=24, loc='upper left')
        
    
    # Adjust layout
    plt.tight_layout()

    #increase the row spacing 
    plt.subplots_adjust(hspace=0.4)
    
    # Save the plot
    plt.savefig('inlet_discharge_pairs.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_parameter_samples(config):
    """
    Generate the parameter samples based on the configuration.

    Parameters:
    - config (dict): Configuration dictionary.

    Returns:
    - Q_samples (numpy.ndarray): Array of shape (n_samples, n_inlet_q) containing the sampled parameter values.
    - Q_config (dict): Dictionary containing the parameter configuration.
    - header_string (str): String containing the parameter names.
    - fileName_parameters (str): String containing the file name of the sampled parameter values.
    """

    # Generate samples
    print("Generating the parameter samples ...")
    
    #note: the returned samples dictionary contains the following keys: discharges, Q_config. The shape of discharges is (n_samples, n_inlet_q).
    samples = generate_discharge_samples(config)
    
    # Print sample statistics
    Q_samples = samples["discharges"]   
    Q_config = samples["Q_config"]

    #print the first 5 samples of each for inspection
    print("Some Q samples:", Q_samples[:5])
    print("Some Q config:", Q_config)

     # Create a header string with the parameter names
    header = []

    #print the statistics of the samples for each inlet-q boundary
    inlet_q_index = 0
    for param_id in Q_config.keys():
        print(f"Parameter ID: {param_id}")
        header.append("Q_ID_"+str(param_id))
        print(f"Mean: {np.mean(Q_samples[:, inlet_q_index]):.3f}")
        print(f"Min: {np.min(Q_samples[:, inlet_q_index]):.3f}")
        print(f"Max: {np.max(Q_samples[:, inlet_q_index]):.3f}")
        inlet_q_index += 1

    #combine the list of header strings into a single string
    header_string = " ".join(header)
    
    #save the sampled parameter values for record
    date_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")    #take the current date and time
    fileName_parameters = "sampledParameters_"+date_time+".dat"
    save_samples_to_file(fileName_parameters, header_string, Q_samples)

    #save the Q_config to a json file
    with open('Q_config.json', 'w') as f:
        json.dump(Q_config, f, indent=4)
    print("Q_config saved to Q_config.json")

    return Q_samples, Q_config, header_string, fileName_parameters


def make_plots(Q_config_train_val_test, Q_config_application, fileName_parameters_train_val_test, fileName_parameters_application):
    """
    Make the plots for the parameter samples.
    
    Parameters:
    - Q_config (dict): Configuration dictionary.
    - fileName_parameters_train_val_test (str): String containing the file name of the sampled parameter values for the train, val, and test sets.
    - fileName_parameters_application (str): String containing the file name of the sampled parameter values for the application set.
    """
    # Load the sampled parameters
    # Read the header (first line) to get column names
    with open(fileName_parameters_train_val_test, 'r') as f:
        header_line = f.readline().strip()
        column_names_train_val_test = header_line.split()

    print("Column names: ", column_names_train_val_test)
    
    Q_samples_train_val_test = np.loadtxt(fileName_parameters_train_val_test, skiprows=1)
    Q_samples_application = np.loadtxt(fileName_parameters_application, skiprows=1)

    #plot the histograms and pairwise scatter matrix of the samples for visual inspection
    print("Plotting the samples ...")     
    
    # Plot inlet discharge pairs    
    plot_inlet_discharge_pairs(Q_samples_train_val_test, Q_samples_application)


if __name__ == "__main__":
  
    # read the Q_config from the json files
    with open('Q_config_train_val_test.json', 'r') as f:
        Q_config_train_val_test = json.load(f) 
    with open('Q_config_application.json', 'r') as f:
        Q_config_application = json.load(f) 

    #set fileName_parameters to the file name of the sampled parameter values
    fileName_parameters_train_val_test = "sampledParameters_2025_11_14-11_40_47_AM.dat"
    fileName_parameters_application = "applicationParameters_2025_11_25-02_08_27_PM.dat"

    make_plots(Q_config_train_val_test, Q_config_application, fileName_parameters_train_val_test, fileName_parameters_application)

    print("All done!")



