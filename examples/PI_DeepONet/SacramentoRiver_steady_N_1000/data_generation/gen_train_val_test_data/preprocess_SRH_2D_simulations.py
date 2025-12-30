"""

This script is used to preprocess, i.e., to sample parameters for simulation with the SRH_2D_Model and SRH_2D_Data classes. The example code samples three parameters (which can be optionally turned off):

1. Manning's n in the main channel
2. Upstream boundary discharges
3. Downstream water elevation

For each parameter, it is assumed to follow a specified distribution. 

The script then generates a set of random samples from the distribution and saves them to a file.

"""

import numpy as np
from scipy import stats
from scipy.stats import truncnorm, lognorm, pearson3, t, chi2, norm, uniform
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

import h5py

import os
import re
import json
from typing import Dict, Any

# Set the random seed before generating any random numbers
np.random.seed(123456)  # You can use any integer as the seed

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

def extract_parameter_id(param_name):
    match = re.search(r'\d+$', param_name)
    return int(match.group()) if match else None
    
def generate_discharge_samples(params):
    """
    Generate parameter samples for discharges at the inlet-q boundaries.

    This function only allows for inlet discharges at the inlet-q boundaries, not other parameters.
       
    Parameters:
    - params (dict): Dictionary containing parameters for discharges at the inlet-q boundaries.

    Returns:
    - samples (dict): Dictionary containing the discharges samples and the parameter configuration.
    """

    #get the number of samples to generate
    #print("params: ", params["parameter_specs"])
    n_samples = params["parameter_specs"]["n_samples"]

    #get the parameters
    parameters = params["parameter_specs"]["parameters"]

    #number of inlet-q boundaries
    n_inlet_q = len(parameters)

    #check if the parameters are correctly specified: there should be n_inlet_q Q parameters
    has_Q = any("Q_ID" in param for param in parameters.keys())

    if not has_Q:
        raise ValueError("Check the parameter names in the config file. Only n_inlet_q inlet discharges at the inlet-q boundaries are supported.")

    #get the part for each parameter. The name should be Q_ID_1 or Q_ID_2, etc., where the number at the end of the string indicates the inlet BC number. 
    Q_config = {}

    #discharges array with shape (n_samples, n_inlet_q)
    discharges = np.zeros((n_samples, n_inlet_q))

    # Generate LHS samples for n_inlet_q parameters (Q)
    lhs_samples = _generate_lhs_samples(n_samples, n_inlet_q)

    # Loop through all inlet discharge parameters
    inlet_q_index = 0
    for param_name, param_data in parameters.items():
        print("inlet_q_index: ", inlet_q_index)
        param_id = extract_parameter_id(param_name)
        param_type = param_name.split('_')[0]  # Gets 'Q'.
        print(f"Parameter: {param_name}, Type: {param_type}, ID: {param_id}")

        if param_type == "Q":
            #check that the distribution is correctly specified
            if param_data['distribution'] != "uniform":
                raise ValueError("Only uniform distribution is supported for inlet discharge.")
         
            Q_config[param_id] = param_data
        else:
            raise ValueError("Only inlet discharges at the inlet-q boundaries are supported.")

        # Given sample statistics for Q parameters
        if Q_config[param_id]['distribution'] == "uniform":
            Q_min = Q_config[param_id]['min']  # minimum value
            Q_max = Q_config[param_id]['max']  # maximum value
        else:
            raise ValueError("Only uniform distribution is supported for inlet discharge.")
    
        # Transform LHS samples to respective distributions
        discharges[:, inlet_q_index] = Q_min + (Q_max - Q_min) * lhs_samples[:, inlet_q_index]
        inlet_q_index += 1

   
    #assemble the samples into a dictionary; also add the parameter specs to the dictionary
    #note: the shape of discharges is (n_samples, n_inlet_q)
    samples = {
        "discharges": discharges,
        "Q_config": Q_config
    } 

    return samples

def _generate_lhs_samples(n_samples: int, n_parameters: int) -> np.ndarray:
    """Generate Latin Hypercube Samples."""
    # Generate the intervals
    cut_points = np.linspace(0, 1, n_samples + 1)
    
    # Generate samples for each parameter
    samples = np.zeros((n_samples, n_parameters))
    
    for i in range(n_parameters):
        # Generate random positions within each interval
        samples[:, i] = cut_points[:-1] + np.random.rand(n_samples) * (cut_points[1] - cut_points[0])
        
        # Randomly shuffle the samples
        np.random.shuffle(samples[:, i])
    
    return samples

def save_samples_to_file(filename, header, samples):
    """
    Save sampled values to a text file.

    In the dictionary samples, the keys are discharges, wse_samples, and manning_n. The shape of discharges is (n_samples), the shape of wse_samples is (n_samples), and the shape of manning_n is (n_samples). We only save discharges, wse_samples, and Manning's n. All arrays are flattened and combined to a 2D array.
    
    Parameters:
        filename (str): The path to the output text file.
        samples (dict): Dictionary of parameter samples as returned by generate_parameter_samples.
    """

    discharges = samples
       
    # Save to a text file using numpy.savetxt
    np.savetxt(filename, discharges, header=header, comments='')
    print(f"Samples saved to {filename}")


def compute_sample_statistics(samples):
    """
    Compute the sample statistics of the discharges.
    """
    n = len(samples)
    
    est_min = np.min(samples)
    est_max = np.max(samples)
    est_mean = np.mean(samples)
    est_std = np.std(samples, ddof=1)   #use ddof=1 for sample std (following Stedinger et al. Chapter 18, 1993)
    est_skew = ((samples - est_mean)**3).sum() * n / ((n-1) * (n-2) * est_std**3)  # Skewness formula

    return est_min, est_max, est_mean, est_std, est_skew
    
def plot_discharge_sample_pdf(inlet_q_index, discharges, Q_config, plot_save_filename):
    """
    Plot the sample probability density function (PDF) of the discharges.

    Parameters:
    - inlet_q_index (int): Index of the inlet-q boundary to plot.
    - discharges (array): Array of discharges.
    - Q_config (dict): Dictionary of Q configuration.
    - plot_save_filename (str): Filename to save the plot.
    """

    #get the ground truth uniform parameters
    true_min = Q_config['min']
    true_max = Q_config['max']
   
    # Estimate uniform parameters from sample data
    # number of samples
    n = len(discharges)
    est_min_q, est_max_q, est_mean_q, est_std_q, est_skew_q = compute_sample_statistics(discharges)

    # Compute histograms using ground truth range for proper comparison
    # Use the ground truth range to ensure histogram and PDF are normalized over the same range
    hist_counts, bin_edges = np.histogram(discharges, bins=20, range=(true_min, true_max), density=True)
    
    print("\n=== Histogram Statistics ===")
    print(f"All discharges - Max density: {np.max(hist_counts):.4f}")
    print(f"All discharges - Total area: {np.sum(hist_counts * np.diff(bin_edges)):.4f}")
    print(f"Sample range: [{np.min(discharges):.2f}, {np.max(discharges):.2f}]")
    print(f"Ground truth range: [{true_min:.2f}, {true_max:.2f}]")
    print("\n")
    
    # Convert bin edges to real space for plotting
    bin_edges_real = bin_edges
    bin_centers_real = (bin_edges[1:] + bin_edges[:-1])/2

    # Define uniform distributions from the sample data and the ground truth
    # Note: stats.uniform(loc, scale) where loc=min and scale=max-min
    estimated_uniform_dist = stats.uniform(loc=est_min_q, scale=est_max_q - est_min_q)
    ground_truth_uniform_dist = stats.uniform(loc=true_min, scale=true_max - true_min)

    # Generate theoretical PDF values using ground truth range for proper comparison
    x_values = np.linspace(true_min, true_max, 1000)

    est_uniform_pdf = estimated_uniform_dist.pdf(x_values) 
    true_uniform_pdf = ground_truth_uniform_dist.pdf(x_values)


    # Plot sample and theoretical PDFs
    plt.figure(figsize=(8, 6))    
    plt.plot(x_values[1:-1], est_uniform_pdf[1:-1], 'r-', linewidth=2, label="Estimated uniform PDF from sample")
    plt.plot(x_values, true_uniform_pdf, 'g--', linewidth=2, label="Uniform PDF from ground truth")

    #plot histogram of the discharges
    plt.bar(bin_centers_real, hist_counts, width=np.diff(bin_edges_real), 
            alpha=0.5, label="Discharge sample distribution", align='center')       

    plt.xlabel("Discharge (cms)")
    plt.ylabel("Probability (%)")
    #plt.xlim(true_min, true_max)  # Set x-axis to ground truth range for proper comparison
    #plt.ylim(0, 7)
    #plt.title("Comparison of Sample, Estimated, and Ground Truth LP3 PDF (Fixed)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(plot_save_filename, dpi=300, bbox_inches='tight', pad_inches=0)

    #plt.show()


def plot_inlet_discharge_pairs(Q_samples):
    """
    Plot pairwise relationships between inlet discharges from Q_samples array.
    
    Parameters:
    -----------
    Q_samples : numpy.ndarray
        2D array of shape (n_samples, 7) containing discharge samples for 7 inlets
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import combinations
    
    # Create labels for inlets
    inlet_labels = [f'Inlet {i+1}' for i in range(7)]
    
    # Create figure with subplots for each pair
    n_pairs = len(list(combinations(range(Q_samples.shape[1]), 2)))
    n_cols = 1  # Number of columns in the subplot grid
    n_rows = 3 # (n_pairs + n_cols - 1) // n_cols  # Calculate number of rows needed

    #plot a 2by2 subplot grid with the fixed selection of pairs: (1,2), (1,3), (1,4), (2,3)
    pairs = [(0,1), (0,2), (1,2)]

    fig = plt.figure(figsize=(15, 5*n_rows))
    for idx, (i, j) in enumerate(pairs):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        #ax.scatter(Q_samples[:100, i], Q_samples[:100, j], c='blue', s=30, alpha=0.6)
        ax.scatter(Q_samples[:, i], Q_samples[:, j], c='blue', s=30, alpha=0.6)

        # Add labels
        ax.set_xlabel(f'$Q_{i+1}$ (m³/s)', fontsize=28)
        ax.set_ylabel(f'$Q_{j+1}$ (m³/s)', fontsize=28)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add correlation coefficient
        correlation = np.corrcoef(Q_samples[:, i], Q_samples[:, j])[0,1]
        ax.text(0.05, 0.9, f'Correlation coefficient: {correlation:.3f}', 
                transform=ax.transAxes, fontsize=24,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Format the x and y axis labels
        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.tick_params(axis='both', which='minor', labelsize=24)
        
    
    # Adjust layout
    plt.tight_layout()
    
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


def make_plots(Q_config, fileName_parameters):
    """
    Make the plots for the parameter samples.
    
    Parameters:
    - Q_config (dict): Configuration dictionary.
    - fileName_parameters (str): String containing the file name of the sampled parameter values.
    """
    # Load the sampled parameters
    # Read the header (first line) to get column names
    with open(fileName_parameters, 'r') as f:
        header_line = f.readline().strip()
        column_names = header_line.split()

    print("Column names: ", column_names)
    
    Q_samples = np.loadtxt(fileName_parameters, skiprows=1)

    #plot the histograms and pairwise scatter matrix of the samples for visual inspection
    print("Plotting the samples ...")
    #loop through all the inlet-q boundaries and plot the discharge sample pdf
    inlet_q_index = 0   
    for param_name in column_names:
        param_id = extract_parameter_id(param_name)
        print("param_id: ", param_id)
        plot_save_filename = "discharge_sample_pdf_"+param_name+".png"
        plot_discharge_sample_pdf(inlet_q_index, Q_samples[:, inlet_q_index], Q_config[str(param_id)], plot_save_filename)
        inlet_q_index += 1
  
    
    # Plot inlet discharge pairs    
    plot_inlet_discharge_pairs(Q_samples)


if __name__ == "__main__":

    # Load configuration
    with open("simulations_config.json", "r") as f:
        config = json.load(f)
    
    # Generate the parameter samples
    #Q_samples, Q_config, header_string, fileName_parameters = generate_parameter_samples(config)

    # Make the plots
    # read the Q_config from the json file
    with open('Q_config.json', 'r') as f:
        Q_config = json.load(f) 

    #set fileName_parameters to the file name of the sampled parameter values
    fileName_parameters = "sampledParameters_2025_11_14-11_40_47_AM.dat"

    make_plots(Q_config, fileName_parameters)

    print("All done!")



