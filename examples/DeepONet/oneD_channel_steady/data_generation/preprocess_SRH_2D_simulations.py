"""

This script is used to preprocess, i.e., to sample parameters for simulation with the SRH_2D_Model and SRH_2D_Data classes. The example code samples three parameters (which can be optionally turned off):

1. Manning's n in the main channel
2. Upstream boundary discharge
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
import pandas as pd
import seaborn as sns

import h5py

import os
import re
import json
from typing import Dict, Any

# Set the random seed before generating any random numbers
np.random.seed(123456)  # You can use any integer as the seed

plt.rc('text', usetex=False)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

def extract_parameter_id(param_name):
    match = re.search(r'\d+$', param_name)
    return int(match.group()) if match else None
    
def generate_parameter_samples(params, filename):
    """
    Generate parameter samples for discharge, Manning's n, and/or exit WSE.

    Currently, the parameters to be sampled are limited to one inlet discharge, one Manning's n, and one exit WSE. For the exit WSE, a rating curve can be used to compute the WSE for each discharge of the inlet.
       
    Parameters:
    - params (dict): Dictionary containing parameters for discharge, Manning's n, and exit WSE.
    - filename (str): Output file name for saving results in HDF5 format.

    Returns:
    - None (saves results to a file).
    """

    #get the number of samples to generate
    #print("params: ", params["parameter_specs"])
    n_samples = params["parameter_specs"]["n_samples"]

    #get the parameters
    parameters = params["parameter_specs"]["parameters"]

    #print("parameters: ", parameters)

    #check how many parameters are to be sampled: currently only one inlet discharge, one Manning's n, and one exit WSE are supported
    if len(parameters) != 3:
        raise ValueError("Only one inlet discharge, one Manning's n, and one exit WSE are supported.")

    #check if the parameters are correctly specified: there should be one Q, one Manning's n, and one WSE
    has_Q = any("Q_ID" in param for param in parameters.keys())
    has_Manning = any("Manning_n_ID" in param for param in parameters.keys())
    has_WSE = any("WSE_ID" in param for param in parameters.keys())

    if not (has_Q and has_Manning and has_WSE):
        raise ValueError("Check the parameter names in the config file. Only one inlet discharge, one Manning's n, and one exit WSE are supported.")

    #get the part for each parameter. The name should be Q_ID_1 or Q_ID_2, etc., where the number at the end of the string indicates the inlet BC number. 
    Q_config = None
    Manning_n_config = None
    WSE_config = None

    # Loop through all parameters
    for param_name, param_data in parameters.items():
        param_id = extract_parameter_id(param_name)
        param_type = param_name.split('_')[0]  # Gets 'Q', 'Manning', 'WSE', etc.
        print(f"Parameter: {param_name}, Type: {param_type}, ID: {param_id}")

        if param_type == "Q":
            #check that the distribution is correctly specified
            if param_data['distribution'] != "uniform":
                raise ValueError("Only uniform distribution is supported for inlet discharge.")
         
            Q_config = param_data
        elif param_type == "Manning":
            Manning_n_config = param_data
        elif param_type == "WSE":
            #for exit WSE, currently only rating curve is supported
            if param_data['distribution'] != "rating_curve":
                raise ValueError("Currently only rating curve is supported for exit WSE.")

            #in addition, the rating curve should be specified in a file. The file should be in the same directory as the config file.
            if param_data['curve_type'] != "from_file":
                raise ValueError("Currently only rating curve from file is supported for exit WSE.")

            if not os.path.exists(param_data['file_name']):
                raise ValueError(f"The rating curve file {param_data['file_name']} does not exist.")

            #check that Q is specified for the rating curve
            if param_data['Q_parameter_name'] is None:
                raise ValueError("Q is not specified for the rating curve.")

            WSE_config = param_data
    
    # Given sample statistics for Q parameters
    if Q_config['distribution'] == "uniform":
        Q_min = Q_config['min']  # minimum value
        Q_max = Q_config['max']  # maximum value
    else:
        raise ValueError("Only uniform distribution is supported for inlet discharge.")

    # Manning's n parameters (Truncated Normal)
    if Manning_n_config['distribution'] == "truncated_normal":
        n_mean = Manning_n_config['mean']  # Example mean value for Manning's n
        n_std = Manning_n_config['std']   # Standard deviation
        n_min = Manning_n_config['min']   # Lower bound
        n_max = Manning_n_config['max']   # Upper bound
    elif Manning_n_config['distribution'] == "log_normal":
        n_est = Manning_n_config['estimate']  # Estimated value for Manning's n
        n_cov = Manning_n_config['cov']   # Covariance for the normal distribution for log(n)
        #n_min = Manning_n_config['min']   # Lower bound
        #n_max = Manning_n_config['max']   # Upper bound
    else:
        raise ValueError("Only truncated normal or log normal distribution is supported for Manning's n.")

    if WSE_config['distribution'] == "rating_curve":
        # Get the rating curve file
        rating_curve_file = WSE_config['file_name']
    else:
        raise ValueError("Only rating curve is supported for exit WSE.")

    # Generate LHS samples for 2 parameters (Q and Manning's n)
    lhs_samples = _generate_lhs_samples(n_samples, 2)

    # Transform LHS samples to respective distributions

    discharges = Q_min + (Q_max - Q_min) * lhs_samples[:, 0]

    # Manning's n follows a truncated normal distribution or log normal distribution
    if Manning_n_config['distribution'] == "truncated_normal":
        a, b = (n_min - n_mean) / n_std, (n_max - n_mean) / n_std  # Bounds in standard deviation units
        n_samples = truncnorm.ppf(lhs_samples[:, 1], a, b, loc=n_mean, scale=n_std)    
    elif Manning_n_config['distribution'] == "log_normal":
        #compute the mean and standard deviation of the normal distribution for log(n)
        normal_mean = (-1.0 + np.sqrt(1.0+2*np.log(n_est)*n_cov**2))/n_cov**2
        normal_std = abs(normal_mean*n_cov)   #abs is used to ensure the standard deviation is positive

        print("n_est:",n_est)
        print("n_cov:",n_cov)
        print("normal_mean:",normal_mean)
        print("normal_std:",normal_std)

        n_samples = _generate_lognormal(lhs_samples[:, 1], normal_mean, normal_std)
    else:
        raise ValueError("Currently only truncated normal or log normal distribution is supported for Manning's n.")

    # Interpolate WSE for each discharge. wse_samples is a 1D array with shape (n_samples)
    wse_samples = np.array([_interpolate_wse(discharge, rating_curve_file)
                           for discharge in discharges])
    
    print("WSE samples shape:",wse_samples.shape)
    print("Manning's n samples shape:",n_samples.shape)

    #assemble the samples into a dictionary; also add the parameter specs to the dictionary
    #note: the shape of discharges is (n_samples), the shape of wse_samples is (n_samples), and the shape of manning_n is (n_samples).
    samples = {
        "discharges": discharges,
        "wse_samples": wse_samples,
        "manning_n": n_samples,
        "Q_config": Q_config,
        "Manning_n_config": Manning_n_config,
        "WSE_config": WSE_config
    }

    # Save results in an HDF5 file for efficient storage
    with h5py.File(filename+".h5", "w") as f:
        f.create_dataset("discharges", data=discharges)
        f.create_dataset("wse_samples", data=wse_samples)
        f.create_dataset("manning_n", data=n_samples)

    # Save the discharge and max discharge results in a csv file
    np.savetxt(filename+"_discharges.csv", discharges.flatten(), delimiter=",")
    np.savetxt(filename+"_wse_samples.csv", wse_samples.flatten(), delimiter=",")
    np.savetxt(filename+"_manning_n.csv", n_samples.flatten(), delimiter=",")


    print("\n")
    print("length of discharges flattened:",len(discharges.flatten()))
    print("length of wse samples flattened:",len(wse_samples.flatten()))
    print("length of manning's n flattened:",len(n_samples.flatten()))

    print(f"Samples generated saved to {filename}")

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

def _generate_truncated_normal(uniform_samples: np.ndarray, min_val: float, max_val: float, 
                             mean: float, std: float) -> np.ndarray:
    """Transform uniform samples to truncated normal distribution."""
    a = (min_val - mean) / std
    b = (max_val - mean) / std
    return truncnorm.ppf(uniform_samples, a, b, loc=mean, scale=std)

def _generate_lognormal(uniform_samples: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Transform uniform samples to lognormal distribution.

    Parameters:
        uniform_samples: uniform samples
        mean: mean of the normal distribution for log(x)
        std: standard deviation of the normal distribution for log(x)
    
    Returns:
        lognormal samples
    """
    
    normal_std = std
    normal_mean = mean
    
    return lognorm.ppf(uniform_samples, normal_std, scale=np.exp(normal_mean))

def _generate_log_pearson3(uniform_samples: np.ndarray, mean: float, std: float, 
                          skew: float) -> np.ndarray:
    """
    Transform uniform samples to Log-Pearson Type III distribution.

    This function is not used in the current implementation.
    """

    # Convert to log space
    log_mean = np.log10(mean)
    log_std = std / (mean * np.log(10))
    
    # Generate Pearson Type III samples
    samples = pearson3.ppf(uniform_samples, skew, loc=log_mean, scale=log_std)
    
    # Convert back from log space
    return np.power(10, samples)

# Function to compute discharge using LP3 formula
def _compute_discharge(mu, sigma, gamma, num_draws):    
    #randomly draw nu from a uniform distribution over the interval [0.0001,0.9999)
    nu = np.random.uniform(0.0001, 0.9999, num_draws)

    #Kn = -norm.ppf(nu)
    Kn = norm.ppf(nu)
    K = (2 / gamma) * (((Kn - gamma/6)*gamma/6 + 1)**3 - 1)  #This is the formula from the Genex report
    #K = (2 / gamma) * ((1 + gamma*Kn/6-gamma**2/36)**3 - 1)  #This is the formula from Stedinger et al. (1993)
    log_Q = mu + sigma * K

    return 10**log_Q  # Convert from log-space to linear space


def _interpolate_wse(discharge, rating_curve_file):
    """
    Interpolate water surface elevation (WSE) for a given discharge using a rating curve.
    
    Parameters:
    -----------
    discharge : float or numpy.ndarray
        Discharge value(s) to interpolate WSE for
    rating_curve_file : str
        Path to the rating curve file (CSV or txt with Q and WSE columns)
        
    Returns:
    --------
    float or numpy.ndarray
        Interpolated WSE value(s)
    """
    # Read rating curve data
    try:
        rating_data = np.loadtxt(rating_curve_file, skiprows=1)
        Q = rating_data[:, 0]  # First column: discharge
        WSE = rating_data[:, 1]  # Second column: water surface elevation
    except Exception as e:
        raise ValueError(f"Error reading rating curve file: {e}")
    
    # Ensure Q is sorted in ascending order
    sort_idx = np.argsort(Q)
    Q = Q[sort_idx]
    WSE = WSE[sort_idx]
    
    # Create interpolation function
    interpolated_wse = np.interp(discharge, Q, WSE, 
                            left=WSE[0],    # Clamp to minimum WSE
                            right=WSE[-1])  # Clamp to maximum WSE
    
    return interpolated_wse

def save_samples_to_file(filename, samples):
    """
    Save sampled values to a text file.

    In the dictionary samples, the keys are discharges, wse_samples, and manning_n. The shape of discharges is (n_samples), the shape of wse_samples is (n_samples), and the shape of manning_n is (n_samples). We only save discharges, wse_samples, and Manning's n. All arrays are flattened and combined to a 2D array.
    
    Parameters:
        filename (str): The path to the output text file.
        samples (dict): Dictionary of parameter samples as returned by generate_parameter_samples.
    """

    discharges = samples["discharges"].flatten()
    wse_samples = samples["wse_samples"].flatten()
    manning_n = samples["manning_n"].flatten()

    #combine the arrays into a 2D array
    data = np.column_stack([discharges, wse_samples, manning_n])
    
    # Create a header string with the parameter names
    header = "Q WSE Manning_n"
    
    # Save to a text file using numpy.savetxt
    np.savetxt(filename, data, header=header, comments='')
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
    
def plot_discharge_sample_pdf(discharges, Q_config, plot_save_filename):
    """
    Plot the sample probability density function (PDF) of the Monte Carlo-generated discharges,
    compare it with the estimated Log Pearson Type III (LP3) PDF, and add the ground truth LP3 PDF.

    Parameters:
    - discharges (array): Array of discharges.
    """

    #get the ground truth LP3 parameters
    true_min = Q_config['min']
    true_max = Q_config['max']
   
    # Estimate uniform parameters from sample data
    # number of samples
    n = len(discharges)
    est_min_q, est_max_q, est_mean_q, est_std_q, est_skew_q = compute_sample_statistics(discharges)

    # Compute histograms (one for all discharges and one for the max discharges)
    hist_counts, bin_edges = np.histogram(discharges, bins=50, density=True)
    
    print("\n=== Histogram Statistics ===")
    print(f"All discharges - Max density: {np.max(hist_counts):.4f}")
    print(f"All discharges - Total area: {np.sum(hist_counts * np.diff(bin_edges)):.4f}")
    print("\n")
    
    # Convert bin edges to real space for plotting
    bin_edges_real = bin_edges
    bin_centers_real = (bin_edges[1:] + bin_edges[:-1])/2

    # Define uniform distributions from the sample data and the ground truth
    estimated_uniform_dist = stats.uniform(est_min_q, est_max_q)
    ground_truth_uniform_dist = stats.uniform(true_min, true_max)

    # Generate theoretical PDF values (in log-space)
    x_values = np.linspace(min(discharges), max(discharges), 1000)

    est_uniform_pdf = estimated_uniform_dist.pdf(x_values) 
    true_uniform_pdf = ground_truth_uniform_dist.pdf(x_values)


    # Plot sample and theoretical PDFs
    plt.figure(figsize=(8, 6))    
    plt.plot(x_values, est_uniform_pdf, 'r-', linewidth=2, label="Estimated uniform PDF from sample")
    plt.plot(x_values, true_uniform_pdf, 'g--', linewidth=2, label="uniform PDF from ground truth")


    #plot histogram of the discharges
    plt.bar(bin_centers_real, hist_counts, width=np.diff(bin_edges_real), 
            alpha=0.5, label="Discharge sample distribution", align='center')    
   

    plt.xlabel("Discharge (cms)")
    plt.ylabel("Probability (%)")
    #plt.xlim(0, 120000)
    #plt.ylim(0, 7)
    #plt.title("Comparison of Sample, Estimated, and Ground Truth LP3 PDF (Fixed)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(plot_save_filename, dpi=300, bbox_inches='tight', pad_inches=0)

    #plt.show()

def plot_ManningN_sample_pdf(manning_n, manning_n_config, plot_save_filename):
    """
    Plot the sample probability density function (PDF) of the Monte Carlo-generated Manning's n.

    Parameters:
    - manning_n (array): Array of Manning's n.
    - manning_n_config (dict): Dictionary of Manning's n configuration.
    """

    #get the ground truth log normal parameters
    true_estimate = manning_n_config['estimate']
    true_cov = manning_n_config['cov']

    #compute the mean and standard deviation of the normal distribution for log(n)
    normal_mean = (-1.0 + np.sqrt(1.0+2*np.log(true_estimate)*true_cov**2))/true_cov**2
    normal_std = abs(normal_mean*true_cov)   #abs is used to ensure the standard deviation is positive
   
    # Estimate log normal parameters from sample data
    # number of samples
    n = len(manning_n)
    log_n = np.log(manning_n)  # Convert to log-space
    sample_min_log_n, sample_max_log_n, sample_mean_log_n, sample_std_log_n, sample_skew_log_n = compute_sample_statistics(log_n)

    # Compute histogram
    hist_counts, bin_edges = np.histogram(log_n, bins=50, density=True)
    
    print("\n=== Histogram Statistics ===")
    print(f"All Manning's n - Max density: {np.max(hist_counts):.4f}")
    print(f"All Manning's n - Total area: {np.sum(hist_counts * np.diff(bin_edges)):.4f}")
    print("\n")
    
    # Convert bin edges to real space for plotting
    bin_edges_real = np.exp(bin_edges)
    bin_centers_real = np.exp((bin_edges[1:] + bin_edges[:-1])/2)

    # Define log normal distributions from the sample data and the ground truth
    #define a normal distribution from the sample data 
    sample_log_normal_dist = stats.norm(loc=sample_mean_log_n, scale=sample_std_log_n)
    ground_truth_log_normal_dist = stats.norm(loc=normal_mean, scale=normal_std)

    # Generate theoretical PDF values (in log-space)
    x_values = np.linspace(min(manning_n), max(manning_n), 1000)
    log_x_values = np.log(x_values)  # Convert to log-space

    sample_log_normal_pdf = sample_log_normal_dist.pdf(log_x_values) 
    ground_truth_log_normal_pdf = ground_truth_log_normal_dist.pdf(log_x_values)


    # Plot sample and theoretical PDFs
    plt.figure(figsize=(8, 6))    
    plt.plot(x_values, sample_log_normal_pdf, 'r-', linewidth=2, label="Estimated log normal PDF from sample")
    plt.plot(x_values, ground_truth_log_normal_pdf, 'g--', linewidth=2, label="log normal PDF from ground truth")


    #plot histogram of the discharges
    plt.bar(bin_centers_real, hist_counts, width=np.diff(bin_edges_real), 
            alpha=0.5, label="Manning's n sample distribution", align='center')    
    

    plt.xlabel("Manning's n")
    plt.ylabel("Probability (%)")
    #plt.xlim(0, 120000)
    #plt.ylim(0, 7)
    #plt.title("Comparison of Sample, Estimated, and Ground Truth LP3 PDF (Fixed)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(plot_save_filename, dpi=300, bbox_inches='tight', pad_inches=0)

    #plt.show()



def plot_histograms_all_parameters(discharges, max_discharges, wse_samples, manning_n):
    """
    Plot histograms of the samples for all parameters: discharges, max_discharges, wse_samples, and manning_n.
    
    Parameters:
        samples (dict): Dictionary with parameter names as keys and sample arrays as values.
    """
    keys = ["discharges", "max_discharges", "wse_samples", "manning_n"]
    n_params = 4 #len(keys)

    samples = {
        "discharges": discharges,
        "max_discharges": max_discharges,
        "wse_samples": wse_samples,
        "manning_n": manning_n
    }
    
    # Create subplots for each parameter
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # If there's only one parameter, axs might not be an array
    if n_params == 1:
        axs = [axs]
        
    #make 2x2 subplots
    for i, key in enumerate(keys):
        i_row = i // 2
        i_col = i % 2
        axs[i_row, i_col].hist(samples[key], bins=30, alpha=0.7, edgecolor='black')
        axs[i_row, i_col].set_title(f"Histogram of {key}", fontsize=14)
        axs[i_row, i_col].set_xlabel(key, fontsize=12)
        axs[i_row, i_col].set_ylabel("Frequency", fontsize=12)

        #compute statistics
        if key == "discharges" or key == "max_discharges":
            est_min, est_max, est_mean, est_std, est_skew = compute_sample_statistics(np.log10(samples[key]))

            # Prepare stats text using the param_specs
            stats_text = (f"log10 min: {est_min:.3f}\n"
                      f"log10 max: {est_max:.3f}\n"
                      f"log10 mean: {est_mean:.3f}\n"
                      f"log10 std: {est_std:.3f}\n"
                      f"log10 skew: {est_skew:.3f}")
        else:
            est_min, est_max, est_mean, est_std, est_skew = compute_sample_statistics(samples[key])

            # Prepare stats text using the param_specs
            stats_text = (f"min: {est_min:.3f}\n"
                      f"max: {est_max:.3f}\n"
                      f"mean: {est_mean:.3f}\n"
                      f"std: {est_std:.3f}\n"
                      f"skew: {est_skew:.3f}")

        # Add text annotation in the top right corner of the plot
        axs[i_row, i_col].text(0.95, 0.95, stats_text, transform=axs[i_row, i_col].transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))    
        
        # Set font size of axis labels, title, and tick labels
        axs[i_row, i_col].tick_params(axis='both', which='major', labelsize=12)
        axs[i_row, i_col].tick_params(axis='both', which='minor', labelsize=10)


    plt.tight_layout()
    plt.savefig("histograms_of_samples_all_parameters.png", dpi=300, bbox_inches='tight', pad_inches=0)
    #plt.show()

def plot_rating_curve_and_wse_histogram(rating_curve_file, wse_samples, q_discharges, plot_save_filename):
    """
    Create a two-panel plot showing the rating curve and WSE histogram.
    
    Parameters:
    -----------
    rating_curve_file : str
        Path to the rating curve file
    wse_samples : numpy.ndarray
        Array of interpolated WSE values
    plot_save_filename : str, optional
        If provided, save the plot to this file
    """
    # Read rating curve data
    rating_data = np.loadtxt(rating_curve_file, skiprows=1)
    Q = rating_data[:, 0]
    WSE = rating_data[:, 1]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Rating Curve
    ax1.plot(Q, WSE, 'b-', linewidth=2, label='Rating Curve')
    ax1.scatter(Q, WSE, color='blue', s=30, alpha=0.5)
    ax1.scatter(q_discharges, wse_samples, color='red', s=30, alpha=0.5, label='MC samples')
    ax1.set_xlabel('Discharge (cms)')
    ax1.set_ylabel('Water Surface Elevation (m)')
    ax1.set_title('Rating Curve')
    ax1.grid(True, alpha=0.3)
    
    # Add min/max lines
    ax1.axhline(y=WSE[0], color='r', linestyle='--', alpha=0.5, label='Min WSE')
    ax1.axhline(y=WSE[-1], color='g', linestyle='--', alpha=0.5, label='Max WSE')
    ax1.legend()
    
    # Plot 2: WSE Histogram
    hist_counts, bin_edges, _ = ax2.hist(wse_samples, bins=100, density=True, 
                                       alpha=0.7, color='blue', label='WSE Samples')
    ax2.set_xlabel('Water Surface Elevation (m)')
    ax2.set_ylabel('Density')
    ax2.set_title('WSE Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Add vertical lines for min/max WSE
    ax2.axvline(x=WSE[0], color='r', linestyle='--', alpha=0.5, label='Min WSE')
    ax2.axvline(x=WSE[-1], color='g', linestyle='--', alpha=0.5, label='Max WSE')
    
    # Add statistics to the histogram
    mean_wse = np.mean(wse_samples)
    std_wse = np.std(wse_samples)
    ax2.axvline(x=mean_wse, color='g', linestyle='-', alpha=0.8, label=f'Mean: {mean_wse:.2f}')
    
    # Add text box with statistics
    stats_text = f'Mean: {mean_wse:.2f}\nStd: {std_wse:.2f}\nMin: {np.min(wse_samples):.2f}\nMax: {np.max(wse_samples):.2f}'
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.legend()

    #add subfigure label (a) and (b)
    ax1.text(-0.06, 1.02, '(a)', transform=ax1.transAxes, fontsize=16, ha='center', va='center')
    ax2.text(-0.06, 1.02, '(b)', transform=ax2.transAxes, fontsize=16, ha='center', va='center')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if filename provided
    plt.savefig(plot_save_filename, dpi=300, bbox_inches='tight')

    print(f"Plot saved as {plot_save_filename}")

if __name__ == "__main__":

    # Load configuration
    with open("simulations_config.json", "r") as f:
        config = json.load(f)
    
    # Generate samples
    print("Generating the parameter samples ...")
    
    #note: the returned samples dictionary contains the following keys: discharges, wse_samples, and manning_n. The shape of discharges is (n_samples), the shape of wse_samples is (n_samples), and the shape of manning_n is (n_samples).
    samples = generate_parameter_samples(config, "parameter_samples")
    
    # Print sample statistics
    for param_name, param_samples in samples.items():
        print(f"\nParameter: {param_name}")
        if param_name == "discharges" or param_name == "max_discharges" or param_name == "wse_samples" or param_name == "manning_n":
            print(f"Mean: {np.mean(param_samples):.3f}")
            print(f"Min: {np.min(param_samples):.3f}")
            print(f"Max: {np.max(param_samples):.3f}")
    
    # Now you have samples for each parameter:
    manning_n_samples = samples["manning_n"]
    Q_samples = samples["discharges"]
    WSE_samples = samples["wse_samples"]    

    #get the parameter specs from the samples dictionary
    Q_config = samples["Q_config"]
    Manning_n_config = samples["Manning_n_config"]
    WSE_config = samples["WSE_config"]

    # For example, print the first 5 samples of each for inspection
    print("Some Manning's n samples:", manning_n_samples[:5])
    print("Some Q samples:", Q_samples[:5])
    print("Some WSE samples:", WSE_samples[:5])

    #save the sampled parameter values for record
    date_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")    #take the current date and time
    fileName_parameters = "sampledParameters_"+date_time+".dat"
    save_samples_to_file(fileName_parameters, samples)

    #plot the histograms and pairwise scatter matrix of the samples for visual inspection
    print("Plotting the samples ...")
    plot_discharge_sample_pdf(Q_samples, Q_config, "discharge_sample_pdf.png")
        
    print("Plotting the rating curve and WSE histogram")
    plot_save_filename = "rating_curve_and_exit_wse_histogram.png"
    plot_rating_curve_and_wse_histogram(WSE_config['file_name'], WSE_samples.flatten(), Q_samples.flatten(), plot_save_filename)
    

    print("All done!")



