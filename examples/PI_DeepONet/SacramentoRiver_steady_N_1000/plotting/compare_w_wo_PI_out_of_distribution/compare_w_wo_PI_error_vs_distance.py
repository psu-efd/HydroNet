# This script is used to compare the results of the PI-DeepONet and the DeepONet without the PI-DeepONet

import os
import json
import matplotlib.pyplot as plt

import numpy as np

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

def compute_pi_deeponet_metrics_difference(variable_name, RMSE_1, RMSE_2, distance,
                                n_bins=6,
                                failure_threshold=0.5,
                                model_1_name="model_1",
                                model_2_name="model_2",
                                save_file_name="metrics_difference_quantification.json"):
    """
    Compute quantitative metrics comparing two models
    given their MSE and a distance measure. This is used to quantify the functional relationship between the error and the distance. 

    Parameters
    ----------
    RMSE_1 : list or array
        Errors for model 1, for example without physics constraint.
    RMSE_2 : list or array
        Errors for model 2, for example with physics constraint.
    distance : list or array
        For example, Wasserstein distances, same length and order as MSE lists.
    n_bins : int, optional
        Number of bins along the distance axis for binned statistics.
    failure_threshold : float, optional
        Threshold on MSE used to define breakdown distance.
    model_1_name : str, optional
        Name of model 1.
    model_2_name : str, optional
        Name of model 2.
    save_file_name : str, optional
        Name of the file to save the metrics.

    Returns
    -------
    metrics : dict
        Dictionary containing:
          - slopes and intercepts of linear fits
          - slope difference
          - Pearson correlations
          - error ratio statistics
          - binned mean errors and improvements
          - breakdown distances
    """
    # convert to numpy arrays
    RMSE_1 = np.asarray(RMSE_1, dtype=float)
    RMSE_2 = np.asarray(RMSE_2, dtype=float)
    distance = np.asarray(distance, dtype=float)

    assert RMSE_1.shape == RMSE_2.shape == distance.shape, \
        "All inputs must have the same shape"

    metrics = {
        "variable_name": variable_name,
        "model_1_name": model_1_name,
        "model_2_name": model_2_name,
    }

    # 1. Linear regression: RMSE ~ a * W + b
    a1, b1 = np.polyfit(distance, RMSE_1, 1)
    a2, b2 = np.polyfit(distance, RMSE_2, 1)

    metrics["slope_model_1"] = a1
    metrics["intercept_model_1"] = b1
    metrics["slope_model_2"] = a2
    metrics["intercept_model_2"] = b2
    metrics["slope_difference"] = a1 - a2   # positive means model 2 has flatter slope

    # 2. Pearson correlation between distance and error
    metrics["pearson_model_1"] = np.corrcoef(distance, RMSE_1)[0, 1]
    metrics["pearson_model_2"] = np.corrcoef(distance, RMSE_2)[0, 1]

    # 3. Error ratios r = MSE_2 / MSE_1
    ratio = RMSE_2 / RMSE_1
    metrics["ratio_mean"] = float(np.mean(ratio))
    metrics["ratio_median"] = float(np.median(ratio))
    metrics["ratio_std"] = float(np.std(ratio, ddof=1))
    metrics["ratio_frac_less_than_one"] = float(np.mean(ratio < 1.0))

    # 4. Binned statistics along distance
    d_min, d_max = float(distance.min()), float(distance.max())
    bin_edges = np.linspace(d_min, d_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    mean1_per_bin = []
    mean2_per_bin = []
    improvement_per_bin = []
    counts_per_bin = []

    for i in range(n_bins):
        left, right = bin_edges[i], bin_edges[i + 1]
        mask = (distance >= left) & (distance < right) if i < n_bins - 1 else \
               (distance >= left) & (distance <= right)
        if np.any(mask):
            m1 = float(np.mean(RMSE_1[mask]))
            m2 = float(np.mean(RMSE_2[mask]))
            mean1_per_bin.append(m1)
            mean2_per_bin.append(m2)
            improvement_per_bin.append(m1 - m2)  # positive means model 2 better
            counts_per_bin.append(int(mask.sum()))
        else:
            mean1_per_bin.append(np.nan)
            mean2_per_bin.append(np.nan)
            improvement_per_bin.append(np.nan)
            counts_per_bin.append(0)

    metrics["bin_centers"] = bin_centers.tolist()
    metrics["bin_mean_RMSE_model_1"] = mean1_per_bin
    metrics["bin_mean_RMSE_model_2"] = mean2_per_bin
    metrics["bin_improvement"] = improvement_per_bin
    metrics["bin_counts"] = counts_per_bin

    # Overall average improvement, weighted by counts
    valid = ~np.isnan(improvement_per_bin)
    if np.any(valid):
        weights = np.array(counts_per_bin, dtype=float)[valid]
        improvements = np.array(improvement_per_bin, dtype=float)[valid]
        metrics["weighted_mean_improvement"] = float(
            np.sum(weights * improvements) / np.sum(weights)
        )
    else:
        metrics["weighted_mean_improvement"] = np.nan

    # 5. Breakdown distance at given threshold
    def breakdown_distance(rmse, dist, thr):
        mask = rmse > thr
        if not np.any(mask):
            return None
        return float(dist[mask].min())

    metrics["breakdown_distance_model_1"] = breakdown_distance(RMSE_1, distance, failure_threshold)
    metrics["breakdown_distance_model_2"] = breakdown_distance(RMSE_2, distance, failure_threshold)

    # Save the metrics to a json file
    with open(save_file_name, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {save_file_name}")

    return metrics


def compare_w_wo_PI_plot_difference_metrics_against_parameter_distance(model_1_dir, model_2_dir, model_1_name, model_2_name, special_case_indices):
    """
    Plot the difference metrics against the parameter distance from the training data.

    The distance is in file wasserstein_distance_results.json.
    The difference metrics are in file diff_lists.json.

    Args:
        model_1_dir : str
            Directory of model 1, for example without physics constraint.
        model_2_dir : str
            Directory of model 2, for example with physics constraint.
        model_1_name : str
            Name of model 1, for example without physics constraint.
        model_2_name : str
            Name of model 2, for example with physics constraint.
        special_case_indices : list
            List of indices of the special cases to plot. Indices are already 0-based.
            For example, [70, 1, 73] for the minimum distance, average distance, and maximum distance.
    """

    # Get the application directory
    application_dir_wo_PI = model_1_dir
    application_dir_w_PI = model_2_dir

    # Load the difference lists from the json file
    with open(os.path.join(application_dir_wo_PI, "diff_lists.json"), "r") as f:
        diff_lists_wo_PI = json.load(f)

    with open(os.path.join(application_dir_w_PI, "diff_lists.json"), "r") as f:
        diff_lists_w_PI = json.load(f)

    # Extract the data (only the normalized difference lists)
    diff_h_normalized_rmse_list_wo_PI = diff_lists_wo_PI["diff_h_normalized_rmse_list"]
    diff_u_normalized_rmse_list_wo_PI = diff_lists_wo_PI["diff_u_normalized_rmse_list"]
    diff_v_normalized_rmse_list_wo_PI = diff_lists_wo_PI["diff_v_normalized_rmse_list"]
    diff_velocity_magnitude_normalized_rmse_list_wo_PI = diff_lists_wo_PI["diff_velocity_magnitude_normalized_rmse_list"]

    diff_h_normalized_rmse_list_w_PI = diff_lists_w_PI["diff_h_normalized_rmse_list"]
    diff_u_normalized_rmse_list_w_PI = diff_lists_w_PI["diff_u_normalized_rmse_list"]
    diff_v_normalized_rmse_list_w_PI = diff_lists_w_PI["diff_v_normalized_rmse_list"]
    diff_velocity_magnitude_normalized_rmse_list_w_PI = diff_lists_w_PI["diff_velocity_magnitude_normalized_rmse_list"]

    # Load the distance from the json file
    with open(os.path.join(application_dir_wo_PI, "wasserstein_distance_results.json"), "r") as f:
        wasserstein_distance_results = json.load(f)
    distance_list = wasserstein_distance_results["wasserstein_distance_per_case"]

    #diff value lists and distance list should have the same length
    if len(diff_h_normalized_rmse_list_wo_PI) != len(distance_list):
        raise ValueError("The length of diff_h_normalized_rmse_list_wo_PI and distance_list should be the same")
    if len(diff_velocity_magnitude_normalized_rmse_list_wo_PI) != len(distance_list):
        raise ValueError("The length of diff_velocity_magnitude_normalized_rmse_list_wo_PI and distance_list should be the same")

    # Extract the data for the special cases
    diff_h_normalized_rmse_list_wo_PI_special = [diff_h_normalized_rmse_list_wo_PI[i] for i in special_case_indices]
    diff_u_normalized_rmse_list_wo_PI_special = [diff_u_normalized_rmse_list_wo_PI[i] for i in special_case_indices]
    diff_v_normalized_rmse_list_wo_PI_special = [diff_v_normalized_rmse_list_wo_PI[i] for i in special_case_indices]
    diff_velocity_magnitude_normalized_rmse_list_wo_PI_special = [diff_velocity_magnitude_normalized_rmse_list_wo_PI[i] for i in special_case_indices]
    distance_list_wo_PI_special = [distance_list[i] for i in special_case_indices]

    diff_h_normalized_rmse_list_w_PI_special = [diff_h_normalized_rmse_list_w_PI[i] for i in special_case_indices]
    diff_u_normalized_rmse_list_w_PI_special = [diff_u_normalized_rmse_list_w_PI[i] for i in special_case_indices]
    diff_v_normalized_rmse_list_w_PI_special = [diff_v_normalized_rmse_list_w_PI[i] for i in special_case_indices]
    diff_velocity_magnitude_normalized_rmse_list_w_PI_special = [diff_velocity_magnitude_normalized_rmse_list_w_PI[i] for i in special_case_indices]
    distance_list_w_PI_special = [distance_list[i] for i in special_case_indices]

    print(f"diff_h_normalized_rmse_list_wo_PI_special: {diff_h_normalized_rmse_list_wo_PI_special}")
    print(f"diff_u_normalized_rmse_list_wo_PI_special: {diff_u_normalized_rmse_list_wo_PI_special}")
    print(f"diff_v_normalized_rmse_list_wo_PI_special: {diff_v_normalized_rmse_list_wo_PI_special}")
    print(f"diff_velocity_magnitude_normalized_rmse_list_wo_PI_special: {diff_velocity_magnitude_normalized_rmse_list_wo_PI_special}")
    print(f"distance_list_wo_PI_special: {distance_list_wo_PI_special}")
    print(f"diff_h_normalized_rmse_list_w_PI_special: {diff_h_normalized_rmse_list_w_PI_special}")
    print(f"diff_u_normalized_rmse_list_w_PI_special: {diff_u_normalized_rmse_list_w_PI_special}")
    print(f"diff_v_normalized_rmse_list_w_PI_special: {diff_v_normalized_rmse_list_w_PI_special}")
    print(f"diff_velocity_magnitude_normalized_rmse_list_w_PI_special: {diff_velocity_magnitude_normalized_rmse_list_w_PI_special}")
    print(f"distance_list_w_PI_special: {distance_list_w_PI_special}")

    # Order the diff value lists and distance list by the distance
    diff_h_normalized_rmse_list_wo_PI = [x for _, x in sorted(zip(distance_list, diff_h_normalized_rmse_list_wo_PI))]
    diff_u_normalized_rmse_list_wo_PI = [x for _, x in sorted(zip(distance_list, diff_u_normalized_rmse_list_wo_PI))]
    diff_v_normalized_rmse_list_wo_PI = [x for _, x in sorted(zip(distance_list, diff_v_normalized_rmse_list_wo_PI))]
    diff_velocity_magnitude_normalized_rmse_list_wo_PI = [x for _, x in sorted(zip(distance_list, diff_velocity_magnitude_normalized_rmse_list_wo_PI))]
    distance_list_wo_PI = sorted(distance_list)

    diff_h_normalized_rmse_list_w_PI = [x for _, x in sorted(zip(distance_list, diff_h_normalized_rmse_list_w_PI))]
    diff_u_normalized_rmse_list_w_PI = [x for _, x in sorted(zip(distance_list, diff_u_normalized_rmse_list_w_PI))]
    diff_v_normalized_rmse_list_w_PI = [x for _, x in sorted(zip(distance_list, diff_v_normalized_rmse_list_w_PI))]
    diff_velocity_magnitude_normalized_rmse_list_w_PI = [x for _, x in sorted(zip(distance_list, diff_velocity_magnitude_normalized_rmse_list_w_PI))]
    distance_list_w_PI = sorted(distance_list)

    # Compute the difference metrics
    metrics_h = compute_pi_deeponet_metrics_difference("h", diff_h_normalized_rmse_list_wo_PI, diff_h_normalized_rmse_list_w_PI, distance_list_wo_PI, model_1_name=model_1_name, model_2_name=model_2_name, save_file_name=model_1_name + "_" + model_2_name + "_performance_h.json")


    metrics_u = compute_pi_deeponet_metrics_difference("u", diff_u_normalized_rmse_list_wo_PI, diff_u_normalized_rmse_list_w_PI, distance_list_wo_PI, model_1_name=model_1_name, model_2_name=model_2_name, save_file_name=model_1_name + "_" + model_2_name + "_performance_u.json")

    metrics_v = compute_pi_deeponet_metrics_difference("v", diff_v_normalized_rmse_list_wo_PI, diff_v_normalized_rmse_list_w_PI, distance_list_wo_PI, model_1_name=model_1_name, model_2_name=model_2_name, save_file_name=model_1_name + "_" + model_2_name + "_performance_v.json")

    metrics_velocity_magnitude = compute_pi_deeponet_metrics_difference("velocity_magnitude", diff_velocity_magnitude_normalized_rmse_list_wo_PI, diff_velocity_magnitude_normalized_rmse_list_w_PI, distance_list_wo_PI, model_1_name=model_1_name, model_2_name=model_2_name, save_file_name=model_1_name + "_" + model_2_name + "_performance_Umag.json")

    # Plot (scatter) the difference lists (h, u, v and velocity magnitude in four subplots; the top three subplots share the same x axis as the bottom subplot)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
    ax1, ax2, ax3, ax4 = axes.flatten()

    scatter_size = 40
    special_case_scatter_size = 5*scatter_size
    
    # Define colors for special cases (one color per point)
    n_special = 3  #len(special_case_indices); we have 3 special cases: average distance, minimum distance, and maximum distance    
    special_colors = ['red', 'green', 'blue']
    
    # Plot water depth MSE
    ax1.scatter(distance_list_wo_PI, diff_h_normalized_rmse_list_wo_PI, color='black', marker='o', s=scatter_size, label='SWE-DeepONet')
    ax1.scatter(distance_list_w_PI, diff_h_normalized_rmse_list_w_PI, facecolors='none', edgecolors='black', marker='o', s=scatter_size, label='PI-SWE-DeepONet')

    # Mark the special cases with different colored dots
    ax1.scatter(distance_list_wo_PI_special, diff_h_normalized_rmse_list_wo_PI_special, facecolors='none', edgecolors=special_colors, marker='o', s=special_case_scatter_size)
    ax1.scatter(distance_list_w_PI_special, diff_h_normalized_rmse_list_w_PI_special, facecolors='none', edgecolors=special_colors, marker='o', s=special_case_scatter_size)

    ax1.set_xlabel('Wasserstein Distance', fontsize=24)
    ax1.set_ylabel('Normalized RMSE for $h$', fontsize=24)
    ax1.set_title('Water depth $h$', fontsize=24, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=18)
    #ax1.set_xlim(0, max(distance_list))
    ax1.tick_params(axis='both', labelsize=22)
    #ax1.text(-0.15, 1.1, '(a)', fontsize=18, fontweight='bold', transform=ax1.transAxes, horizontalalignment='left', verticalalignment='top')

    # Plot x velocity MSE
    ax2.scatter(distance_list_wo_PI, diff_u_normalized_rmse_list_wo_PI, color='black', marker='o', s=scatter_size, label='SWE-DeepONet')
    ax2.scatter(distance_list_w_PI, diff_u_normalized_rmse_list_w_PI, facecolors='none', edgecolors='black', marker='o', s=scatter_size, label='PI-SWE-DeepONet')

    # Mark the special cases with dots with different colors (each scatter has a different color)
    ax2.scatter(distance_list_wo_PI_special, diff_u_normalized_rmse_list_wo_PI_special, facecolors='none', edgecolors=special_colors, marker='o', s=special_case_scatter_size)
    ax2.scatter(distance_list_w_PI_special, diff_u_normalized_rmse_list_w_PI_special, facecolors='none', edgecolors=special_colors, marker='o', s=special_case_scatter_size)

    ax2.set_xlabel('Wasserstein Distance', fontsize=24)
    ax2.set_ylabel('Normalized RMSE for $u$', fontsize=24)
    ax2.set_title('Velocity component $u$', fontsize=24, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=18)
    #ax2.set_xlim(0, max(distance_list))
    ax2.tick_params(axis='both', labelsize=22)
    #ax2.text(-0.15, 1.1, '(b)', fontsize=18, fontweight='bold', transform=ax2.transAxes, horizontalalignment='left', verticalalignment='top')

    # Plot y velocity MSE
    ax3.scatter(distance_list_wo_PI, diff_v_normalized_rmse_list_wo_PI, color='black', marker='o', s=scatter_size, label='SWE-DeepONet')
    ax3.scatter(distance_list_w_PI, diff_v_normalized_rmse_list_w_PI, facecolors='none', edgecolors='black', marker='o', s=scatter_size, label='PI-SWE-DeepONet')

    # Mark the special cases with different colored dots
    ax3.scatter(distance_list_wo_PI_special, diff_v_normalized_rmse_list_wo_PI_special, facecolors='none', edgecolors=special_colors, marker='o', s=special_case_scatter_size)
    ax3.scatter(distance_list_w_PI_special, diff_v_normalized_rmse_list_w_PI_special, facecolors='none', edgecolors=special_colors, marker='o', s=special_case_scatter_size)

    ax3.set_xlabel('Wasserstein Distance', fontsize=24)
    ax3.set_ylabel('Normalized RMSE for $v$', fontsize=24)
    ax3.set_title('Velocity component $v$', fontsize=24, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=18)
    #ax3.set_xlim(0, max(distance_list))
    ax3.tick_params(axis='both', labelsize=22)
    #ax3.text(-0.15, 1.1, '(c)', fontsize=18, fontweight='bold', transform=ax3.transAxes, horizontalalignment='left', verticalalignment='top')
    
    # Plot velocity magnitude MSE
    ax4.scatter(distance_list_wo_PI, diff_velocity_magnitude_normalized_rmse_list_wo_PI, color='black', marker='o', s=scatter_size, label='SWE-DeepONet')
    ax4.scatter(distance_list_w_PI, diff_velocity_magnitude_normalized_rmse_list_w_PI, facecolors='none', edgecolors='black', marker='o', s=scatter_size, label='PI-SWE-DeepONet')

    # Mark the special cases with different colored dots
    ax4.scatter(distance_list_wo_PI_special, diff_velocity_magnitude_normalized_rmse_list_wo_PI_special, facecolors='none', edgecolors=special_colors, marker='o', s=special_case_scatter_size)
    ax4.scatter(distance_list_w_PI_special, diff_velocity_magnitude_normalized_rmse_list_w_PI_special, facecolors='none', edgecolors=special_colors, marker='o', s=special_case_scatter_size)

    ax4.set_xlabel('Wasserstein Distance', fontsize=24)
    ax4.set_ylabel('Normalized RMSE for $|\\mathbf{u}|$', fontsize=24)
    ax4.set_title('Velocity magnitude $|\\mathbf{u}|$', fontsize=24, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(fontsize=18)
    #ax4.set_xlim(0, max(distance_list))
    ax4.tick_params(axis='both', labelsize=22)
    #ax4.text(-0.15, 1.1, '(d)', fontsize=18, fontweight='bold', transform=ax4.transAxes, horizontalalignment='left', verticalalignment='top')
    
    # Adjust layout to prevent overlap and increase space between columns
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    
    # Save the figure
    output_file = os.path.join('metric_distance_'+model_1_name+'_vs_'+model_2_name+'.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Show the plot
    #plt.show()

    #plot the improvement ratio (RMSE_2 / RMSE_1) against the distance
    #For h, u, v and velocity magnitude, plot the improvement ratio against the distance in a 2x2 subplot.
    improvement_ratio_h = [diff_h_normalized_rmse_list_w_PI[i] / diff_h_normalized_rmse_list_wo_PI[i] for i in range(len(diff_h_normalized_rmse_list_wo_PI))]
    improvement_ratio_u = [diff_u_normalized_rmse_list_w_PI[i] / diff_u_normalized_rmse_list_wo_PI[i] for i in range(len(diff_u_normalized_rmse_list_wo_PI))]
    improvement_ratio_v = [diff_v_normalized_rmse_list_w_PI[i] / diff_v_normalized_rmse_list_wo_PI[i] for i in range(len(diff_v_normalized_rmse_list_wo_PI))]
    improvement_ratio_velocity_magnitude = [diff_velocity_magnitude_normalized_rmse_list_w_PI[i] / diff_velocity_magnitude_normalized_rmse_list_wo_PI[i] for i in range(len(diff_velocity_magnitude_normalized_rmse_list_wo_PI))]

    #divide the error ratio into two groups: less than 1 and greater than 1; 
    # also record the error ratio for each group
    improvement_ratio_h_less_than_one = [improvement_ratio_h[i] for i in range(len(improvement_ratio_h)) if improvement_ratio_h[i] < 1]
    improvement_ratio_h_greater_than_one = [improvement_ratio_h[i] for i in range(len(improvement_ratio_h)) if improvement_ratio_h[i] > 1]
    improvement_ratio_u_less_than_one = [improvement_ratio_u[i] for i in range(len(improvement_ratio_u)) if improvement_ratio_u[i] < 1]
    improvement_ratio_u_greater_than_one = [improvement_ratio_u[i] for i in range(len(improvement_ratio_u)) if improvement_ratio_u[i] > 1]
    improvement_ratio_v_less_than_one = [improvement_ratio_v[i] for i in range(len(improvement_ratio_v)) if improvement_ratio_v[i] < 1]
    improvement_ratio_v_greater_than_one = [improvement_ratio_v[i] for i in range(len(improvement_ratio_v)) if improvement_ratio_v[i] > 1]
    improvement_ratio_velocity_magnitude_less_than_one = [improvement_ratio_velocity_magnitude[i] for i in range(len(improvement_ratio_velocity_magnitude)) if improvement_ratio_velocity_magnitude[i] < 1]
    improvement_ratio_velocity_magnitude_greater_than_one = [improvement_ratio_velocity_magnitude[i] for i in range(len(improvement_ratio_velocity_magnitude)) if improvement_ratio_velocity_magnitude[i] > 1]

    #record the distance for each group
    distance_h_less_than_one = [distance_list_wo_PI[i] for i in range(len(distance_list_wo_PI)) if improvement_ratio_h[i] < 1]
    distance_h_greater_than_one = [distance_list_wo_PI[i] for i in range(len(distance_list_wo_PI)) if improvement_ratio_h[i] > 1]
    distance_u_less_than_one = [distance_list_wo_PI[i] for i in range(len(distance_list_wo_PI)) if improvement_ratio_u[i] < 1]
    distance_u_greater_than_one = [distance_list_wo_PI[i] for i in range(len(distance_list_wo_PI)) if improvement_ratio_u[i] > 1]
    distance_v_less_than_one = [distance_list_wo_PI[i] for i in range(len(distance_list_wo_PI)) if improvement_ratio_v[i] < 1]
    distance_v_greater_than_one = [distance_list_wo_PI[i] for i in range(len(distance_list_wo_PI)) if improvement_ratio_v[i] > 1]
    distance_velocity_magnitude_less_than_one = [distance_list_wo_PI[i] for i in range(len(distance_list_wo_PI)) if improvement_ratio_velocity_magnitude[i] < 1]
    distance_velocity_magnitude_greater_than_one = [distance_list_wo_PI[i] for i in range(len(distance_list_wo_PI)) if improvement_ratio_velocity_magnitude[i] > 1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # plot the error ratio for h
    ax1.scatter(distance_h_less_than_one, improvement_ratio_h_less_than_one, facecolors='none', edgecolors='blue', marker='o', s=scatter_size, label='$r < 1$')    
    ax1.scatter(distance_h_greater_than_one, improvement_ratio_h_greater_than_one, color='red', marker='o', s=scatter_size, label='$r > 1$')    
    #set y axis range 
    #ax1.set_ylim(0.5, 1.7)
    #draw a horizontal line at y = 1
    ax1.axhline(y=1, color='black', linestyle='--', linewidth=2)
    ax1.set_xlabel('Wasserstein Distance', fontsize=24)
    ax1.set_ylabel('Error Ratio for $h$', fontsize=24)
    ax1.set_title('Water depth $h$', fontsize=24, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=18, loc='upper right')
    ax1.tick_params(axis='both', labelsize=22)

    # plot the error ratio for u
    ax2.scatter(distance_u_less_than_one, improvement_ratio_u_less_than_one, facecolors='none', edgecolors='blue', marker='o', s=scatter_size, label='$r < 1$')    
    ax2.scatter(distance_u_greater_than_one, improvement_ratio_u_greater_than_one, facecolors='red', edgecolors='red', marker='o', s=scatter_size, label='$r > 1$')    
    #ax2.set_ylim(0.7, 1.1)
    ax2.axhline(y=1, color='black', linestyle='--', linewidth=2)
    ax2.set_xlabel('Wasserstein Distance', fontsize=24)
    ax2.set_ylabel('Error Ratio for $u$', fontsize=24)
    ax2.set_title('Velocity component $u$', fontsize=24, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=18, loc='upper right')
    ax2.tick_params(axis='both', labelsize=22)

    # plot the error ratio for v
    ax3.scatter(distance_v_less_than_one, improvement_ratio_v_less_than_one, facecolors='none', edgecolors='blue', marker='o', s=scatter_size, label='$r < 1$')    
    ax3.scatter(distance_v_greater_than_one, improvement_ratio_v_greater_than_one, facecolors='red', edgecolors='red', marker='o', s=scatter_size, label='$r > 1$')    
    #ax3.set_ylim(0.7, 1.2)
    ax3.axhline(y=1, color='black', linestyle='--', linewidth=2)
    ax3.set_xlabel('Wasserstein Distance', fontsize=24)
    ax3.set_ylabel('Error Ratio for $v$', fontsize=24)
    ax3.set_title('Velocity component $v$', fontsize=24, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=18, loc='upper right')
    ax3.tick_params(axis='both', labelsize=22)

    # plot the error ratio for velocity magnitude
    ax4.scatter(distance_velocity_magnitude_less_than_one, improvement_ratio_velocity_magnitude_less_than_one, facecolors='none', edgecolors='blue', marker='o', s=scatter_size, label='$r < 1$')    
    ax4.scatter(distance_velocity_magnitude_greater_than_one, improvement_ratio_velocity_magnitude_greater_than_one, facecolors='red', edgecolors='red', marker='o', s=scatter_size, label='$r > 1$')    
    #ax4.set_ylim(0.7, 1.1)
    ax4.axhline(y=1, color='black', linestyle='--', linewidth=2)
    ax4.set_xlabel('Wasserstein Distance', fontsize=24)
    ax4.set_ylabel('Error Ratio for $|\\mathbf{u}|$', fontsize=24)
    ax4.set_title('Velocity magnitude $|\\mathbf{u}|$', fontsize=24, fontweight='bold')

    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(fontsize=18, loc='upper right')
    ax4.tick_params(axis='both', labelsize=22)

    # Adjust layout to prevent overlap and increase space between columns
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    # Save the figure
    output_file = os.path.join('improvement_ratio_distance_'+model_1_name+'_vs_'+model_2_name+'.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")



if __name__ == "__main__":


    model_1_name = 'DeepONet'
    model_2_name = 'PI_DeepONet_3'
    model_1_dir = '../../'+model_1_name+'/application'
    model_2_dir = '../../'+model_2_name+'/application'

    print(f"model_1_dir: {model_1_dir}")
    print(f"model_2_dir: {model_2_dir}")

    # Special case indices: 1 (average distance), 70 (minimum distance), 73 (maximum distance)
    special_case_indices = [x - 1 for x in [70, 1, 73]]  # subtract 1 to get the indices in the array

    compare_w_wo_PI_plot_difference_metrics_against_parameter_distance(model_1_dir, model_2_dir, model_1_name, model_2_name, special_case_indices)

    print("Done")
