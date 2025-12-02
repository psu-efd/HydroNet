# This script is used to compare the results of the PI-DeepONet and the DeepONet without the PI-DeepONet

import os
import json
import matplotlib.pyplot as plt

import numpy as np

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

def compute_pi_deeponet_metrics_difference(variable_name, MSE_1, MSE_2, distance,
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
    MSE_1 : list or array
        Errors for model 1, for example without physics constraint.
    MSE_2 : list or array
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
    MSE_1 = np.asarray(MSE_1, dtype=float)
    MSE_2 = np.asarray(MSE_2, dtype=float)
    distance = np.asarray(distance, dtype=float)

    assert MSE_1.shape == MSE_2.shape == distance.shape, \
        "All inputs must have the same shape"

    metrics = {
        "variable_name": variable_name,
        "model_1_name": model_1_name,
        "model_2_name": model_2_name,
    }

    # 1. Linear regression: MSE ~ a * W + b
    a1, b1 = np.polyfit(distance, MSE_1, 1)
    a2, b2 = np.polyfit(distance, MSE_2, 1)

    metrics["slope_model_1"] = a1
    metrics["intercept_model_1"] = b1
    metrics["slope_model_2"] = a2
    metrics["intercept_model_2"] = b2
    metrics["slope_difference"] = a1 - a2   # positive means model 2 has flatter slope

    # 2. Pearson correlation between distance and error
    metrics["pearson_model_1"] = np.corrcoef(distance, MSE_1)[0, 1]
    metrics["pearson_model_2"] = np.corrcoef(distance, MSE_2)[0, 1]

    # 3. Error ratios r = MSE_2 / MSE_1
    ratio = MSE_2 / MSE_1
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
            m1 = float(np.mean(MSE_1[mask]))
            m2 = float(np.mean(MSE_2[mask]))
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
    metrics["bin_mean_MSE_model_1"] = mean1_per_bin
    metrics["bin_mean_MSE_model_2"] = mean2_per_bin
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
    def breakdown_distance(mse, dist, thr):
        mask = mse > thr
        if not np.any(mask):
            return None
        return float(dist[mask].min())

    metrics["breakdown_distance_model_1"] = breakdown_distance(MSE_1, distance, failure_threshold)
    metrics["breakdown_distance_model_2"] = breakdown_distance(MSE_2, distance, failure_threshold)

    # Save the metrics to a json file
    with open(save_file_name, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {save_file_name}")

    return metrics


def compare_w_wo_PI_plot_difference_metrics_against_parameter_distance(model_1_dir, model_2_dir, model_1_name, model_2_name):
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
    diff_h_normalized_mse_list_wo_PI = diff_lists_wo_PI["diff_h_normalized_mse_list"]
    diff_u_normalized_mse_list_wo_PI = diff_lists_wo_PI["diff_u_normalized_mse_list"]
    diff_v_normalized_mse_list_wo_PI = diff_lists_wo_PI["diff_v_normalized_mse_list"]
    diff_velocity_magnitude_normalized_mse_list_wo_PI = diff_lists_wo_PI["diff_velocity_magnitude_normalized_mse_list"]

    diff_h_normalized_mse_list_w_PI = diff_lists_w_PI["diff_h_normalized_mse_list"]
    diff_u_normalized_mse_list_w_PI = diff_lists_w_PI["diff_u_normalized_mse_list"]
    diff_v_normalized_mse_list_w_PI = diff_lists_w_PI["diff_v_normalized_mse_list"]
    diff_velocity_magnitude_normalized_mse_list_w_PI = diff_lists_w_PI["diff_velocity_magnitude_normalized_mse_list"]

    # Load the distance from the json file
    with open(os.path.join(application_dir_wo_PI, "wasserstein_distance_results.json"), "r") as f:
        wasserstein_distance_results = json.load(f)
    distance_list = wasserstein_distance_results["wasserstein_distance_per_case"]

    #diff value lists and distance list should have the same length
    if len(diff_h_normalized_mse_list_wo_PI) != len(distance_list):
        raise ValueError("The length of diff_h_normalized_mse_list_wo_PI and distance_list should be the same")
    if len(diff_velocity_magnitude_normalized_mse_list_wo_PI) != len(distance_list):
        raise ValueError("The length of diff_velocity_magnitude_normalized_mse_list_wo_PI and distance_list should be the same")

    # Order the diff value lists and distance list by the distance
    diff_h_normalized_mse_list_wo_PI = [x for _, x in sorted(zip(distance_list, diff_h_normalized_mse_list_wo_PI))]
    diff_u_normalized_mse_list_wo_PI = [x for _, x in sorted(zip(distance_list, diff_u_normalized_mse_list_wo_PI))]
    diff_v_normalized_mse_list_wo_PI = [x for _, x in sorted(zip(distance_list, diff_v_normalized_mse_list_wo_PI))]
    diff_velocity_magnitude_normalized_mse_list_wo_PI = [x for _, x in sorted(zip(distance_list, diff_velocity_magnitude_normalized_mse_list_wo_PI))]
    distance_list_wo_PI = sorted(distance_list)

    diff_h_normalized_mse_list_w_PI = [x for _, x in sorted(zip(distance_list, diff_h_normalized_mse_list_w_PI))]
    diff_u_normalized_mse_list_w_PI = [x for _, x in sorted(zip(distance_list, diff_u_normalized_mse_list_w_PI))]
    diff_v_normalized_mse_list_w_PI = [x for _, x in sorted(zip(distance_list, diff_v_normalized_mse_list_w_PI))]
    diff_velocity_magnitude_normalized_mse_list_w_PI = [x for _, x in sorted(zip(distance_list, diff_velocity_magnitude_normalized_mse_list_w_PI))]
    distance_list_w_PI = sorted(distance_list)

    # Compute the difference metrics
    metrics_h = compute_pi_deeponet_metrics_difference("h", diff_h_normalized_mse_list_wo_PI, diff_h_normalized_mse_list_w_PI, distance_list_wo_PI, model_1_name=model_1_name, model_2_name=model_2_name, save_file_name=model_1_name + "_" + model_2_name + "_performance_h.json")


    metrics_u = compute_pi_deeponet_metrics_difference("u", diff_u_normalized_mse_list_wo_PI, diff_u_normalized_mse_list_w_PI, distance_list_wo_PI, model_1_name=model_1_name, model_2_name=model_2_name, save_file_name=model_1_name + "_" + model_2_name + "_performance_u.json")

    metrics_v = compute_pi_deeponet_metrics_difference("v", diff_v_normalized_mse_list_wo_PI, diff_v_normalized_mse_list_w_PI, distance_list_wo_PI, model_1_name=model_1_name, model_2_name=model_2_name, save_file_name=model_1_name + "_" + model_2_name + "_performance_v.json")

    metrics_velocity_magnitude = compute_pi_deeponet_metrics_difference("velocity_magnitude", diff_velocity_magnitude_normalized_mse_list_wo_PI, diff_velocity_magnitude_normalized_mse_list_w_PI, distance_list_wo_PI, model_1_name=model_1_name, model_2_name=model_2_name, save_file_name=model_1_name + "_" + model_2_name + "_performance_Umag.json")

    # Plot (scatter) the difference lists (h, u, v and velocity magnitude in four subplots; the top three subplots share the same x axis as the bottom subplot)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
    ax1, ax2, ax3, ax4 = axes.flatten()

    scatter_size = 40
    
    # Plot water depth MSE
    ax1.scatter(distance_list_wo_PI, diff_h_normalized_mse_list_wo_PI, color='black', marker='o', s=scatter_size, label='without physics constraint')
    ax1.scatter(distance_list_w_PI, diff_h_normalized_mse_list_w_PI, facecolors='none', edgecolors='black', marker='o', s=scatter_size, label='with physics constraint')
    ax1.set_xlabel('Wasserstein Distance', fontsize=18)
    ax1.set_ylabel('Normalized MSE for $h$', fontsize=18)
    ax1.set_title('Water depth $h$', fontsize=18, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=14)
    #ax1.set_xlim(0, max(distance_list))
    ax1.tick_params(axis='both', labelsize=16)
    ax1.text(-0.15, 1.1, '(a)', fontsize=18, fontweight='bold', transform=ax1.transAxes, horizontalalignment='left', verticalalignment='top')

    # Plot x velocity MSE
    ax2.scatter(distance_list_wo_PI, diff_u_normalized_mse_list_wo_PI, color='black', marker='o', s=scatter_size, label='without physics constraint')
    ax2.scatter(distance_list_w_PI, diff_u_normalized_mse_list_w_PI, facecolors='none', edgecolors='black', marker='o', s=scatter_size, label='with physics constraint')
    ax2.set_xlabel('Wasserstein Distance', fontsize=18)
    ax2.set_ylabel('Normalized MSE for $u$', fontsize=18)
    ax2.set_title('Velocity component $u$', fontsize=18, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=14)
    #ax2.set_xlim(0, max(distance_list))
    ax2.tick_params(axis='both', labelsize=16)
    ax2.text(-0.15, 1.1, '(b)', fontsize=18, fontweight='bold', transform=ax2.transAxes, horizontalalignment='left', verticalalignment='top')

    # Plot y velocity MSE
    ax3.scatter(distance_list_wo_PI, diff_v_normalized_mse_list_wo_PI, color='black', marker='o', s=scatter_size, label='without physics constraint')
    ax3.scatter(distance_list_w_PI, diff_v_normalized_mse_list_w_PI, facecolors='none', edgecolors='black', marker='o', s=scatter_size, label='with physics constraint')
    ax3.set_xlabel('Wasserstein Distance', fontsize=18)
    ax3.set_ylabel('Normalized MSE for $v$', fontsize=18)
    ax3.set_title('Velocity component $v$', fontsize=18, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=14)
    #ax3.set_xlim(0, max(distance_list))
    ax3.tick_params(axis='both', labelsize=16)
    ax3.text(-0.15, 1.1, '(c)', fontsize=18, fontweight='bold', transform=ax3.transAxes, horizontalalignment='left', verticalalignment='top')
    
    # Plot velocity magnitude MSE
    ax4.scatter(distance_list_wo_PI, diff_velocity_magnitude_normalized_mse_list_wo_PI, color='black', marker='o', s=scatter_size, label='without physics constraint')
    ax4.scatter(distance_list_w_PI, diff_velocity_magnitude_normalized_mse_list_w_PI, facecolors='none', edgecolors='black', marker='o', s=scatter_size, label='with physics constraint')
    ax4.set_xlabel('Wasserstein Distance', fontsize=18)
    ax4.set_ylabel('Normalized MSE for $|\\mathbf{u}|$', fontsize=18)
    ax4.set_title('Velocity magnitude $|\\mathbf{u}|$', fontsize=18, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(fontsize=14)
    ax4.set_xlim(0, max(distance_list))
    ax4.tick_params(axis='both', labelsize=16)
    ax4.text(-0.15, 1.1, '(d)', fontsize=18, fontweight='bold', transform=ax4.transAxes, horizontalalignment='left', verticalalignment='top')
    
    # Adjust layout to prevent overlap and increase space between columns
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    
    # Save the figure
    output_file = os.path.join('comparison_w_wo_PI_difference_metrics_against_parameter_distance.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Show the plot
    #plt.show()

if __name__ == "__main__":

    model_1_dir = '../../SacramentoRiver_steady_DeepONet/application'
    model_2_dir = '../../SacramentoRiver_steady_PI_DeepONet/application'
    model_1_name = 'DeepONet'
    model_2_name = 'PI-DeepONet'
    compare_w_wo_PI_plot_difference_metrics_against_parameter_distance(model_1_dir, model_2_dir, model_1_name, model_2_name)

    print("Done")