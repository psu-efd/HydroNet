# This script is used to compare the results of the PI-DeepONet and the DeepONet without the PI-DeepONet

import os
import json
import matplotlib.pyplot as plt

def compare_w_wo_PI_plot_difference_metrics_against_parameter_distance():
    """
    Plot the difference metrics against the parameter distance from the training data.

    The distance is in file wasserstein_distance_results.json.
    The difference metrics are in file diff_lists.json.
    """

    # Get the application directory
    application_dir_wo_PI = '../SacramentoRiver_steady_DeepONet/application'
    application_dir_w_PI = '../SacramentoRiver_steady_PI_DeepONet_2/application'

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

    # Plot (scatter) the difference lists (h, u, v and velocity magnitude in four subplots; the top three subplots share the same x axis as the bottom subplot)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16), sharex=True)

    scatter_size = 40
    
    # Plot water depth MSE
    ax1.scatter(distance_list_wo_PI, diff_h_normalized_mse_list_wo_PI, color='blue', marker='o', s=scatter_size, label='Water Depth (without physics constraint)')
    ax1.scatter(distance_list_w_PI, diff_h_normalized_mse_list_w_PI, facecolors='none', edgecolors='blue', marker='o', s=scatter_size, label='Water Depth (with physics constraint)')
    #ax1.set_xlabel('Wasserstein Distance', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Normalized MSE', fontsize=18)
    #ax1.set_title('Water Depth Prediction Error (MSE)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=16)
    #ax1.set_xlim(0, max(distance_list))
    ax1.tick_params(axis='both', labelsize=16)

    # Plot x velocity MSE
    ax2.scatter(distance_list_wo_PI, diff_u_normalized_mse_list_wo_PI, color='green', marker='o', s=scatter_size, label='X Velocity (without physics constraint)')
    ax2.scatter(distance_list_w_PI, diff_u_normalized_mse_list_w_PI, facecolors='none', edgecolors='green', marker='o', s=scatter_size, label='X Velocity (with physics constraint)')
    #ax2.set_xlabel('Wasserstein Distance', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Normalized MSE', fontsize=18)
    #ax2.set_title('X Velocity Prediction Error (MSE)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=16)
    #ax2.set_xlim(0, max(distance_list))
    ax2.tick_params(axis='both', labelsize=16)

    # Plot y velocity MSE
    ax3.scatter(distance_list_wo_PI, diff_v_normalized_mse_list_wo_PI, color='red', marker='o', s=scatter_size, label='Y Velocity (without physics constraint)')
    ax3.scatter(distance_list_w_PI, diff_v_normalized_mse_list_w_PI, facecolors='none', edgecolors='red', marker='o', s=scatter_size, label='Y Velocity (with physics constraint)')
    #ax3.set_xlabel('Wasserstein Distance', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Normalized MSE', fontsize=18)
    #ax3.set_title('Y Velocity Prediction Error (MSE)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=16)
    #ax3.set_xlim(0, max(distance_list))
    ax3.tick_params(axis='both', labelsize=16)
    
    # Plot velocity magnitude MSE
    ax4.scatter(distance_list_wo_PI, diff_velocity_magnitude_normalized_mse_list_wo_PI, color='purple', marker='o', s=scatter_size, label='Velocity Magnitude (without physics constraint)')
    ax4.scatter(distance_list_w_PI, diff_velocity_magnitude_normalized_mse_list_w_PI, facecolors='none', edgecolors='purple', marker='o', s=scatter_size, label='Velocity Magnitude (with physics constraint)')
    ax4.set_xlabel('Wasserstein Distance', fontsize=18)
    ax4.set_ylabel('Normalized MSE', fontsize=18)
    #ax4.set_title('Velocity Magnitude Prediction Error (MSE)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(fontsize=16)
    ax4.set_xlim(0, max(distance_list))
    ax4.tick_params(axis='both', labelsize=16)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join('comparison_w_wo_PI_difference_metrics_against_parameter_distance.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Show the plot
    #plt.show()

if __name__ == "__main__":
    compare_w_wo_PI_plot_difference_metrics_against_parameter_distance()

    print("Done")