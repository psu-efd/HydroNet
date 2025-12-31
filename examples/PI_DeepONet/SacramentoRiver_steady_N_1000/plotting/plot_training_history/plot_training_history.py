#This script plots the training history of the DeepONet and PI-DeepONet models.
#It has three subplots:
#1. Training and validation losses: SWE_DeepONet vs. PI_SWE_DeepONet
#2. For PI_SWE_DeepONet only, the physics-informed loss and the data loss are plotted.
#3. For PI_SWE_DeepONet only, the PDE components of the physics-informed loss are plotted.

import os
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

def plot_training_history(deeponet_training_history_file, pi_deeponet_training_history_file):
    """
    Plots the training history of the DeepONet and PI-DeepONet models.
    Args:
        deeponet_training_history_file: The file name of the training history file of the DeepONet model.
        pi_deeponet_training_history_file: The file name of the training history file of the PI-DeepONet model.
    Returns:
        None
    """
    # Load history files
    with open(deeponet_training_history_file, 'r') as f:
        deeponet_history = json.load(f)
    
    with open(pi_deeponet_training_history_file, 'r') as f:
        pi_deeponet_history = json.load(f)
    
    # Extract data
    # DeepONet
    deeponet_train_loss = np.array(deeponet_history['training_loss_history'])
    deeponet_val_loss = np.array(deeponet_history['validation_loss_history'])
    deeponet_epochs = np.arange(1, len(deeponet_train_loss) + 1)
    
    # PI-DeepONet
    pi_deeponet_train_loss = np.array(pi_deeponet_history['training_loss_history'])
    pi_deeponet_val_loss = np.array(pi_deeponet_history['validation_loss_history'])
    pi_deeponet_epochs = np.arange(1, len(pi_deeponet_train_loss) + 1)
    
    # PI-DeepONet component losses for training
    pi_component_loss_training = pi_deeponet_history['training_component_loss_history']
    pi_data_loss_training = np.array(pi_component_loss_training['deeponet_data_loss'])
    pi_pde_loss_training = np.array(pi_component_loss_training['pinn_pde_loss'])
    pi_pde_cty_training = np.array(pi_component_loss_training['pinn_pde_loss_cty'])
    pi_pde_mom_x_training = np.array(pi_component_loss_training['pinn_pde_loss_mom_x'])
    pi_pde_mom_y_training = np.array(pi_component_loss_training['pinn_pde_loss_mom_y'])

    # PI-DeepONet component losses for validation
    pi_component_loss_validation = pi_deeponet_history['validation_component_loss_history']
    pi_data_loss_validation = np.array(pi_component_loss_validation['deeponet_data_loss'])
    pi_pde_loss_validation = np.array(pi_component_loss_validation['pinn_pde_loss'])
    pi_pde_cty_validation = np.array(pi_component_loss_validation['pinn_pde_loss_cty'])
    pi_pde_mom_x_validation = np.array(pi_component_loss_validation['pinn_pde_loss_mom_x'])
    pi_pde_mom_y_validation = np.array(pi_component_loss_validation['pinn_pde_loss_mom_y'])

    # PI-DeepONet weights 
    pi_deeponet_data_loss_weight = pi_deeponet_history['adaptive_weight_history']['deeponet_data_loss_weight']
    pi_deeponet_pde_loss_weight = pi_deeponet_history['adaptive_weight_history']['deeponet_pinn_loss_weight']
    
    # Create figure with 3 subplots (3 rows, 1 column)
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=False, facecolor='w')
    
    # Subplot 1: Training and validation losses for both models
    axs[0].plot(deeponet_epochs, deeponet_train_loss, 'b-', linewidth=1.5, label='SWE-DeepONet (training)', alpha=0.8)
    axs[0].plot(deeponet_epochs, deeponet_val_loss, 'b--', linewidth=1.5, label='SWE-DeepONet (validation)', alpha=0.8)
    #axs[0].plot(pi_deeponet_epochs, pi_deeponet_train_loss, 'r-', linewidth=1.5, label='PI-SWE-DeepONet (training)', alpha=0.8)
    axs[0].plot(pi_deeponet_epochs, pi_data_loss_training, 'r-', linewidth=1.5, label='PI-SWE-DeepONet (training)', alpha=0.8)  #For PI-SWE-DeepONet, use the data loss as the training loss (to exclude the physics-informed loss)
    axs[0].plot(pi_deeponet_epochs, pi_data_loss_validation, 'r-.', linewidth=1.5, label='PI-SWE-DeepONet (validation)', alpha=0.8)
    axs[0].set_ylim(1e-2, 1)
    axs[0].set_xlabel('Epoch', fontsize=32)
    axs[0].set_ylabel('Loss', fontsize=32)
    axs[0].set_title('Training and Validation Losses', fontsize=24)
    axs[0].legend(fontsize=18, loc='upper right', framealpha=0.5)
    axs[0].grid(True, alpha=0.3, linestyle='--')
    axs[0].tick_params(axis='both', labelsize=28)
    axs[0].set_yscale('log')  # Use log scale for better visualization
    
    # Subplot 2: PI-DeepONet physics-informed loss and data loss (weighted by the data loss weight and the PDE loss weight)
    axs[1].plot(pi_deeponet_epochs, pi_data_loss_training * pi_deeponet_data_loss_weight[:len(pi_data_loss_training)], 'b-', linewidth=1.5, label='Data Loss', alpha=0.8)
    axs[1].plot(pi_deeponet_epochs, pi_pde_loss_training * pi_deeponet_pde_loss_weight[:len(pi_pde_loss_training)], 'r--', linewidth=1.5, label='PDE Loss', alpha=0.8)
    axs[1].set_xlabel('Epoch', fontsize=32)
    axs[1].set_ylabel('Loss', fontsize=32)
    axs[1].set_title('PI-SWE-DeepONet: Data Loss vs. PDE Loss (weighted)', fontsize=24)
    axs[1].legend(fontsize=24, loc='upper right', framealpha=0.5)
    axs[1].grid(True, alpha=0.3, linestyle='--')
    axs[1].tick_params(axis='both', labelsize=28)
    axs[1].set_yscale('log')
    
    # Subplot 3: PI-DeepONet PDE components
    axs[2].plot(pi_deeponet_epochs, pi_pde_cty_training, 'k-', linewidth=1.5, label='Continuity', alpha=0.8)
    axs[2].plot(pi_deeponet_epochs, pi_pde_mom_x_training, 'b--', linewidth=1.5, label='Momentum $x$', alpha=0.8)
    axs[2].plot(pi_deeponet_epochs, pi_pde_mom_y_training, 'r-.', linewidth=1.5, label='Momentum $y$', alpha=0.8)
    axs[2].set_xlabel('Epoch', fontsize=32)
    axs[2].set_ylabel('Loss', fontsize=32)
    axs[2].set_title('PI-SWE-DeepONet: PDE Component Losses (unweighted)', fontsize=24)
    axs[2].legend(fontsize=24, loc='center right', framealpha=0.5)
    axs[2].grid(True, alpha=0.3, linestyle='--')
    axs[2].tick_params(axis='both', labelsize=28)
    axs[2].set_yscale('log')

    #add text for (a), (b), (c) in the top left corner of each subplot
    axs[0].text(-0.2, 1.05, '(a)', fontsize=34, transform=axs[0].transAxes, verticalalignment='top', horizontalalignment='left', fontweight='bold')
    axs[1].text(-0.2, 1.05, '(b)', fontsize=34, transform=axs[1].transAxes, verticalalignment='top', horizontalalignment='left', fontweight='bold')
    axs[2].text(-0.2, 1.05, '(c)', fontsize=34, transform=axs[2].transAxes, verticalalignment='top', horizontalalignment='left', fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    #plt.subplots_adjust(hspace=0.5)

    # Save figure
    plt.savefig("training_history.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    

if __name__ == "__main__":
    # The file names of the training history files
    deeponet_training_history_file = '../../DeepONet/history_20251230_171752.json'
    pi_deeponet_training_history_file = '../../PI_DeepONet_1/history_20251230_181047.json'

    plot_training_history(deeponet_training_history_file, pi_deeponet_training_history_file)

    print("Plotting training history completed.")