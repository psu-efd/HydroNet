"""
Visualization utilities for HydroNet.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.tri import Triangulation


def setup_plotting_style():
    """Set the plotting style for consistent visualization."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12


def plot_training_history(loss_history, val_loss_history=None, title="Training Loss", save_path=None):
    """
    Plot the training and validation loss history.
    
    Args:
        loss_history (list): List of training losses.
        val_loss_history (list, optional): List of validation losses.
        title (str, optional): Plot title.
        save_path (str, optional): Path to save the figure. If None, the figure is shown instead.
    """
    setup_plotting_style()
    plt.figure()
    
    epochs = np.arange(1, len(loss_history) + 1)
    plt.semilogy(epochs, loss_history, 'b-', label='Training Loss')
    
    if val_loss_history is not None:
        plt.semilogy(epochs, val_loss_history, 'r-', label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_2d_contour(x, y, z, title="2D Contour Plot", cmap='viridis', colorbar_label=None, save_path=None):
    """
    Create a 2D contour plot.
    
    Args:
        x (np.ndarray): x-coordinates.
        y (np.ndarray): y-coordinates.
        z (np.ndarray): Values to plot.
        title (str, optional): Plot title.
        cmap (str, optional): Colormap.
        colorbar_label (str, optional): Label for the colorbar.
        save_path (str, optional): Path to save the figure. If None, the figure is shown instead.
    """
    setup_plotting_style()
    plt.figure()
    
    contour = plt.tricontourf(x, y, z, 50, cmap=cmap)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    
    cbar = plt.colorbar(contour)
    if colorbar_label:
        cbar.set_label(colorbar_label)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_comparison(x, y, z_true, z_pred, title="Comparison", cmap='viridis', colorbar_label=None, save_path=None):
    """
    Create a comparison plot between true and predicted values.
    
    Args:
        x (np.ndarray): x-coordinates.
        y (np.ndarray): y-coordinates.
        z_true (np.ndarray): True values.
        z_pred (np.ndarray): Predicted values.
        title (str, optional): Plot title.
        cmap (str, optional): Colormap.
        colorbar_label (str, optional): Label for the colorbar.
        save_path (str, optional): Path to save the figure. If None, the figure is shown instead.
    """
    setup_plotting_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # True values
    cont1 = axes[0].tricontourf(x, y, z_true, 50, cmap=cmap)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title(f'{title} (True)')
    
    # Predicted values
    cont2 = axes[1].tricontourf(x, y, z_pred, 50, cmap=cmap)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title(f'{title} (Prediction)')
    
    # Error
    error = np.abs(z_true - z_pred)
    cont3 = axes[2].tricontourf(x, y, error, 50, cmap='hot')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_title(f'{title} (Absolute Error)')
    
    # Add colorbars
    cbar1 = plt.colorbar(cont1, ax=axes[0])
    cbar2 = plt.colorbar(cont2, ax=axes[1])
    cbar3 = plt.colorbar(cont3, ax=axes[2])
    
    if colorbar_label:
        cbar1.set_label(colorbar_label)
        cbar2.set_label(colorbar_label)
        cbar3.set_label(f'|{colorbar_label}|')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_vector_field(x, y, u, v, title="Vector Field", cmap='viridis', density=1.0, save_path=None):
    """
    Plot a vector field (e.g., velocity field).
    
    Args:
        x (np.ndarray): x-coordinates.
        y (np.ndarray): y-coordinates.
        u (np.ndarray): x-component of the vector field.
        v (np.ndarray): y-component of the vector field.
        title (str, optional): Plot title.
        cmap (str, optional): Colormap.
        density (float, optional): Density of arrows in the quiver plot.
        save_path (str, optional): Path to save the figure. If None, the figure is shown instead.
    """
    setup_plotting_style()
    plt.figure()
    
    # Calculate the magnitude of the vector field
    magnitude = np.sqrt(u**2 + v**2)
    
    # Create a colored quiver plot
    plt.quiver(x[::int(1/density)], y[::int(1/density)], 
              u[::int(1/density)], v[::int(1/density)], 
              magnitude[::int(1/density)], cmap=cmap, 
              scale=25, width=0.002)
    
    plt.colorbar(label='Magnitude')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show() 