"""
Training utilities for PINN models.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import time
import shutil

from ...utils.config import Config
from .data import PINNDataset
from .model import SWE_PINN  # Direct import instead of relative import

class PINNTrainer:
    """
    Trainer for Physics-Informed Neural Network (PINN) models.
    """
    def __init__(self, model, dataset, config):
        """
        Initialize the PINN trainer.
        
        Args:
            model (nn.Module): PINN model to train.
            dataset (PINNDataset): Dataset containing all collocation points and data.
            config (Config): Configuration object.
        """

        #check and ensure that the model is a SWE_PINN
        if not isinstance(model, SWE_PINN):
            raise ValueError("model must be a SWE_PINN")
        
        #check and ensure that the dataset is a PINNDataset
        if not isinstance(dataset, PINNDataset):
            raise ValueError("dataset must be a PINNDataset")
        
        self.model = model
        self.dataset = dataset

        # Get model config
        self.bPDE_loss, self.bInitial_loss, self.bBoundary_loss, self.bData_loss = self.model.get_loss_flags()
        self.bSteady = not self.bInitial_loss

        # Load configuration
        if config is not None:
            self.config = config
        else:
            raise ValueError("config must be provided and not None")
            
        # Set device
        self.device = self.model.get_device()       

        # Note: Model should already be on the correct device from initialization
        # Only move if it's not already on the device
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)

        print(f"Device: {self.device}") 
        
        # Training parameters
        self.batch_size = int(self.config.get_required_config('training.batch_size'))
        self.epochs = int(self.config.get_required_config('training.epochs'))
        self.learning_rate = float(self.config.get_required_config('training.learning_rate'))
        self.weight_decay = float(self.config.get_required_config('training.weight_decay'))

        # Loss function (for simple data loss computation)
        # Note: Full training uses model.compute_total_loss() which includes physics constraints
        self.loss_fn = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        use_scheduler = self.config.get('training.scheduler.use_scheduler', False)
        if use_scheduler:
            scheduler_type = self.config.get_required_config('training.scheduler.scheduler_type')
            if scheduler_type == 'ReduceLROnPlateau':
                patience = int(self.config.get_required_config('training.scheduler.patience'))
                factor = float(self.config.get_required_config('training.scheduler.factor'))
                min_lr = float(self.config.get_required_config('training.scheduler.min_lr'))
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    patience=patience,
                    factor=factor,
                    min_lr=min_lr
                )
            elif scheduler_type == 'ExponentialLR':
                gamma = float(self.config.get_required_config('training.scheduler.gamma'))
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer,
                    gamma=gamma
                )
            else:
                raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        else:
            self.scheduler = None
        
        # Early stopping
        use_early_stopping = self.config.get_required_config('training.early_stopping.use_early_stopping')
        if use_early_stopping:
            patience = int(self.config.get_required_config('training.early_stopping.patience'))
            min_delta = float(self.config.get_required_config('training.early_stopping.min_delta'))
            self.early_stopping = EarlyStopping(patience, min_delta)
        else:
            self.early_stopping = None
        
        # Adaptive loss balancing
        self.use_adaptive_loss_balancing = self.config.get_required_config('training.loss_weights.use_adaptive_balancing')
        self.use_adaptive_pde_component_balancing = self.config.get_required_config('training.loss_weights.use_adaptive_pde_component_balancing')
        
        # Adaptive balancing frequency: update weights every N epochs throughout training
        # If not specified, falls back to old behavior (only first N epochs)
        self.adaptive_balancing_frequency = self.config.get_required_config('training.loss_weights.adaptive_balancing_frequency')        
        
        # If frequency is not specified, use old behavior (only first N epochs)
        if self.adaptive_balancing_frequency is None:
            self.adaptive_balancing_frequency = 0  # 0 means use old behavior
        
        # PDE tolerance-based weight control
        self.pde_tolerance = float(self.config.get('training.loss_weights.pde_tolerance', 1e-4))
        self.pde_weight_decay_factor = float(self.config.get('training.loss_weights.pde_weight_decay_factor', 0.9))
        self.pde_weight_decay_frequency = int(self.config.get('training.loss_weights.pde_weight_decay_frequency', 5))
        self.pde_weight_moderation_factor = float(self.config.get('training.loss_weights.pde_weight_moderation_factor', 0.5))
        self.pde_weight_frozen = False  # Track if PDE weight is frozen due to low loss
        self.pde_decay_epoch_counter = 0  # Counter for decay frequency
        
        self.initial_loss_magnitudes = {}
        self.loss_balancing_factors = {}
        
        # Adaptive weight history tracking
        self.adaptive_weight_history = {
            'pde_loss_weight': [],
            'boundary_loss_weight': [],
            'data_loss_weight': [],
            'pde_continuity_weight': [],
            'pde_momentum_x_weight': [],
            'pde_momentum_y_weight': []
        }
        if not self.bSteady:
            self.adaptive_weight_history['initial_loss_weight'] = []
            
        # Logging        
        log_dir = self.config.get_required_config('training.logging.tensorboard_log_dir')
        self.writer = SummaryWriter(log_dir)
        
        # Checkpointing
        self.checkpoint_dir = self.config.get_required_config('training.logging.checkpoint_dir')
        self.save_freq = self.config.get_required_config('training.logging.save_freq')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training history
        self.loss_history = []
        self.component_loss_history = {}
        
        # Get training parameters from config
        self.print_freq = self.config.get_required_config('training.logging.print_freq')        

    def train(self):
        """
        Train the PINN model using full batch training.        
        """

        # Get stats
        mesh_stats = self.dataset.get_mesh_stats()
        data_stats = self.dataset.get_data_stats()

        # Get all points at once - full batch training
        # Get pde points
        if self.bPDE_loss:
            pde_points, pde_data = self.dataset.get_pde_points()
            if pde_points is not None and pde_data is not None:
                pde_points = pde_points.to(self.device)
                pde_data = pde_data.to(self.device)
            else:
                pde_points = None
                pde_data = None
        else:
            pde_points = None
            pde_data = None

        # Get initial points if it is included in loss, available and for transient problems
        if self.bInitial_loss:
            initial_points, initial_values = self.dataset.get_initial_points()
            if initial_points is not None:
                initial_points = initial_points.to(self.device)
                initial_values = initial_values.to(self.device)
            else:
                initial_points = None
                initial_values = None
        else:
            initial_points = None
            initial_values = None
        
        # Get boundary points
        if self.bBoundary_loss:
            boundary_points, boundary_ids, boundary_z, boundary_normals, boundary_lengths, boundary_ManningN = self.dataset.get_boundary_points()

            if boundary_points is not None:
                boundary_points = boundary_points.to(self.device)
            else:
                boundary_points = None
            
            if boundary_ids is not None:
                boundary_ids = boundary_ids.to(self.device)
            else:
                boundary_ids = None

            if boundary_z is not None:
                boundary_z = boundary_z.to(self.device)
            else:
                boundary_z = None
            
            if boundary_normals is not None:
                boundary_normals = boundary_normals.to(self.device)
            else:
                boundary_normals = None
            
            if boundary_lengths is not None:
                boundary_lengths = boundary_lengths.to(self.device)
            else:
                boundary_lengths = None

            if boundary_ManningN is not None:
                boundary_ManningN = boundary_ManningN.to(self.device)
            else:
                boundary_ManningN = None
        else:
            boundary_points = None
            boundary_ids = None
            boundary_z = None
            boundary_normals = None
            boundary_lengths = None
            boundary_ManningN = None

        # Get data points if it is included in loss and available
        if self.bData_loss:
            data_points, data_values, data_flags = self.dataset.get_data_points()

            if data_points is not None:
                data_points = data_points.to(self.device)
                data_values = data_values.to(self.device)
                data_flags = data_flags.to(self.device)
            else:
                data_points = None
                data_values = None
                data_flags = None
        else:
            data_points = None
            data_values = None
            data_flags = None

        print(f"\nStarting full batch training ...")

        if pde_points is not None:
            print(f"Number of PDE points: {len(pde_points)}")
        if initial_points is not None:
            print(f"Number of initial points: {len(initial_points)}")
        if boundary_points is not None:
            print(f"Number of boundary points: {len(boundary_points)}")
        if data_points is not None:
            print(f"Number of data points: {len(data_points)}")
        
        start_time = time.time()
        best_loss = float('inf')

        #print the total number of epochs
        print(f"Total number of epochs: {self.epochs}")

        # Adaptive loss balancing: compute initial loss magnitudes
        if self.use_adaptive_loss_balancing:
            print("Computing initial loss magnitudes for adaptive balancing...")
            boundary_info_tuple = (boundary_points, boundary_ids, boundary_z, boundary_normals, boundary_lengths, boundary_ManningN)
            self._compute_initial_loss_magnitudes(
                pde_points, pde_data, initial_points, initial_values,
                boundary_info_tuple, data_points, data_values, data_flags, mesh_stats, data_stats
            )
            self._update_loss_weights_from_balancing(current_pde_loss=None)
        
        # Record initial weights
        self._record_adaptive_weights()

        # Training loop
        for epoch in range(self.epochs):              

            # Pack boundary info
            boundary_info=(boundary_points, boundary_ids, boundary_z, boundary_normals, boundary_lengths, boundary_ManningN)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass and compute loss using all points
            total_loss, loss_components, predictions_and_true_values = self.model.compute_total_loss(
                    pde_points,
                    pde_data,
                    initial_points,
                    initial_values,
                    boundary_info,
                    data_points,
                    data_values,
                    data_flags,
                    mesh_stats,
                    data_stats
                )
                
            # Backward pass
            total_loss.backward()
                
            # Gradient clipping (to avoid exploding gradients)           
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Update weights
            self.optimizer.step()
                
            # Adaptive loss balancing: update weights periodically
            should_update_weights = False
            if self.use_adaptive_loss_balancing:
                if self.adaptive_balancing_frequency > 0:
                    # New behavior: update every N epochs throughout training
                    should_update_weights = (epoch % self.adaptive_balancing_frequency == 0)
                else:
                    print("adaptive_balancing_frequency is negative or zero, so weights will not be updated")
                
            if should_update_weights:
                self._update_adaptive_loss_weights(loss_components, epoch)
                
            # Record weights at every epoch (for complete history)
            self._record_adaptive_weights()

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(total_loss)
                else:
                    self.scheduler.step()
                
            # Save history and log progress
            self._log_training_step(epoch, total_loss, loss_components)
                
            # Save checkpoints
            if (epoch + 1) % self.save_freq == 0:
                self._save_checkpoint(epoch + 1)
                
            # Save best model
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                self._save_checkpoint('best')

            # Print progress
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {total_loss.item():.6f}")

            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(total_loss.item()):
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Training finished
        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time:.2f} seconds")
        print(f"Best loss achieved: {best_loss:.6f}")
        
        return {
            'loss_history': self.loss_history,
            'component_loss_history': self.component_loss_history            
        }, predictions_and_true_values
        
    def _save_checkpoint(self, epoch):
        """
        Save a model checkpoint.
        
        Args:
            epoch (int or str): Current epoch or 'final' or 'best'.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"pinn_epoch_{epoch}.pt")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'component_loss_history': self.component_loss_history,
            'epoch': epoch
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # On Windows, if the file exists and is locked (e.g., by antivirus or file explorer),
        # we need to handle this gracefully. Use a temporary file and atomic rename.
        temp_path = checkpoint_path + '.tmp'
        
        try:
            # Save to a temporary file first
            torch.save(checkpoint, temp_path)
            
            # If the target file exists, try to remove it first
            if os.path.exists(checkpoint_path):
                try:
                    os.remove(checkpoint_path)
                except (OSError, PermissionError):
                    # File is locked - we'll try to overwrite it directly below
                    pass
            
            # Try atomic rename (works on Windows if target doesn't exist or was removed)
            try:
                os.rename(temp_path, checkpoint_path)
            except (OSError, PermissionError):
                # If rename fails (target file is locked), try to overwrite directly
                # This might work in some cases even if the file appears locked
                try:
                    shutil.copy2(temp_path, checkpoint_path)
                    os.remove(temp_path)
                except (OSError, PermissionError) as copy_error:
                    # Clean up temp file
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                    raise RuntimeError(
                        f"Cannot save checkpoint to {checkpoint_path}. "
                        f"The file may be locked by another process (antivirus, file explorer, etc.). "
                        f"Error: {copy_error}"
                    )
                
        except Exception as e:
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            
            # If temp file approach failed, try direct save as last resort
            if not isinstance(e, RuntimeError):
                try:
                    torch.save(checkpoint, checkpoint_path)
                except Exception as save_error:
                    raise RuntimeError(
                        f"Cannot save checkpoint to {checkpoint_path}. "
                        f"Original error: {e}. Direct save error: {save_error}. "
                        f"This may be due to file locking by antivirus or another process. "
                        f"Try closing file explorer windows or temporarily disabling antivirus."
                    )
            else:
                raise
            
        if epoch!="best":
            print(f"Checkpoint saved to {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path):
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.loss_history = checkpoint.get('loss_history', [])
        self.component_loss_history = checkpoint.get('component_loss_history', {})
        
        #print(f"Loaded checkpoint from {checkpoint_path}")
        
    def predict(self, x):
        """
        Make predictions with the model.
        
        Args:
            x (torch.Tensor or numpy.ndarray): Input tensor of shape [batch_size, 3] containing (x, y, t) coordinates.
            
        Returns:
            numpy.ndarray: Model predictions (it is normalized if bNormalize is True)
        """
        self.model.eval()
        
        # Convert to torch tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
            
        # Move to device
        x = x.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(x)
            
        # Convert to numpy
        output = output.cpu().numpy()
        
        return output

    def _log_training_step(self, epoch, total_loss, loss_components):
        """
        Log the training progress.
        
        Args:
            epoch (int): Current epoch number
            total_loss (float): Total loss value (weighted)
            loss_components (dict): Dictionary containing individual loss components
        """
        # Store losses in history
        self.loss_history.append(total_loss.item())
        
        # Store component losses
        for key, value in loss_components.items():           
                
            if isinstance(value, dict):
                # For nested dictionaries, create sub-dictionaries
                if key not in self.component_loss_history:
                    self.component_loss_history[key] = {}
                    
                for subkey, subvalue in value.items():
                    if subkey not in self.component_loss_history[key]:
                        self.component_loss_history[key][subkey] = []
                    self.component_loss_history[key][subkey].append(subvalue.item() if torch.is_tensor(subvalue) else subvalue)
            else:
                if key not in self.component_loss_history:
                    self.component_loss_history[key] = []

                self.component_loss_history[key].append(value.item() if torch.is_tensor(value) else value)
        
        # Print progress
        if (epoch + 1) % self.print_freq == 0:
            print(f"\nEpoch [{epoch+1}/{self.epochs}]")
            print(f"Weighted Total Loss: {loss_components['weighted_total_loss']:.6f}")
            print(f"Unweighted Total Loss: {loss_components['unweighted_total_loss']:.6f}")
            print("\nLoss components:")
            for key, value in loss_components.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for subkey, subvalue in value.items():
                        if torch.is_tensor(subvalue):
                            print(f"    {subkey}: {subvalue.item():.6f}")
                        else:
                            print(f"    {subkey}: {subvalue:.6f}")
                else:
                    if torch.is_tensor(value):
                        print(f"  {key}: {value.item():.6f}")
                    else:
                        print(f"  {key}: {value:.6f}")
    
    def _compute_initial_loss_magnitudes(self, pde_points, pde_data, initial_points, initial_values,
                                        boundary_info, data_points, data_values, data_flags,
                                        mesh_stats, data_stats):
        """
        Compute initial loss magnitudes for adaptive loss balancing.
        
        This runs a few forward passes to estimate the typical magnitudes of different loss components.
        """
        self.model.eval()
        
        pde_losses = []
        initial_losses = []
        boundary_losses = []
        data_losses = []
        pde_continuity_losses = []
        pde_momentum_x_losses = []
        pde_momentum_y_losses = []
        
        # Compute PDE loss if enabled: use the whole pde_points set to compute the loss
        # Note: We need gradients enabled for PDE loss computation (it uses autograd.grad),
        # but we don't call backward() so no gradients are accumulated
        if self.bPDE_loss and pde_points is not None and pde_data is not None:
            # Enable gradients for PDE loss computation
            pde_loss, pde_loss_components, _, _, _ = self.model.compute_pde_loss(
                pde_points, pde_data, mesh_stats, data_stats
            )
            pde_losses.append(pde_loss.detach().item())
            
            # Store PDE component losses for adaptive balancing
            if self.use_adaptive_pde_component_balancing:
                pde_continuity_losses.append(pde_loss_components['continuity_loss'])
                pde_momentum_x_losses.append(pde_loss_components['momentum_x_loss'])
                pde_momentum_y_losses.append(pde_loss_components['momentum_y_loss'])
        
        # Compute initial loss if enabled
        if self.bInitial_loss and initial_points is not None and initial_values is not None:
            with torch.no_grad():
                initial_loss, _, _ = self.model.compute_initial_loss(initial_points, initial_values)
                initial_losses.append(initial_loss.item())
        
        # Compute boundary loss if enabled
        # Note: We need gradients enabled for boundary loss computation (it uses autograd.grad),
        # but we don't call backward() so no gradients are accumulated
        if self.bBoundary_loss and boundary_info[0] is not None:
            # Enable gradients for boundary loss computation
            boundary_loss, _, _, _, _ = self.model.compute_boundary_loss(
                boundary_info, mesh_stats, data_stats
            )
            boundary_losses.append(boundary_loss.detach().item())
        
        # Compute data loss if enabled
        if self.bData_loss and data_points is not None and data_values is not None:
            with torch.no_grad():
                data_loss, _, _, _, _ = self.model.compute_data_loss(
                    data_points, data_values, data_flags, mesh_stats, data_stats
                )
                data_losses.append(data_loss.item())
        
        # Store initial magnitudes (use median to be robust to outliers)
        if pde_losses:
            self.initial_loss_magnitudes['pde_loss'] = np.median(pde_losses)
        if initial_losses:
            self.initial_loss_magnitudes['initial_loss'] = np.median(initial_losses)
        if boundary_losses:
            self.initial_loss_magnitudes['boundary_loss'] = np.median(boundary_losses)
        if data_losses:
            self.initial_loss_magnitudes['data_loss'] = np.median(data_losses)
        
        # Store PDE component loss magnitudes
        if self.use_adaptive_pde_component_balancing:
            if pde_continuity_losses:
                self.initial_loss_magnitudes['pde_continuity_loss'] = np.median(pde_continuity_losses)
            if pde_momentum_x_losses:
                self.initial_loss_magnitudes['pde_momentum_x_loss'] = np.median(pde_momentum_x_losses)
            if pde_momentum_y_losses:
                self.initial_loss_magnitudes['pde_momentum_y_loss'] = np.median(pde_momentum_y_losses)
        
        self.model.train()
        
        print(f"Initial loss magnitudes: {self.initial_loss_magnitudes}")
    
    def _update_loss_weights_from_balancing(self, current_pde_loss=None):
        """
        Update loss weights based on computed balancing factors.
        
        The balancing factor for each loss is computed as:
        balancing_factor = reference_loss_magnitude / loss_magnitude
        
        This ensures losses are on similar scales.
        
        Args:
            current_pde_loss (float, optional): Current PDE loss value for tolerance-based control.
        """
        if not self.initial_loss_magnitudes:
            return
        
        # Use data loss as reference, or pde_loss if data_loss not available
        if 'data_loss' in self.initial_loss_magnitudes:
            reference_magnitude = self.initial_loss_magnitudes['data_loss']
        elif 'pde_loss' in self.initial_loss_magnitudes:
            reference_magnitude = self.initial_loss_magnitudes['pde_loss']
        else:
            reference_magnitude = 1.0
        
        # Compute balancing factors with tolerance-based control for PDE loss
        if 'pde_loss' in self.initial_loss_magnitudes:
            pde_magnitude = self.initial_loss_magnitudes['pde_loss']
            
            # Check if current PDE loss is below tolerance
            if current_pde_loss is not None and current_pde_loss < self.pde_tolerance:
                # PDE loss is satisfied - freeze or decay the weight
                if not self.pde_weight_frozen:
                    print(f"PDE loss ({current_pde_loss:.6e}) below tolerance ({self.pde_tolerance:.6e}). "
                          f"Freezing PDE weight updates.")
                    self.pde_weight_frozen = True
                
                # Keep the current weight (don't update based on balancing)
                # The weight will be decayed separately if needed
                if pde_magnitude > 0:
                    # Still compute the factor, but we'll use it differently
                    self.loss_balancing_factors['pde_loss'] = reference_magnitude / pde_magnitude
                else:
                    self.loss_balancing_factors['pde_loss'] = 1.0
            else:
                # PDE loss is above tolerance - allow moderate adaptive updates
                if self.pde_weight_frozen:
                    pde_loss_str = f"{current_pde_loss:.6e}" if current_pde_loss is not None else "N/A"
                    print(f"PDE loss ({pde_loss_str}) above tolerance ({self.pde_tolerance:.6e}). "
                          f"Resuming adaptive PDE weight updates with moderation.")
                    self.pde_weight_frozen = False
                    self.pde_decay_epoch_counter = 0  # Reset decay counter
                
                # Moderate adaptive update: blend current factor with new factor
                if pde_magnitude > 0:
                    new_factor = reference_magnitude / pde_magnitude
                    # If we already have a factor, moderate the update
                    if 'pde_loss' in self.loss_balancing_factors:
                        current_factor = self.loss_balancing_factors['pde_loss']
                        # Blend: use moderation_factor of new factor and (1-moderation_factor) of current
                        self.loss_balancing_factors['pde_loss'] = (
                            self.pde_weight_moderation_factor * new_factor + 
                            (1 - self.pde_weight_moderation_factor) * current_factor
                        )
                    else:
                        self.loss_balancing_factors['pde_loss'] = new_factor
                else:
                    self.loss_balancing_factors['pde_loss'] = 1.0
        
        if 'initial_loss' in self.initial_loss_magnitudes:
            initial_magnitude = self.initial_loss_magnitudes['initial_loss']
            if initial_magnitude > 0:
                self.loss_balancing_factors['initial_loss'] = reference_magnitude / initial_magnitude
            else:
                self.loss_balancing_factors['initial_loss'] = 1.0
        
        if 'boundary_loss' in self.initial_loss_magnitudes:
            boundary_magnitude = self.initial_loss_magnitudes['boundary_loss']
            if boundary_magnitude > 0:
                self.loss_balancing_factors['boundary_loss'] = reference_magnitude / boundary_magnitude
            else:
                self.loss_balancing_factors['boundary_loss'] = 1.0
        
        if 'data_loss' in self.initial_loss_magnitudes:
            data_magnitude = self.initial_loss_magnitudes['data_loss']
            if data_magnitude > 0:
                self.loss_balancing_factors['data_loss'] = reference_magnitude / data_magnitude
            else:
                self.loss_balancing_factors['data_loss'] = 1.0
        
        # Apply balancing factors to model's loss weights
        base_pde_weight = self.config.get_required_config("training.loss_weights.pinn.pde_loss")
        base_boundary_weight = self.config.get_required_config("training.loss_weights.pinn.boundary_loss")
        base_data_weight = self.config.get_required_config("training.loss_weights.pinn.data_loss")
        
        if 'pde_loss' in self.loss_balancing_factors:
            if self.pde_weight_frozen and current_pde_loss is not None and current_pde_loss < self.pde_tolerance:
                # PDE loss is satisfied - freeze the weight (keep current value)
                # Weight will be decayed separately in _update_adaptive_loss_weights
                current_weight = self.model.loss_weight_pde_loss.item()
                # Don't update the weight based on balancing, just keep it frozen
                pass  # Weight remains unchanged
            else:
                # Normal adaptive update (with moderation if above tolerance)
                new_pde_weight = base_pde_weight * self.loss_balancing_factors['pde_loss']
                self.model.loss_weight_pde_loss.data = torch.tensor(
                    float(new_pde_weight), dtype=torch.float32, device=self.device
                )
        
        if 'boundary_loss' in self.loss_balancing_factors:
            new_boundary_weight = base_boundary_weight * self.loss_balancing_factors['boundary_loss']
            self.model.loss_weight_boundary_loss.data = torch.tensor(
                float(new_boundary_weight), dtype=torch.float32, device=self.device
            )
        
        if 'data_loss' in self.loss_balancing_factors:
            new_data_weight = base_data_weight * self.loss_balancing_factors['data_loss']
            self.model.loss_weight_data_loss.data = torch.tensor(
                float(new_data_weight), dtype=torch.float32, device=self.device
            )
        
        if not self.bSteady and 'initial_loss' in self.loss_balancing_factors:
            base_initial_weight = self.config.get_required_config("training.loss_weights.pinn.initial_loss")
            new_initial_weight = base_initial_weight * self.loss_balancing_factors['initial_loss']
            self.model.loss_weight_initial_loss.data = torch.tensor(
                float(new_initial_weight), dtype=torch.float32, device=self.device
            )
        
        # Balance PDE component losses if enabled
        if self.use_adaptive_pde_component_balancing:
            # Use the median magnitude as reference (more robust than max)
            pde_component_magnitudes = {}
            if 'pde_continuity_loss' in self.initial_loss_magnitudes:
                pde_component_magnitudes['continuity'] = self.initial_loss_magnitudes['pde_continuity_loss']
            if 'pde_momentum_x_loss' in self.initial_loss_magnitudes:
                pde_component_magnitudes['momentum_x'] = self.initial_loss_magnitudes['pde_momentum_x_loss']
            if 'pde_momentum_y_loss' in self.initial_loss_magnitudes:
                pde_component_magnitudes['momentum_y'] = self.initial_loss_magnitudes['pde_momentum_y_loss']
            
            if pde_component_magnitudes:
                reference_pde_magnitude = np.median(list(pde_component_magnitudes.values()))
                
                # Compute and apply balancing factors
                base_continuity_weight = self.config.get_required_config("training.loss_weights.pde.continuity")
                base_momentum_x_weight = self.config.get_required_config("training.loss_weights.pde.momentum_x")
                base_momentum_y_weight = self.config.get_required_config("training.loss_weights.pde.momentum_y")
                
                if 'continuity' in pde_component_magnitudes and pde_component_magnitudes['continuity'] > 0:
                    factor = reference_pde_magnitude / pde_component_magnitudes['continuity']
                    new_weight = base_continuity_weight * factor
                    self.model.loss_weight_pde_continuity.data = torch.tensor(
                        float(new_weight), dtype=torch.float32, device=self.device
                    )
                    self.loss_balancing_factors['pde_continuity'] = factor
                
                if 'momentum_x' in pde_component_magnitudes and pde_component_magnitudes['momentum_x'] > 0:
                    factor = reference_pde_magnitude / pde_component_magnitudes['momentum_x']
                    new_weight = base_momentum_x_weight * factor
                    self.model.loss_weight_pde_momentum_x.data = torch.tensor(
                        float(new_weight), dtype=torch.float32, device=self.device
                    )
                    self.loss_balancing_factors['pde_momentum_x'] = factor
                
                if 'momentum_y' in pde_component_magnitudes and pde_component_magnitudes['momentum_y'] > 0:
                    factor = reference_pde_magnitude / pde_component_magnitudes['momentum_y']
                    new_weight = base_momentum_y_weight * factor
                    self.model.loss_weight_pde_momentum_y.data = torch.tensor(
                        float(new_weight), dtype=torch.float32, device=self.device
                    )
                    self.loss_balancing_factors['pde_momentum_y'] = factor
                
                print(f"Updated PDE component weights - Continuity: {self.model.loss_weight_pde_continuity.item():.6f}, "
                      f"Momentum X: {self.model.loss_weight_pde_momentum_x.item():.6f}, "
                      f"Momentum Y: {self.model.loss_weight_pde_momentum_y.item():.6f}")
        
        print(f"Updated loss weights - PDE: {self.model.loss_weight_pde_loss.item():.6f}, "
              f"Boundary: {self.model.loss_weight_boundary_loss.item():.6f}, "
              f"Data: {self.model.loss_weight_data_loss.item():.6f}")

        if not self.bSteady:
            print(f"Initial: {self.model.loss_weight_initial_loss.item():.6f}")
    
    def _update_adaptive_loss_weights(self, loss_components, epoch):
        """
        Update loss weights adaptively during training.
        
        This uses exponential moving average to smooth the updates.
        """
        alpha = 0.1  # Smoothing factor
        
        # Update magnitudes with exponential moving average
        if 'pde_loss' in loss_components.get('loss_components', {}):
            current_magnitude = loss_components['loss_components']['pde_loss']
            if 'pde_loss' not in self.initial_loss_magnitudes:
                self.initial_loss_magnitudes['pde_loss'] = current_magnitude
            else:
                self.initial_loss_magnitudes['pde_loss'] = (
                    alpha * current_magnitude + (1 - alpha) * self.initial_loss_magnitudes['pde_loss']
                )
        
        if not self.bSteady and 'initial_loss' in loss_components.get('loss_components', {}):
            current_magnitude = loss_components['loss_components']['initial_loss']
            if 'initial_loss' not in self.initial_loss_magnitudes:
                self.initial_loss_magnitudes['initial_loss'] = current_magnitude
            else:
                self.initial_loss_magnitudes['initial_loss'] = (
                    alpha * current_magnitude + (1 - alpha) * self.initial_loss_magnitudes['initial_loss']
                )
        
        if 'boundary_loss' in loss_components.get('loss_components', {}):
            current_magnitude = loss_components['loss_components']['boundary_loss']
            if 'boundary_loss' not in self.initial_loss_magnitudes:
                self.initial_loss_magnitudes['boundary_loss'] = current_magnitude
            else:
                self.initial_loss_magnitudes['boundary_loss'] = (
                    alpha * current_magnitude + (1 - alpha) * self.initial_loss_magnitudes['boundary_loss']
                )
        
        if 'data_loss' in loss_components.get('loss_components', {}):
            current_magnitude = loss_components['loss_components']['data_loss']
            if 'data_loss' not in self.initial_loss_magnitudes:
                self.initial_loss_magnitudes['data_loss'] = current_magnitude
            else:
                self.initial_loss_magnitudes['data_loss'] = (
                    alpha * current_magnitude + (1 - alpha) * self.initial_loss_magnitudes['data_loss']
                )
        
        # Update PDE component loss magnitudes if adaptive balancing is enabled
        if self.use_adaptive_pde_component_balancing:
            if 'continuity_loss' in loss_components.get('pde_loss_components', {}):
                current_magnitude = loss_components['pde_loss_components']['continuity_loss']
                if 'pde_continuity_loss' not in self.initial_loss_magnitudes:
                    self.initial_loss_magnitudes['pde_continuity_loss'] = current_magnitude
                else:
                    self.initial_loss_magnitudes['pde_continuity_loss'] = (
                        alpha * current_magnitude + (1 - alpha) * self.initial_loss_magnitudes['pde_continuity_loss']
                    )
            
            if 'momentum_x_loss' in loss_components.get('pde_loss_components', {}):
                current_magnitude = loss_components['pde_loss_components']['momentum_x_loss']
                if 'pde_momentum_x_loss' not in self.initial_loss_magnitudes:
                    self.initial_loss_magnitudes['pde_momentum_x_loss'] = current_magnitude
                else:
                    self.initial_loss_magnitudes['pde_momentum_x_loss'] = (
                        alpha * current_magnitude + (1 - alpha) * self.initial_loss_magnitudes['pde_momentum_x_loss']
                    )
            
            if 'momentum_y_loss' in loss_components.get('pde_loss_components', {}):
                current_magnitude = loss_components['pde_loss_components']['momentum_y_loss']
                if 'pde_momentum_y_loss' not in self.initial_loss_magnitudes:
                    self.initial_loss_magnitudes['pde_momentum_y_loss'] = current_magnitude
                else:
                    self.initial_loss_magnitudes['pde_momentum_y_loss'] = (
                        alpha * current_magnitude + (1 - alpha) * self.initial_loss_magnitudes['pde_momentum_y_loss']
                    )
        
        # Get current PDE loss for tolerance-based control
        current_pde_loss = None
        if 'pde_loss' in loss_components.get('loss_components', {}):
            current_pde_loss = loss_components['loss_components']['pde_loss']
        
        # Recompute balancing factors
        # If using frequency-based updates, update immediately
        # Otherwise, update every 10 epochs 
        if self.adaptive_balancing_frequency > 0:
            # Frequency-based: update immediately when this method is called
            self._update_loss_weights_from_balancing(current_pde_loss=current_pde_loss)
        elif (epoch + 1) % 10 == 0:
            # Old behavior: update every 5 epochs
            self._update_loss_weights_from_balancing(current_pde_loss=current_pde_loss)
        
        # Apply decay to frozen PDE weight if needed
        if self.pde_weight_frozen and current_pde_loss is not None and current_pde_loss < self.pde_tolerance:
            self.pde_decay_epoch_counter += 1
            if self.pde_decay_epoch_counter >= self.pde_weight_decay_frequency:
                current_weight = self.model.loss_weight_pde_loss.item()
                new_weight = current_weight * self.pde_weight_decay_factor
                self.model.loss_weight_pde_loss.data = torch.tensor(
                    float(new_weight), dtype=torch.float32, device=self.device
                )
                print(f"Decaying frozen PDE weight: {current_weight:.6f} -> {new_weight:.6f} "
                      f"(factor: {self.pde_weight_decay_factor}, PDE loss: {current_pde_loss:.6e})")
                self.pde_decay_epoch_counter = 0  # Reset counter
    
    def _record_adaptive_weights(self):
        """
        Record current adaptive loss weights to history.
        """
        self.adaptive_weight_history['pde_loss_weight'].append(
            self.model.loss_weight_pde_loss.item()
        )
        self.adaptive_weight_history['boundary_loss_weight'].append(
            self.model.loss_weight_boundary_loss.item()
        )
        self.adaptive_weight_history['data_loss_weight'].append(
            self.model.loss_weight_data_loss.item()
        )
        if not self.bSteady:
            self.adaptive_weight_history['initial_loss_weight'].append(
                self.model.loss_weight_initial_loss.item()
            )
        self.adaptive_weight_history['pde_continuity_weight'].append(
            self.model.loss_weight_pde_continuity.item()
        )
        self.adaptive_weight_history['pde_momentum_x_weight'].append(
            self.model.loss_weight_pde_momentum_x.item()
        )
        self.adaptive_weight_history['pde_momentum_y_weight'].append(
            self.model.loss_weight_pde_momentum_y.item()
        )


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """
    def __init__(self, patience=500, min_delta=1e-5, verbose=True):
        """
        Initialize early stopping.
        
        Args:
            patience (int): Number of epochs to wait before stopping.
            min_delta (float): Minimum change in loss to be considered as improvement.
            verbose (bool): Whether to print messages.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss):
        """
        Check if training should stop.
        
        Args:
            val_loss (float): Validation loss.
            
        Returns:
            bool: True if training should stop, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose and self.counter % 100 == 0:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered")
                    
        return self.early_stop 