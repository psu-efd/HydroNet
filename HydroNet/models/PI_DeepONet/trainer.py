"""
Training utilities for Physics-Informed SWE_DeepONet models.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import time
import shutil

from ...utils.config import Config
from ...models.PINN.data import PINNDataset
from ...models.PI_DeepONet.model import PI_SWE_DeepONetModel


class PI_SWE_DeepONetTrainer:
    """
    Trainer for Physics-Informed SWE_DeepONet models.
    """
    def __init__(self, model, config: Config):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): Physics-Informed DeepONet model to train.
            config (Config): Configuration object.
        """
        # Check if model is an instance of PI_SWE_DeepONetModel
        if not isinstance(model, PI_SWE_DeepONetModel):
            raise ValueError("model must be an instance of PI_SWE_DeepONetModel")

        if not isinstance(config, Config):
            raise ValueError("config must be a Config object.")

        self.model = model
        self.config = config
            
        # Set device
        self.device = self.config.get_device()
        
        # Note: Model should already be on the correct device from initialization
        # Only move if it's not already on the device
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)

        print(f"Device: {self.device}")
        
        # Training parameters
        self.batch_size = self.config.get_required_config('training.batch_size')
        self.epochs = self.config.get_required_config('training.epochs')
        self.learning_rate = float(self.config.get_required_config('training.learning_rate'))
        self.weight_decay = float(self.config.get_required_config('training.weight_decay'))
        
        # Loss function (for simple data loss computation)
        # Note: Full training uses model.compute_total_loss() which includes physics constraints
        self.loss_fn = nn.MSELoss()
        
        # Optimizer
        optimizer_type = self.config.get_required_config('training.optimizer.optimizer_type')
        self.optimizer_type = optimizer_type  # Store for use in training loop
        if optimizer_type == 'Adam':
            print(f"Using Adam optimizer")
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_type == 'AdamW':
            print(f"Using AdamW optimizer")
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_type == 'LBFGS':
            print(f"Using LBFGS optimizer (full-batch optimizer)")
            print(f"Warning: LBFGS does not support weight_decay. Weight decay will be ignored.")
            print(f"Note: LBFGS requires full-batch training. All batches will be concatenated into a single batch per epoch.")
            
            # Read LBFGS options from config
            lbfgs_options = self.config.get('training.optimizer.LBFGS_options', {})
            max_iter = int(lbfgs_options.get('max_iter', 20))
            max_eval = lbfgs_options.get('max_eval', None)
            if max_eval is not None:
                max_eval = int(max_eval)
            tolerance_grad = float(lbfgs_options.get('tolerance_grad', 1e-07))
            tolerance_change = float(lbfgs_options.get('tolerance_change', 1e-09))
            history_size = int(lbfgs_options.get('history_size', 100))
            line_search_fn = lbfgs_options.get('line_search_fn', None)
            # Convert string "null" or "None" to Python None
            if line_search_fn in [None, "null", "None", ""]:
                line_search_fn = None
            elif line_search_fn == "strong_wolfe":
                line_search_fn = "strong_wolfe"
            else:
                raise ValueError(f"Invalid line_search_fn: {line_search_fn}. Must be 'strong_wolfe' or null")
            
            # Store tolerance values and options for debugging
            self.lbfgs_tolerance_grad = tolerance_grad
            self.lbfgs_tolerance_change = tolerance_change
            self.lbfgs_max_iter = max_iter
            self.lbfgs_history_size = history_size
            self.lbfgs_line_search_fn = line_search_fn
            
            print(f"LBFGS options: history_size={history_size}, max_iter={max_iter}, max_eval={max_eval}")
            
            self.optimizer = optim.LBFGS(
                self.model.parameters(),
                lr=self.learning_rate,
                max_iter=max_iter,
                max_eval=max_eval,
                tolerance_grad=tolerance_grad,
                tolerance_change=tolerance_change,
                history_size=history_size,
                line_search_fn=line_search_fn
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # Learning rate scheduler
        use_scheduler = self.config.get_required_config('training.scheduler.use_scheduler')
        print(f"Use scheduler: {use_scheduler}")
        
        if use_scheduler:
            scheduler_type = self.config.get_required_config('training.scheduler.scheduler_type')
            if scheduler_type == 'ReduceLROnPlateau':
                patience = int(self.config.get_required_config('training.scheduler.patience'))
                factor = float(self.config.get_required_config('training.scheduler.factor'))
                min_lr = float(self.config.get_required_config('training.scheduler.min_lr'))
                # Required: starting epoch to begin tracking best validation loss
                # Useful when starting from pre-trained models where validation loss may initially increase
                self.scheduler_start_epoch = int(self.config.get_required_config('training.scheduler.start_epoch'))
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    patience=patience,
                    factor=factor,
                    min_lr=min_lr
                )

                #print for learning rate scheduler
                print(f"Learning rate scheduler: {scheduler_type}")
                print(f"Learning rate: {self.learning_rate}")
                print(f"Learning rate scheduler patience: {patience}")
                print(f"Learning rate scheduler factor: {factor}")
                print(f"Learning rate scheduler min_lr: {min_lr}")
                print(f"Learning rate scheduler start epoch: {self.scheduler_start_epoch} (will begin tracking best validation loss from this epoch)")

            elif scheduler_type == 'ExponentialLR':
                gamma = self.config.get_required_config('training.scheduler.gamma')
                self.scheduler = optim.lr_scheduler.ExponentialLR(
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
            
        # Logging
        log_dir = self.config.get_required_config('training.logging.tensorboard_log_dir')
        self.writer = SummaryWriter(log_dir)
        
        # Checkpointing
        self.checkpoint_dir = self.config.get_required_config('training.logging.checkpoint_dir')
        self.save_freq = self.config.get_required_config('training.logging.save_freq')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training and validation history
        self.training_loss_history = []
        self.validation_loss_history = []

        # Training component loss history
        self.training_component_loss_history = {
            'deeponet_data_loss': [],
            'pinn_pde_loss': [],
            'pinn_pde_loss_cty': [],
            'pinn_pde_loss_mom_x': [],
            'pinn_pde_loss_mom_y': [],
            'pinn_initial_loss': [],
            'pinn_boundary_loss': [],
            'total_loss': []
        }
        
        # Validation component loss history
        self.validation_component_loss_history = {
            'deeponet_data_loss': [],
            'pinn_pde_loss': [],
            'pinn_pde_loss_cty': [],
            'pinn_pde_loss_mom_x': [],
            'pinn_pde_loss_mom_y': [],
            'pinn_initial_loss': [],
            'pinn_boundary_loss': [],
            'total_loss': []
        }
        
        # Adaptive loss balancing
        self.use_adaptive_loss_balancing = self.config.get_required_config('training.loss_weights.use_adaptive_balancing')
        self.use_adaptive_pde_component_balancing = self.config.get_required_config('training.loss_weights.use_adaptive_pde_component_balancing')
        
        # Adaptive balancing frequency: update weights every N epochs throughout training
        # If not specified, falls back to old behavior (only first N epochs)
        self.adaptive_balancing_frequency = self.config.get_required_config('training.loss_weights.adaptive_balancing_frequency')
        
        # If frequency is not specified, use old behavior (only first N epochs)
        if self.adaptive_balancing_frequency is None:
            self.adaptive_balancing_frequency = 0  # 0 means use old behavior
        
        self.initial_loss_magnitudes = {}
        self.loss_balancing_factors = {}
        
        # Adaptive weight history tracking
        self.adaptive_weight_history = {
            'deeponet_data_loss_weight': [],
            'deeponet_pinn_loss_weight': [],
            'pde_continuity_weight': [],
            'pde_momentum_x_weight': [],
            'pde_momentum_y_weight': []
        }
        
        # PDE loss weight ramping
        ramping_config = self.config.get('training.loss_weights.ramping_pde_loss_weights', {})
        self.use_pde_loss_ramping = ramping_config.get('use_ramping', False)
        self.ramping_epochs = int(ramping_config.get('ramping_epochs', 10)) if self.use_pde_loss_ramping else 0
        
        # Store the target PDE weight (will be updated by adaptive balancing if enabled)
        # This is the weight we want to ramp to
        self.target_pde_loss_weight = None
        
        if self.use_pde_loss_ramping:
            print(f"PDE loss weight ramping enabled: ramping from 0 to full weight over {self.ramping_epochs} epochs")
        
    def train(self, train_loader, val_loader, physics_dataset=None):
        """
        Train the Physics-Informed SWE_DeepONet model.
        
        Args:
            train_loader (DataLoader): Training data loader for DeepONet data.
            val_loader (DataLoader): Validation data loader for DeepONet data.
            physics_dataset (PINNDataset, optional): Dataset for physics constraints.
                Should have methods: get_pde_points(), get_initial_points(), get_boundary_points(),
                get_mesh_stats(), get_data_stats().
            
        Returns:
            dict: Training history containing loss_history and component_loss_history.
        """
            
        # Get branch input dimension from data if needed
        batch = next(iter(train_loader))  #branch_input(batch_size, n_features), trunk_input(nCells, n_coords), output(batch_size, nCells, n_outputs)
        branch_input = batch[0]   #branch_input(batch_size, n_features)
        branch_dim = branch_input.shape[1]  #branch_dim = n_features

        if self.model.branch_input_dim == 0:            
            self.model.set_branch_input_dim(branch_dim)
            print(f"Set branch input dimension to {branch_dim}")

        # Check if the model branch input dimension is the same as the branch input dimension of the data
        if self.model.branch_input_dim != branch_dim:
            raise ValueError(f"Model branch input dimension {self.model.branch_input_dim} is not the same as the branch input dimension {branch_dim} of the data.")
            
        # Check if physics-informed loss is enabled and physics dataset is provided
        if self.model.use_physics_loss and physics_dataset is None:
            raise ValueError("physics_dataset must be provided when physics-informed loss is enabled.")

        # Get physics collocation points and data
        if self.model.use_physics_loss:
            pde_points_data = physics_dataset.get_pde_points()
            if pde_points_data is not None:
                pde_points, pde_data = pde_points_data
                pde_points = pde_points.to(self.device)
                pde_data = pde_data.to(self.device)
            else:
                pde_points = None
                pde_data = None

            initial_points_data = physics_dataset.get_initial_points()
            if initial_points_data is not None:
                initial_points, initial_values = initial_points_data
                initial_points = initial_points.to(self.device)
                initial_values = initial_values.to(self.device)
            else:
                initial_points = None
                initial_values = None

            boundary_points_data = physics_dataset.get_boundary_points()
            if boundary_points_data is not None:
                (
                    boundary_points,
                    boundary_ids,
                    boundary_z,
                    boundary_normals,
                    boundary_lengths,
                    boundary_ManningN,
                ) = boundary_points_data
                boundary_points = boundary_points.to(self.device)
            else:
                boundary_points = None

            all_pinn_points_stats = physics_dataset.get_all_pinn_stats()
        else:
            pde_points = None
            pde_data = None
            initial_points = None
            initial_values = None
            boundary_points = None
            all_pinn_points_stats = None
        
        # Get DeepONet stats 
        # Handle case where dataset might be a Subset wrapper
        train_dataset = train_loader.dataset
        if hasattr(train_dataset, 'dataset'):
            # It's a Subset, get the underlying dataset
            train_dataset = train_dataset.dataset
        all_deeponet_points_stats = train_dataset.get_deeponet_stats()

        if all_deeponet_points_stats is None:
            raise ValueError("train_loader.dataset must provide DeepONet point statistics.")

        if self.model.use_physics_loss:
            if all_pinn_points_stats is None:
                raise ValueError("physics_dataset must provide PINN point statistics when physics loss is enabled.")
        
        # Adaptive loss balancing: compute initial loss magnitudes
        if self.use_adaptive_loss_balancing:
            print("Computing initial loss magnitudes for adaptive balancing...")
            self._compute_initial_loss_magnitudes(
                train_loader, pde_points, pde_data, all_deeponet_points_stats, all_pinn_points_stats
            )
            self._update_loss_weights_from_balancing()
        
        # Initialize target PDE loss weight for ramping (after adaptive balancing if enabled)
        # If adaptive balancing is disabled, use the current weight from config
        if self.use_pde_loss_ramping:
            if self.target_pde_loss_weight is None:
                self.target_pde_loss_weight = self.model.loss_weight_deeponet_pinn_loss.item()
            #print(f"Target PDE loss weight for ramping: {self.target_pde_loss_weight:.6f}")
        
        # Record initial weights
        self._record_adaptive_weights()
        
        # Training loop
        print(f"Starting training for {self.epochs} epochs...")
        start_time = time.time()
        
        best_loss = float('inf')
        
        for epoch in range(self.epochs):
            # Apply PDE loss weight ramping at the start of each epoch (before training)
            # This ensures training uses the ramped weight
            print(f"PDE loss weight before ramping: {self.model.loss_weight_deeponet_pinn_loss.item():.6f}")
            if self.use_pde_loss_ramping:
                # Ensure target weight is set (fallback to current weight if somehow not set)
                if self.target_pde_loss_weight is None:
                    self.target_pde_loss_weight = self.model.loss_weight_deeponet_pinn_loss.item()

                print(f"Target PDE loss weight for ramping: {self.target_pde_loss_weight:.6f}")
                
                if epoch < self.ramping_epochs:
                    # Linear ramping from 0 to target weight
                    # epoch 0: factor = 0, epoch ramping_epochs-1: factor = 1.0
                    ramping_factor = epoch / self.ramping_epochs
                    ramped_weight = self.target_pde_loss_weight * ramping_factor
                    self.model.loss_weight_deeponet_pinn_loss.data = torch.tensor(
                        float(ramped_weight), dtype=torch.float32, device=self.device
                    )
                else:
                    # After ramping period, ensure weight is at target
                    self.model.loss_weight_deeponet_pinn_loss.data = torch.tensor(
                        float(self.target_pde_loss_weight), dtype=torch.float32, device=self.device
                    )

                print(f"PDE loss weight after ramping: {self.model.loss_weight_deeponet_pinn_loss.item():.6f}")
            
            # Training step
            # LBFGS requires full-batch training, so route to appropriate method
            if self.optimizer_type == 'LBFGS':
                train_loss, loss_components = self._train_epoch_lbfgs(
                    train_loader, 
                    pde_points, 
                    pde_data,
                    initial_points,
                    initial_values,
                    boundary_points,
                    all_deeponet_points_stats,
                    all_pinn_points_stats
                )
            else:
                train_loss, loss_components = self._train_epoch(
                    train_loader, 
                    pde_points, 
                    pde_data,
                    initial_points,
                    initial_values,
                    boundary_points,
                    all_deeponet_points_stats,
                    all_pinn_points_stats
                )
            
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
                # Update target weight for ramping if adaptive balancing changed it
                if self.use_pde_loss_ramping:
                    self.target_pde_loss_weight = self.model.loss_weight_deeponet_pinn_loss.item()
                    # Re-apply ramping if we're still in the ramping period
                    if epoch < self.ramping_epochs:
                        ramping_factor = epoch / self.ramping_epochs
                        ramped_weight = self.target_pde_loss_weight * ramping_factor
                        self.model.loss_weight_deeponet_pinn_loss.data = torch.tensor(
                            float(ramped_weight), dtype=torch.float32, device=self.device
                        )
            
            # Record weights at every epoch (for complete history)
            # This captures the current state of weights after any potential updates
            self._record_adaptive_weights()
            
            # Update training history
            self.training_loss_history.append(train_loss)
            for key in loss_components:
                if key not in self.training_component_loss_history:
                    raise ValueError(f"Key {key} not found in training_component_loss_history")
                self.training_component_loss_history[key].append(loss_components[key])
            
            # Validation step
            val_loss, val_loss_components = self._validate_epoch(
                val_loader,
                pde_points,
                pde_data,
                initial_points,
                initial_values,
                boundary_points,
                all_deeponet_points_stats,
                all_pinn_points_stats
            )
            self.validation_loss_history.append(val_loss)
            
            # Update validation component loss history
            for key in val_loss_components:
                if key not in self.validation_component_loss_history:
                    raise ValueError(f"Key {key} not found in validation_component_loss_history")
                self.validation_component_loss_history[key].append(val_loss_components[key])

            # Update learning rate scheduler if needed
            if self.scheduler is not None and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                old_lr = self.optimizer.param_groups[0]['lr']
                old_best = self.scheduler.best if hasattr(self.scheduler, 'best') else None
                old_num_bad_epochs = self.scheduler.num_bad_epochs if hasattr(self.scheduler, 'num_bad_epochs') else None
                
                # If we're before the start epoch, don't let the scheduler track best validation loss
                # This is useful when starting from pre-trained models where validation loss may initially increase
                if epoch < self.scheduler_start_epoch:
                    # Don't call scheduler.step() before start_epoch to avoid accumulating state
                    # The scheduler will be initialized at start_epoch
                    pass  # Skip scheduler update entirely
                elif epoch == self.scheduler_start_epoch:
                    # At start epoch, initialize the scheduler's best to current validation loss
                    # This ensures the scheduler starts tracking from this point
                    self.scheduler.best = val_loss
                    self.scheduler.num_bad_epochs = 0
                    # Now call step to properly initialize the scheduler state
                    self.scheduler.step(val_loss)
                    print(f"  Learning rate scheduler: Starting to track best validation loss from epoch {self.scheduler_start_epoch}")
                    print(f"    Initial best validation loss set to: {val_loss:.6f}")
                else:
                    # Normal operation: scheduler tracks best validation loss
                    self.scheduler.step(val_loss)
                
                new_lr = self.optimizer.param_groups[0]['lr']
                new_best = self.scheduler.best if hasattr(self.scheduler, 'best') else None
                new_num_bad_epochs = self.scheduler.num_bad_epochs if hasattr(self.scheduler, 'num_bad_epochs') else None
                
                if old_lr != new_lr:
                    print(f"  Learning rate scheduler: Reduced LR from {old_lr:.2e} to {new_lr:.2e}")
                    print(f"    Best validation loss: {new_best:.6f}")
                    print(f"    Epochs without improvement: {new_num_bad_epochs}/{self.scheduler.patience}")
                else:
                    # Print scheduler status every epoch
                    if epoch < self.scheduler_start_epoch:
                        print(f"  Learning rate scheduler: Current LR = {new_lr:.2e}, Not tracking yet (start epoch: {self.scheduler_start_epoch})")
                    else:
                        print(f"  Learning rate scheduler: Current LR = {new_lr:.2e}, Best val loss = {new_best:.6f}, Epochs without improvement = {new_num_bad_epochs}/{self.scheduler.patience}")
            elif self.scheduler is not None and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step()
                new_lr = self.optimizer.param_groups[0]['lr']
                if old_lr != new_lr:
                    print(f"  Learning rate scheduler: LR changed from {old_lr:.2e} to {new_lr:.2e}")
                    
            # Print progress
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
            # Print training component losses
            print(f"  Train - Data Loss: {loss_components.get('deeponet_data_loss', 0.0):.6f}")
            if self.model.use_physics_loss:
                print(f"  Train - PDE Loss: {loss_components.get('pinn_pde_loss', 0.0):.6f}")
            if initial_points is not None:
                print(f"  Train - Initial Loss: {loss_components.get('pinn_initial_loss', 0.0):.6f}")
            if boundary_points is not None:
                print(f"  Train - Boundary Loss: {loss_components.get('pinn_boundary_loss', 0.0):.6f}")
            
            # Print validation component losses
            print(f"  Val - Data Loss: {val_loss_components.get('deeponet_data_loss', 0.0):.6f}")
            if self.model.use_physics_loss:
                print(f"  Val - PDE Loss: {val_loss_components.get('pinn_pde_loss', 0.0):.6f}")
            if initial_points is not None:
                print(f"  Val - Initial Loss: {val_loss_components.get('pinn_initial_loss', 0.0):.6f}")
            if boundary_points is not None:
                print(f"  Val - Boundary Loss: {val_loss_components.get('pinn_boundary_loss', 0.0):.6f}")
            
            # Print loss weights
            #if self.use_adaptive_loss_balancing:
            print(f"  Loss Weights - Data: {self.model.loss_weight_deeponet_data_loss.item():.6f}, "
                f"PDE: {self.model.loss_weight_deeponet_pinn_loss.item():.6f}")
                
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
                
            # Log training component losses
            self.writer.add_scalar('Loss/train_data', loss_components.get('deeponet_data_loss', 0.0), epoch)
            if self.model.use_physics_loss:
                self.writer.add_scalar('Loss/train_pde', loss_components.get('pinn_pde_loss', 0.0), epoch)
            if initial_points is not None:
                self.writer.add_scalar('Loss/train_initial', loss_components.get('pinn_initial_loss', 0.0), epoch)
            if boundary_points is not None:
                self.writer.add_scalar('Loss/train_boundary', loss_components.get('pinn_boundary_loss', 0.0), epoch)
            
            # Log validation component losses
            self.writer.add_scalar('Loss/val_data', val_loss_components.get('deeponet_data_loss', 0.0), epoch)
            if self.model.use_physics_loss:
                self.writer.add_scalar('Loss/val_pde', val_loss_components.get('pinn_pde_loss', 0.0), epoch)
            if initial_points is not None:
                self.writer.add_scalar('Loss/val_initial', val_loss_components.get('pinn_initial_loss', 0.0), epoch)
            if boundary_points is not None:
                self.writer.add_scalar('Loss/val_boundary', val_loss_components.get('pinn_boundary_loss', 0.0), epoch)
                
            # Save checkpoint
            if (epoch + 1) % self.save_freq == 0:
                self._save_checkpoint(epoch + 1)
                
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                self._save_checkpoint('best')
                
            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_loss):
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                    
        # Save final model
        self._save_checkpoint('final')
        
        # Training finished
        elapsed_time = time.time() - start_time
        print(f"Training completed in {elapsed_time:.2f} seconds")
        
        return {
            'training_loss_history': self.training_loss_history,
            'validation_loss_history': self.validation_loss_history,
            'training_component_loss_history': self.training_component_loss_history,
            'validation_component_loss_history': self.validation_component_loss_history,
            'adaptive_weight_history': self.adaptive_weight_history
        }
        
    def _train_epoch(self, train_loader, pde_points=None, pde_data=None,
                    initial_points=None, initial_values=None,
                    boundary_points=None, all_deeponet_points_stats=None, all_pinn_points_stats=None):
        """
        Train for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader.
            pde_points (torch.Tensor, optional): Points inside the domain for physics constraints.
            pde_data (torch.Tensor, optional): PDE data (zb, Sx, Sy, ManningN) for physics constraints.
            initial_points (torch.Tensor, optional): Points at initial time.
            initial_values (torch.Tensor, optional): Initial values at initial points.
            boundary_points (torch.Tensor, optional): Points on the boundary.
            all_deeponet_points_stats (dict, optional): Statistics of DeepONet points for normalization.
            all_pinn_points_stats (dict, optional): Statistics of PINN points for normalization.
            
        Returns:
            tuple: (average_loss, loss_components)
        """
        # For Adam/AdamW, use standard mini-batch training
        # Note: LBFGS is handled in the main training loop
        self.model.train()
        
        total_loss = 0
        total_components = {
            'deeponet_data_loss': 0.0,      # DeepONet loss
            'pinn_pde_loss': 0.0,       # PINN loss (include PDE and BC/IC if applicable)
            'pinn_pde_loss_cty': 0.0,
            'pinn_pde_loss_mom_x': 0.0,
            'pinn_pde_loss_mom_y': 0.0,
            'pinn_initial_loss': 0.0,
            'pinn_boundary_loss': 0.0,
            'total_loss': 0.0
        }
        
        for batch in tqdm(train_loader, desc="Training", leave=False, file=sys.stdout, dynamic_ncols=True, mininterval=0.1, ncols=None):
            
            # Get batch data (batching over cases)
            # branch_input: (batch_size_cases, n_features)
            # trunk_input: (batch_size_cases, nCells, n_coords) - DataLoader stacks the same trunk_input for each case
            # target: (batch_size_cases, nCells, n_outputs)
            branch_input, trunk_input, target = batch
            branch_input = branch_input.to(self.device)  # (batch_size_cases, n_features)
            trunk_input = trunk_input.to(self.device)  # (batch_size_cases, nCells, n_coords)
            target = target.to(self.device)  # (batch_size_cases, nCells, n_outputs)
            
            # Get the batch size for this batch (number of cases)
            batch_size_cases = branch_input.shape[0]
            nCells = trunk_input.shape[1]  # Number of cells (same for all cases)
            
            # Reshape inputs to match model expectations (point-based batching)
            # Expand branch_input: (batch_size_cases, n_features) -> (batch_size_cases * nCells, n_features)
            # Repeat each case's branch_input nCells times
            branch_input_expanded = branch_input.repeat_interleave(nCells, dim=0)  # (batch_size_cases * nCells, n_features)
            
            # Expand trunk_input: (batch_size_cases, nCells, n_coords) -> (batch_size_cases * nCells, n_coords)
            trunk_input_expanded = trunk_input.view(-1, trunk_input.shape[-1])  # (batch_size_cases * nCells, n_coords)
            
            # Flatten target: (batch_size_cases, nCells, n_outputs) -> (batch_size_cases * nCells, n_outputs)
            target_flattened = target.view(-1, target.shape[-1])  # (batch_size_cases * nCells, n_outputs)
            
            # Set PINN mesh points for PINN physics (PDEs: SWEs) constraints
            physics_branch_input = None
            physics_trunk_input = None
            physics_pde_data = None
            
            if self.model.use_physics_loss and pde_points is not None and pde_data is not None:
                # For case-based batching, we need to repeat PDE points batch_size_cases times
                # Each case in the batch needs its own set of PDE points
                # Sample random PDE points (same number for all cases, but can be different points)
                n_pde_points_per_case = len(pde_points)  # Use all PDE points

                #print(f"  Training epoch: Using {n_pde_points_per_case} PDE points per case")
                
                # Sample PDE points for each case (can use same or different points per case)
                # For now, use the same set of PDE points for all cases (but repeat for each case)
                #if n_pde_points_per_case > len(pde_points):
                    # Sample with replacement if needed
                #    physics_indices = torch.randint(0, len(pde_points), (n_pde_points_per_case,), device=pde_points.device, dtype=torch.long)
                #else:
                # Use all PDE points
                physics_indices = torch.arange(len(pde_points), device=pde_points.device, dtype=torch.long)[:n_pde_points_per_case]
                
                pde_points_sampled = pde_points[physics_indices]  # (n_pde_points_per_case, n_coords)
                pde_data_sampled = pde_data[physics_indices]  # (n_pde_points_per_case, 4)
                
                # Repeat PDE points and data for each case in the batch
                # Shape: (batch_size_cases, n_pde_points_per_case, ...) -> (batch_size_cases * n_pde_points_per_case, ...)
                physics_trunk_input = pde_points_sampled.unsqueeze(0).repeat(batch_size_cases, 1, 1)  # (batch_size_cases, n_pde_points_per_case, n_coords)
                physics_trunk_input = physics_trunk_input.view(-1, physics_trunk_input.shape[-1])  # (batch_size_cases * n_pde_points_per_case, n_coords)
                
                physics_pde_data = pde_data_sampled.unsqueeze(0).repeat(batch_size_cases, 1, 1)  # (batch_size_cases, n_pde_points_per_case, 4)
                physics_pde_data = physics_pde_data.view(-1, physics_pde_data.shape[-1])  # (batch_size_cases * n_pde_points_per_case, 4)
                
                # Repeat branch_input for each PDE point (same branch_input for all PDE points of a case)
                physics_branch_input = branch_input.repeat_interleave(n_pde_points_per_case, dim=0)  # (batch_size_cases * n_pde_points_per_case, n_features)
            
            # Sample initial points if provided
            initial_trunk_input = None
            initial_batch_values = None
            if initial_points is not None and initial_values is not None:
                initial_indices = torch.randint(0, len(initial_points), (batch_size_cases,))
                initial_trunk_input = initial_points[initial_indices]
                initial_batch_values = initial_values[initial_indices]
                
            # Sample boundary points if provided
            # Note: For complex boundary conditions (inlet-q, exit-h, wall), 
            # a more sophisticated sampling approach may be needed
            boundary_trunk_input = None
            boundary_batch_values = None
            if boundary_points is not None:
                boundary_indices = torch.randint(0, len(boundary_points), (batch_size_cases,))
                boundary_trunk_input = boundary_points[boundary_indices]
                # For now, we assume boundary_values would be provided separately
                # This can be extended later for more complex boundary conditions
                
            # Forward pass and compute loss            
            self.optimizer.zero_grad()

            total_batch_loss, loss_components = self.model.compute_total_loss(
                branch_input_expanded,
                trunk_input_expanded,
                target=target_flattened,
                physics_branch_input=physics_branch_input,
                physics_trunk_input=physics_trunk_input,
                pde_data=physics_pde_data,                
                all_deeponet_points_stats=all_deeponet_points_stats,
                all_pinn_points_stats=all_pinn_points_stats
            )
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping (to avoid exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update weights (Adam/AdamW)
            self.optimizer.step()
            
            # Extract loss value before deleting tensor
            batch_loss_value = total_batch_loss.item()
            
            # Explicitly delete intermediate tensors to help free memory
            del total_batch_loss
            if 'physics_indices' in locals() and physics_indices is not None:
                del physics_indices
            
            # Update total loss
            total_loss += batch_loss_value
            
            # Update component losses
            for key in total_components:
                if key in loss_components:
                    total_components[key] += loss_components[key]
                
        # Compute average losses
        avg_loss = total_loss / len(train_loader)
        avg_components = {key: total_components[key] / len(train_loader) for key in total_components}
        
        return avg_loss, avg_components
    
    def _train_epoch_lbfgs(self, train_loader, pde_points=None, pde_data=None,
                           initial_points=None, initial_values=None,
                           boundary_points=None, all_deeponet_points_stats=None, all_pinn_points_stats=None):
        """
        Train for one epoch using LBFGS (full-batch optimizer).
        
        LBFGS requires the full dataset, so we concatenate all batches into a single batch.
        
        Args:
            train_loader (DataLoader): Training data loader.
            pde_points (torch.Tensor, optional): Points inside the domain for physics constraints.
            pde_data (torch.Tensor, optional): PDE data (zb, Sx, Sy, ManningN) for physics constraints.
            initial_points (torch.Tensor, optional): Points at initial time.
            initial_values (torch.Tensor, optional): Initial values at initial points.
            boundary_points (torch.Tensor, optional): Points on the boundary.
            all_deeponet_points_stats (dict, optional): Statistics of DeepONet points for normalization.
            all_pinn_points_stats (dict, optional): Statistics of PINN points for normalization.
            
        Returns:
            tuple: (average_loss, loss_components)
        """
        self.model.train()
        
        # Concatenate all batches into a single full batch for LBFGS
        # Note: Batches are case-based, so we concatenate cases
        print("  LBFGS: Concatenating all batches into full batch...")
        all_branch_inputs = []
        all_trunk_inputs = []
        all_targets = []
        
        for batch in train_loader:
            # Get batch data (batching over cases)
            # branch_input: (batch_size_cases, n_features)
            # trunk_input: (batch_size_cases, nCells, n_coords)
            # target: (batch_size_cases, nCells, n_outputs)
            branch_input, trunk_input, target = batch
            all_branch_inputs.append(branch_input)
            all_trunk_inputs.append(trunk_input)
            all_targets.append(target)
        
        # Concatenate all batches (case-based)
        full_branch_input = torch.cat(all_branch_inputs, dim=0).to(self.device)  # (n_all_cases, n_features)
        full_trunk_input = torch.cat(all_trunk_inputs, dim=0).to(self.device)  # (n_all_cases, nCells, n_coords)
        full_target = torch.cat(all_targets, dim=0).to(self.device)  # (n_all_cases, nCells, n_outputs)
        
        n_all_cases = full_branch_input.shape[0]
        nCells = full_trunk_input.shape[1]
        print(f"  LBFGS: Full batch - Cases: {n_all_cases}, Cells per case: {nCells}, Total points: {n_all_cases * nCells}")
        
        # Reshape inputs to match model expectations (point-based batching)
        # Expand branch_input: (n_all_cases, n_features) -> (n_all_cases * nCells, n_features)
        branch_input_expanded = full_branch_input.repeat_interleave(nCells, dim=0)  # (n_all_cases * nCells, n_features)
        
        # Expand trunk_input: (n_all_cases, nCells, n_coords) -> (n_all_cases * nCells, n_coords)
        trunk_input_expanded = full_trunk_input.view(-1, full_trunk_input.shape[-1])  # (n_all_cases * nCells, n_coords)
        
        # Flatten target: (n_all_cases, nCells, n_outputs) -> (n_all_cases * nCells, n_outputs)
        target_flattened = full_target.view(-1, full_target.shape[-1])  # (n_all_cases * nCells, n_outputs)
        
        # Set PINN mesh points for PINN physics (PDEs: SWEs) constraints
        physics_branch_input = None
        physics_trunk_input = None
        physics_pde_data = None
        
        if self.model.use_physics_loss and pde_points is not None and pde_data is not None:
            # For case-based batching, we need to repeat PDE points n_all_cases times
            # Each case in the batch needs its own set of PDE points
            # Use all PDE points for each case
            n_pde_points_per_case = len(pde_points)  # Use all PDE points

            #print(f"  LBFGS: Using {n_pde_points_per_case} PDE points per case")
            
            # Use all PDE points
            physics_indices = torch.arange(len(pde_points), device=pde_points.device, dtype=torch.long)[:n_pde_points_per_case]
            
            pde_points_sampled = pde_points[physics_indices]  # (n_pde_points_per_case, n_coords)
            pde_data_sampled = pde_data[physics_indices]  # (n_pde_points_per_case, 4)
            
            # Repeat PDE points and data for each case in the batch
            # Shape: (n_all_cases, n_pde_points_per_case, ...) -> (n_all_cases * n_pde_points_per_case, ...)
            physics_trunk_input = pde_points_sampled.unsqueeze(0).repeat(n_all_cases, 1, 1)  # (n_all_cases, n_pde_points_per_case, n_coords)
            physics_trunk_input = physics_trunk_input.view(-1, physics_trunk_input.shape[-1])  # (n_all_cases * n_pde_points_per_case, n_coords)
            
            physics_pde_data = pde_data_sampled.unsqueeze(0).repeat(n_all_cases, 1, 1)  # (n_all_cases, n_pde_points_per_case, 4)
            physics_pde_data = physics_pde_data.view(-1, physics_pde_data.shape[-1])  # (n_all_cases * n_pde_points_per_case, 4)
            
            # Repeat branch_input for each PDE point (same branch_input for all PDE points of a case)
            physics_branch_input = full_branch_input.repeat_interleave(n_pde_points_per_case, dim=0)  # (n_all_cases * n_pde_points_per_case, n_features)
        
        # Sample initial points if provided
        initial_trunk_input = None
        initial_batch_values = None
        if initial_points is not None and initial_values is not None:
            initial_indices = torch.randint(0, len(initial_points), (n_all_cases,), device=initial_points.device)
            initial_trunk_input = initial_points[initial_indices]
            initial_batch_values = initial_values[initial_indices]
        
        # Sample boundary points if provided
        boundary_trunk_input = None
        boundary_batch_values = None
        if boundary_points is not None:
            boundary_indices = torch.randint(0, len(boundary_points), (n_all_cases,), device=boundary_points.device)
            boundary_trunk_input = boundary_points[boundary_indices]
        
        # LBFGS requires a closure function
        closure_call_count = [0]  # Use list to allow modification in closure
        closure_losses = []  # Track losses during LBFGS iterations
        
        def closure():
            closure_call_count[0] += 1
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss on full batch (using point-based reshaped inputs)
            loss, loss_components = self.model.compute_total_loss(
                branch_input_expanded,
                trunk_input_expanded,
                target=target_flattened,
                physics_branch_input=physics_branch_input,
                physics_trunk_input=physics_trunk_input,
                pde_data=physics_pde_data,
                all_deeponet_points_stats=all_deeponet_points_stats,
                all_pinn_points_stats=all_pinn_points_stats
            )
            
            loss_value = loss.item()
            closure_losses.append(loss_value)
            
            # Backward pass
            loss.backward()
            
            # Debug: Check gradient magnitudes on first call and track loss changes
            if closure_call_count[0] == 1:
                total_grad_norm = 0.0
                max_grad = 0.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        param_grad_norm = param.grad.data.norm(2)
                        total_grad_norm += param_grad_norm.item() ** 2
                        max_grad = max(max_grad, param.grad.data.abs().max().item())
                total_grad_norm = total_grad_norm ** 0.5
                print(f"    LBFGS closure call {closure_call_count[0]}: loss = {loss_value:.9f}")
                print(f"    Gradient norm: {total_grad_norm:.6e}, Max gradient: {max_grad:.6e}")
                print(f"    Tolerance check: grad_norm <= {self.lbfgs_tolerance_grad}? {total_grad_norm <= self.lbfgs_tolerance_grad}")
                print(f"    Learning rate: {self.learning_rate}")
            else:
                # Print loss every call with change from previous
                prev_loss = closure_losses[-2] if len(closure_losses) > 1 else initial_loss_value
                loss_delta = loss_value - prev_loss
                direction = "↑" if loss_delta > 0 else "↓" if loss_delta < 0 else "="
                print(f"    LBFGS closure call {closure_call_count[0]}: loss = {loss_value:.9f} ({direction} {loss_delta:+.9f})")
            
            return loss
        
        # Single optimizer step for the full batch
        print("  LBFGS: Running optimizer step on full batch...")
        initial_loss = closure()  # Get initial loss before optimization
        initial_loss_value = initial_loss.item()
        print(f"    Initial loss before LBFGS step: {initial_loss_value:.6f}")
        
        # Store initial parameter values for comparison
        initial_params = [p.data.clone() for p in self.model.parameters() if p.requires_grad]
        
        final_loss = self.optimizer.step(closure)
        
        print(f"    LBFGS closure called {closure_call_count[0]} times")
        
        # Get the actual final loss from the last closure call (more accurate than optimizer.step() return value)
        actual_final_loss = closure_losses[-1] if closure_losses else initial_loss_value
        
        if final_loss is not None:
            final_loss_value = final_loss.item()
            loss_change = initial_loss_value - final_loss_value
            actual_loss_change = initial_loss_value - actual_final_loss
            
            # Show loss progression
            if len(closure_losses) > 1:
                min_loss = min(closure_losses)
                max_loss = max(closure_losses)
                print(f"    Loss during LBFGS: min={min_loss:.9f}, max={max_loss:.9f}, range={max_loss-min_loss:.9f}")
                print(f"    First 5 losses: {[f'{l:.9f}' for l in closure_losses[:5]]}")
                if len(closure_losses) > 5:
                    print(f"    Last 5 losses: {[f'{l:.9f}' for l in closure_losses[-5:]]}")
            
            print(f"    Final loss (from optimizer.step): {final_loss_value:.9f}")
            print(f"    Final loss (from last closure call): {actual_final_loss:.9f}")
            print(f"    Loss change (optimizer): {loss_change:.9f}")
            print(f"    Loss change (actual): {actual_loss_change:.9f}")
            
            # Check if line search rejected the step
            if abs(final_loss_value - initial_loss_value) < 1e-8 and abs(actual_final_loss - initial_loss_value) > 1e-6:
                print(f"    WARNING: Line search may have rejected steps! Final loss reverted to initial.")
                if hasattr(self, 'lbfgs_line_search_fn') and self.lbfgs_line_search_fn == "strong_wolfe":
                    print(f"      This suggests 'strong_wolfe' line search is too strict.")
                    print(f"      Consider: line_search_fn: null (disable line search)")
                else:
                    print(f"      LBFGS backtracking line search rejected the step (loss got worse).")
                    print(f"      Consider: increasing learning rate or relaxing tolerance_change")
            
            # Check if parameters actually changed
            param_changes = []
            for i, (init_param, curr_param) in enumerate(zip(initial_params, [p.data for p in self.model.parameters() if p.requires_grad])):
                param_change = (curr_param - init_param).abs().max().item()
                param_changes.append(param_change)
            max_param_change = max(param_changes) if param_changes else 0.0
            print(f"    Max parameter change: {max_param_change:.6e}")
            
            if loss_change == 0.0 and max_param_change < 1e-8:
                print(f"    WARNING: LBFGS made no progress! Consider:")
                print(f"      - Increasing learning rate (current: {self.learning_rate})")
                if hasattr(self, 'lbfgs_tolerance_change'):
                    print(f"      - Relaxing tolerance_change (current: {self.lbfgs_tolerance_change:.2e}, try 1e-07)")
                if hasattr(self, 'lbfgs_max_iter'):
                    print(f"      - Increasing max_iter (current: {self.lbfgs_max_iter})")
                if hasattr(self, 'lbfgs_history_size'):
                    print(f"      - Increasing history_size (current: {self.lbfgs_history_size})")
        else:
            print(f"    Final loss: None (LBFGS may have converged or hit tolerance)")
        
        # Compute loss components for reporting (run one more forward pass)
        # Use the reshaped point-based inputs, not the case-based ones
        # Note: We don't use torch.no_grad() here because PDE residual computation
        # requires gradients. We won't call backward() or optimizer.step() after this.
        _, loss_components = self.model.compute_total_loss(
            branch_input_expanded,
            trunk_input_expanded,
            target=target_flattened,
            physics_branch_input=physics_branch_input,
            physics_trunk_input=physics_trunk_input,
            pde_data=physics_pde_data,
            all_deeponet_points_stats=all_deeponet_points_stats,
            all_pinn_points_stats=all_pinn_points_stats
        )
        
        # Convert loss components to Python floats
        avg_components = {
            'deeponet_data_loss': loss_components.get('deeponet_data_loss', 0.0),
            'pinn_pde_loss': loss_components.get('pinn_pde_loss', 0.0),
            'pinn_pde_loss_cty': loss_components.get('pinn_pde_loss_cty', 0.0),
            'pinn_pde_loss_mom_x': loss_components.get('pinn_pde_loss_mom_x', 0.0),
            'pinn_pde_loss_mom_y': loss_components.get('pinn_pde_loss_mom_y', 0.0),
            'pinn_initial_loss': loss_components.get('pinn_initial_loss', 0.0),
            'pinn_boundary_loss': loss_components.get('pinn_boundary_loss', 0.0),
            'total_loss': final_loss.item() if final_loss is not None else loss_components.get('total_loss', 0.0)
        }
        
        avg_loss = avg_components['total_loss']
        
        return avg_loss, avg_components
        
    def _validate_epoch(self, val_loader, pde_points=None, pde_data=None,
                    initial_points=None, initial_values=None,
                    boundary_points=None, all_deeponet_points_stats=None, all_pinn_points_stats=None):
        """
        Validate for one epoch.
        
        Args:
            val_loader (DataLoader): Validation data loader.
            pde_points (torch.Tensor, optional): Points inside the domain for physics constraints.
            pde_data (torch.Tensor, optional): PDE data (zb, Sx, Sy, ManningN) for physics constraints.
            initial_points (torch.Tensor, optional): Points at initial time.
            initial_values (torch.Tensor, optional): Initial values at initial points.
            boundary_points (torch.Tensor, optional): Points on the boundary.
            all_deeponet_points_stats (dict, optional): Statistics of DeepONet points for normalization.
            all_pinn_points_stats (dict, optional): Statistics of PINN points for normalization.
            
        Returns:
            tuple: (average_loss, loss_components)
        """
        self.model.eval()
        
        total_loss = 0
        total_components = {
            'deeponet_data_loss': 0.0,      # DeepONet loss
            'pinn_pde_loss': 0.0,       # PINN loss (include PDE and BC/IC if applicable)
            'pinn_pde_loss_cty': 0.0,
            'pinn_pde_loss_mom_x': 0.0,
            'pinn_pde_loss_mom_y': 0.0,
            'pinn_initial_loss': 0.0,
            'pinn_boundary_loss': 0.0,
            'total_loss': 0.0
        }
        
        # Note: We don't use torch.no_grad() here because PDE residual computation
        # requires gradients even during validation. We're still in eval() mode and
        # won't update parameters since we don't call backward() or optimizer.step()
        for batch in tqdm(val_loader, desc="Validating", leave=False, file=sys.stdout, dynamic_ncols=True, mininterval=0.1, ncols=None):
                # Get batch data (batching over cases)
                branch_input, trunk_input, target = batch
                branch_input = branch_input.to(self.device)  # (batch_size_cases, n_features)
                trunk_input = trunk_input.to(self.device)  # (batch_size_cases, nCells, n_coords)
                target = target.to(self.device)  # (batch_size_cases, nCells, n_outputs)
                
                # Get the batch size for this batch (number of cases)
                batch_size_cases = branch_input.shape[0]
                nCells = trunk_input.shape[1]  # Number of cells (same for all cases)
                
                # Reshape inputs to match model expectations (point-based batching)
                # Expand branch_input: (batch_size_cases, n_features) -> (batch_size_cases * nCells, n_features)
                branch_input_expanded = branch_input.repeat_interleave(nCells, dim=0)  # (batch_size_cases * nCells, n_features)
                
                # Expand trunk_input: (batch_size_cases, nCells, n_coords) -> (batch_size_cases * nCells, n_coords)
                trunk_input_expanded = trunk_input.view(-1, trunk_input.shape[-1])  # (batch_size_cases * nCells, n_coords)
                
                # Flatten target: (batch_size_cases, nCells, n_outputs) -> (batch_size_cases * nCells, n_outputs)
                target_flattened = target.view(-1, target.shape[-1])  # (batch_size_cases * nCells, n_outputs)
                
                # Randomly sample PINN mesh points for PINN physics (PDEs: SWEs) constraints
                physics_branch_input = None
                physics_trunk_input = None
                physics_pde_data = None
                
                if self.model.use_physics_loss and pde_points is not None and pde_data is not None:
                    # For case-based batching, we need to repeat PDE points batch_size_cases times
                    n_pde_points_per_case = len(pde_points)  # Use all PDE points

                    #print(f"  Validation epoch: Using {n_pde_points_per_case} PDE points per case")
                    
                    # Sample PDE points (use all for validation)
                    physics_indices = torch.arange(len(pde_points), device=pde_points.device, dtype=torch.long)[:n_pde_points_per_case]
                    
                    pde_points_sampled = pde_points[physics_indices]  # (n_pde_points_per_case, n_coords)
                    pde_data_sampled = pde_data[physics_indices]  # (n_pde_points_per_case, 4)
                    
                    # Repeat PDE points and data for each case in the batch
                    physics_trunk_input = pde_points_sampled.unsqueeze(0).repeat(batch_size_cases, 1, 1)  # (batch_size_cases, n_pde_points_per_case, n_coords)
                    physics_trunk_input = physics_trunk_input.view(-1, physics_trunk_input.shape[-1])  # (batch_size_cases * n_pde_points_per_case, n_coords)
                    
                    physics_pde_data = pde_data_sampled.unsqueeze(0).repeat(batch_size_cases, 1, 1)  # (batch_size_cases, n_pde_points_per_case, 4)
                    physics_pde_data = physics_pde_data.view(-1, physics_pde_data.shape[-1])  # (batch_size_cases * n_pde_points_per_case, 4)
                    
                    # Repeat branch_input for each PDE point
                    physics_branch_input = branch_input.repeat_interleave(n_pde_points_per_case, dim=0)  # (batch_size_cases * n_pde_points_per_case, n_features)
                
                # Sample initial points if provided
                initial_trunk_input = None
                initial_batch_values = None
                if initial_points is not None and initial_values is not None:
                    initial_indices = torch.randint(0, len(initial_points), (batch_size_cases,))
                    initial_trunk_input = initial_points[initial_indices]
                    initial_batch_values = initial_values[initial_indices]
                    
                # Sample boundary points if provided
                boundary_trunk_input = None
                boundary_batch_values = None
                if boundary_points is not None:
                    boundary_indices = torch.randint(0, len(boundary_points), (batch_size_cases,))
                    boundary_trunk_input = boundary_points[boundary_indices]
                
                # Forward pass and compute loss
                total_batch_loss, loss_components = self.model.compute_total_loss(
                    branch_input_expanded,
                    trunk_input_expanded,
                    target=target_flattened,
                    physics_branch_input=physics_branch_input,
                    physics_trunk_input=physics_trunk_input,
                    pde_data=physics_pde_data,
                    all_deeponet_points_stats=all_deeponet_points_stats,
                    all_pinn_points_stats=all_pinn_points_stats
                )
                
                # Update total loss
                total_loss += total_batch_loss.item()
                
                # Update component losses
                for key in total_components:
                    if key in loss_components:
                        total_components[key] += loss_components[key]
                
        # Compute average losses
        avg_loss = total_loss / len(val_loader)
        avg_components = {key: total_components[key] / len(val_loader) for key in total_components}
        
        return avg_loss, avg_components
        
    def _save_checkpoint(self, epoch):
        """
        Save a model checkpoint.
        
        Args:
            epoch (int or str): Current epoch or 'final' or 'best'.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"pi_deeponet_epoch_{epoch}.pt")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_loss_history': self.training_loss_history,
            'validation_loss_history': self.validation_loss_history,
            'training_component_loss_history': self.training_component_loss_history,
            'validation_component_loss_history': self.validation_component_loss_history,
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
            
        print(f"Checkpoint saved to {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path, mode):
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file.
            mode (str): Loading mode. Options:
                - 'initial_condition': Only load model weights, ignore optimizer, scheduler, and history.
                                      Use this when you want to use pre-trained weights as initialization
                                      but start training from scratch with new hyperparameters.
                - 'resume_training': Load everything (model weights, optimizer state, scheduler state, history).
                                    Use this when you want to continue training from where you left off.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        if mode not in ['initial_condition', 'resume_training']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'initial_condition' or 'resume_training'")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Always load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Verify weights were loaded correctly
        first_weight_after_load = next(iter(self.model.branch_net.net[0].fc.parameters()))
        print(f"Model weights after loading checkpoint: first weight = {first_weight_after_load.data[0, 0].item():.8f}")
        
        if mode == 'initial_condition':
            # Only load model weights, ignore everything else
            # This allows starting fresh training with new hyperparameters
            print(f"Loaded model weights from {checkpoint_path} (initial condition mode - optimizer, scheduler, and history ignored)")
        elif mode == 'resume_training':
            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Update learning rate from config to override checkpoint learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
                print(f"Loaded optimizer state (learning rate updated to {self.learning_rate} from config)")
            else:
                print("Warning: No optimizer state found in checkpoint")
            
            # Load scheduler state
            if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("Loaded scheduler state")
            elif 'scheduler_state_dict' in checkpoint and self.scheduler is None:
                print("Warning: Scheduler state found in checkpoint but no scheduler is configured")
            
            # Load history
            self.training_loss_history = checkpoint.get('training_loss_history', [])
            self.validation_loss_history = checkpoint.get('validation_loss_history', [])
            self.training_component_loss_history = checkpoint.get('training_component_loss_history', {
                'deeponet_data_loss': [],
                'pinn_pde_loss': [],
                'pinn_pde_loss_cty': [],
                'pinn_pde_loss_mom_x': [],
                'pinn_pde_loss_mom_y': [],
                'pinn_initial_loss': [],
                'pinn_boundary_loss': [],
                'total_loss': []
            })
            self.validation_component_loss_history = checkpoint.get('validation_component_loss_history', {
                'deeponet_data_loss': [],
                'pinn_pde_loss': [],
                'pinn_pde_loss_cty': [],
                'pinn_pde_loss_mom_x': [],
                'pinn_pde_loss_mom_y': [],
                'pinn_initial_loss': [],
                'pinn_boundary_loss': [],
                'total_loss': []
            })
            print(f"Loaded training history from {checkpoint_path} (resume training mode)")
        
    def predict(self, branch_input, trunk_input):
        """
        Make predictions with the model.
        
        Args:
            branch_input (torch.Tensor or numpy.ndarray): Input function for branch net.
            trunk_input (torch.Tensor or numpy.ndarray): Coordinates for trunk net.
            
        Returns:
            numpy.ndarray: Model predictions.
        """
        self.model.eval()
        
        # Convert to torch tensors if needed
        if isinstance(branch_input, np.ndarray):
            branch_input = torch.tensor(branch_input, dtype=torch.float32)
        if isinstance(trunk_input, np.ndarray):
            trunk_input = torch.tensor(trunk_input, dtype=torch.float32)
            
        # Move to device
        branch_input = branch_input.to(self.device)
        trunk_input = trunk_input.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(branch_input, trunk_input)
            
        # Convert to numpy
        output = output.cpu().numpy()
        
        return output
    
    def _compute_initial_loss_magnitudes(self, train_loader, pde_points, pde_data, 
                                         all_deeponet_points_stats, all_pinn_points_stats):
        """
        Compute initial loss magnitudes for adaptive loss balancing.
        
        This runs a few forward passes to estimate the typical magnitudes of different loss components.
        """
        self.model.eval()
        
        
        #num_samples = min(5, len(train_loader)) #only use the first 5 samples to compute the initial loss magnitudes
        num_samples = len(train_loader)   # Use all the samples to compute the initial loss magnitudes
        data_losses = []
        pde_losses = []
        pde_continuity_losses = []
        pde_momentum_x_losses = []
        pde_momentum_y_losses = []
        
        # Accumulate total squared error and total samples for consistent loss computation across batch sizes
        total_squared_error = 0.0
        total_samples = 0
        
        # Note: We need gradients enabled for PDE loss computation (it uses autograd.grad),
        # but we don't call backward() so no gradients are accumulated
        for i, batch in enumerate(train_loader):
            if i >= num_samples:
                break
                
            # Get batch data (batching over cases)
            branch_input, trunk_input, target = batch
            branch_input = branch_input.to(self.device)  # (batch_size_cases, n_features)
            trunk_input = trunk_input.to(self.device)  # (batch_size_cases, nCells, n_coords)
            target = target.to(self.device)  # (batch_size_cases, nCells, n_outputs)
            
            # Get the batch size for this batch (number of cases)
            batch_size_cases = branch_input.shape[0]
            nCells = trunk_input.shape[1]  # Number of cells (same for all cases)
            
            # Reshape inputs to match model expectations (point-based batching)
            # Expand branch_input: (batch_size_cases, n_features) -> (batch_size_cases * nCells, n_features)
            branch_input_expanded = branch_input.repeat_interleave(nCells, dim=0)  # (batch_size_cases * nCells, n_features)
            
            # Expand trunk_input: (batch_size_cases, nCells, n_coords) -> (batch_size_cases * nCells, n_coords)
            trunk_input_expanded = trunk_input.view(-1, trunk_input.shape[-1])  # (batch_size_cases * nCells, n_coords)
            
            # Flatten target: (batch_size_cases, nCells, n_outputs) -> (batch_size_cases * nCells, n_outputs)
            target_flattened = target.view(-1, target.shape[-1])  # (batch_size_cases * nCells, n_outputs)
            
            # Compute data loss (no gradients needed, but doesn't hurt)
            with torch.no_grad():
                # Verify model is in eval mode
                if not self.model.training == False:
                    print(f"  WARNING: Model is not in eval mode! Setting to eval mode...")
                    self.model.eval()
                
                data_loss, _ = self.model.compute_deeponet_data_loss(branch_input_expanded, trunk_input_expanded, target_flattened, all_deeponet_points_stats)
                
                data_losses.append(data_loss.item())              
            
            # Compute PDE loss if enabled (gradients needed for autograd.grad in compute_pde_residuals)
            if self.model.use_physics_loss and pde_points is not None and pde_data is not None:
                # For case-based batching, we need to repeat PDE points batch_size_cases times
                n_pde_points_per_case = len(pde_points)  # Use all PDE points
                
                # Sample PDE points (use all for initial loss computation)
                physics_indices = torch.arange(len(pde_points), device=pde_points.device, dtype=torch.long)[:n_pde_points_per_case]
                
                pde_points_sampled = pde_points[physics_indices]  # (n_pde_points_per_case, n_coords)
                pde_data_sampled = pde_data[physics_indices]  # (n_pde_points_per_case, 4)
                
                # Repeat PDE points and data for each case in the batch
                physics_trunk_input = pde_points_sampled.unsqueeze(0).repeat(batch_size_cases, 1, 1)  # (batch_size_cases, n_pde_points_per_case, n_coords)
                physics_trunk_input = physics_trunk_input.view(-1, physics_trunk_input.shape[-1])  # (batch_size_cases * n_pde_points_per_case, n_coords)
                
                physics_pde_data = pde_data_sampled.unsqueeze(0).repeat(batch_size_cases, 1, 1)  # (batch_size_cases, n_pde_points_per_case, 4)
                physics_pde_data = physics_pde_data.view(-1, physics_pde_data.shape[-1])  # (batch_size_cases * n_pde_points_per_case, 4)
                
                # Repeat branch_input for each PDE point
                physics_branch_input = branch_input.repeat_interleave(n_pde_points_per_case, dim=0)  # (batch_size_cases * n_pde_points_per_case, n_features)
                
                # Enable gradients for PDE loss computation
                pde_loss, pde_loss_components, _, _, _ = self.model.compute_pde_loss(
                    physics_branch_input, physics_trunk_input, physics_pde_data,
                    all_deeponet_points_stats, all_pinn_points_stats
                )
                pde_losses.append(pde_loss.detach().item())
                
                # Store PDE component losses for adaptive balancing
                if self.use_adaptive_pde_component_balancing:
                    pde_continuity_losses.append(pde_loss_components['continuity_loss'])
                    pde_momentum_x_losses.append(pde_loss_components['momentum_x_loss'])
                    pde_momentum_y_losses.append(pde_loss_components['momentum_y_loss'])
        
        # Store initial magnitudes (mean of the losses across all batches-to be consistent with the loss computation in the training loop)
        if data_losses:
            self.initial_loss_magnitudes['deeponet_data_loss'] = np.mean(data_losses)           
        if pde_losses:
            self.initial_loss_magnitudes['pinn_pde_loss'] = np.mean(pde_losses)
        
        # Store PDE component loss magnitudes (mean of the losses across all batches-to be consistent with the loss computation in the training loop)
        if self.use_adaptive_pde_component_balancing:
            if pde_continuity_losses:
                self.initial_loss_magnitudes['pde_continuity_loss'] = np.mean(pde_continuity_losses)
            if pde_momentum_x_losses:
                self.initial_loss_magnitudes['pde_momentum_x_loss'] = np.mean(pde_momentum_x_losses)
            if pde_momentum_y_losses:
                self.initial_loss_magnitudes['pde_momentum_y_loss'] = np.mean(pde_momentum_y_losses)
        
        self.model.train()
        
        print(f"Initial loss magnitudes: {self.initial_loss_magnitudes}")
    
    def _update_loss_weights_from_balancing(self):
        """
        Update loss weights based on computed balancing factors.
        
        The balancing factor for each loss is computed as:
        balancing_factor = reference_loss_magnitude / loss_magnitude
        
        This ensures losses are on similar scales.
        
        Note: Uses current loss magnitudes (stored in self.initial_loss_magnitudes,
        which are updated by _update_adaptive_loss_weights() during training).
        """
        if not self.initial_loss_magnitudes:
            return
        
        # Use data loss as reference (typically the smallest)
        # Use the current magnitude (which may have been updated by adaptive balancing)
        reference_magnitude = self.initial_loss_magnitudes.get('deeponet_data_loss', 1.0)
        
        # Compute balancing factors based on current magnitudes
        if 'deeponet_data_loss' in self.initial_loss_magnitudes:
            data_magnitude = self.initial_loss_magnitudes['deeponet_data_loss']
            if data_magnitude > 0:
                # Data loss weight should be 1.0 (reference)
                self.loss_balancing_factors['deeponet_data_loss'] = reference_magnitude / data_magnitude
            else:
                self.loss_balancing_factors['deeponet_data_loss'] = 1.0
        
        if 'pinn_pde_loss' in self.initial_loss_magnitudes:
            pde_magnitude = self.initial_loss_magnitudes['pinn_pde_loss']
            if pde_magnitude > 0:
                # Balance PDE loss to match data loss scale
                self.loss_balancing_factors['pinn_pde_loss'] = reference_magnitude / pde_magnitude
            else:
                self.loss_balancing_factors['pinn_pde_loss'] = 1.0
        
        # Apply balancing factors to model's loss weights
        base_data_weight = self.config.get("training.loss_weights.deeponet.data_loss", 1.0)
        base_pde_weight = self.config.get("training.loss_weights.deeponet.pinn_loss", 1.0)
        
        # Debug: print current magnitudes and balancing factors
        if 'deeponet_data_loss' in self.initial_loss_magnitudes and 'pinn_pde_loss' in self.initial_loss_magnitudes:
            print(f"  Adaptive balancing - Data loss magnitude: {self.initial_loss_magnitudes['deeponet_data_loss']:.6f}, "
                  f"PDE loss magnitude: {self.initial_loss_magnitudes['pinn_pde_loss']:.6f}")
            if 'pinn_pde_loss' in self.loss_balancing_factors:
                print(f"  Adaptive balancing - Balancing factor: {self.loss_balancing_factors['pinn_pde_loss']:.6f}")
        
        if 'deeponet_data_loss' in self.loss_balancing_factors:
            new_data_weight = base_data_weight * self.loss_balancing_factors['deeponet_data_loss']
            self.model.loss_weight_deeponet_data_loss.data = torch.tensor(
                float(new_data_weight), dtype=torch.float32, device=self.device
            )
        
        if 'pinn_pde_loss' in self.loss_balancing_factors:
            new_pde_weight = base_pde_weight * self.loss_balancing_factors['pinn_pde_loss']
            self.model.loss_weight_deeponet_pinn_loss.data = torch.tensor(
                float(new_pde_weight), dtype=torch.float32, device=self.device
            )
            # Update target weight for ramping if enabled
            if self.use_pde_loss_ramping:
                self.target_pde_loss_weight = new_pde_weight
        
        # Balance PDE component losses if enabled
        if self.use_adaptive_pde_component_balancing:
            # Use the largest PDE component loss as reference to balance them
            pde_component_magnitudes = {}
            if 'pde_continuity_loss' in self.initial_loss_magnitudes:
                pde_component_magnitudes['continuity'] = self.initial_loss_magnitudes['pde_continuity_loss']
            if 'pde_momentum_x_loss' in self.initial_loss_magnitudes:
                pde_component_magnitudes['momentum_x'] = self.initial_loss_magnitudes['pde_momentum_x_loss']
            if 'pde_momentum_y_loss' in self.initial_loss_magnitudes:
                pde_component_magnitudes['momentum_y'] = self.initial_loss_magnitudes['pde_momentum_y_loss']
            
            if pde_component_magnitudes:
                # Use the median magnitude as reference (more robust than max)
                reference_pde_magnitude = np.median(list(pde_component_magnitudes.values()))
                
                # Compute and apply balancing factors
                base_continuity_weight = self.config.get("training.loss_weights.pde.continuity", 1.0)
                base_momentum_x_weight = self.config.get("training.loss_weights.pde.momentum_x", 1.0)
                base_momentum_y_weight = self.config.get("training.loss_weights.pde.momentum_y", 1.0)
                
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
        
        print(f"Updated loss weights - Data: {self.model.loss_weight_deeponet_data_loss.item():.6f}, "
              f"PDE: {self.model.loss_weight_deeponet_pinn_loss.item():.6f}")
    
    def _update_adaptive_loss_weights(self, loss_components, epoch):
        """
        Update loss weights adaptively during training.
        
        This uses exponential moving average to smooth the updates.
        """
        alpha = 0.1  # Smoothing factor
        
        # Update magnitudes with exponential moving average
        if 'deeponet_data_loss' in loss_components:
            current_magnitude = loss_components['deeponet_data_loss']
            if 'deeponet_data_loss' not in self.initial_loss_magnitudes:
                self.initial_loss_magnitudes['deeponet_data_loss'] = current_magnitude
            else:
                self.initial_loss_magnitudes['deeponet_data_loss'] = (
                    alpha * current_magnitude + (1 - alpha) * self.initial_loss_magnitudes['deeponet_data_loss']
                )
        
        if 'pinn_pde_loss' in loss_components:
            current_magnitude = loss_components['pinn_pde_loss']
            if 'pinn_pde_loss' not in self.initial_loss_magnitudes:
                self.initial_loss_magnitudes['pinn_pde_loss'] = current_magnitude
            else:
                self.initial_loss_magnitudes['pinn_pde_loss'] = (
                    alpha * current_magnitude + (1 - alpha) * self.initial_loss_magnitudes['pinn_pde_loss']
                )
        
        # Update PDE component loss magnitudes if adaptive balancing is enabled
        if self.use_adaptive_pde_component_balancing:
            if 'pinn_pde_loss_cty' in loss_components:
                current_magnitude = loss_components['pinn_pde_loss_cty']
                if 'pde_continuity_loss' not in self.initial_loss_magnitudes:
                    self.initial_loss_magnitudes['pde_continuity_loss'] = current_magnitude
                else:
                    self.initial_loss_magnitudes['pde_continuity_loss'] = (
                        alpha * current_magnitude + (1 - alpha) * self.initial_loss_magnitudes['pde_continuity_loss']
                    )
            
            if 'pinn_pde_loss_mom_x' in loss_components:
                current_magnitude = loss_components['pinn_pde_loss_mom_x']
                if 'pde_momentum_x_loss' not in self.initial_loss_magnitudes:
                    self.initial_loss_magnitudes['pde_momentum_x_loss'] = current_magnitude
                else:
                    self.initial_loss_magnitudes['pde_momentum_x_loss'] = (
                        alpha * current_magnitude + (1 - alpha) * self.initial_loss_magnitudes['pde_momentum_x_loss']
                    )
            
            if 'pinn_pde_loss_mom_y' in loss_components:
                current_magnitude = loss_components['pinn_pde_loss_mom_y']
                if 'pde_momentum_y_loss' not in self.initial_loss_magnitudes:
                    self.initial_loss_magnitudes['pde_momentum_y_loss'] = current_magnitude
                else:
                    self.initial_loss_magnitudes['pde_momentum_y_loss'] = (
                        alpha * current_magnitude + (1 - alpha) * self.initial_loss_magnitudes['pde_momentum_y_loss']
                    )
        
        # Recompute balancing factors
        # If using frequency-based updates, update immediately
        # Otherwise, update every 5 epochs (old behavior for backward compatibility)
        if self.adaptive_balancing_frequency > 0:
            # Frequency-based: update immediately when this method is called
            self._update_loss_weights_from_balancing()
        elif (epoch + 1) % 5 == 0:
            # Old behavior: update every 5 epochs
            self._update_loss_weights_from_balancing()
        # Note: Weights are recorded at every epoch in the training loop, so no need to record here
    
    def _record_adaptive_weights(self):
        """
        Record current adaptive loss weights to history.
        """
        self.adaptive_weight_history['deeponet_data_loss_weight'].append(
            self.model.loss_weight_deeponet_data_loss.item()
        )
        self.adaptive_weight_history['deeponet_pinn_loss_weight'].append(
            self.model.loss_weight_deeponet_pinn_loss.item()
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
    def __init__(self, patience=50, min_delta=1e-5, verbose=True):
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
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered")
                    
        return self.early_stop 