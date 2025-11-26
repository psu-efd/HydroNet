"""
Training utilities for Physics-Informed SWE_DeepONet models.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import time

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
        self.batch_size = self.config.get('training.batch_size', 32)
        self.epochs = self.config.get('training.epochs', 2000)
        self.learning_rate = float(self.config.get('training.learning_rate', 0.001))
        self.weight_decay = float(self.config.get('training.weight_decay', 1e-5))
        
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
            scheduler_type = self.config.get('training.scheduler.scheduler_type', 'ReduceLROnPlateau')
            if scheduler_type == 'ReduceLROnPlateau':
                patience = int(self.config.get('training.scheduler.patience', 20))
                factor = float(self.config.get('training.scheduler.factor', 0.5))
                min_lr = float(self.config.get('training.scheduler.min_lr', 1e-6))
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    patience=patience,
                    factor=factor,
                    min_lr=min_lr
                )
            elif scheduler_type == 'ExponentialLR':
                gamma = self.config.get('training.scheduler.gamma', 0.999)
                self.scheduler = optim.lr_scheduler.ExponentialLR(
                    self.optimizer,
                    gamma=gamma
                )
        else:
            self.scheduler = None
            
        # Early stopping
        use_early_stopping = self.config.get('training.early_stopping.use_early_stopping', True)
        if use_early_stopping:
            patience = int(self.config.get('training.early_stopping.patience', 50))
            min_delta = float(self.config.get('training.early_stopping.min_delta', 1e-5))
            self.early_stopping = EarlyStopping(patience, min_delta)
        else:
            self.early_stopping = None
            
        # Logging
        log_dir = self.config.get('logging.tensorboard_log_dir', './logs/tensorboard')
        self.writer = SummaryWriter(log_dir)
        
        # Checkpointing
        self.checkpoint_dir = self.config.get('paths.checkpoint_dir', './checkpoints')
        self.save_freq = self.config.get('logging.save_freq', 10)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training and validation history
        self.training_loss_history = []
        self.validation_loss_history = []

        # Training component loss history (no validation component loss history because for validation, we only compute the DeepONet data loss)
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
        
        # Adaptive loss balancing
        self.use_adaptive_loss_balancing = self.config.get('training.loss_weights.use_adaptive_balancing', False)
        self.use_adaptive_pde_component_balancing = self.config.get('training.loss_weights.use_adaptive_pde_component_balancing', False)
        
        # Adaptive balancing frequency: update weights every N epochs throughout training
        # If not specified, falls back to old behavior (only first N epochs)
        self.adaptive_balancing_frequency = self.config.get('training.loss_weights.adaptive_balancing_frequency', None)
        self.adaptive_balancing_epochs = self.config.get('training.loss_weights.adaptive_balancing_epochs', 10)
        
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
        batch = next(iter(train_loader))
        branch_input = batch[0]
        branch_dim = branch_input.shape[1]

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

            all_pinn_points_stats = physics_dataset.get_all_pinn_points_stats()
        else:
            pde_points = None
            pde_data = None
            initial_points = None
            initial_values = None
            boundary_points = None
            all_pinn_points_stats = None
        
        # Get DeepONet stats 
        all_deeponet_points_stats = train_loader.dataset.get_deeponet_stats()

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
        
        # Record initial weights
        self._record_adaptive_weights()
        
        # Training loop
        print(f"Starting training for {self.epochs} epochs...")
        start_time = time.time()
        
        best_loss = float('inf')
        
        for epoch in range(self.epochs):
            # Training step
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
                    # Old behavior: only update during first N epochs
                    should_update_weights = (epoch < self.adaptive_balancing_epochs)
            
            if should_update_weights:
                self._update_adaptive_loss_weights(loss_components, epoch)
            
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
            val_loss = self._validate_epoch(val_loader)
            self.validation_loss_history.append(val_loss)

            # Update learning rate scheduler if needed
            if self.scheduler is not None and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(val_loss)
                new_lr = self.optimizer.param_groups[0]['lr']
                if old_lr != new_lr:
                    print(f"  Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")
            elif self.scheduler is not None and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
                    
            # Print progress
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
            # Print component losses
            print(f"  Data Loss: {loss_components.get('deeponet_data_loss', 0.0):.6f}")
            if self.model.use_physics_loss:
                print(f"  PDE Loss: {loss_components.get('pinn_pde_loss', 0.0):.6f}")
            if initial_points is not None:
                print(f"  Initial Loss: {loss_components.get('pinn_initial_loss', 0.0):.6f}")
            if boundary_points is not None:
                print(f"  Boundary Loss: {loss_components.get('pinn_boundary_loss', 0.0):.6f}")
            
            # Print loss weights if using adaptive balancing
            if self.use_adaptive_loss_balancing:
                print(f"  Loss Weights - Data: {self.model.loss_weight_deeponet_data_loss.item():.6f}, "
                      f"PDE: {self.model.loss_weight_deeponet_pinn_loss.item():.6f}")
                
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
                
            self.writer.add_scalar('Loss/data', loss_components.get('deeponet_data_loss', 0.0), epoch)
            if self.model.use_physics_loss:
                self.writer.add_scalar('Loss/pde', loss_components.get('pinn_pde_loss', 0.0), epoch)
            if initial_points is not None:
                self.writer.add_scalar('Loss/initial', loss_components.get('pinn_initial_loss', 0.0), epoch)
            if boundary_points is not None:
                self.writer.add_scalar('Loss/boundary', loss_components.get('pinn_boundary_loss', 0.0), epoch)
                
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
        
        for batch in tqdm(train_loader, desc="Training", leave=False):
            
            # Get batch data
            branch_input, trunk_input, target = batch
            branch_input = branch_input.to(self.device)
            trunk_input = trunk_input.to(self.device)
            target = target.to(self.device)
            
            # Get the batch size for this batch
            batch_size = branch_input.shape[0]
            
            # Randomly sample PINN mesh points for PINN physics (PDEs: SWEs) constraints
            physics_branch_input = None
            physics_trunk_input = None
            physics_pde_data = None
            
            if self.model.use_physics_loss and pde_points is not None and pde_data is not None:
                # Sample random PDE points for this batch
                # Note: Advanced indexing (tensor indices) creates a copy, not a view.
                # The copy inherits requires_grad from pde_points, but compute_pde_residuals
                # will clone, detach, and set requires_grad=True anyway, so we don't need to
                # explicitly set requires_grad here.
                physics_indices = torch.randint(0, len(pde_points), (batch_size,))
                physics_trunk_input = pde_points[physics_indices]
                physics_pde_data = pde_data[physics_indices]
                physics_branch_input = branch_input  # Use the same branch input as the DeepONet branch input, meaning the PDEs have to be satisfied for all the Branch inputs.
            
            # Sample initial points if provided
            initial_trunk_input = None
            initial_batch_values = None
            if initial_points is not None and initial_values is not None:
                initial_indices = torch.randint(0, len(initial_points), (batch_size,))
                initial_trunk_input = initial_points[initial_indices]
                initial_batch_values = initial_values[initial_indices]
                
            # Sample boundary points if provided
            # Note: For complex boundary conditions (inlet-q, exit-h, wall), 
            # a more sophisticated sampling approach may be needed
            boundary_trunk_input = None
            boundary_batch_values = None
            if boundary_points is not None:
                boundary_indices = torch.randint(0, len(boundary_points), (batch_size,))
                boundary_trunk_input = boundary_points[boundary_indices]
                # For now, we assume boundary_values would be provided separately
                # This can be extended later for more complex boundary conditions
                
            # Forward pass and compute loss            
            self.optimizer.zero_grad()

            total_batch_loss, loss_components = self.model.compute_total_loss(
                branch_input,
                trunk_input,
                target=target,
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
            
            # Update weights
            self.optimizer.step()
            
            # Update total loss
            total_loss += total_batch_loss.item()
            
            # Update component losses
            for key in total_components:
                if key in loss_components:
                    total_components[key] += loss_components[key]
                
        # Compute average losses
        avg_loss = total_loss / len(train_loader)
        avg_components = {key: total_components[key] / len(train_loader) for key in total_components}
        
        return avg_loss, avg_components
        
    def _validate_epoch(self, val_loader):
        """
        Validate for one epoch.
        
        Args:
            val_loader (DataLoader): Validation data loader.
            
        Returns:
            float: Average validation loss for the epoch.
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                # Get batch data
                branch_input, trunk_input, target = batch
                branch_input = branch_input.to(self.device)
                trunk_input = trunk_input.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                output = self.model(branch_input, trunk_input)
                
                # Compute loss (data loss only for validation)
                loss = torch.mean((output - target)**2)
                
                # Update total loss
                total_loss += loss.item()
                
        # Compute average loss
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss
        
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
            'epoch': epoch
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        torch.save(checkpoint, checkpoint_path)
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
        
        #print(f"Loaded checkpoint from {checkpoint_path}")
        
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
        
        # Note: We need gradients enabled for PDE loss computation (it uses autograd.grad),
        # but we don't call backward() so no gradients are accumulated
        for i, batch in enumerate(train_loader):
            if i >= num_samples:
                break
                
            branch_input, trunk_input, target = batch
            branch_input = branch_input.to(self.device)
            trunk_input = trunk_input.to(self.device)
            target = target.to(self.device)
            
            batch_size = branch_input.shape[0]
            
            # Compute data loss (no gradients needed, but doesn't hurt)
            with torch.no_grad():
                data_loss, _ = self.model.compute_deeponet_data_loss(branch_input, trunk_input, target)
                data_losses.append(data_loss.item())
            
            # Compute PDE loss if enabled (gradients needed for autograd.grad in compute_pde_residuals)
            if self.model.use_physics_loss and pde_points is not None and pde_data is not None:
                physics_indices = torch.randint(0, len(pde_points), (batch_size,))
                physics_trunk_input = pde_points[physics_indices]
                physics_pde_data = pde_data[physics_indices]
                physics_branch_input = branch_input
                
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
        
        # Store initial magnitudes (use median to be robust to outliers)
        if data_losses:
            self.initial_loss_magnitudes['deeponet_data_loss'] = np.median(data_losses)
        if pde_losses:
            self.initial_loss_magnitudes['pinn_pde_loss'] = np.median(pde_losses)
        
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
    
    def _update_loss_weights_from_balancing(self):
        """
        Update loss weights based on computed balancing factors.
        
        The balancing factor for each loss is computed as:
        balancing_factor = reference_loss_magnitude / loss_magnitude
        
        This ensures losses are on similar scales.
        """
        if not self.initial_loss_magnitudes:
            return
        
        # Use data loss as reference (typically the smallest)
        reference_magnitude = self.initial_loss_magnitudes.get('deeponet_data_loss', 1.0)
        
        # Compute balancing factors
        if 'deeponet_data_loss' in self.initial_loss_magnitudes:
            data_magnitude = self.initial_loss_magnitudes['deeponet_data_loss']
            if data_magnitude > 0:
                self.loss_balancing_factors['deeponet_data_loss'] = reference_magnitude / data_magnitude
            else:
                self.loss_balancing_factors['deeponet_data_loss'] = 1.0
        
        if 'pinn_pde_loss' in self.initial_loss_magnitudes:
            pde_magnitude = self.initial_loss_magnitudes['pinn_pde_loss']
            if pde_magnitude > 0:
                self.loss_balancing_factors['pinn_pde_loss'] = reference_magnitude / pde_magnitude
            else:
                self.loss_balancing_factors['pinn_pde_loss'] = 1.0
        
        # Apply balancing factors to model's loss weights
        base_data_weight = self.config.get("training.loss_weights.deeponet.data_loss", 1.0)
        base_pde_weight = self.config.get("training.loss_weights.deeponet.pinn_loss", 1.0)
        
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