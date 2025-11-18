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
from ...data import PINNDataset
from ...models.PI_DeepONet.model import PI_SWE_DeepONetModel


class PI_SWE_DeepONetTrainer:
    """
    Trainer for Physics-Informed SWE_DeepONet models.
    """
    def __init__(self, model, config):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): Physics-Informed DeepONet model to train.
            config (Config): Configuration object.
        """
        # Check if model is an instance of PI_SWE_DeepONetModel
        if not isinstance(model, PI_SWE_DeepONetModel):
            raise ValueError("model must be an instance of PI_SWE_DeepONetModel")

        self.model = model
        
        # Load configuration
        if not isinstance(config, Config):
            raise ValueError("config must be a Config object")

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
                    min_lr=min_lr,
                    verbose=True
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
        
        # Training history
        self.loss_history = []
        self.component_loss_history = []
        
    def train(self, train_loader, val_loader, physics_dataset):
        """
        Train the Physics-Informed SWE_DeepONet model.
        
        Args:
            train_loader (DataLoader): Training data loader for DeepONet data.
            val_loader (DataLoader): Validation data loader for DeepONet data.
            physics_dataset (PINNDataset): Dataset for physics constraints.
                Should have methods: get_pde_points(), get_initial_points(), get_boundary_points(),
                get_mesh_stats(), get_data_stats().
            
        Returns:
            dict: Training history containing loss_history and component_loss_history.
        """
            
        # Get branch input dimension from data if needed
        if self.model.branch_input_dim == 0:
            batch = next(iter(train_loader))
            branch_input = batch[0]
            branch_dim = branch_input.shape[1]
            self.model.set_branch_input_dim(branch_dim)
            print(f"Set branch input dimension to {branch_dim}")
            
        # Get physics collocation points and data
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
            boundary_points, boundary_ids, boundary_z, boundary_normals, boundary_lengths, boundary_ManningN = boundary_points_data
            boundary_points = boundary_points.to(self.device)
        else:
            boundary_points = None
        
        # Get DeepONet and PINN points stats for normalization
        all_deeponet_points_stats = train_loader.dataset.get_deeponet_stats()
        all_pinn_points_stats = physics_dataset.get_all_pinn_points_stats()
        
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
            
            self.loss_history.append(train_loss)
            self.component_loss_history.append(loss_components)
            
            # Validation step
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)
                
                # Update learning rate scheduler if needed
                if self.scheduler is not None and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
            else:
                val_loss = None
                if self.scheduler is not None and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                    
            # Print progress
            if val_loss is not None:
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.6f}")
                
            # Print component losses
            print(f"  Data Loss: {loss_components.get('data_loss', 0.0):.6f}")
            print(f"  PDE Loss: {loss_components.get('pde_loss', 0.0):.6f}")
            if initial_points is not None:
                print(f"  Initial Loss: {loss_components.get('initial_loss', 0.0):.6f}")
            if boundary_points is not None:
                print(f"  Boundary Loss: {loss_components.get('boundary_loss', 0.0):.6f}")
                
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            if val_loss is not None:
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                
            self.writer.add_scalar('Loss/data', loss_components.get('data_loss', 0.0), epoch)
            self.writer.add_scalar('Loss/pde', loss_components.get('pde_loss', 0.0), epoch)
            if initial_points is not None:
                self.writer.add_scalar('Loss/initial', loss_components.get('initial_loss', 0.0), epoch)
            if boundary_points is not None:
                self.writer.add_scalar('Loss/boundary', loss_components.get('boundary_loss', 0.0), epoch)
                
            # Save checkpoint
            if (epoch + 1) % self.save_freq == 0:
                self._save_checkpoint(epoch + 1)
                
            # Save best model
            current_loss = val_loss if val_loss is not None else train_loss
            if current_loss < best_loss:
                best_loss = current_loss
                self._save_checkpoint('best')
                
            # Early stopping
            if self.early_stopping is not None and val_loss is not None:
                if self.early_stopping(val_loss):
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                    
        # Save final model
        self._save_checkpoint('final')
        
        # Training finished
        elapsed_time = time.time() - start_time
        print(f"Training completed in {elapsed_time:.2f} seconds")
        
        return {
            'loss_history': self.loss_history,
            'component_loss_history': self.component_loss_history
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
            
            if pde_points is not None and pde_data is not None:
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
            'loss_history': self.loss_history,
            'component_loss_history': self.component_loss_history,
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
            
        self.loss_history = checkpoint.get('loss_history', [])
        self.component_loss_history = checkpoint.get('component_loss_history', [])
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        
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