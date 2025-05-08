"""
Training utilities for Physics-Informed DeepONet models.
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
from ...data import DeepONetDataset, PINNDataset


class PI_DeepONetTrainer:
    """
    Trainer for Physics-Informed DeepONet models.
    """
    def __init__(self, model, config_file=None, config=None):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): Physics-Informed DeepONet model to train.
            config_file (str, optional): Path to configuration file.
            config (Config, optional): Configuration object.
        """
        self.model = model
        
        # Load configuration
        if config is not None:
            self.config = config
        elif config_file is not None:
            self.config = Config(config_file)
        else:
            self.config = model.config
            
        # Set device
        self.device = self.config.get_device()
        self.model.to(self.device)
        
        # Training parameters
        self.batch_size = self.config.get('training.batch_size', 32)
        self.epochs = self.config.get('training.epochs', 2000)
        self.learning_rate = self.config.get('training.learning_rate', 0.001)
        self.weight_decay = self.config.get('training.weight_decay', 1e-5)
        
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
                patience = self.config.get('training.scheduler.patience', 20)
                factor = self.config.get('training.scheduler.factor', 0.5)
                min_lr = self.config.get('training.scheduler.min_lr', 1e-6)
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
            patience = self.config.get('training.early_stopping.patience', 50)
            min_delta = self.config.get('training.early_stopping.min_delta', 1e-5)
            self.early_stopping = EarlyStopping(patience, min_delta)
        else:
            self.early_stopping = None
            
        # Logging
        log_dir = self.config.get('logging.tensorboard_log_dir', '../logs/tensorboard')
        self.writer = SummaryWriter(log_dir)
        
        # Checkpointing
        self.checkpoint_dir = self.config.get('paths.checkpoint_dir', '../checkpoints')
        self.save_freq = self.config.get('logging.save_freq', 10)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training history
        self.loss_history = []
        self.component_loss_history = []
        
    def train(self, train_loader=None, val_loader=None, physics_dataset=None, 
             initial_conditions=None, boundary_conditions=None):
        """
        Train the Physics-Informed DeepONet model.
        
        Args:
            train_loader (DataLoader, optional): Training data loader.
            val_loader (DataLoader, optional): Validation data loader.
            physics_dataset (PINNDataset, optional): Dataset for physics constraints.
            initial_conditions (callable, optional): Function that returns initial conditions.
            boundary_conditions (callable, optional): Function that returns boundary conditions.
            
        Returns:
            dict: Training history.
        """
        # Load data if not provided
        if train_loader is None or val_loader is None:
            train_loader, val_loader = self._load_data()
            
        # Create physics dataset if not provided
        if physics_dataset is None:
            physics_dataset = self._create_physics_dataset()
            
        # Get branch input dimension from data if needed
        if self.model.branch_dim == 0:
            batch = next(iter(train_loader))
            branch_input = batch[0]
            branch_dim = branch_input.shape[1]
            self.model.set_branch_dim(branch_dim)
            print(f"Set branch input dimension to {branch_dim}")
            
        # Get physics collocation points
        domain_points = physics_dataset.get_pde_points().to(self.device)
        initial_points = physics_dataset.get_initial_points().to(self.device) if initial_conditions is not None else None
        boundary_points = physics_dataset.get_boundary_points().to(self.device) if boundary_conditions is not None else None
        
        # Training loop
        print(f"Starting training for {self.epochs} epochs...")
        start_time = time.time()
        
        best_loss = float('inf')
        
        for epoch in range(self.epochs):
            # Training step
            train_loss, loss_components = self._train_epoch(
                train_loader, 
                domain_points, 
                initial_points, 
                boundary_points, 
                initial_conditions, 
                boundary_conditions
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
            print(f"  Data Loss: {loss_components['data_loss']:.6f}")
            print(f"  PDE Loss: {loss_components['pde_loss']:.6f}")
            if initial_points is not None:
                print(f"  Initial Loss: {loss_components['initial_loss']:.6f}")
            if boundary_points is not None:
                print(f"  Boundary Loss: {loss_components['boundary_loss']:.6f}")
                
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            if val_loss is not None:
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                
            self.writer.add_scalar('Loss/data', loss_components['data_loss'], epoch)
            self.writer.add_scalar('Loss/pde', loss_components['pde_loss'], epoch)
            if initial_points is not None:
                self.writer.add_scalar('Loss/initial', loss_components['initial_loss'], epoch)
            if boundary_points is not None:
                self.writer.add_scalar('Loss/boundary', loss_components['boundary_loss'], epoch)
                
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
        
    def _train_epoch(self, train_loader, domain_points, initial_points=None, 
                    boundary_points=None, initial_conditions=None, boundary_conditions=None):
        """
        Train for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader.
            domain_points (torch.Tensor): Points inside the domain for physics constraints.
            initial_points (torch.Tensor, optional): Points at initial time.
            boundary_points (torch.Tensor, optional): Points on the boundary.
            initial_conditions (callable, optional): Function that returns initial conditions.
            boundary_conditions (callable, optional): Function that returns boundary conditions.
            
        Returns:
            tuple: (average_loss, loss_components)
        """
        self.model.train()
        total_loss = 0
        total_components = {
            'data_loss': 0.0,
            'pde_loss': 0.0,
            'initial_loss': 0.0,
            'boundary_loss': 0.0,
            'total_loss': 0.0
        }
        
        for batch in tqdm(train_loader, desc="Training", leave=False):
            # Get batch data
            branch_input, trunk_input, target = batch
            branch_input = branch_input.to(self.device)
            trunk_input = trunk_input.to(self.device)
            target = target.to(self.device)
            
            # Sample physics points for this batch
            batch_size = branch_input.shape[0]
            
            # Sample domain points for physics constraints
            physics_indices = torch.randint(0, len(domain_points), (batch_size,))
            physics_trunk_input = domain_points[physics_indices]
            physics_branch_input = branch_input  # Use the same branch input
            
            # Sample initial points if provided
            if initial_points is not None and initial_conditions is not None:
                initial_indices = torch.randint(0, len(initial_points), (batch_size,))
                initial_trunk_input = initial_points[initial_indices]
            else:
                initial_trunk_input = None
                
            # Sample boundary points if provided
            if boundary_points is not None and boundary_conditions is not None:
                boundary_indices = torch.randint(0, len(boundary_points), (batch_size,))
                boundary_trunk_input = boundary_points[boundary_indices]
            else:
                boundary_trunk_input = None
                
            # Forward pass and compute loss
            self.optimizer.zero_grad()
            
            total_batch_loss, loss_components = self.model.compute_total_loss(
                branch_input,
                trunk_input,
                target=target,
                physics_branch_input=physics_branch_input,
                physics_trunk_input=physics_trunk_input,
                initial_trunk_input=initial_trunk_input,
                boundary_trunk_input=boundary_trunk_input,
                initial_conditions=initial_conditions,
                boundary_conditions=boundary_conditions
            )
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Update total loss
            total_loss += total_batch_loss.item()
            
            # Update component losses
            for key in total_components:
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
        
    def _load_data(self):
        """
        Load training and validation data.
        
        Returns:
            tuple: (train_loader, val_loader)
        """
        # Get data paths from config
        data_dir = self.config.get('data.train_data_path', '../data/train')
        val_dir = self.config.get('data.val_data_path', '../data/val')
        
        # Create datasets
        train_dataset = DeepONetDataset(data_dir, split='train', normalize=True)
        val_dataset = DeepONetDataset(val_dir, split='val', normalize=True)
        
        # Create data loaders
        train_loader = get_deeponet_dataloader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = get_deeponet_dataloader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        return train_loader, val_loader
        
    def _create_physics_dataset(self):
        """
        Create a dataset for physics constraints.
        
        Returns:
            PINNDataset: Dataset containing collocation points.
        """
        # Get domain parameters from config
        domain_params = self.config.get('sampling.domain', {
            'x_min': 0.0,
            'x_max': 1.0,
            'y_min': 0.0,
            'y_max': 1.0,
            't_min': 0.0,
            't_max': 1.0
        })
        
        # Get sampling parameters from config
        num_domain_points = self.config.get('sampling.num_domain_points', 5000)
        num_boundary_points = self.config.get('sampling.num_boundary_points', 1000)
        num_initial_points = self.config.get('sampling.num_initial_points', 1000)
        
        # Create dataset
        dataset = PINNDataset(
            domain_params,
            num_domain_points=num_domain_points,
            num_boundary_points=num_boundary_points,
            num_initial_points=num_initial_points
        )
        
        return dataset
        
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