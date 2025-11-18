"""
Training utilities for DeepONet models.
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
from .model import BranchNet, SWE_DeepONetModel


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """
    def __init__(self, patience=30, min_delta=1e-5, verbose=True):
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


class SWE_DeepONetTrainer:
    """
    Trainer for SWE_DeepONet models.
    """
    def __init__(self, model, config):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): SWE_DeepONet model to train.
            config (Config): Configuration object.
        """

        # Check if model is an instance of SWE_DeepONetModel
        if not isinstance(model, SWE_DeepONetModel):
            raise ValueError("model must be an instance of SWE_DeepONetModel")

        self.model = model
        
        # Load configuration
        if not isinstance(config, Config):
            raise ValueError("config must be an instance of Config")

        self.config = config
            
        # Set device
        self.device = self.config.get_device()
        self.model.to(self.device)

        print(f"Device: {self.device}")
        
        # Training parameters (with default values)
        self.batch_size = self.config.get('training.batch_size', 32)
        self.epochs = self.config.get('training.epochs', 1000)
        self.learning_rate = float(self.config.get('training.learning_rate', 0.001))
        self.weight_decay = float(self.config.get('training.weight_decay', 1e-5))
        
        # Loss function
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
                patience = int(self.config.get('training.scheduler.patience', 10))
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
                gamma = float(self.config.get('training.scheduler.gamma', 0.999))
                self.scheduler = optim.lr_scheduler.ExponentialLR(
                    self.optimizer,
                    gamma=gamma
                )
        else:
            self.scheduler = None
            
        # Early stopping
        use_early_stopping = self.config.get('training.early_stopping.use_early_stopping', True)
        if use_early_stopping:
            patience = int(self.config.get('training.early_stopping.patience', 30))
            min_delta = float(self.config.get('training.early_stopping.min_delta', 1e-5))
            self.early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
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
        self.train_losses = []
        self.val_losses = []
        
    def train(self, train_loader, val_loader):
        """
        Train the SWE_DeepONet model.
        
        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            
        Returns:
            dict: Training history.
        """
            
        # Get sample from loader to determine input dimensions if needed
        if self.model.branch_input_dim == 0:
            for batch in train_loader:
                branch_input, _, _ = batch
                self.model.branch_input_dim = branch_input.shape[1]
                # Reconstruct branch network with correct input dimension
                branch_layers = self.config.get('model.branch_net.hidden_layers', [128, 128, 128])
                branch_activation = self.config.get('model.branch_net.activation', 'relu')
                branch_dropout = self.config.get('model.branch_net.dropout_rate', 0.0)
                self.model.branch_net = BranchNet(
                    self.model.branch_input_dim, 
                    self.model.hidden_dim, 
                    branch_layers, 
                    branch_activation, 
                    branch_dropout
                )
                self.model.branch_net.to(self.device)
                break
            
        # Setup for training
        history = {'train_loss': [], 'val_loss': [], 'best_val_loss': float('inf')}        
        
        # Early stopping
        use_early_stopping = self.config.get('training.early_stopping.use_early_stopping', True)
        if use_early_stopping:
            # Use self.early_stopping that was already initialized in __init__
            if self.early_stopping is None:
                patience = int(self.config.get('training.early_stopping.patience', 30))
                min_delta = float(self.config.get('training.early_stopping.min_delta', 1e-5))
                self.early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
            early_stopping = self.early_stopping
            
        # TensorBoard logging
        log_dir = 'logs/swe_deeponet_' + time.strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir)
        
        # Training loop
        print(f"Starting training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            # Train for one epoch
            train_loss = self._train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validate
            val_loss = self._validate_epoch(val_loader)
            history['val_loss'].append(val_loss)
            
            # TensorBoard logging
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.epochs} - Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}")
            
            # Learning rate scheduler step
            if hasattr(self, 'scheduler'):
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                    
            # Save best model
            if val_loss < history['best_val_loss']:
                history['best_val_loss'] = val_loss
                self._save_checkpoint(epoch)
                print(f"Saved checkpoint at epoch {epoch+1} with val_loss: {val_loss:.6f}")
                
            # Early stopping check
            if use_early_stopping and early_stopping(val_loss):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
        writer.close()
        
        print("Training completed.")

        return history
        
    def _train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader.
            
        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training", leave=False):
            # Get batch data
            branch_input, trunk_input, target = batch
            branch_input = branch_input.to(self.device)
            trunk_input = trunk_input.to(self.device)
            target = target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(branch_input, trunk_input)
            
            # Compute loss
            loss = self.loss_fn(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update total loss
            total_loss += loss.item()
            
        # Compute average loss
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss
        
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
                
                # Compute loss
                loss = self.loss_fn(output, target)
                
                # Update total loss
                total_loss += loss.item()
                
        # Compute average loss
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss
        
    def _save_checkpoint(self, epoch):
        """
        Save a model checkpoint.
        
        Args:
            epoch (int or str): Current epoch or 'final'.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"deeponet_epoch_{epoch}.pt")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
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
            
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
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