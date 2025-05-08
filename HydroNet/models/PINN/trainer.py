"""
Training utilities for PINN models.
"""
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import time

from ...utils.config import Config
from ...data import PINNDataset


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
        
        self.model = model
        self.dataset = dataset

        # Load configuration
        if config is not None:
            self.config = config
        else:
            raise ValueError("config must be provided and not None")
            
        # Set device
        self.device = self.model.get_device()        
        
        # Training parameters
        #self.batch_size = self.config.get('training.batch_size', 1024)
        #self.epochs = self.config.get('training.epochs', 10000)
        #self.learning_rate = self.config.get('training.learning_rate', 0.001)
        try:
            self.weight_decay = self.config.get('training.weight_decay')
            if self.weight_decay is None:
                raise ValueError("weight_decay must be specified in config file")
        except KeyError:
            raise ValueError("weight_decay must be specified in config file")
        
        # Get optimizer and epoch lists
        try:
            self.optimizer_names = self.config.get('training.optimizers')
            if self.optimizer_names is None:
                raise ValueError("optimizers must be specified in config file")
        except KeyError:
            raise ValueError("optimizers must be specified in config file")
        
        try:
            self.epoch_list = self.config.get('training.epochs')
            if self.epoch_list is None:
                raise ValueError("epochs must be specified in config file")
        except KeyError:
            raise ValueError("epochs must be specified in config file")
        
        # Get learning rates for each optimizer
        try:            
            self.learning_rates = self.config.get('training.learning_rates')
        except KeyError:
            raise ValueError("learning_rates must be specified in config file")
        
        # Verify lists have same length
        if not (len(self.optimizer_names) == len(self.epoch_list) == len(self.learning_rates)):
            raise ValueError("Number of optimizers, epochs, and learning rates must match")
        
        # Initialize first optimizer
        self.current_optimizer_idx = 0
        self.optimizer = self._create_optimizer(
            self.optimizer_names[0],
            self.learning_rates[0]
        )
        
        # Create learning rate scheduler
        if self.config.get('training.scheduler.type') == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('training.scheduler.step_size', 1000),
                gamma=self.config.get('training.scheduler.gamma', 0.5)
            )
        elif self.config.get('training.scheduler.type') == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('training.scheduler.T_max', 1000)
            )
        else:
            self.scheduler = None

        # Log initial learning rate
        if self.scheduler is not None:
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"Initial learning rate: {current_lr}")
        
        # Early stopping
        use_early_stopping = self.config.get('training.early_stopping.use_early_stopping', True)
        if use_early_stopping:
            patience = self.config.get('training.early_stopping.patience', 500)
            min_delta = self.config.get('training.early_stopping.min_delta', 1e-5)
            self.early_stopping = EarlyStopping(patience, min_delta)
        else:
            self.early_stopping = None
            
        # Logging
        log_dir = self.config.get('logging.tensorboard_log_dir', './logs/tensorboard')
        self.writer = SummaryWriter(log_dir)
        
        # Checkpointing
        self.checkpoint_dir = self.config.get('paths.checkpoint_dir', './checkpoints')
        self.save_freq = self.config.get('logging.save_freq', 100)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training history
        self.loss_history = []
        self.component_loss_history = {}
        
        # Get training parameters from config
        self.print_freq = self.config.get('training.print_freq', 100)

        # Get model config
        self.bPDE_loss, self.bInitial_loss, self.bBoundary_loss, self.bData_loss = self.model.get_loss_flags()
        self.bSteady = not self.bInitial_loss
        
    def _create_optimizer(self, optimizer_name, learning_rate):
        """
        Create an optimizer based on name.
        
        Args:
            optimizer_name (str): Name of the optimizer
            learning_rate (float): Learning rate for this optimizer
        """
        if optimizer_name.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=float(learning_rate),
                weight_decay=self.weight_decay
            )
        elif optimizer_name.lower() == 'lbfgs':
            # L-BFGS doesn't support weight_decay directly
            return optim.LBFGS(
                self.model.parameters(),
                lr=float(learning_rate),
                max_iter=20,
                history_size=50,
                line_search_fn='strong_wolfe'
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def train(self):
        """
        Train the PINN model using full batch training with sequential optimizers.        
        """
        # Get all points at once - full batch training
        # Get pde points
        if self.bPDE_loss:
            pde_points = self.dataset.get_pde_points()
            if pde_points is not None:
                pde_points = pde_points.to(self.device)
            else:
                pde_points = None
        else:
            pde_points = None

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
            boundary_points, boundary_ids, boundary_normals, boundary_lengths = self.dataset.get_boundary_points()

            if boundary_points is not None:
                boundary_points = boundary_points.to(self.device)
            else:
                boundary_points = None
            
            if boundary_ids is not None:
                boundary_ids = boundary_ids.to(self.device)
            else:
                boundary_ids = None
            
            if boundary_normals is not None:
                boundary_normals = boundary_normals.to(self.device)
            else:
                boundary_normals = None
            
            if boundary_lengths is not None:
                boundary_lengths = boundary_lengths.to(self.device)
            else:
                boundary_lengths = None
        else:
            boundary_points = None
            boundary_ids = None
            boundary_normals = None
            boundary_lengths = None

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

        print(f"\nStarting full batch training with {len(self.optimizer_names)} optimizers...")

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
        total_epochs = 0

        #compute the total number of epochs
        n_epochs_all_optimizers = sum(self.epoch_list)
        print(f"Total number of epochs for all optimizers: {n_epochs_all_optimizers}")

        # Loop over each optimizer
        for opt_idx, (optimizer_name, n_epochs, lr) in enumerate(
            zip(self.optimizer_names, self.epoch_list, self.learning_rates)):
            
            print(f"\nStarting training with {optimizer_name}")
            print(f"Learning rate: {lr}")
            print(f"Number of epochs: {n_epochs}")
            
            # Create new optimizer with its specific learning rate
            self.optimizer = self._create_optimizer(optimizer_name, lr)
            
            # Create new scheduler for this optimizer if needed
            if self.scheduler is not None:
                scheduler_type = self.config.get('training.scheduler.scheduler_type', 'ExponentialLR')
                if scheduler_type == 'ExponentialLR':
                    gamma = self.config.get('training.scheduler.gamma', 0.999)
                    self.scheduler = optim.lr_scheduler.ExponentialLR(
                        self.optimizer,
                        gamma=gamma
                    )
                elif scheduler_type == 'ReduceLROnPlateau':
                    patience = self.config.get('training.scheduler.patience', 10)
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

            # Training loop for current optimizer
            for epoch in range(n_epochs):
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass and compute loss using all points
                total_loss, loss_components, predictions_and_true_values = self.model.compute_total_loss(
                    pde_points,
                    initial_points=initial_points,
                    initial_values=initial_values,
                    boundary_info=(boundary_points, boundary_ids, boundary_normals, boundary_lengths),
                    data_points=data_points,
                    data_values=data_values,
                    data_flags=data_flags
                )
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update weights
                if optimizer_name.lower() == 'lbfgs':
                    # L-BFGS requires a closure
                    def closure():
                        self.optimizer.zero_grad()
                        loss, _ = self.model.compute_total_loss(
                            pde_points,
                            initial_points=initial_points,
                            initial_values=initial_values,
                            boundary_info=(boundary_points, boundary_ids, boundary_normals, boundary_lengths),
                            data_points=data_points,
                            data_values=data_values
                        )
                        loss.backward()
                        return loss
                    self.optimizer.step(closure)
                else:
                    self.optimizer.step()
                
                # Update scheduler if needed
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(total_loss)
                    else:
                        self.scheduler.step()
                
                # Save history and log progress
                self._log_training_step(total_epochs, total_loss, loss_components, n_epochs, n_epochs_all_optimizers, optimizer_name)
                
                # Save checkpoints
                if (total_epochs + 1) % self.save_freq == 0:
                    self._save_checkpoint(total_epochs + 1)
                
                # Save best model
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    self._save_checkpoint('best')
                
                # Check early stopping
                if self.early_stopping is not None and self.early_stopping(total_loss.item()):
                    print(f"Early stopping triggered at epoch {total_epochs+1}")
                    return {
                        'loss_history': self.loss_history,
                        'component_loss_history': self.component_loss_history
                    }
                
                total_epochs += 1

        # Training finished
        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time:.2f} seconds")
        print(f"Best loss achieved: {best_loss:.6f}")
        
        return {
            'loss_history': self.loss_history,
            'component_loss_history': self.component_loss_history            
        }, predictions_and_true_values
        
    def _create_dataset(self):
        """
        Create a PINN dataset for training.
        
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
        num_domain_points = self.config.get('sampling.num_domain_points', 20000)
        num_boundary_points = self.config.get('sampling.num_boundary_points', 5000)
        num_initial_points = self.config.get('sampling.num_initial_points', 5000)
        
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
        self.component_loss_history = checkpoint.get('component_loss_history', {})
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        
    def predict(self, x):
        """
        Make predictions with the model.
        
        Args:
            x (torch.Tensor or numpy.ndarray): Input tensor of shape [batch_size, 3] containing (x, y, t) coordinates.
            
        Returns:
            numpy.ndarray: Model predictions.
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

    def _log_training_step(self, epoch, total_loss, loss_components, n_epochs, n_epochs_all_optimizers, optimizer_name):
        """
        Log the training progress.
        
        Args:
            epoch (int): Current epoch number
            total_loss (float): Total loss value
            loss_components (dict): Dictionary containing individual loss components
            n_epochs (int): Total number of epochs for current optimizer
            optimizer_name (str): Name of current optimizer
        """
        # Store losses in history
        self.loss_history.append(total_loss.item())
        
        # Store component losses
        for key, value in loss_components.items():
            if key not in self.component_loss_history:
                self.component_loss_history[key] = []
            if isinstance(value, dict):
                # For nested dictionaries, save each of the items and values in sub-dictionaries
                for subkey, subvalue in value.items():
                    if subkey not in self.component_loss_history:
                        self.component_loss_history[subkey] = []
                    self.component_loss_history[subkey].append(subvalue.item() if torch.is_tensor(subvalue) else subvalue)
            else:
                self.component_loss_history[key].append(value.item() if torch.is_tensor(value) else value)
        
        # Print progress
        if (epoch + 1) % self.print_freq == 0:
            print(f"\nEpoch [{epoch+1}/{n_epochs_all_optimizers}] with {optimizer_name}")
            print(f"Total Loss: {total_loss.item():.6f}")
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