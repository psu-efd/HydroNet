"""
Loss weight scheduling strategies for PINN training.
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Callable

class LossWeightScheduler:
    """
    Base class for loss weight scheduling strategies.
    """
    def __init__(self, initial_weights: Dict[str, float], bSteady: bool = True):
        """
        Initialize the loss weight scheduler.
        
        Args:
            initial_weights (Dict[str, float]): Initial weights for each loss component.
            bSteady (bool): Whether this is a steady-state problem.
        """
        # Filter out initial condition weight for steady-state problems
        if bSteady and 'initial' in initial_weights:
            initial_weights = {k: v for k, v in initial_weights.items() if k != 'initial'}
            
        self.initial_weights = initial_weights.copy()
        self.current_weights = initial_weights.copy()
        self.bSteady = bSteady

        
    def step(self, epoch: int, loss_components: Dict[str, float], model: torch.nn.Module) -> Dict[str, float]:
        """
        Update loss weights based on the current epoch and loss components.
        
        Args:
            epoch (int): Current training epoch.
            loss_components (Dict[str, float]): Current loss component values.
            model (torch.nn.Module): The neural network model.              

        Returns:
            Dict[str, float]: Updated loss weights.
        """
        return self.current_weights

class ConstantWeightScheduler(LossWeightScheduler):
    """
    Constant loss weights that don't change during training.
    """
    def __init__(self, weights: Dict[str, float], bSteady: bool = True):
        """
        Initialize constant weight scheduler.
        
        Args:
            weights (Dict[str, float]): Fixed weights for each loss component.
            bSteady (bool): Whether this is a steady-state problem.
        """
        super().__init__(weights, bSteady)
        
    def step(self, epoch: int, loss_components: Dict[str, float], model: torch.nn.Module) -> Dict[str, float]:
        """
        Return constant weights.
        
        Args:
            epoch (int): Current training epoch.
            loss_components (Dict[str, float]): Current loss component values.
            model (torch.nn.Module): The neural network model.

        Returns:
            Dict[str, float]: Constant loss weights.
        """
        return self.current_weights

class ManualWeightScheduler(LossWeightScheduler):
    """
    Manual weight scheduling with predefined curriculum.
    """
    def __init__(self, 
                 initial_weights: Dict[str, float],
                 schedule: List[Dict[str, Union[float, int]]],
                 bSteady: bool = True):
        """
        Initialize manual weight scheduler.
        
        Args:
            initial_weights (Dict[str, float]): Initial weights for each loss component.
            schedule (List[Dict[str, Union[float, int]]]): List of dictionaries containing epoch numbers and weight updates.
                Each dict should have 'epoch' key and weight keys for components to update.
                Example: [{'epoch': 100, 'pde': 0.5}, {'epoch': 200, 'data': 2.0}]
            bSteady (bool): Whether this is a steady-state problem.
        """
        super().__init__(initial_weights, bSteady)
        self.schedule = sorted(schedule, key=lambda x: x['epoch'])
        self.current_schedule_idx = 0
        
    def step(self, epoch: int, loss_components: Dict[str, float], model: torch.nn.Module) -> Dict[str, float]:
        """
        Update weights according to the manual schedule.
        
        Args:
            epoch (int): Current training epoch.
            loss_components (Dict[str, float]): Current loss component values.
            model (torch.nn.Module): The neural network model.

        Returns:
            Dict[str, float]: Updated loss weights.
        """
        # Check if we need to update weights based on the current epoch
        while (self.current_schedule_idx < len(self.schedule) and 
               epoch >= self.schedule[self.current_schedule_idx]['epoch']):
            update = self.schedule[self.current_schedule_idx]
            for key, value in update.items():
                if key != 'epoch':
                    # Map the config keys to our internal keys
                    if key == 'initial_condition':
                        if not self.bSteady:  # Only update initial weight if not steady-state
                            self.current_weights['initial_loss'] = value
                    elif key == 'boundary_condition':
                        self.current_weights['boundary_loss'] = value
                    elif key == 'data_points':
                        self.current_weights['data_loss'] = value
                    elif key == 'pde':
                        self.current_weights['pde_loss'] = value
                    else:
                        #error
                        raise ValueError(f"Invalid key: {key} in ManualWeightScheduler")
                    
            self.current_schedule_idx += 1
            
        return self.current_weights.copy()

class GradNormScheduler(LossWeightScheduler):
    """
    Gradient normalization based weight scheduling following the GradNorm algorithm.
    """
    def __init__(self, 
                 initial_weights: Dict[str, float],
                 alpha: float = 1.0,
                 learning_rate: float = 0.001,
                 bSteady: bool = True):
        """
        Initialize GradNorm scheduler.
        
        Args:
            initial_weights (Dict[str, float]): Initial weights for each loss component.
            alpha (float): Balancing factor for GradNorm.
            learning_rate (float): Learning rate for weight updates.
            bSteady (bool): Whether this is a steady-state problem.
        """
        super().__init__(initial_weights, bSteady)
        self.alpha = alpha
        self.lr = learning_rate
        self.initial_losses = None
        self.min_weight = 0.01
        self.max_weight = 10.0
        self.shared_params = None  # Will store reference to shared parameters
        self.loss_tensors = None  # Will store the actual loss tensors
        print("\nInitializing GradNormScheduler with weights:", initial_weights)
        
    def set_shared_params(self, params):
        """
        Set the shared parameters for gradient computation.
        
        Args:
            params: List of shared parameters (typically from model.parameters())
        """
        self.shared_params = params
        
    def set_loss_tensors(self, loss_tensors: Dict[str, torch.Tensor]):
        """
        Set the actual loss tensors from the model's forward pass.
        
        Args:
            loss_tensors: Dictionary of loss tensors for each component
        """
        self.loss_tensors = loss_tensors
        
    def step(self, epoch: int, loss_components: Dict[str, float], model: torch.nn.Module) -> Dict[str, float]:
        """
        Update weights using GradNorm algorithm.
        
        Args:
            epoch (int): Current training epoch.
            loss_components (Dict[str, float]): Current loss component values.
            model (torch.nn.Module): The neural network model.
            
        Returns:
            Dict[str, float]: Updated loss weights.
        """
        if self.shared_params is None:
            raise ValueError("Shared parameters not set. Call set_shared_params first.")
            
        if self.loss_tensors is None:
            raise ValueError("Loss tensors not set. Call set_loss_tensors first.")
            
        if epoch % 100 == 0:
            print(f"\nEpoch {epoch} - Loss components:")
            for key, value in loss_components.items():
                print(f"{key}: {value}")
            print("\nCurrent weights:")
            for key, value in self.current_weights.items():
                print(f"{key}: {value}")
            if self.initial_losses is not None:
                print("\nInitial losses:")
                for key, value in self.initial_losses.items():
                    print(f"{key}: {value}")
            
        if self.initial_losses is None:
            print("\nSetting initial losses from components:", loss_components)
            self.initial_losses = {}
            for loss_key, loss_value in loss_components.items():                
                self.initial_losses[loss_key] = loss_value
            print("Initial losses set to:", self.initial_losses)
            return self.current_weights
            
        # Compute relative inverse training rates
        relative_inverse_rates = {}
        valid_components = 0
        
        for loss_key in self.current_weights:
            if loss_key not in loss_components:
                raise ValueError(f"Loss key {loss_key} not found in loss components")
            
            if loss_components[loss_key] == 0 and self.initial_losses[loss_key] == 0:
                relative_inverse_rates[loss_key] = 1.0
                valid_components += 1
            elif self.initial_losses[loss_key] == 0:
                relative_inverse_rates[loss_key] = 1.5
                valid_components += 1
            elif loss_components[loss_key] == 0:
                relative_inverse_rates[loss_key] = 0.75
                valid_components += 1
            else:
                eps = 1e-8
                ratio = (loss_components[loss_key] + eps) / (self.initial_losses[loss_key] + eps)
                relative_inverse_rates[loss_key] = ratio ** self.alpha
                valid_components += 1
        
        if valid_components == 0:
            print("\nNo valid components found, returning current weights")
            return self.current_weights
                
        mean_rate = sum(relative_inverse_rates.values()) / valid_components
        
        # Compute gradient norms for each task
        grad_norms = {}
        for loss_key in self.current_weights:
            if loss_key in self.loss_tensors:
                # Use the actual loss tensor from the model
                loss_tensor = self.loss_tensors[loss_key]
                
                # Compute gradients for this task's loss
                model.zero_grad()
                weighted_loss = self.current_weights[loss_key] * loss_tensor
                weighted_loss.backward(retain_graph=True)
                
                # Compute gradient norm for shared parameters
                grad_norm = 0.0
                for param in self.shared_params:
                    if param.grad is not None:
                        grad_norm += torch.norm(param.grad).item() ** 2
                grad_norm = grad_norm ** 0.5
                grad_norms[loss_key] = grad_norm
            else:
                raise ValueError(f"Loss key {loss_key} not found in loss tensors")
        
        # Compute target gradient norm
        target_grad_norm = mean_rate * sum(grad_norms.values()) / len(grad_norms)
        
        # Update weights to minimize difference between actual and target gradient norms
        for key in self.current_weights:
            if key in grad_norms:
                grad_diff = grad_norms[key] - target_grad_norm
                self.current_weights[key] *= (1.0 - self.lr * grad_diff)
                self.current_weights[key] = max(min(self.current_weights[key], self.max_weight), self.min_weight)
        
        # Normalize weights to sum to 1
        total = sum(self.current_weights.values())
        if total > 0:
            self.current_weights = {k: v/total for k, v in self.current_weights.items()}
        
        if epoch % 100 == 0:
            print("\nGradient norms:")
            for key, value in grad_norms.items():
                print(f"{key}: {value:.4f}")
            print(f"Target gradient norm: {target_grad_norm:.4f}")
            print("\nUpdated weights:")
            for key, value in self.current_weights.items():
                print(f"{key}: {value:.4f}")
        
        return self.current_weights

class SoftAdaptScheduler(LossWeightScheduler):
    """
    SoftAdapt weight scheduling based on exponential moving average of loss ratios.
    """
    def __init__(self, 
                 initial_weights: Dict[str, float],
                 beta: float = 0.9,
                 bSteady: bool = True):
        """
        Initialize SoftAdapt scheduler.
        
        Args:
            initial_weights (Dict[str, float]): Initial weights for each loss component.
            beta (float): Smoothing factor for exponential moving average.
            bSteady (bool): Whether this is a steady-state problem.
        """
        super().__init__(initial_weights, bSteady)
        self.beta = beta
        self.previous_losses = None
        self.ema_ratios = {k: 1.0 for k in initial_weights}
        
    def step(self, epoch: int, loss_components: Dict[str, float]) -> Dict[str, float]:
        """
        Update weights using SoftAdapt algorithm.
        
        Args:
            epoch (int): Current training epoch.
            loss_components (Dict[str, float]): Current loss component values.
            
        Returns:
            Dict[str, float]: Updated loss weights.
        """
        if self.previous_losses is None:
            self.previous_losses = {k: v for k, v in loss_components.items()}
            return self.current_weights
            
        # Calculate loss ratios and update EMA
        for key in self.current_weights:
            if key in loss_components and key in self.previous_losses:
                if self.previous_losses[key] > 0:
                    ratio = loss_components[key] / self.previous_losses[key]
                    self.ema_ratios[key] = (self.beta * self.ema_ratios[key] + 
                                          (1 - self.beta) * ratio)
                    
        # Update weights based on EMA ratios
        total = sum(self.ema_ratios.values())
        if total > 0:
            self.current_weights = {k: v/total for k, v in self.ema_ratios.items()}
            
        # Update previous losses
        self.previous_losses = {k: v for k, v in loss_components.items()}
        
        return self.current_weights 