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
    def __init__(self, initial_weights: Dict[str, float]):
        """
        Initialize the loss weight scheduler.
        
        Args:
            initial_weights (Dict[str, float]): Initial weights for each loss component.
        """
        self.initial_weights = initial_weights.copy()
        self.current_weights = initial_weights.copy()
        
    def step(self, epoch: int, loss_components: Dict[str, float]) -> Dict[str, float]:
        """
        Update loss weights based on the current epoch and loss components.
        
        Args:
            epoch (int): Current training epoch.
            loss_components (Dict[str, float]): Current loss component values.
            
        Returns:
            Dict[str, float]: Updated loss weights.
        """
        return self.current_weights

class ConstantWeightScheduler(LossWeightScheduler):
    """
    Constant loss weights that don't change during training.
    """
    def __init__(self, weights: Dict[str, float]):
        """
        Initialize constant weight scheduler.
        
        Args:
            weights (Dict[str, float]): Fixed weights for each loss component.
        """
        super().__init__(weights)
        
    def step(self, epoch: int, loss_components: Dict[str, float]) -> Dict[str, float]:
        """
        Return constant weights.
        
        Args:
            epoch (int): Current training epoch.
            loss_components (Dict[str, float]): Current loss component values.
            
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
                 schedule: List[Dict[str, Union[float, int]]]):
        """
        Initialize manual weight scheduler.
        
        Args:
            initial_weights (Dict[str, float]): Initial weights for each loss component.
            schedule (List[Dict[str, Union[float, int]]]): List of dictionaries containing epoch numbers and weight updates.
                Each dict should have 'epoch' key and weight keys for components to update.
                Example: [{'epoch': 100, 'pde': 0.5}, {'epoch': 200, 'data': 2.0}]
        """
        super().__init__(initial_weights)
        self.schedule = sorted(schedule, key=lambda x: x['epoch'])
        self.current_schedule_idx = 0
        
    def step(self, epoch: int, loss_components: Dict[str, float]) -> Dict[str, float]:
        """
        Update weights according to the manual schedule.
        
        Args:
            epoch (int): Current training epoch.
            loss_components (Dict[str, float]): Current loss component values.
            
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
                        self.current_weights['initial'] = value
                    elif key == 'boundary_condition':
                        self.current_weights['boundary'] = value
                    elif key == 'data_points':
                        self.current_weights['data'] = value
                    else:
                        self.current_weights[key] = value
            self.current_schedule_idx += 1
            
        return self.current_weights.copy()

class GradNormScheduler(LossWeightScheduler):
    """
    Gradient normalization based weight scheduling.
    """
    def __init__(self, 
                 initial_weights: Dict[str, float],
                 alpha: float = 1.5,
                 learning_rate: float = 0.01):
        """
        Initialize GradNorm scheduler.
        
        Args:
            initial_weights (Dict[str, float]): Initial weights for each loss component.
            alpha (float): Balancing factor for GradNorm.
            learning_rate (float): Learning rate for weight updates.
        """
        super().__init__(initial_weights)
        self.alpha = alpha
        self.lr = learning_rate
        self.initial_losses = None
        
    def step(self, epoch: int, loss_components: Dict[str, float]) -> Dict[str, float]:
        """
        Update weights using GradNorm algorithm.
        
        Args:
            epoch (int): Current training epoch.
            loss_components (Dict[str, float]): Current loss component values.
            
        Returns:
            Dict[str, float]: Updated loss weights.
        """
        if self.initial_losses is None:
            self.initial_losses = {k: v for k, v in loss_components.items() 
                                 if k in self.current_weights}
            return self.current_weights
            
        # Calculate relative inverse training rates
        relative_inverse_rates = {}
        for key in self.current_weights:
            if key in loss_components and key in self.initial_losses:
                relative_inverse_rates[key] = (loss_components[key] / 
                                             self.initial_losses[key]) ** self.alpha
                
        # Calculate mean relative inverse rate
        mean_rate = np.mean(list(relative_inverse_rates.values()))
        
        # Update weights
        for key in self.current_weights:
            if key in relative_inverse_rates:
                grad_norm = abs(relative_inverse_rates[key] - mean_rate)
                self.current_weights[key] += self.lr * grad_norm
                
        # Normalize weights
        total = sum(self.current_weights.values())
        self.current_weights = {k: v/total for k, v in self.current_weights.items()}
        
        return self.current_weights

class SoftAdaptScheduler(LossWeightScheduler):
    """
    SoftAdapt weight scheduling based on exponential moving average of loss ratios.
    """
    def __init__(self, 
                 initial_weights: Dict[str, float],
                 beta: float = 0.9):
        """
        Initialize SoftAdapt scheduler.
        
        Args:
            initial_weights (Dict[str, float]): Initial weights for each loss component.
            beta (float): Smoothing factor for exponential moving average.
        """
        super().__init__(initial_weights)
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
            self.previous_losses = {k: v for k, v in loss_components.items() 
                                  if k in self.current_weights}
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
        self.previous_losses = {k: v for k, v in loss_components.items() 
                              if k in self.current_weights}
        
        return self.current_weights 