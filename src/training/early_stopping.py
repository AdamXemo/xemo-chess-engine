"""
Early stopping utility for training.

Monitors validation loss and stops training if no improvement is seen
for a specified number of epochs.
"""


class EarlyStopping:
    """
    Early stopping to terminate training when validation loss stops improving.
    
    Monitors a metric (typically validation loss) and stops training if no
    improvement is observed for `patience` epochs.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for metrics that should decrease (loss), 
                  'max' for metrics that should increase (accuracy)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.improved = False
        
        # Set comparison function based on mode
        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
            self.best_score = float('inf')
        elif mode == 'max':
            self.is_better = lambda current, best: current > best + min_delta
            self.best_score = float('-inf')
        else:
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value (e.g., validation loss)
            
        Returns:
            True if training should stop, False otherwise
        """
        self.improved = False
        
        # Check if this is an improvement
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self.improved = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.early_stop = False
        self.improved = False
        if self.mode == 'min':
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')
    
    def state_dict(self) -> dict:
        """
        Get state dictionary for checkpointing.
        
        Returns:
            Dictionary containing early stopping state
        """
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'mode': self.mode,
        }
    
    def load_state_dict(self, state_dict: dict):
        """
        Load state from dictionary.
        
        Args:
            state_dict: Dictionary containing early stopping state
        """
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.early_stop = state_dict['early_stop']
        self.patience = state_dict['patience']
        self.min_delta = state_dict['min_delta']
        self.mode = state_dict['mode']
        
        # Reset comparison function
        if self.mode == 'min':
            self.is_better = lambda current, best: current < best - self.min_delta
        else:
            self.is_better = lambda current, best: current > best + self.min_delta

