from pathlib import Path
import math
import random

import torch
import numpy as np
import mlflow

def log_training_metrics():
    # Compute metrics
    def mean_safe(x) -> float:
        return np.mean(x) if len(x) > 0 else 0.0    


def set_global_seeds(seed=42):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)