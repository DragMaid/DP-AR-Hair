import logging
from training.trainer import MLFlowManager
from src.configs import model_config


class Trainer:
    """
    Trainer for the DP-AR Hair model
    """
    
    def __init__(self, config):
        self.cfg = model_config
        self.mlflow_manager = MLFlowManager(config)
        logging.info(f"Running on device: {self.cfg.HairGanParams.device}")
        
    def train(self):
        """Execute the main training loop."""
        
    