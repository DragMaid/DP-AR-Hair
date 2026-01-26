import mlflow
import logging
from dataclasses import dataclass

@dataclass
class MLFlowConfig:
    tracking_uri: str = "http://localhost:5000"

class MLFlowManager:
    """
    Manages MLFlow experiment tracking.
    """
    
    def __init__(self, config: MLFlowConfig, experiment_name: str = "dp-hair-training"):
        self.experiment_name = experiment_name
        self.config = config
        self._setup_mlflow()
        
    def _setup_mlflow(self):
        """Configure MLFlow tracking URI and experiment."""
        uri = self.config.tracking_uri
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(self.experiment_name)
        logging.info(f"Server is running at {uri}")
        
    def start_run(self):
        """Start MLflow and launch UI."""
        mlflow.start_run()
    
    def log_metric(self, metric: str, value: float, step: int = None):
        """Log a metric to MLFlow"""
        mlflow.log_metric(metric, value, step)
            
    def log_params(self, params: dict):
        """Log parameters to MLFlow."""
        mlflow.log_params(params)
        
    def end_run(self):
        """End current MLFlow."""
        mlflow.end_run()
        