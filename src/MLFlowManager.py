import mlflow
import logging

class MLFlowManager:
    """
    Manages MLFlow experiment tracking.
    """
    # TODO: Add config (if neccessary)
    def __init__(self, experiment_name: str = "dp-hair-training"):
        self.experiment_name = experiment_name
        self._setup_mlflow()
        
    def _setup_mlflow(self):
        """Configure MLFlow tracking URI and experiment."""
        url = "http://localhost:5000"
        mlflow.set_tracking_uri(url)
        mlflow.set_experiment(self.experiment_name)
        logging.info(f"Server is running at {url}")
        
    def start_run(self):
        """Start MLflow and launch UI."""
        mlflow.start_run()
        
    def log_metrics(self, metrics: dict, step: int):
        """Log metrics to MLFlow."""
        mlflow.log_metrics(metrics, step=step)
        
    def log_params(self, params: dict):
        """Log parameters to MLFlow."""
        mlflow.log_params(params)
        
    def end_run(self):
        """End current MLFlow."""
        mlflow.end_run()
        