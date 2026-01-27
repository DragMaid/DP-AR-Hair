import mlflow
import logging


class MLFlowManager:
    """
    Manages MLFlow experiment tracking.
    """

    def __init__(self, uri: str = None, experiment_name: str = "dp-hair-training"):
        self.uri = "http://localhost:5000" if not uri else uri
        self.experiment_name = experiment_name
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Configure MLFlow tracking URI and experiment."""
        mlflow.set_tracking_uri(self.uri)
        mlflow.set_experiment(self.experiment_name)
        logging.info(f"MLflow Tracking Server is running at {self.uri}")

    def __enter__(self):
        """Start MLflow logging"""
        mlflow.start_run()

    def __exit__(self):
        """Stop MLflow logging"""
        mlflow.end_run()

    def log_metric(self, metric: str, value: float, step: int = None):
        """Log a metric to MLFlow"""
        mlflow.log_metric(metric, value, step)

    def log_params(self, params: dict):
        """Log parameters to MLFlow."""
        mlflow.log_params(params)


if __name__ == "__main__":
    from time import sleep
    from random import randint
    mlflow_manager = MLFlowManager()

    with mlflow_manager:
        step = 0
        while True:
            step += 1
            logs = {"feature1": randint(0, 10),
                    "feature2": randint(0, 10)}
            for metric, value in logs.items():
                mlflow_manager.log_metric(metric, value, step)
            sleep(1)
