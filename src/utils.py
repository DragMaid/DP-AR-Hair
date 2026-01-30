import mlflow
import logging


class MLFlowManager:
    """
    Manages MLFlow experiment tracking.
    """

    def __init__(self,
                 uri: str = None,
                 enabled=True,
                 experiment_name: str = "dp-hair-training"):
        self.uri = "http://localhost:5000" if not uri else uri
        self.enabled = enabled
        self.experiment_name = experiment_name
        self._setup_mlflow()

    def start_run(self):
        if self.enabled:
            return mlflow.start_run()

    def _setup_mlflow(self):
        """Configure MLFlow tracking URI and experiment."""
        if self.enabled:
            mlflow.set_tracking_uri(self.uri)
            mlflow.set_experiment(self.experiment_name)
            logging.info(f"MLflow Tracking Server is running at {self.uri}")

    def log_metric(self, *args, **kwargs):
        if self.enabled:
            mlflow.log_metric(*args, **kwargs)

    def log_artifact(self, *args, **kwargs):
        if self.enabled:
            mlflow.log_artifact(*args, **kwargs)


if __name__ == "__main__":
    from time import sleep
    from random import randint
    mlflow_manager = MLFlowManager()

    with mlflow_manager.start_run():
        step = 0
        while True:
            step += 1
            logs = {"a1/gen/feature1": randint(0, 10),
                    "a2/disc/feature2": randint(0, 10),
                    "a2/disc/feature3": randint(0, 10),
                    "a2/gen/feature3": randint(0, 10)}
            for metric, value in logs.items():
                mlflow_manager.log_metric(metric, value, step=step)
            sleep(1)
