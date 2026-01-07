import logging

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)-8s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler()  # Log to the console
    ]
)
