import logging
import os

# Define log directory
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Configure logging
def get_logger(name):
    """Returns a configured logger."""
    logger = logging.getLogger(name)
    
    # Set the logging level
    logger.setLevel(logging.INFO)
    
    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "app.log"))
    file_handler.setLevel(logging.INFO)
    
    # Create a stream handler to output logs to the terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    
    # Define the log format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    
    # Apply formatter to both handlers
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger
