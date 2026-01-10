import logging
from pathlib import Path

def get_logger(name="project_logger", log_file="logs/project.log", level=logging.INFO):
    """
    Returns a configured logger that writes to a log file and prints to console.
    """
    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(exist_ok=True, parents=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:  # prevent duplicate handlers if called multiple times
        # File handler
        fh = logging.FileHandler(log_file, mode='a')
        fh.setLevel(level)
        fh_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

    return logger
