from datetime import datetime
import logging
from pathlib import Path

folder = Path(__file__).resolve().parent / "logger_folder"
folder.mkdir(parents=True, exist_ok=True)

log_filename = folder / f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

logger = logging.getLogger("open_posttraining_logger")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler(log_filename)
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.propagate = False

