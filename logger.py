from datetime import datetime
import logging
from pathlib import Path

folder = Path("logger_folder")
folder.mkdir(parents=True, exist_ok=True)

log_filename = folder / f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

