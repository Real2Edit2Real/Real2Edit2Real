import sys
import logging
import atexit
import os
from datetime import datetime


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def print_and_save_logging(log_dir, rank):
    """
    Simultaneously save all print/logging output to a log file and the screen.
    The log filename automatically includes the rank and timestamp.
    """
    os.makedirs(log_dir, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"log_rank{rank}_{now}.txt")

    log_file = open(log_filename, "w", buffering=1)
    atexit.register(log_file.close)

    # Output to screen and file simultaneously
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    # Bind logging output to the log file as well (console still displays through print)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(log_file)
        ]
    )

    print(f"[INFO] Logging initialized at: {log_filename}")
    return log_file
