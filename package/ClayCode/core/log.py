import logging
import logging.config
import shutil
from pathlib import Path

__all__ = ["logger"]


class ClayCodeLogger(logging.Logger):
    logging.basicConfig(format="%(message)s", level=logging.INFO, force=True)
    logging.captureWarnings(True)

    def __init__(self, name, level=logging.INFO):
        super().__init__(name, level=level)
        self.logfilename = Path(".claycode.log")
        file_handler = logging.FileHandler(self.logfilename, "w")
        file_handler.setLevel(level=level)
        self.addHandler(file_handler)

    def set_file_name(self, new_filepath, new_filename):
        new_filename = (Path(new_filepath) / new_filename).with_suffix(".log")
        if len(self.handlers) != 0:
            for handler in self.handlers:
                if isinstance(handler, logging.FileHandler):
                    file_handler = handler
                    break
            if file_handler:
                file_handler.close()
                shutil.move(self.logfilename, new_filename)
                self.removeHandler(file_handler)
        new_file_handler = logging.FileHandler(new_filename, "a")
        # Set the same formatter and level as the old file handler
        new_file_handler.setFormatter(file_handler.formatter)
        new_file_handler.setLevel(file_handler.level)
        # Add the new file handler to the logger
        self.addHandler(new_file_handler)


logging.setLoggerClass(ClayCodeLogger)
logger = logging.getLogger("ClayCode")
