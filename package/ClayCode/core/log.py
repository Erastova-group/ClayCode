import logging
import logging.config
import os
import random
import re
from pathlib import Path

from ClayCode.core.consts import exec_date, exec_datetime, exec_time

__all__ = ["ClayCodeLogger"]


class ClayCodeLogger(logging.Logger):
    logging.basicConfig(format="%(message)s", level=logging.INFO, force=False)
    logging.captureWarnings(True)
    _logfilename = Path(".logfile")
    _logfilename = _logfilename.with_stem(
        f"{_logfilename.stem}_{random.randrange(0, 9999):04}"
    )
    _name = __package__.split(".")[0]

    def __init__(self, name, level=logging.INFO):
        super().__init__(name, level=level)
        self.logfilename = self.__class__._logfilename.with_suffix(".log")
        file_handler = logging.FileHandler(
            self.logfilename, "w", encoding="UTF-8"
        )
        file_handler.setLevel(level=level)
        self.addHandler(file_handler)
        self.info(f"{self.__class__._name} - {exec_date} - {exec_time}")

    def set_file_name(self, new_filepath=None, new_filename=None, final=False):
        if new_filename is None:
            new_filename = self.logfilename.name
            new_filepath = self.logfilename.parent
        elif new_filepath is None:
            new_filename = Path(new_filename)
            new_filepath = new_filename.parent
            new_filename = new_filename.name
        if not final:
            stem_suffix = f"_{exec_datetime}"
        else:
            stem_suffix = ""
        new_filename = (
            Path(new_filepath) / f"{new_filename}{stem_suffix}"
        ).with_suffix(".log")
        parent_instance = self
        while parent_instance.parent.name != "root":
            parent_instance = parent_instance.parent
        child_loggers = [
            logger_instance
            for logger, logger_instance in parent_instance.manager.loggerDict.items()
            if (
                re.match(self._name, logger) is not None
                and type(logger_instance) != logging.PlaceHolder
            )
        ]
        for instance in [parent_instance, *child_loggers]:
            if instance.logfilename != new_filename:
                if len(instance.handlers) != 0:
                    for handler in instance.handlers:
                        if isinstance(handler, logging.FileHandler):
                            file_handler = handler
                            break
                    if file_handler:
                        file_handler.close()
                        try:
                            with open(instance.logfilename, "r") as old_file:
                                old_log = old_file.read()
                            parent_instance.debug(
                                f"{instance.name}: Moving {instance.logfilename.resolve()} to {new_filename.resolve()}"
                            )
                        except FileNotFoundError:
                            pass
                        else:
                            with open(new_filename, "a") as new_file:
                                new_file.write(old_log)
                                os.unlink(instance.logfilename)
                        instance.logfilename = new_filename
                        instance.removeHandler(file_handler)
                new_file_handler = logging.FileHandler(new_filename, "a")
                # Set the same formatter and level as the old file handler
                new_file_handler.setFormatter(file_handler.formatter)
                new_file_handler.setLevel(file_handler.level)
                # Add the new file handler to the logger
                instance.addHandler(new_file_handler)
        parent_instance.__class__._logfilename = (
            self.logfilename
        ) = new_filename
