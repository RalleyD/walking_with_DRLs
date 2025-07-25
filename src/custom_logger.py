import logging
import sys
from time import strftime, localtime
from pathlib import Path
from src.util.plotter import PRJ_ROOT


class CustomLogger(object):
    """
    A singular logger for the project.
    Consolidating logging from all modules in the package.

    methods:
        get_logger: return a logging instance from the object instance
        get_project_logger: return a logging instance from the class object.
    """

    def __init__(self, name="default"):
        """
        Class constructor

        Args:
            name (str): Friendly name for the logging instance
        """
        self._name = name
        self._logger = logging.getLogger(self._name)
        self._logger.setLevel(logging.DEBUG)

    @classmethod
    def get_project_logger(cls, name="walker-drl") -> logging.Logger:
        """
        Provides a singular logging instance for the class object

        For the project, this is the way it should be used.
        It provides a logging instance which can be used by all modules
        in the project. Hence, the classmethod.

        Sets up the logger if one is not already created.
        Args:
            name (str): name of the logging instance. Default is 'default'
        Returns:
            class logging instance.
        """
        # setup logger instance
        instance = cls(name)
        logger = instance._logger

        if len(logger.handlers) == 0:
            # setup output formatter
            format_s = "%(asctime)s,%(msecs)03d:%(module)s:%(funcName)s:%(levelname)s: %(message)s"
            format_hdlr = logging.Formatter(
                format_s,
                style='%',
                # human readable!
                datefmt="%Y-%b-%d %H:%M:%S"
            )

            # setup stdout streamer
            console_hdlr = logging.StreamHandler()
            console_hdlr.setFormatter(format_hdlr)

            # setup file streamer
            file_hdlr = instance._set_file_stream(format_hdlr, name)

            # add handlers to project logger
            logger.addHandler(console_hdlr)
            logger.addHandler(file_hdlr)
            # don't propagate to the root handler, this avoids the root logger sending our logs to stderr
            logger.propagate = False

        return logger

    def _set_file_stream(self, fmt_hdlr: logging.Formatter, name="walker-drl") -> logging.FileHandler:
        """
        Setup the logging file stream handler

        Args:
            format_hdlr (Formatter): logging Formatter instance.
            name (str): name of the logger.

        Returns:
            FileHandler instance
        """
        # find logging directory
        log_dir = "logs"
        log_path = PRJ_ROOT / log_dir

        log_path.mkdir(exist_ok=True)

        # create filename
        log_name = f"{strftime('%Y-%m-%d_%H-%M-%S', localtime())}-{name}.log"

        # create file handler
        file_hdlr = logging.FileHandler(
            filename=f"{log_path}/{log_name}",
            mode='w',
            encoding='utf8'
        )

        # set log level
        file_hdlr.setLevel(logging.DEBUG)
        # apply formatter
        file_hdlr.setFormatter(fmt_hdlr)

        return file_hdlr
