"""
Package of Signal Processing Utilities.
author: Craig Weaver
"""
import logging
from rich.logging import RichHandler
from rich.console import Console
#from rich import inspect

VERSION = '0.1.0'

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

loglevels = {
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}

loggers_propagate = {
    "PIL": False,
    "matplotlib": False,
}


def setup_logging(rootlogger, level, logfile=None):

    rootlogger.setLevel(logging.DEBUG)

    fmt_str = "[%(levelname)6s] (%(filename)s:%(lineno)s) %(message)s"

    if logfile:
        fh = logging.FileHandler(logfile, mode='w')
        fmt = logging.Formatter(fmt=fmt_str)

        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        rootlogger.addHandler(fh)

    con = Console()
    if con.is_terminal:
        ch = RichHandler(rich_tracebacks=True, show_time=False)
    else:
        ch = logging.StreamHandler()
        fmt = logging.Formatter(fmt=fmt_str)
        ch.setFormatter(fmt)

    ch.setLevel(loglevels[level])
    rootlogger.addHandler(ch)

    for logger_name in loggers_propagate:
        l = logging.getLogger(logger_name)
        l.propagate = loggers_propagate[logger_name]
