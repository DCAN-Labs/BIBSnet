import logging
import sys

VERBOSE_LEVEL_NUM = 15
SUBPROCESS_LEVEL_NUM = VERBOSE_LEVEL_NUM -1

def make_logger():

    # Add new logging level between DEBUG and INFO
    logging.addLevelName(VERBOSE_LEVEL_NUM, "VERBOSE")
    def verbose(self, message, *args, **kws):
        if self.isEnabledFor(VERBOSE_LEVEL_NUM):
            self._log(VERBOSE_LEVEL_NUM, message, args, **kws)
    logging.Logger.verbose = verbose

    # Add new logging level for subprocesses
    logging.addLevelName(SUBPROCESS_LEVEL_NUM, "SUBPROCESS")
    def subprocess(self, message, *args, **kws):
        if self.isEnabledFor(SUBPROCESS_LEVEL_NUM):
            self._log(SUBPROCESS_LEVEL_NUM, message, args, **kws)
    logging.Logger.subprocess = subprocess
    
    log = logging.getLogger("BIBSnet")

    # Create standard format for log statements
    format = "\n%(levelname)s %(asctime)s: %(message)s"
    formatter = logging.Formatter(format)
    subprocess_format = "%(id)s %(asctime)s: %(message)s"
    subprocess_formatter = logging.Formatter(subprocess_format)

    # Redirect INFO and DEBUG to stdout
    handle_out = logging.StreamHandler(sys.stdout)
    handle_out.setLevel(logging.DEBUG)
    handle_out.addFilter(lambda record: record.levelno <= logging.INFO)
    handle_out.addFilter(lambda record: record.levelno != SUBPROCESS_LEVEL_NUM)
    handle_out.setFormatter(formatter)
    log.addHandler(handle_out)

    # Set special format for subprocess level
    handle_subprocess = logging.StreamHandler(sys.stdout)
    handle_subprocess.setLevel(SUBPROCESS_LEVEL_NUM)
    handle_subprocess.addFilter(lambda record: record.levelno <= SUBPROCESS_LEVEL_NUM)
    handle_subprocess.setFormatter(subprocess_formatter)
    log.addHandler(handle_subprocess)

    # Redirect WARNING+ to stderr
    handle_err = logging.StreamHandler(sys.stderr)
    handle_err.setLevel(logging.WARNING)
    handle_err.setFormatter(formatter)
    log.addHandler(handle_err)

    return log

LOGGER = make_logger()
