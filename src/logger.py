import logging
import sys

VERBOSE_LEVEL_NUM = 15

def make_logger(name):

    # Add new logging level between DEBUG and INFO
    logging.addLevelName(VERBOSE_LEVEL_NUM, "VERBOSE")
    def verbose(self, message, *args, **kws):
        if self.isEnabledFor(VERBOSE_LEVEL_NUM):
            self._log(VERBOSE_LEVEL_NUM, message, args, **kws)
    logging.Logger.verbose = verbose
    
    log = logging.getLogger(name)

    # Create standard format for log statements
    format = "\n%(name)s %(levelname)s %(asctime)s: %(message)s"
    formatter = logging.Formatter(format)

    # Redirect INFO and DEBUG to stdout
    handle_out = logging.StreamHandler(sys.stdout)
    handle_out.setLevel(logging.DEBUG)
    handle_out.addFilter(lambda record: record.levelno <= logging.INFO)
    handle_out.setFormatter(formatter)
    log.addHandler(handle_out)

    # Redirect WARNING+ to stderr
    handle_err = logging.StreamHandler(sys.stderr)
    handle_err.setLevel(logging.WARNING)
    handle_err.setFormatter(formatter)
    log.addHandler(handle_err)

    return log

LOGGER = make_logger("BIBSnet")
