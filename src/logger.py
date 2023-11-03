import logging
import sys

IMPORTANT_LEVEL_NUM = 25

def make_logger():

    # Add new logging level between INFO and WARNING
    logging.addLevelName(IMPORTANT_LEVEL_NUM, "IMPORTANT")
    def important(self, message, *args, **kws):
        if self.isEnabledFor(IMPORTANT_LEVEL_NUM):
            self._log(IMPORTANT_LEVEL_NUM, message, args, **kws)
    logging.Logger.important = important
    
    log = logging.getLogger("bibsnet")

    # Create standard format for log statements
    format = "\n%(levelname)s %(asctime)s: %(message)s"
    formatter = logging.Formatter(format)

    # Redirect INFO and DEBUG to stdout
    handle_out = logging.StreamHandler(sys.stdout)
    handle_out.setLevel(logging.DEBUG)
    handle_out.addFilter(lambda record: record.levelno <= IMPORTANT_LEVEL_NUM)
    handle_out.setFormatter(formatter)
    log.addHandler(handle_out)

    # Redirect WARNING+ to stderr
    handle_err = logging.StreamHandler(sys.stderr)
    handle_err.setLevel(logging.WARNING)
    handle_err.setFormatter(formatter)
    log.addHandler(handle_err)

    return log

LOGGER = make_logger()