import logging
import sys
import os

def make_logger_old():
    """
    Make logger to log status updates, warnings, and other important info
    :return: logging.Logger able to print info to stdout and problems to stderr
    """  # TODO Incorporate pprint to make printed JSONs/dicts more readable
    fmt = "\n%(levelname)s %(asctime)s: %(message)s"
    logging.basicConfig(stream=sys.stdout, format=fmt, level=logging.INFO)  
    logging.basicConfig(stream=sys.stderr, format=fmt, level=logging.ERROR)
    logging.basicConfig(stream=sys.stderr, format=fmt, level=logging.WARNING)
    return logging.getLogger(os.path.basename(sys.argv[0]))

def make_logger():
    log = logging.getLogger("log-test")

    # Create standard format for log statements
    format = "\n%(levelname)s %(asctime)s: %(message)s"
    formatter = logging.Formatter(format)

    # Set log level based on user input
    # if args.verbose:
    #     level = logging.INFO
    # elif args.debug:
    #     level = logging.DEBUG
    # else:
    #     level = logging.ERROR
    # log.setLevel(level)

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

LOGGER = make_logger()