import os
import sys
import logging


# Logging formatter supporting colored output
class LogFormatter(logging.Formatter):
    COLOR_CODES = {
        logging.CRITICAL: "\033[1;35m",  # bright/bold magenta
        logging.ERROR: "\033[1;31m",  # bright/bold red
        logging.WARNING: "\033[1;33m",  # bright/bold yellow
        logging.INFO: "\033[1;37m",  # white / light gray
        logging.DEBUG: "\033[0;37m"  # bright/bold black / dark gray
    }

    RESET_CODE = "\033[0m"

    def __init__(self, color, *args, **kwargs):
        super(LogFormatter, self).__init__(*args, **kwargs)
        self.color = color

    def format(self, record, *args, **kwargs):
        if self.color == True and record.levelno in self.COLOR_CODES:
            record.color_on = self.COLOR_CODES[record.levelno]
            record.color_off = self.RESET_CODE
        else:
            record.color_on = ""
            record.color_off = ""
        return super(LogFormatter, self).format(record, *args, **kwargs)


# Setup logging
def setup_logging(console_log_output, console_log_level, console_log_color, logfile_file, logfile_log_level,
                  logfile_log_color, log_line_template):
    # Create logger
    # For simplicity, we use the root logger, i.e. call 'logging.getLogger()'
    # without name argument. This way we can simply use module methods for
    # for logging throughout the script. An alternative would be exporting
    # the logger, i.e. 'global logger; logger = logging.getLogger("<name>")'
    #logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    # Set global log level to 'debug' (required for handler levels to work)
    logger.setLevel(logging.INFO)

    # Create console handler
    console_log_output = console_log_output.lower()
    if console_log_output == "stdout":
        console_log_output = sys.stdout
    elif console_log_output == "stderr":
        console_log_output = sys.stderr
    else:
        print("Failed to set console output: invalid output: '%s'" % console_log_output)
        return False
    console_handler = logging.StreamHandler(console_log_output)

    # Set console log level
    try:
        console_handler.setLevel(console_log_level.upper())  # only accepts uppercase level names
    except:
        print("Failed to set console log level: invalid level: '%s'" % console_log_level)
        return False

    # Create and set formatter, add console handler to logger
    console_formatter = LogFormatter(fmt=log_line_template, color=console_log_color)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # Create log file handler
    try:
        logfile_handler = logging.FileHandler(logfile_file)
    except Exception as exception:
        print("Failed to set up log file: %s" % str(exception))
        return False

    # Set log file log level
    try:
        logfile_handler.setLevel(logfile_log_level.upper())  # only accepts uppercase level names
    except:
        print("Failed to set log file log level: invalid level: '%s'" % logfile_log_level)
        return False

    # Create and set formatter, add log file handler to logger
    logfile_formatter = LogFormatter(fmt=log_line_template, color=logfile_log_color)
    logfile_handler.setFormatter(logfile_formatter)
    logger.addHandler(logfile_handler)

    # Success
    return True


def live_trader_setup_logging(script_name: str):
    if (not setup_logging(console_log_output="stdout", console_log_level="warning", console_log_color=True,
                          logfile_file=script_name + ".log", logfile_log_level="debug", logfile_log_color=False,
                          log_line_template="%(color_on)s%(asctime)s  %(levelname)-8s  %(threadName)s  %(message)s%(color_off)s")):
        print("Failed to setup logging, aborting.")
        quit()



if __name__ == '__main__':
    live_trader_setup_logging(script_name=os.path.splitext(os.path.basename(sys.argv[0]))[0])

    # Log some messages
    """logging.critical("Critical message")
    logging.error("Error message")
    logging.warning("Warning message")
    logging.info("Info message")
    logging.debug("Debug message")
    print("End")
    quit()
    """

    #logging.basicConfig(
    #    format='%(asctime)s %(levelname)-8s %(message)s',
    #    level=logging.INFO,
    #    datefmt='%Y-%m-%d %H:%M:%S')
    #logging.Formatter.converter = time.gmtime
    #main()
