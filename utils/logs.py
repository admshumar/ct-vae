import os
import sys


def begin_logging(directory):
    log_filename = os.path.abspath(os.path.join(directory, 'experiment.log'))
    log_error_filename = os.path.abspath(os.path.join(directory, 'error.log'))
    sys.stdout = open(log_filename, "w")
    sys.stderr = open(log_error_filename, "w")


def end_logging():
    sys.stdout.close()
