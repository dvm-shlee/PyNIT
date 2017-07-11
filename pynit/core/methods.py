from __future__ import print_function
import os
import shutil
import sys
import logging
import logging.handlers
import messages
import shlex
from subprocess import PIPE, Popen


def splitnifti(path):
    while '.nii' in path:
        path = os.path.splitext(path)[0]
    return str(path)


def shell(cmd):
    """Execute shell command

    :param cmd: str, command to execute
    :return: stdout, error
    """
    try:
        processor = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
        out, err = processor.communicate()
        return out, err
    except OSError as e:
        raiseerror(messages.Errors.InputValueError, 'Command can not be executed.')


def get_logger(path, name):
    # create logger
    logger = logging.getLogger('{0}Logger'.format(name))
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(path, '{0}.log'.format(name)))
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handler to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def raiseerror(exception, message):
    """ Raise User friendly error message
    """
    try:
        raise exception(message)
    except Exception as e:
        sys.stderr.write("ERROR({0}): {1}\n".format(e.__doc__, e.message))
        messages.warnings.simplefilter("ignore")
        sys.exit()


def mkdir(*paths):
    """ Make list of directories
    """
    for path in paths:
        try:
            os.mkdir(path)
        except:
            pass


def path_splitter(path):
    """ Split path structure into list
    """
    return path.strip(os.sep).split(os.sep)


def copyfile(output_path, input_path):
    """ Copy File
    """
    shutil.copyfile(input_path, output_path)