from __future__ import print_function
import os
import pip
import shutil
import sys
import logging
import logging.handlers
import messages
import shlex
import datetime
from subprocess import PIPE, Popen


def splitnifti(path):
    while '.nii' in path:
        path = os.path.splitext(path)[0]
    return str(path)


def splitext(path):
    while '.nii' in path:
        path = os.path.splitext(path)[0]
    return str(path)


def shell(cmd):
    """ Execute shell command

    :param cmd: str, command to execute
    :return: stdout, error
    """
    try:
        processor = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
        out, err = processor.communicate()
        return out, err
    except OSError as e:
        raiseerror(messages.Errors.InputValueError, 'Command can not be executed.')


def scheduler(cmd, type='slurm'):
    """ Execute shell command through scheduler for cluster computing
    :param cmd: command to execute
    :return: stdout, error
    """
    if type == 'slurm':
        pass
    else:
        pass
    return cmd


def get_logger(path, name):
    today = "".join(str(datetime.date.today()).split('-'))
    # create logger
    logger = logging.getLogger('{0}Logger'.format(name))
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(path, '{0}-{1}.log'.format(name, today)))
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
    """ make all given directories
    :param paths: directories want to make
    :type paths: str[,str,..]
    """
    for path in paths:
        basedir = os.path.dirname(path)
        if basedir:
            if not os.path.exists(basedir):
                parentdir = os.path.dirname(basedir)
                if not os.path.exists(parentdir):
                    try:
                        os.mkdir(parentdir)
                    except:
                        raiseerror(messages.InputPathError, '{} is not exists'.format(os.path.dirname(parentdir)))
                try:
                    os.mkdir(basedir)
                except:
                    pass
        if not (os.path.exists(path) and os.path.isdir(path)):
            try:
                os.mkdir(path)
            except:
                pass


def path_splitter(path):
    """ Split path sting into list

    :param path: path string want to split
    :type path: str
    """
    return path.strip(os.sep).split(os.sep)


def copyfile(output_path, input_path):
    """ copy files from input_path to output_path

    :param output_path: destination path
    :param input_path: origin path
    :type output_path: str
    :type input_path: str
    """
    shutil.copyfile(input_path, output_path)


def update(*args):
    """ Upgrade python module

    :param args: list of python modules want to upgrade. if not given, this method upgrade pynit module
    :type args: str, ...
    """
    if not len(args):
        pip.main(['install', '--upgrade', 'pynit'])
    else:
        pip.main(['install', '--upgrade'] + args)


def install(*args):
    """ Install python module
    :param args: list of python modules want to install
    :type args: str, ...
    """
    pip.main(['install'] + args)