import os
import re
import sys
from collections import namedtuple
import multiprocessing
from multiprocessing.pool import ThreadPool
from ..tools import messages
from ..tools import methods

#########################################
# The imported modules belows           #
# check jupyter notebook environment    #
#########################################
jupyter_env = False
try:
    if len([key for key in sys.modules.keys() if 'ipykernel' in key]):
        from tqdm import tqdm_notebook as progressbar
        from ipywidgets import widgets
        from ipywidgets.widgets import HTML as title
        from IPython.display import display, display_html
        jupyter_env = True
    else:
        from tqdm import tqdm as progressbar
except:
    pass

#########################################
# The imported modules belows are       #
# utilized in Step instance             #
#########################################
import json
import numpy as np
from tempfile import mkdtemp
from StringIO import StringIO
from time import sleep
from shutil import rmtree
import pandas as pd


def check_dataclass(path):
    """ This method checks the dataclass of input path and
    return the dataclass classifier and basepath name of the input path

    :param path: input path
    :type path: str
    :return: dataclass, path
    :rtype: int, str
    """
    if os.path.exists(path):
        dataclass = 1
        path = methods.path_splitter(path)
        path = path[-1]
    else:
        dataclass = 0
    return dataclass, path


def get_step_name(procobj, step, results=False, verbose=None):
    """ This method checks if the input step had been executed or not.
    If the step already executed, return the name of existing folder,
    if not, the number of order is attached as prefix and will be returned

    :param procobj: process class instance
    :param step: name of the step
    :param results:
    :param verbose: print information
    :type procobj: pynit.Process instance
    :type step: str
    :type results: bool
    :type verbose: bool
    :return: name of the step
    :rtype: str
    """
    if results:
        idx = 2
    else:
        idx = 1
    processing_path = os.path.join(procobj._prjobj.path,
                                   procobj._prjobj.ds_type[idx],
                                   procobj.processing)
    executed_steps = [f for f in os.listdir(processing_path) if os.path.isdir(os.path.join(processing_path, f))]
    if len(executed_steps):
        overlapped = [old_step for old_step in executed_steps if step in old_step]
        if len(overlapped):
            if verbose:
                print('Notice: existing path')
            checked_files = []
            for f in os.walk(os.path.join(processing_path, overlapped[0])):
                checked_files.extend(f[2])
            if len(checked_files):
                if verbose:
                    print('Notice: Last step path is not empty')
            return overlapped[0]
        else:
            return "_".join([str(len(executed_steps) + 1).zfill(3), step])
    else:
        if verbose:
            print('The pipeline [{pipeline}] is initiated'.format(pipeline=procobj.processing))
        return "_".join([str(1).zfill(3), step])


def retreive_obj_from_command(command):
    tmpobjs = [obj.strip('{}') for obj in re.findall(r"[{\w'}]+", command) if obj[0] == '{' and obj[-1] == '}']
    objs = []
    for o in tmpobjs:
        if "}{" in o:
            objs.extend(o.split('}{'))
        else:
            objs.append(o)
    return objs


class Step(object):
    """ Template for a processing step

    This class simply allows you to design processing steps, that needs to combine multiple command line tools in
    several fMRI imaging package such as AFNI, ANTs, and FSL.
    The fundamental mechanism is that by applying given inputs, outputs, and command, this class generating
    customized function and executed it.

    - data structure -
    'dataset'   : the template that storing the structure of an input source
    'oppset'    : the template that storing the structure of an output parameter, such as motion parameter
                and transformation profiles.
    'cmdset'    : the template that storing the structure of executing commands including variables

    """
    dataset = namedtuple('Dataset', ['name', 'input_path', 'static'])       # dataset template
    oppset = namedtuple('OutputParam', ['name', 'prefix', 'ext'])           # output_param template
    cmdset = namedtuple('Command', ['name', 'command', 'option'])           # command template
    mthset = namedtuple('Method', ['name', 'args', 'kwargs'])               # method template

    def __init__(self, procobj, subjects=None, sessions=None):
        """Initiate Step class

        :param procobj:
        """
        self._procobj = procobj                         # load Process object
        self._processing = procobj.processing           # read Process name
        self._tempfiles = []                            # temp file handler
        self._mainset = None                            # main input handler
        self._sidesets = []                             # side inputs handler
        self._staticinput = {}                          # static input handler
        self._outparam = {}                             # output_param handler
        self._cmdstdout = []                            # cmdstdout handler
        if subjects:
            residuals = [subj for subj in subjects if subj not in procobj.subjects]
            if len(residuals):
                methods.raiseerror(messages.Errors.InputValueError,
                                   '{} is(are) not available subject(s)')
            else:
                self._subjects = subjects
        else:
            self._subjects = procobj.subjects[:]            # load all subject list from process object
        self._outputs = {}                              # output handler
        try:
            if sessions:
                residuals = [sess for sess in sessions if sess not in procobj.sessions]
                if len(residuals):
                    methods.raiseerror(messages.Errors.InputValueError,
                                       '{} is(are) not available session(s)')
                else:
                    self._sessions = sessions
            else:
                self._sessions = procobj.sessions[:]        # check single session or not
        except:
            self._sessions = None
        self._commands = []                             # executing commands handler
        self._filters = {'main':[], 'sides':{}, 'extra':{}}         # handler for project obj filtering

    def set_variable(self, name, value=None):
        """

        :param name:
        :param value:
        :return:
        """
        self._filters['extra'][name] = value

    def set_input(self, name, input_path, filters=None, static=False, side=False):
        """Import input dataset

        :param name: str, datatype or absolute path
        :param input_path: str
        :param filters: dict, kw_argment filters
        :param static: boolean, True, if this object need to be looped, if not, only use first index
        :param side: boolean, True, if this object is side prjobj
        :return: None
        """
        dc, ipath = check_dataclass(input_path)
        if side:
            self._sidesets.append(self.dataset(name=name, input_path=ipath, static=static))
            self._filters['sides'][name] = self.get_filtercode(str(dc), ipath, filters)
        else:
            self._mainset = self.dataset(name=name, input_path=input_path, static=static)
            self._filters['main'] = self.get_filtercode(str(dc), ipath, filters)

    def set_staticinput(self, name, value):
        """Import static file

        :param name:
        :param value:
        """
        if isinstance(value, str):
            if os.path.exists(value):
                value = '"{}"'.format(value)
            else:
                value = '{}'.format(value)
        self._staticinput[name] = value

    def set_outparam(self, name, ext, prefix=None):
        """This method set file name of output parameters

        :param name: Variables for parameter output file
        :param ext: Extension of the output file
        :param prefix: If prefix has value, add prefix to output file
        :type name: str
        :type ext: str
        :type prefix: str
        """
        self._outparam[name] = (self.oppset(name=name, prefix=prefix, ext=ext))

    def set_execmethod(self, command, var=None):
        """Set structured command on command handler

        :param command: Structured command with input and output variables
        :param var: Name of variable
        :param idx: Index for replacing the commands on handler
        :type command: str
        :type var: str
        :type idx: idx
        """
        self._commands.append((command, [var]))

    def set_command(self, command, verbose=False, stdout=None ):
        """Set structured command on command handler

        :param command  : str
            Structured command with input and output variables
        :param verbose  : boolean
        :param idx      : int
            Index for replacing the commands on handler
        :param stdout   : str or None
            if True, the input string can be used the variable for stand output results of a command
        :return:
        """
        objs = retreive_obj_from_command(command)
        total = dict([(sideobj.name, sideobj.static) for sideobj in self._sidesets])
        total[self._mainset.name] = self._mainset.static
        if stdout:
            total[stdout] = False
        self._cmdstdout.append(stdout)
        try:
            totalobjs = total.keys()[:]
        except:
            totalobjs = []

        # Get list of residual inputs
        residuals = [obj for obj in sorted(list(set(objs))) if obj not in totalobjs]
        residuals = [obj for obj in residuals if 'temp' not in obj]
        residuals = [obj for obj in residuals if 'output' not in obj]
        residuals = [obj for obj in residuals if 'prefix' not in obj]
        residuals = [obj for obj in residuals if 'sub_path' not in obj]
        residuals = [obj for obj in residuals if obj not in self._staticinput.keys()]
        residuals = [obj for obj in residuals if obj not in self._outparam.keys()]
        residuals = [obj for obj in residuals if obj not in self._cmdstdout]

        # Check accuracy
        if len(residuals):
            methods.raiseerror(ValueError, 'Too many inputs :{0}'.format(str(residuals)))
        output = "'{0}'.format(".format(command)
        str_format = []
        for obj in objs:
            if obj == 'output' or obj == 'prefix' or obj == 'sub_path':
                str_format.append("{0}={0}".format(obj))
            else:
                if 'temp' in obj:
                    str_format.append("{0}=os.path.join(temppath, '{0}.nii')".format(obj))
                    self._tempfiles.append(obj)
                elif obj in self._staticinput.keys():
                    str_format.append("{0}={1}".format(obj, self._staticinput[obj]))
                elif obj in self._outparam.keys():
                    if self._outparam[obj].prefix:
                        str_format.append("{0}='{1}_'+ methods.splitnifti(output)+'{2}'".format(obj,
                                                                                                self._outparam[obj].prefix,
                                                                                                self._outparam[obj].ext))
                    else:
                        str_format.append("{0}=methods.splitnifti(output)+'{1}'".format(obj, self._outparam[obj].ext))
                else:
                    try:
                        if total[obj]:
                            str_format.append("{0}={1}.Abspath".format(obj, obj))
                        else:
                            str_format.append("{0}={1}[i].Abspath".format(obj, obj))
                    except:
                        if obj in self._cmdstdout:
                            str_format.append("{0}={1}".format(obj, obj))
                        else:
                            methods.raiseerror(messages.Errors.InputValueError, "Exception case occured!")

        output = "{0}{1})".format(output, ", ".join(list(set(str_format))))
        self._commands.append((output, stdout))
        if self._tempfiles:
            self._tempfiles = sorted(list(set(self._tempfiles)))
        if verbose:
            return output

    def get_inputcode(self):
        """Put the set inputs values on the lists as a building block of customized function

        :return: str
        """
        inputcode = []
        mainobj = self._mainset
        try:
            sideobjs = self._sidesets[:]
        except:
            sideobjs = None
        try:
            if mainobj.static:
                inputcode = ['{0} = self._prjobj({1})[0]'.format(mainobj.name, self._filters['main'])]
            else:
                inputcode = ['{0} = self._prjobj({1})'.format(mainobj.name, self._filters['main'])]
        except:
            methods.raiseerror(NameError, 'Main input is not defined')
        if sideobjs:
            for sideobj in sideobjs:
                name = sideobj.name
                if sideobj.static:
                    inputcode.append('{0} = self._prjobj({1})[0]'.format(name, self._filters['sides'][name]))
                else:
                    inputcode.append('{0} = self._prjobj({1})'.format(name, self._filters['sides'][name]))
        else:
            pass
        if self._filters['extra']:
            for extra in sorted(self._filters['extra'].keys()):
                inputcode.append('{0} = {1}'.format(extra, self._filters['extra'][extra]))
        return inputcode

    def get_filtercode(self, dataclass, input_path, filters):
        """Generate list of filtering based keywords based on set input values

        :param dataclass:
        :param input_path:
        :param filters:
        :return: str
        """
        if dataclass == '0':
            output_filters = [dataclass, '"{0}"'.format(input_path)]
        else:
            output_filters = [dataclass, '"{0}"'.format(self._processing), '"{0}"'.format(input_path)]
        if self._sessions:
            output_filters.extend(['subj', 'sess'])
        else:
            output_filters.extend(['subj'])
        if isinstance(filters, dict):
            kwargs = ['{key}="{value}"'.format(key=k, value=v) for k, v in filters.items() if isinstance(v, str)]
            output_filters.extend(kwargs)
            kwargs = ['{key}={value}'.format(key=k, value=v) for k, v in filters.items() if isinstance(v, list)]
            output_filters.extend(kwargs)
        else:
            pass
        return ', '.join(output_filters)

    def get_executefunc(self, name, verbose=False):
        """Step function generator

        :param name: str
        :param verbose: boolean
        :return: str
        """
        # Define inputs
        filters = ['\t{}'.format(input) for input in self.get_inputcode()]

        # Depends on the main input type (static or multiple), different structure of function are generated
        if self._mainset.static:    # if main input datasets only need to process first files for each subject
            body = ['\toutputs = []',
                    '\toutput = os.path.join(sub_path, {0}.Filename)'.format(self._mainset.name),
                    '\tprefix_filter = methods.splitnifti(os.path.basename(output))',
                    '\tprefix = methods.splitnifti(output)',
                    '\tflist = [f for f in os.listdir(sub_path)]',
                    '\tif len([f for f in flist if prefix_filter in f]):',
                    '\t\tself.logger.info("Step::Skipped because the file[{0}] is exist.".format(output))',
                    '\telse:']
            for cmd, stdout in self._commands:
                if isinstance(stdout, str):
                    body += ['\t\t{0}, err = methods.shell({1})'.format(stdout, cmd)]
                elif isinstance(stdout, list):
                    if stdout[0]:
                        body += ['\t\t{0} = {1}'.format(stdout[0], cmd)]
                    else:
                        body += ['\t\t{0}'.format(cmd)]
                else:
                    body += ['\t\toutputs.append(methods.shell({0}))'.format(cmd)]
            if self._tempfiles:
                temp = ['\ttemppath = mkdtemp()',
                        '\tself.logger.info("SYS::TempFolder[{0}] is generated".format(temppath))']
                close = ['\trmtree(temppath)',
                         '\tself.logger.info("SYS::TempFolder[{0}] is closed".format(temppath))']
                body = temp + body + close
            else:
                pass
        else:   # if main input datasets need to be looped for each subject
            loop = ['\toutputs = []',
                    '\tfor i in progressbar(range(len({0})), desc="Files", leave=False):'.format(self._mainset.name)]
            body = ['\t\ttemp_outputs = []',
                    '\t\toutput = os.path.join(sub_path, {0}[i].Filename)'.format(self._mainset.name),
                    '\t\tprefix_filter = methods.splitnifti(os.path.basename(output))',
                    '\t\tprefix = methods.splitnifti(output)',
                    '\t\tflist = [f for f in os.listdir(sub_path)]',
                    '\t\tif len([f for f in flist if prefix_filter in f]):',
                    '\t\t\tself.logger.info("Step::Skipped because the file[{0}] is exist.".format(output))',
                    '\t\t\tsleep(0.08)',
                    '\t\telse:']
            for cmd, stdout in self._commands:
                if isinstance(stdout, str):
                    body += ['\t\t\t{0}, err = methods.shell({1})'.format(stdout, cmd)]
                elif isinstance(stdout, list):
                    if stdout[0]:
                        body += ['\t\t\t{0} = {1}'.format(stdout[0], cmd)]
                    else:
                        body += ['\t\t\t{0}'.format(cmd)]
                else:
                    body += ['\t\t\ttemp_outputs.append(methods.shell({0}))'.format(cmd)]
                self._procobj.logger.info('"{}" command is executed for subjects'.format(cmd))
            body += ['\t\toutputs.append(temp_outputs)']
            if self._tempfiles:
                temp = ['\t\ttemppath = mkdtemp()',
                        '\t\tself.logger.info("SYS::TempFolder[{0}] is generated".format(temppath))']
                close = ['\t\trmtree(temppath)',
                         '\t\tself.logger.info("SYS::TempFolder[{0}] is closed".format(temppath))']
                body = loop + temp + body + close
            else:
                body = loop + body

        if self._sessions: # Check the project is multi-session
            header = ['def {0}(self, output_path, idx, subj, sess):'.format(name),
                      '\tsub_path = os.path.join(output_path, subj, sess)',
                      '\tmethods.mkdir(sub_path)']
        else:
            header = ['def {0}(self, output_path, idx, subj):'.format(name),
                      '\tsub_path = os.path.join(output_path, subj)',
                      '\tmethods.mkdir(sub_path)']
        footer = ['\treturn outputs\n']
        output = header + filters + body + footer
        output = '\n'.join(output)
        if verbose:
            print(unicode(output, "utf-8"))
            return None
        else:
            return output

    def run(self, step_name, surfix, results=False, debug=False):
        """Generate loop commands for step

        :param step_name: str
        :param surfix: str
        :return: None
        """
        if debug:
            return self.get_executefunc('debug', verbose=True)
        else:
            self._procobj._prjobj.reload()
            if self._procobj._parallel:
                thread = multiprocessing.cpu_count()
            else:
                thread = 1
            pool = ThreadPool(thread)
            self._procobj.logger.info("Step::[{0}] is executed with {1} thread(s).".format(step_name, thread))
            output_path = self._procobj.init_step("{0}-{1}".format(step_name, surfix))
            if self._sessions:
                for idx, subj in enumerate(progressbar(self._subjects, desc='Subjects')):
                    methods.mkdir(os.path.join(output_path, subj))
                    iteritem = [(self._procobj, output_path, idx, subj, sess) for sess in self._sessions]
                    for outputs in progressbar(pool.imap_unordered(self.worker, iteritem), desc='Sessions',
                                              leave=False, total=len(iteritem)):
                        if len(outputs):
                            if isinstance(outputs[0], list):
                                all_outputs = []
                                for output in outputs:
                                    all_outputs.extend(['STDOUT:\n{0}\nMessage:\n{1}\n\n'.format(out, err) for out, err in output])
                                outputs = all_outputs[:]
                            else:
                                outputs = ['STDOUT:\n{0}\nMessage:\n{1}\n\n'.format(out, err) for out, err in outputs if outputs]
                            with open(os.path.join(output_path, 'stephistory.log'), 'a') as f:
                                f.write('\n\n'.join(outputs))
                        else:
                            pass
            else:
                dirs = [os.path.join(output_path, subj) for subj in self._subjects]
                methods.mkdir(dirs)
                iteritem = [(self._procobj, output_path, idx, subj) for idx, subj in enumerate(self._subjects)]
                for outputs in progressbar(pool.imap_unordered(self.worker, iteritem), desc='Subjects',
                                          total=len(iteritem)):
                    if outputs != None:
                        if len(outputs):
                            if isinstance(outputs[0], list):
                                all_outputs = []
                                for output in outputs:
                                    all_outputs.extend(['STDOUT:\n{0}\nMessage:\n{1}\n\n'.format(out, err) for out, err in output])
                                outputs = all_outputs[:]
                            else:
                                outputs = ['STDOUT:\n{0}\nMessage:\n{1}\n\n'.format(out, err) for out, err in outputs]
                            with open(os.path.join(output_path, 'stephistory.log'), 'a') as f:
                                f.write('\n\n'.join(outputs))
                        else:
                            pass
                    else:
                        pass
            self._procobj._history[os.path.basename(output_path)] = output_path
            self._procobj.save_history()
            self._procobj._prjobj.reload()
            return output_path

    def worker(self, args):
        """The worker for parallel computing

        :param args: list, Arguments for step execution
        :return: str
        """
        funccode = self.get_executefunc('stepexec')
        output = None
        exec(funccode)    # load step function on memory
        try:
            exec('output = stepexec(*args)')    # execute function
        except Exception as e:
            print(e)
        return output

