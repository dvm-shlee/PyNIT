import os
import re
import sys
import pickle
from collections import namedtuple
import multiprocessing
from multiprocessing.pool import ThreadPool

from ..core import messages
from ..core import methods
from ..core import tools
from .images import TempFile

# Import hidden modules
import numpy as np
from tempfile import mkdtemp
from StringIO import StringIO
from time import sleep

# Import modules for interfacing with jupyter notebook
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


def check_dataclass(datatype):
    if os.path.exists(datatype):
        dataclass = 1
        datatype = methods.path_splitter(datatype)
        datatype = datatype[-1]
    else:
        dataclass = 0
    return dataclass, datatype


def get_step_name(procobj, step, results=False, verbose = None):
    if results:
        idx = 2
    else:
        idx = 1
    processing_path = os.path.join(procobj._prjobj.path, procobj._prjobj.ds_type[idx], procobj.processing)
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

        :param name     : str
            Variables for parameter output file
        :param ext      : str
            Extension of the output file
        :param prefix   : str
            If prefix has value, add prefix to output file
        """
        self._outparam[name] = (self.oppset(name=name, prefix=prefix, ext=ext))

    def set_execmethod(self, command, var=None, idx=None):
        """Set structured command on command handler

        :param command  : str
            Structured command with input and output variables
        :param var      : str
            Name of variable
        :param idx      : int
            Index for replacing the commands on handler
        :return:
        """
        if idx:
            self._commands[idx] = (command, [var])
        else:
            self._commands.append((command, [var]))

    def set_command(self, command, verbose=False, idx=None, stdout=None ):
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
        tmpobjs = [obj.strip('{}') for obj in re.findall(r"[{\w'}]+", command) if obj[0] == '{' and obj[-1] == '}']
        objs = []
        for o in tmpobjs:
            if "}{" in o:
                objs.extend(o.split('}{'))
            else:
                objs.append(o)
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
                            methods.raiseerror(messages.Errors.InputValueError, "Something wrong")

        output = "{0}{1})".format(output, ", ".join(list(set(str_format))))
        if idx:
            self._commands[idx] = (output, stdout)
        else:
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
                    '\t\tself.logger.info("The File[{0}] is already exist.".format(output))',
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
                        '\tself.logger.info("TempFolder[{0}] is generated".format(temppath))']
                close = ['\trmtree(temppath)',
                         '\tself.logger.info("TempFolder[{0}] is closed".format(temppath))']
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
                    '\t\t\tself.logger.info("The File[{0}] is already exist.".format(output))',
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
                        '\t\tself.logger.info("TempFolder[{0}] is generated".format(temppath))']
                close = ['\t\trmtree(temppath)',
                         '\t\tself.logger.info("TempFolder[{0}] is closed".format(temppath))']
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

    def run(self, step_name, surfix, debug=False):
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
            self._procobj.logger.info("Step:[{0}] is executed with {1} thread(s).".format(step_name, thread))
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
                                    all_outputs.extend(['STDOUT:\n{0}\nMessage:\n{1}'.format(out, err) for out, err in output])
                                outputs = all_outputs[:]
                            else:
                                outputs = ['STDOUT:\n{0}\nMessage:\n{1}'.format(out, err) for out, err in outputs if outputs]
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
                                    all_outputs.extend(['STDOUT:\n{0}\nMessage:\n{1}'.format(out, err) for out, err in output])
                                outputs = all_outputs[:]
                            else:
                                outputs = ['STDOUT:\n{0}\nMessage:\n{1}'.format(out, err) for out, err in outputs]
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
        # except IndexError as e:
        #     methods.raiseerror(ImportError,
        #                        '[{}] Parsing input dataset Failed, Please check you put the correct inputs'.format(e))
        return output


class Process(object):
    """Collections of step components for pipelines
    """
    def __init__(self, prjobj, name, parallel=True, logging=True, viewer='itksnap'):
        """

        :param prjobj:
        :param name:
        :param parallel:
        :param logging:
        :param viewer:
        """

        # Prepare inputs
        prjobj.reset_filters()
        self._processing = name
        self._prjobj = prjobj
        path = os.path.join(self._prjobj.path, self._prjobj.ds_type[1])
        self._path = os.path.join(path, self._processing)

        # Initiate logger
        if logging:
            self.logger = methods.get_logger(path, name)

        # Define default arguments
        self._subjects = None
        self._sessions = None
        self._history = {}
        self._parallel = parallel
        self._tempfiles = []
        self._viewer = viewer

        # Initiate
        self.init_proc()

    def check_input(self, input_path):
        """Check input_path and return absolute path

        :param input_path: str, name of the Processing step folder
        :return: str, Absolute path of the step
        """
        if isinstance(input_path, int):
            input_path = self.steps[input_path]
        if input_path in self.executed:
            return self._history[input_path]
        else:
            return input_path

    def check_filters(self, **kwargs):
        """Check filters for input datasets

        :param kwargs:
        :return:
        """
        avail = ['file_tag', 'ignore', 'ext']
        filters = None
        subj = None
        sess = None
        if kwargs:
            if any(key in avail for key in kwargs.keys()):
                filters = dict([(key, value) for key, value in kwargs.items()
                                if any([key in avail])])
            if 'subj' in kwargs.keys():
                subj = kwargs['subj']
            if 'sess' in kwargs.keys():
                sess = kwargs['sess']
        return filters, subj, sess

    def afni_MeanImgCalc(self, func, cbv=False, surfix='func'):
        """Mean image calculation for functional image : Initial preparation

        :param func: str, Name of functional data folder at source location (eg. 'func')
        :param cbv: boolean, True if MION contrast agent is infused
        :param surfix: str, Output folder surfix
        :return: output_path: dict, Absolute path of outputs
        """
        display(title(value='** Processing mean image calculation.....'))
        func = self.check_input(func)
        step = Step(self)
        step.set_input(name='func', input_path=func, static=True)
        step.set_outparam(name='mparam', ext='.1D')
        cmd01 = "3dvolreg -prefix {temp_01} -1Dfile {mparam} -Fourier -verbose -base 0 {func}"
        step.set_command(cmd01)
        if cbv:
            cmd02 = "3dinfo -nv {func}"
            step.set_staticinput('bold', 'int(int(ttime)/3)')
            step.set_staticinput('bold_output', 'methods.splitnifti(output)+"_BOLD.nii.gz"')
            step.set_staticinput('cbv', 'int(int(ttime)*2/3)')
            step.set_staticinput('cbv_output', 'methods.splitnifti(output)+"_CBV.nii.gz"')
            step.set_command(cmd02, stdout='ttime')
            options = ['"[0..{bold}]"',
                       '"[{cbv}..$]"']
            cmd03 = "3dTstat -prefix {bold_output} -mean {temp_01}" + options[0]
            step.set_command(cmd03)
            cmd04 = "3dTstat -prefix {cbv_output} -mean {temp_01}" + options[1]
            step.set_command(cmd04)
            # step.get_executefunc('test', verbose=True)
            output_path = step.run('MeanImgCalc-CBV', surfix)
        else:
            cmd02 = "3dTstat -prefix {output} -mean {temp_01}"
            step.set_command(cmd02)
            output_path = step.run('MeanImgCalc-BOLD', surfix)
        return dict(meanfunc=output_path)

    def afni_SliceTimingCorrection(self, func, tr=None, tpattern=None, surfix='func'):
        """Corrects for slice time differences when individual 2D slices are recorded over a 3D image

        :param func: str,
        :param tr: int,
        :param tpattern: str,
        :param surfix: str,
        :return: output_path: dict, Absolute path of outputs
        """
        display(title(value='** Processing slice timing correction.....'))
        func = self.check_input(func)
        options = str()
        step = Step(self)
        step.set_input(name='func', input_path=func, static=False)
        cmd = "3dTshift -prefix {output}"
        if tr:
            options += " -TR {0}".format(tr)
        if tpattern:
            options += " -tpattern {0}".format(tpattern)
        else:
            options += " -tpattern altplus"
        input_str = " {func}"
        cmd = cmd+options+input_str
        step.set_command(cmd)
        output_path = step.run('SliceTmCorrect', surfix)
        return dict(func=output_path)

    def afni_MotionCorrection(self, func, base, surfix='func'):
        """

        :param func:
        :param base:
        :param surfix:
        :return:
        """
        display(title(value='** Processing motion correction.....'))
        func = self.check_input(func)
        base = self.check_input(base)
        step = Step(self)
        step.set_input(name='func', input_path=func, static=False)
        try:
            if '-CBV-' in base:
                mimg_filters = {'file_tag': '_CBV', 'ignore': 'BOLD'}
                step.set_input(name='base', input_path=base, filters=mimg_filters, static=True, side=True)
            else:
                step.set_input(name='base', input_path=base, static=True, side=True)
            # mimg_path = self.steps[0]
            # if '-CBV-' in mimg_path:
            #     mimg_filters = {'file_tag': '_CBV'}
            #     step.set_input(name='base', input_path=mimg_path, filters=mimg_filters, static=True, side=True)
            # else:
            #     step.set_input(name='base', input_path=mimg_path, static=True, side=True)
        except:
            methods.raiseerror(messages.Errors.MissingPipeline,
                               'Initial Mean image calculation step has not been executed!')
        step.set_outparam(name='mparam', ext='.1D')
        step.set_outparam('transmat', ext='.aff12.1D')
        cmd01 = "3dvolreg -prefix {temp_01} -1Dfile {mparam} -Fourier -verbose -base 0 {func}"
        step.set_command(cmd01)
        cmd02 = "3dTstat -mean -prefix {temp_02} {temp_01}"
        step.set_command(cmd02)
        cmd03 = "3dAllineate -prefix {temp_03} -warp sho -base {base} -1Dmatrix_save {transmat} {temp_02}"
        step.set_command(cmd03)
        cmd04 = '3dAllineate -prefix {output} -1Dmatrix_apply {transmat} -warp sho {temp_01}'
        step.set_command(cmd04)
        output_path = step.run('MotionCorrection', surfix)
        return dict(func=output_path)

    def afni_MaskPrep(self, anat, tmpobj, func=None, surfix='func'):
        """

        :param anat:
        :return:
        """
        display(title(value='** Processing mask image preparation.....'))
        anat = self.check_input(anat)
        step = Step(self)
        mimg_path = None
        try:
            step.set_input(name='anat', input_path=anat, static=True)
        except:
            methods.raiseerror(messages.Errors.MissingPipeline,
                               'No anatomy file!')
        try:
            step.set_staticinput(name='mask', value=str(tmpobj.mask))
        except:
            methods.raiseerror(messages.InputPathError,
                               'No mask template file!')
        cmd01 = "3dAllineate -prefix {temp1} -NN -onepass -EPI -base {anat} -cmass+xy {mask}"
        cmd02 = '3dcalc -prefix {output} -expr "astep(a, 0.5)" -a {temp1}'
        step.set_command(cmd01)
        step.set_command(cmd02)
        anat_mask = step.run('MaskPrep', 'anat')
        step = Step(self)
        try:
            if func:
                mimg_path = self.check_input(func)
            else:
                mimg_path = self.steps[0]
            if '-CBV-' in mimg_path:
                mimg_filters = {'file_tag': '_BOLD'}
                step.set_input(name='func', input_path=mimg_path, filters=mimg_filters, static=True)
            else:
                step.set_input(name='func', input_path=mimg_path, static=True)
        except:
            methods.raiseerror(messages.Errors.MissingPipeline,
                               'Initial Mean image calculation step has not been executed!')
        try:
            step.set_staticinput(name='mask', value=str(tmpobj.mask))
        except:
            methods.raiseerror(messages.InputPathError,
                               'No mask template file!')
        cmd01 = "3dAllineate -prefix {temp1} -NN -onepass -EPI -base {func} -cmass+xy {mask}"
        cmd02 = '3dcalc -prefix {output} -expr "astep(a, 0.5)" -a {temp1}'
        step.set_command(cmd01, idx=0)
        step.set_command(cmd02)
        func_mask = step.run('MaskPrep', surfix)
        if jupyter_env:
            if self._viewer == 'itksnap':
                display(widgets.VBox([title(value='-'*43 + ' Anatomical images ' + '-'*43),
                                      tools.itksnap(self, anat_mask, anat),
                                      title(value='<br>' + '-'*43 + ' Functional images ' + '-'*43),
                                      tools.itksnap(self, func_mask, mimg_path)]))
            elif self._viewer == 'fslview':
                display(widgets.VBox([title(value='-'*43 + ' Anatomical images ' + '-'*43),
                                      tools.fslview(self, anat_mask, anat),
                                      title(value='<br>' + '-'*43 + ' Functional images ' + '-'*43),
                                      tools.fslview(self, func_mask, mimg_path)]))
            else:
                methods.raiseerror(messages.Errors.InputValueError,
                                   '"{}" is not available'.format(self._viewer))
        else:
            return dict(anat_mask=anat_mask, func_mask=func_mask)

    def afni_SkullStrip(self, anat, func, surfix='func'):
        """ The pre-defined step for skull stripping with AFNI

        :param anat:
        :param func:
        :return:
        """
        display(title(value='** Processing skull stripping.....'))
        anat = self.check_input(anat)
        func = self.check_input(func)
        anat_mask = [self.steps[idx] for idx, step in self.executed.items() if 'MaskPrep-anat' in step][0]
        anat_mask = self.check_input(anat_mask)
        func_mask = [self.steps[idx] for idx, step in self.executed.items() if 'MaskPrep-{}'.format(surfix) in step][0]
        func_mask = self.check_input(func_mask)
        step = Step(self)
        step.set_input(name='anat', input_path=anat, static=True)
        step.set_input(name='anat_mask', input_path=anat_mask, static=True, side=True)
        cmd01 = '3dcalc -prefix {output} -expr "a*step(b)" -a {anat} -b {anat_mask}'
        step.set_command(cmd01)
        anat_path = step.run('SkullStrip', 'anat')
        step = Step(self)
        if '-CBV-' in func:
            func_filter = {'file_tag':'_BOLD'}
            step.set_input(name='func', input_path=func, filters=func_filter, static=True)
        else:
            step.set_input(name='func', input_path=func, static=True)
        step.set_input(name='func_mask', input_path=func_mask, static=True, side=True)
        cmd02 = '3dcalc -prefic {output} -expr "a*step(b)" -a {func} -b {func_mask}'
        step.set_command(cmd02, idx=0)
        func_path = step.run('SkullStrip', surfix)
        return dict(anat=anat_path, func=func_path)

    def afni_Coreg(self, anat, meanfunc, surfix='func'):
        """Applying bias field correction with ANTs N4 algorithm and then align funtional image to
        anatomical space using Afni's 3dAllineate command

        :param anat:
        :param meanfunc:
        :param surfix:
        :return:
        """
        display(title(value='** Processing coregistration.....'))
        anat = self.check_input(anat)
        meanfunc = self.check_input(meanfunc)
        step = Step(self)
        step.set_input(name='anat', input_path=anat, static=True, side=True)
        step.set_input(name='func', input_path=meanfunc, static=True)
        step.set_outparam(name='transmat', ext='.aff12.1D')
        cmd01 = "N4BiasFieldCorrection -i {anat} -o {temp_01}"
        step.set_command(cmd01)
        cmd02 = "N4BiasFieldCorrection -i {func} -o {temp_02}"
        step.set_command(cmd02)
        cmd03 = "3dAllineate -prefix {output} -onepass -EPI -base {temp_01} -cmass+xy " \
                "-1Dmatrix_save {transmat} {temp_02}"
        step.set_command(cmd03)
        output_path = step.run('Coregistration', surfix)
        return dict(func=output_path)

    def afni_SkullStripAll(self, func, meanfunc, surfix='func'):
        """Applying arithmetic skull stripping

        :param func:
        :param meanfunc:
        :param surfix:
        :return:
        """
        display(title(value='** Processing skull stripping to all {} data.....'.format(surfix)))
        meanfunc = self.check_input(meanfunc)
        func = self.check_input(func)
        step = Step(self)
        step.set_input(name='mask', input_path=meanfunc, static=True, side=True)
        step.set_input(name='func', input_path=func)
        cmd = '3dcalc -prefix {output} -expr "a*step(b)" -a {func} -b {mask}'
        step.set_command(cmd)
        output_path = step.run('Apply_SkullStrip', surfix)
        return dict(func=output_path)

    def afni_ApplyCoregAll(self, func, coregfunc, surfix='func'):
        """Applying transform matrix to all functional data using Afni's 3dAllineate command

        :param func:
        :param coregfunc:
        :param surfix:
        :return:
        """
        display(title(value='** Applying coregistration to all {} data.....'.format(surfix)))
        coregfunc = self.check_input(coregfunc)
        func = self.check_input(func)
        step = Step(self)
        tform_filters = {'ext':'.aff12.1D'}
        step.set_input(name='tform', input_path=coregfunc, filters=tform_filters, static=True, side=True)
        step.set_input(name='coreg', input_path=coregfunc, static=True, side=True)
        step.set_input(name='func', input_path=func)
        cmd = '3dAllineate -prefix {output} -master {coreg} -1Dmatrix_apply {tform} {func}'
        step.set_command(cmd)
        output_path = step.run('Apply_Coreg', surfix)
        return dict(func=output_path)

    def afni_SpatialNorm(self, anat, tmpobj, surfix='anat'):
        """Align anatomical image to template brain space using Afni's 3dAllineate command

        :param anat:
        :param tmpobj:
        :param surfix:
        :return:
        """
        display(title(value='** Processing spatial normalization.....'))
        anat = self.check_input(anat)
        step = Step(self)
        step.set_input(name='anat', input_path=anat, static=True)
        step.set_staticinput(name='tmpobj', value=tmpobj.template_path)
        step.set_outparam(name='transmat', ext='.aff12.1D')
        cmd = '3dAllineate -prefix {output} -twopass -cmass+xy -zclip -conv 0.01 -base {tmpobj} ' \
              '-cost crM -check nmi -warp shr -1Dmatrix_save {transmat} {anat}'
        step.set_command(cmd)
        output_path = step.run('SpatialNorm', surfix)
        return dict(normanat=output_path)

    def afni_ApplySpatialNorm(self, func, normanat, surfix='func'):
        """Applying transform matrix to all functional data for spatial normalization

        :param func:
        :param normanat:
        :param surfix:
        :return:
        """
        display(title(value='** Applying spatial normalization to all {} data.....'.format(surfix)))
        func = self.check_input(func)
        normanat = self.check_input(normanat)
        step = Step(self)
        step.set_input(name='func', input_path=func)
        step.set_input(name='normanat', input_path=normanat, static=True, side=True)
        transmat_filter = {'ext':'.aff12.1D'}
        step.set_input(name='transmat', input_path=normanat, filters=transmat_filter, static=True, side=True)
        cmd = '3dAllineate -prefix {output} -master {normanat} -warp shr -1Dmatrix_apply {transmat} {func}'
        step.set_command(cmd)
        output_path = step.run('ApplySpatialNorm', surfix)
        return dict(normfunc=output_path)

    def afni_SpatialSmoothing(self, func, fwhm=0.5, tmpobj=None, surfix='func', **kwargs):
        """

        :param func:
        :param fwhm:
        :param tmpobj:
        :param surfix:
        :return:
        """
        display(title(value='** Processing spatial smoothing.....'))
        func = self.check_input(func)
        filters, subj, sess = self.check_filters(**kwargs)
        step = Step(self, subjects=subj, sessions=sess)
        step.set_input(name='func', input_path=func, filters=filters)
        if not fwhm:
            methods.raiseerror(messages.Errors.InputValueError, 'the FWHM value have to specified')
        else:
            step.set_staticinput('fwhm', fwhm)
        cmd = '3dBlurInMask -prefix {output} -FWHM {fwhm}'
        if tmpobj:
            step.set_staticinput('mask', value=str(tmpobj.mask))
            cmd += ' -mask {mask}'
        cmd += ' -quiet {func}'
        step.set_command(cmd)
        output_path = step.run('SpatialSmoothing', surfix)
        return dict(func=output_path)

    def afni_GLManalysis(self, func, paradigm, clip_range=None, surfix='func', **kwargs):
        """

        :param func:
        :param paradigm:
        :param clip_range:
        :param surfix:
        :return:
        """
        display(title(value='** Processing General Linear Analysis'))
        func = self.check_input(func)
        filters, subj, sess = self.check_filters(**kwargs)
        step = Step(self, subjects=subj, sessions=sess)
        step.set_input(name='func', input_path=func, filters=filters)
        step.set_variable(name='paradigm', value=paradigm)
        step.set_staticinput(name='param', value='" ".join(map(str, paradigm[idx][0]))')
        step.set_staticinput(name='model', value='paradigm[idx][1][0]')
        step.set_staticinput(name='mparam', value='" ".join(map(str, paradigm[idx][1][1]))')
        if clip_range:
            cmd01 = '3dDeconvolve -input {func}'
            cmd01 += '"[{}..{}]" '.format(clip_range[0], clip_range[1])
            cmd01 += '-num_stimts 1 -polort 2 -stim_times 1 "1D: {param}" ' \
                    '"{model}({mparam})" -stim_label 1 STIM -tout -bucket {output}'
        else:
            cmd01 = '3dDeconvolve -input {func} -num_stimts 1 -polort 2 -stim_times 1 "1D: {param}" ' \
                    '"{model}({mparam})" -stim_label 1 STIM -tout -bucket {output} -x1D {prefix}'
        step.set_command(cmd01)
        glm = step.run('GLMAnalysis', surfix, debug=False)
        display(title(value='** Estimating the temporal auto-correlation structure'))
        step = Step(self, subjects=subj, sessions=sess)
        step.set_input(name='func', input_path=func, filters=filters)
        filter = dict(ext='.xmat.1D')
        step.set_input(name='glm', input_path=glm, filters=filter, side=True)
        if clip_range:
            cmd02 = '3dREMLfit -matrix {glm} -input {func}'
            cmd02 += '"[{}..{}]" '.format(clip_range[0], clip_range[1])
            cmd02 += '-tout -Rbuck {output} -verb'
        else:
            cmd02 = '3dREMLfit -matrix {glm} -input {func} -tout -Rbuck {output} -verb'
        step.set_command(cmd02)
        output_path = step.run('REMLfit', surfix, debug=False)
        return dict(GLM=output_path)

    def afni_ClusterMap(self, glm, func, tmpobj, pval=0.01, cluster_size=40, surfix='func'):
        """Wrapper method of afni's 3dclust for generating clustered mask

        :param glm:
        :param func:
        :param tmpobj:
        :param pval:
        :param cluster_size:
        :param surfix:
        :return:
        """
        display(title(value='** Generating clustered masks'))
        glm = self.check_input(glm)
        func = self.check_input(func)
        step = Step(self)
        step.set_input(name='glm', input_path=glm)
        step.set_input(name='func', input_path=func, side=True)
        step.set_staticinput(name='pval', value=pval)
        step.set_staticinput(name='csize', value=cluster_size)
        cmd01 = '3dAttribute BRICK_STATAUX {glm}'
        step.set_command(cmd01, stdout='dof')
        step.set_staticinput(name='dof', value='dof.split()[-1]')
        cmd02 = 'cdf -p2t fitt {pval} {dof}'
        step.set_command(cmd02, stdout='tval')
        step.set_execmethod('tval.split("=")[1].strip()', var='tval')
        cmd03 = '3dclust -1Dformat -nosum -1dindex 2 -1tindex 2 -2thresh -{tval} {tval} ' \
                '-dxyz=1 -savemask {output} 1.01 {csize} {glm}'
        step.set_command(cmd03)
        step.set_execmethod('with open(methods.splitnifti(output) + ".json", "wb") as f:')
        step.set_execmethod('\tjson.dump(dict(source=func[i].Abspath), f)')
        output_path = step.run('ClusteredMask', surfix=surfix)
        if jupyter_env:
            if self._viewer == 'itksnap':
                display(tools.itksnap(self, output_path, tmpobj.image.get_filename()))
            elif self._viewer == 'fslview':
                display(tools.fslview(self, output_path, tmpobj.image.get_filename()))
            else:
                methods.raiseerror(messages.Errors.InputValueError,
                                   '"{}" is not available'.format(self._viewer))
        else:
            return dict(mask=output_path)

    def afni_SignalProcessing(self, func, norm=True, ort=None, clip_range=None, mask=None, bpass=None,
                              fwhm=None, dt=None, surfix='func', **kwargs):
        """Wrapper method of afni's 3dTproject for signal processing of resting state data

        :param func:
        :param norm:
        :param ort:
        :param mask:
        :param bpass:
        :param fwhm:
        :param dt:
        :param surfix:
        :return:
        """
        display(title(value='** Run signal processing for resting state data'))
        filters, subj, sess = self.check_filters(**kwargs)
        step = Step(self, subjects=subj, sessions=sess)
        func = self.check_input(func)
        ort = self.check_input(ort)
        step.set_input(name='func', input_path=func, filters=filters)
        cmd = ['3dTproject -prefix {output}']
        orange, irange = None, None         # orange= range of ort file, irange= range of image file
        if bpass:                           # set bandpass filter
            if isinstance(bpass, list) and len(bpass) == 2:
                cmd.append('-passband {} {}'.format(*bpass))
            else:
                pass
        if norm:                            # set signal normalization
            cmd.append('-norm')
        if ort:                             # set ort (nuisance signal regression)
            if clip_range:
                if isinstance(clip_range, list):
                    if len(clip_range) == 2:
                        orange = "'{" + "{}..{}".format(*clip_range) + "}'"
                        irange = "'[" + "{}..{}".format(*clip_range) + "]'"
                        step.set_staticinput(name='orange', value=orange)
                        step.set_staticinput(name='irange', value=irange)

            ort_filter = {'ext': '.1D', 'ignore': ['.aff12']}
            if filters:
                for key in filters.keys():
                    if 'ignore' in key:
                        if isinstance(filters['ignore'], list):
                            ort_filter['ignore'].extend(filters.pop('ignore'))
                        else:
                            ort_filter['ignore'].append(filters.pop('ignore'))
                    if 'ext' in key:
                        filters.pop('ext')
                ort_filter.update(filters)
            if isinstance(ort, dict):
                for key, value in ort.items():
                    ortpath = self.check_input(value)
                    if clip_range:
                        cmd.append('-ort {{}}'.format(key)+'{orange}')
                    else:
                        cmd.append('-ort {{}}'.format(key))
                    step.set_input(name=key, input_path=ortpath, filters=ort_filter, side=True)
            elif isinstance(ort, list):
                for i, o in enumerate(ort):
                    exec('ort_{} = self.check_input({})'.format(str(i), o))
                    ort_name = 'ort_{}'.format(str(i))
                    if clip_range:
                        cmd.append('-ort {}'.format(ort_name)+'{orange}')
                    else:
                        cmd.append('-ort {}'.format(ort_name))
                    step.set_input(name=ort_name, input_path=o, filters=ort_filter, side=True)
            elif isinstance(ort, str):
                ort = self.check_input(ort)
                if clip_range:
                    cmd.append('-ort {ort}"{orange}"')
                else:
                    cmd.append('-ort {ort}')
                step.set_input(name='ort', input_path=ort, filters=ort_filter, side=True)
            else:
                self.logger.debug('TypeError on input ort.')
        if mask:                            # set mask
            if os.path.isfile(mask):
                step.set_staticinput(name='mask', value=mask)
            elif os.path.isdir(mask):
                step.set_input(name='mask', input_path=mask, static=True, side=True)
            else:
                pass
            cmd.append('-mask {mask}')
        if fwhm:                            # set smoothness
            step.set_staticinput(name='fwhm', value=fwhm)
            cmd.append('-blur {fwhm}')
        if dt:                              # set sampling rate (TR)
            step.set_staticinput(name='dt', value=dt)
            cmd.append('-dt {dt}')
        if clip_range:                           # set range
            cmd.append('-input {func}"{irange}"')
        else:
            cmd.append('-input {func}')
        step.set_command(" ".join(cmd))
        output_path = step.run('SignalProcess', surfix=surfix, debug=False)
        return dict(signalprocess=output_path)

    def afni_ROIStats(self, func, rois, cbv=None, clip_range=None, option=None, surfix='func', **kwargs):
        """Extracting time-course data from ROIs

        :param func:    Input path for functional data
        :param roi:     Template instance or mask path
        :param cbv:     [echotime, number of volumes (TR) to average]
        :param clip_range:
        :param option:  bilateral or merge if roi is Template instance
        :param surfix:
        :type func:     str
        :type roi:      Template or str
        :type cbv:      list
        :type surfix:   str
        :return:        Current step path
        :rtype:         dict
        """
        display(title(value='** Extracting time-course data from ROIs'))
        func = self.check_input(func)
        # Check if given rois is Template instance
        tmp = None
        list_of_roi = None
        if not isinstance(rois, str):
            try:
                if option:
                    if option == 'bilateral':
                        tmp = TempFile(rois.atlas, atlas=True, bilateral=True)
                    elif option == 'merge':
                        tmp = TempFile(rois.atlas, atlas=True, merge=True)
                    elif option == 'contra':
                        tmp = TempFile(rois.atlas, atlas=True, flip=True)
                    else:
                        tmp = TempFile(rois.atlas, atlas=True)
                else:
                    tmp = TempFile(rois.atlas, atlas=True)
                rois = str(tmp.path)
                list_of_roi = list(tmp.label)
            except:
                pass
        else:
            pass
        # Check if given rois path is existed in the list of executed steps
        rois = self.check_input(rois)

        # Initiate step instance
        filters, subj, sess = self.check_filters(**kwargs)
        step = Step(self, subjects=subj, sessions=sess)

        # If given roi path is single file
        if os.path.isfile(rois):
            step.set_staticinput(name='rois', value=rois)
            step.set_input(name='func', input_path=func)
            cmd = '3dROIstats -mask {rois} {func}'
        # Else, given roi path is directory
        else:
            step.set_input(name='rois', input_path=rois)
            step.set_input(name='func', input_path=rois, filters=dict(ext='json'), side=True)
            step.set_execmethod('func_path = json.load(open(func[i].Abspath))["source"]')
            step.set_staticinput('func_path', value='func_path')
            cmd = '3dROIstats -mask {rois} {func_path}'
        # If CBV parameters are given, parsing the CBV infusion file path from json file
        if cbv:
            step.set_input(name='cbv', input_path=func, side=True, filters=dict(ext='.json'))
        if clip_range:
            cmd += '"[{}..{}]"'.format(clip_range[0], clip_range[1])
        step.set_command(cmd, stdout='out')
        step.set_execmethod('temp_outputs.append([out, err])')
        step.set_execmethod('pd.read_table(StringIO(out))', var='df')
        step.set_execmethod('df[df.columns[2:]]', var='df')
        # If given roi is Template instance
        if list_of_roi:
            step.set_variable(name='list_roi', value=list_of_roi)
            step.set_execmethod('list_roi', var='df.columns')
        # again, if CBV parameter are given, put commends and methods into custom build function
        if cbv:
            if isinstance(cbv, list) and len(cbv) == 2:
                step.set_variable(name='te', value=cbv[0])
                step.set_variable(name='n_tr', value=cbv[1])
                step.set_execmethod('cbv_path = json.load(open(cbv[i].Abspath))["cbv"]')
                step.set_staticinput(name='cbv_path', value='cbv_path')
                cbv_cmd = '3dROIstats -mask {rois} {cbv_path}'
                step.set_command(cbv_cmd, stdout='cbv_out')
                step.set_execmethod('temp_outputs.append([out, err])')
                step.set_execmethod('pd.read_table(StringIO(cbv_out))', var='cbv_df')
                step.set_execmethod('cbv_df[cbv_df.columns[2:]]', var='cbv_df')
                if list_of_roi:
                    step.set_execmethod('list_roi', var='cbv_df.columns')
            else:
                methods.raiseerror(messages.Errors.InputValueError, 'Please check input CBV parameters')
        step.set_execmethod('if len(df.columns):')
        # again, if CBV parameter are given, correct the CBV changes.
        if cbv:
            step.set_execmethod('\tdR2_mion = (-1 / te) * np.log(df.loc[:n_tr, :].mean(axis=0) / '
                                'cbv_df.loc[:n_tr, :].mean(axis=0))')
            step.set_execmethod('\tdR2_stim = (-1 / te) * np.log(df / df.loc[:n_tr, :].mean(axis=0))')
            step.set_execmethod('\tdf = dR2_stim/dR2_mion')
        # Generating excel files
        step.set_execmethod('\tfname = os.path.splitext(str(func[i].Filename))[0]')
        step.set_execmethod('\tdf.to_excel(os.path.join(sub_path, methods.splitnifti(fname)+".xlsx"), '
                            'index=False)')
        step.set_execmethod('\tpass')
        step.set_execmethod('else:')
        step.set_execmethod('\tpass')

        # Run the steps
        output_path = step.run('ExtractROIs', surfix=surfix)#, debug=True)
        if tmp:
            tmp.close()
        return dict(timecourse=output_path)

    def afni_TemporalClipping(self, func, clip_range, surfix='func', **kwargs):
        """

        :param func:
        :param clip_range:
        :param surfix:
        :param kwargs:
        :return:
        """
        display(title(value='** Temporal clipping of functional image'))
        filters, subj, sess = self.check_filters(**kwargs)
        step = Step(self, subjects=subj, sessions=sess)
        func = self.check_input(func)
        step.set_input(name='func', input_path=func, filters=filters)
        if clip_range:
            if isinstance(clip_range, list):
                if len(clip_range) == 2:
                    irange = "'[" + "{}..{}".format(*clip_range) + "]'"
                    step.set_staticinput(name='irange', value=irange)
        cmd = '3dcalc -prefix {output} -expr "a" -a {func}"{irange}"'
        step.set_command(cmd)
        output_path = step.run('TemporalClipping', surfix, debug=False)
        return dict(clippedfunc=output_path)

    def ants_Coreg(self):
        pass

    def ants_ApplyCoreg(self):
        pass

    def ants_SpatialNorm(self, anat, tmpobj, surfix='anat'):
        """This step align the anatomical data to given template brain space using ANTs non-linear SyN algorithm

        :param anat     :   str or int
            Folder name of anatomical data in Data class or absolute path
            If you put integer, the path will inputted by indexing the executed path with given integer
        :param tmpobj:
        :param surfix:
        :return:
        """

        parallel = False
        if self._parallel:
            parallel = True
        self._parallel = False          # turn of parallel processing mode

        display(title(value='** Processing spatial normalization.....'))
        anat = self.check_input(anat)
        step = Step(self)
        step.set_input(name='anat', input_path=anat, static=True)
        step.set_staticinput(name='tmpobj', value=tmpobj.template_path)
        step.set_staticinput(name='thread', value=multiprocessing.cpu_count())
        cmd = 'antsSyN -f {tmpobj} -m {anat} -o {prefix} -n {thread}'
        step.set_command(cmd)
        output_path = step.run('SpatialNorm', surfix, debug=False)

        if parallel:
            self._parallel = True
        return dict(normanat=output_path)

    def ants_ApplySpatialNorm(self, func, warped, surfix='func', **kwargs):
        """This step applying the non-linear transform matrix from the anatomical image to functional images

        :param func     :   str or int
            Folder name of functional data in Data class or absolute path of one of executed step
            If you put integer, the path will inputted by indexing the executed path with given integer

        :param warped   :   str or int
            Absolute path that ANTs SyN based spatial normalization registration is applied
            If you put integer, the path will inputted by indexing the executed path with given integer

        :param surfix   :   str
            The given string will be set as surfix of output folder

        :param kwargs   :   dict
            This arguments will be used for filtering the input data, available keywords are as below
            'subs', 'sess', 'file_tag', 'ignore', 'ext'

        :return:
            Output path as dictionary format
        """
        display(title(value='** Processing spatial normalization.....'))

        # Check and correct inputs
        func = self.check_input(func)
        warped = self.check_input(warped)
        filters, subj, sess = self.check_filters(**kwargs)
        step = Step(self, subjects=subj, sessions=sess)

        # Set filters for input transform data
        baseimg_filter = dict(ignore=['_1InverseWarp', '_1Warp', '_inversed'])
        dmorph_filter = dict(file_tag='_1Warp')
        tmatrix_filter = dict(ext='.mat')

        # Set inputs
        step.set_input(name='func', input_path=func, filters=filters)
        step.set_input(name='base', input_path=warped, static=True, filters=baseimg_filter, side=True)
        step.set_input(name='morph', input_path=warped, static=True, filters=dmorph_filter, side=True)
        step.set_input(name='mat', input_path=warped, static=True, filters=tmatrix_filter, side=True)

        # Set commend that need to executes for all subjects
        cmd = 'WarpTimeSeriesImageMultiTransform 4 {func} {output} -R {base} {morph} {mat}'
        step.set_command(cmd)
        output_path = step.run('ApplySpatialNorm', surfix, debug=False)
        return dict(normfunc=output_path)

    def pn_MaskPrep(self, anat, tmpobj, func=None, surfix='func'): #TODO: Need to develop own skullstrip alrorithm
        """

        :param anat:
        :return:
        """
        display(title(value='** Processing mask image preparation.....'))
        anat = self.check_input(anat)
        step = Step(self)
        mimg_path = None
        try:
            step.set_input(name='anat', input_path=anat, static=True)
        except:
            methods.raiseerror(messages.Errors.MissingPipeline,
                               'No anatomy file!')
        try:
            step.set_staticinput(name='mask', value=str(tmpobj.mask))
        except:
            methods.raiseerror(messages.InputPathError,
                               'No mask template file!')
        cmd01 = "mask_prep 10 {anat} {temp1}"
        cmd02 = 'mask_reg -o {output} -f {temp1} -m {mask} -n 1'
        step.set_command(cmd01)
        step.set_command(cmd02)
        anat_mask = step.run('MaskPrep', 'anat')
        step = Step(self)
        try:
            if func:
                mimg_path = self.check_input(func)
            else:
                mimg_path = self.steps[0]
            if '-CBV-' in mimg_path:
                mimg_filters = {'file_tag': '_BOLD'}
                step.set_input(name='func', input_path=mimg_path, filters=mimg_filters, static=True)
            else:
                step.set_input(name='func', input_path=mimg_path, static=True)
        except:
            methods.raiseerror(messages.Errors.MissingPipeline,
                               'Initial Mean image calculation step has not been executed!')
        try:
            step.set_staticinput(name='mask', value=str(tmpobj.mask))
        except:
            methods.raiseerror(messages.InputPathError,
                               'No mask template file!')
        cmd01 = "mask_prep -f 10 {func} {temp1}"
        cmd02 = 'mask_reg -o {output} -f {temp1} -m {mask} -n 1'
        step.set_command(cmd01)
        step.set_command(cmd02)
        func_mask = step.run('MaskPrep', surfix)
        if jupyter_env:
            if self._viewer == 'itksnap':
                display(widgets.VBox([title(value='-' * 43 + ' Anatomical images ' + '-' * 43),
                                      tools.itksnap(self, anat_mask, anat),
                                      title(value='<br>' + '-' * 43 + ' Functional images ' + '-' * 43),
                                      tools.itksnap(self, func_mask, mimg_path)]))
            elif self._viewer == 'fslview':
                display(widgets.VBox([title(value='-' * 43 + ' Anatomical images ' + '-' * 43),
                                      tools.fslview(self, anat_mask, anat),
                                      title(value='<br>' + '-' * 43 + ' Functional images ' + '-' * 43),
                                      tools.fslview(self, func_mask, mimg_path)]))
            else:
                methods.raiseerror(messages.Errors.InputValueError,
                                   '"{}" is not available'.format(self._viewer))
        else:
            return dict(anat_mask=anat_mask, func_mask=func_mask)

    def ants_MotionCorrection(self, func, surfix='func', debug=False):
        display(title(value='** Extracting time-course data from ROIs'))
        func = self.check_input(func)
        step = Step(self)
        step.set_input(name='func', input_path=func, static=False)
        cmd01 = "antsMotionCorr -d 3 -a {func} -o {prefix}-avg.nii.gz"
        cmd02 = "antsMotionCorr -d 3 -o [{prefix},{prefix}.nii.gz,{prefix}-avg.nii.gz] " \
                "-m gc[ {prefix}-avg.nii.gz ,{func}, 1, 1, Random, 0.05  ] -t Affine[ 0.005 ] " \
                "-i 20 -u 1 -e 1 -s 0 -f 1 -n 10"
        step.set_command(cmd01)
        step.set_command(cmd02)
        output_path = step.run('MotionCorrection', surfix=surfix, debug=debug)
        return dict(func=output_path)

    def ants_BiasFieldCorrection(self, anat, func):
        anat = self.check_input(anat)
        func = self.check_input(func)
        step1 = Step(self)
        step2 = Step(self)
        filters = dict(file_tag='_mask')
        step1.set_input(name='anat', input_path=anat, static=True)
        step2.set_input(name='func', input_path=func, static=True)
        cmd1 = 'N4BiasFieldCorrection -i {anat} -o {output}'
        cmd2 = 'N4BiasFieldCorrection -i {func} -o {output}'
        step1.set_command(cmd1)
        step2.set_command(cmd2)
        anat_path = step1.run('BiasFiled', 'anat')
        func_path = step2.run('BiasField', 'func')
        return dict(anat=anat_path, func=func_path)

    def fsl_IndividualICA(self, func, tr=2.0, alpha=0.5, bgthreshold=10, surfix='func'):
        func = self.check_input(func)
        step = Step(self)
        step.set_input(name='func', input_path=func)
        cmd = ['melodic -i {func} -o {sub_path}', '--tr={}'.format(tr), '--mmthresh={}'.format(alpha),
               '--bgthreshold={} --nobet --nomask'.format(bgthreshold)]
        step.set_command(' '.join(cmd))
        func_path = step.run('IndividualICA', 'func')
        return dict(func=func_path)

    def fsl_BiasFieldCalculation(self, anat, func, n_class=3, smoothing=2, image_type=2, debug=False):
        anat = self.check_input(anat)
        func = self.check_input(func)
        step1 = Step(self)
        step2 = Step(self)
        step1.set_input(name='anat', input_path=anat, static=True)
        step2.set_input(name='func', input_path=func)
        cmd1 = ['fast --class={} --lowpass={} --type={} -b'.format(n_class, smoothing, image_type),
                '--out={prefix} {anat}']
        cmd2 = ['fast --class={} --lowpass={} --type={} -b'.format(n_class, smoothing, image_type),
                '--out={prefix} {func}']
        step1.set_command(' '.join(cmd1))
        step2.set_command(' '.join(cmd2))
        anat_path = step1.run('BiasFieldCalculation', 'anat', debug=debug)
        func_path = step2.run('BiasFieldCalculation', 'func', debug=debug)
        return dict(anat=anat_path, func=func_path)

    def fsl_BiasFieldCorrection(self, anat, anat_bias, func, func_bias, debug=False):
        anat = self.check_input(anat)
        anat_bias = self.check_input(anat_bias)
        func = self.check_input(func)
        func_bias = self.check_input(func_bias)
        step1 = Step(self)
        step2 = Step(self)
        step1.set_input(name='anat', input_path=anat, static=True)
        step2.set_input(name='func', input_path=func)
        step1.set_input(name='anat_bias', input_path=anat_bias, static=True, filters=dict(file_tag='_bias'), side=True)
        step2.set_input(name='func_bias', input_path=func_bias, filters=dict(file_tag='_bias'), side=True)
        cmd1 = '3dcalc -prefix {output} -expr "a/b" -a {anat} -b {anat_bias}'
        cmd2 = '3dcalc -prefix {output} -expr "a/b" -a {func} -b {func_bias}'
        step1.set_command(cmd1)
        step2.set_command(cmd2)
        anat_path = step1.run('BiasFieldCorrection', 'anat', debug=debug)
        func_path = step2.run('BiasFieldCorrection', 'func', debug=debug)
        return dict(anat=anat_path, func=func_path)

    def fsl_DualRegression(self, func, surfix='func'): #TODO: Implant DualRegression
        pass

    def itksnap(self, idx, base_idx=None):
        """Launch ITK-snap

        :param idx:
        :param base_idx:
        :return:
        """
        if base_idx:
            display(tools.itksnap(self, self.steps[idx], self.steps[base_idx]))
        else:
            display(tools.itksnap(self, self.steps[idx]))

    def afni(self, idx, tmpobj=None):
        """Launch AFNI gui

        :param idx:
        :param tmpobj:
        :return:
        """
        tools.afni(self, self.steps[idx], tmpobj=tmpobj)

    def fslview(self, idx, base_idx=None):
        """Launch fslview

                :param idx:
                :param base_idx:
                :return:
                """
        if base_idx:
            tools.fslview(self, self.steps[idx], self.steps[base_idx])
        else:
            tools.fslview(self, self.steps[idx])


    @property
    def path(self):
        return self._path

    @property
    def processing(self):
        return self._processing

    @property
    def subjects(self):
        return self._subjects

    @property
    def sessions(self):
        return self._sessions

    @property
    def executed(self):
        """Listing out executed steps

        :return:
        """
        exists = [d for d in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, d))]
        for step in self._history.keys():
            if step not in exists:
                del self._history[step]
        n_hist = len(self._history.keys())
        output = zip(range(n_hist), sorted(self._history.keys()))
        if jupyter_env:
            return dict(output)
        else:
            return dict(output)


    @property
    def steps(self):
        return [self._history[step] for step in self.executed.values()]

    def reset(self):
        """reset subject and session information

        :return: None
        """
        if self._prjobj(1).subjects:
            if self._prjobj(0).subjects:
                datasubj = set(list(self._prjobj(0).subjects))
                procsubj = set(list(self._prjobj(1).subjects))
                if datasubj.issubset(procsubj):
                    if not procsubj.issubset(datasubj):
                        try:
                            self._subjects = sorted(self._prjobj(1, self.processing).subjects[:])
                            if not self._prjobj.single_session:
                                self._sessions = sorted(self._prjobj(1, self.processing).sessions[:])
                        except:
                            self._subjects = sorted(self._prjobj(0).subjects[:])
                            if not self._prjobj.single_session:
                                self._sessions = sorted(self._prjobj(0).sessions[:])
                    else:
                        self._subjects = sorted(self._prjobj(0).subjects[:])
                        if not self._prjobj.single_session:
                            self._sessions = sorted(self._prjobj(0).sessions[:])
                else:
                    self._subjects = sorted(self._prjobj(0).subjects[:])
                    if not self._prjobj.single_session:
                        self._sessions = sorted(self._prjobj(0).sessions[:])
            else:
                self._subjects = sorted(self._prjobj(1).subjects[:])
        else:
            self._subjects = sorted(self._prjobj(0).subjects[:])
            if not self._prjobj.single_session:
                self._sessions = sorted(self._prjobj(0).sessions[:])

        self.logger.info('Attributes [subjects, sessions] are reset to default value.')
        self.logger.info('Subject is defined as [{}]'.format(",".join(self._subjects)))
        if self._sessions:
            self.logger.info('Session is defined as [{}]'.format(",".join(self._sessions)))

    def init_proc(self):
        """Initiate process folder

        :return: None
        """
        methods.mkdir(self._path)
        self.logger.info('Process object is initiated with {0}'.format(self.processing))
        history = os.path.join(self._path, '.proc_hisroty')
        if os.path.exists(history):
            with open(history, 'r') as f:
                self._history = pickle.load(f)
            self.logger.info("History file is loaded".format(history))
        else:
            self.save_history()
        self.reset()
        return self._path

    def init_step(self, name):
        """Initiate step

        :param name: str
        :return: str
        """

        if self._processing:
            path = get_step_name(self, name)
            path = os.path.join(self._prjobj.path, self._prjobj.ds_type[1], self._processing, path)
            methods.mkdir(path)
            return path
        else:
            methods.raiseerror(messages.Errors.InitiationFailure, 'Error on initiating step')

    def save_history(self):
        history = os.path.join(self._path, '.proc_hisroty')
        with open(history, 'w') as f:
            pickle.dump(self._history, f)
        self.logger.info("History file is saved at '{0}'".format(history))

