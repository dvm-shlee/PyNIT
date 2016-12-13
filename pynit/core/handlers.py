import os
import sys
import copy
import pickle
import pandas as pd
from tempfile import mkdtemp
from shutil import rmtree
from time import sleep

# from progressbar import ProgressBar, SimpleProgress
try:
    if len([key for key in sys.modules.keys() if key == 'ipykernel']):
        from tqdm import tqdm_notebook as progressbar
    else:
        from tqdm import tqdm as progressbar
except:
    pass

from collections import namedtuple

from .objects import Reference, ImageObj
from .processors import Analysis, Interface, TempFile
from .methods import np
from .visualizers import Viewer
import messages
import methods

import multiprocessing
from multiprocessing.pool import ThreadPool


class Project(object):
    """Project Handler for functional Neuro MRI datasets
    """

    def __init__(self, project_path, ds_ref='NIRAL', img_format='NifTi-1', **kwargs):
        """Load and initiate the project

        Parameters
        ----------
        project_path:   str
            Path of particular project
        ds_ref:         str
            Reference of data structure (default: 'NIRAL')
        img_format:     str
            Reference img format
        """
        # Display options for pandasDataframe
        max_rows = 100
        max_colwidth = 100
        if kwargs:
            if 'max_rows' in kwargs.keys():
                max_rows = kwargs['max_rows']
            if 'max_colwidth' in kwargs.keys():
                max_colwidth = kwargs['max_colwidth']
        pd.options.display.max_rows = max_rows
        pd.options.display.max_colwidth = max_colwidth

        # Define default attributes
        self.single_session = False             # True if project has single session
        self.__empty_project = False            # True if project folder is empty
        self.__filters = [None] * 6
        # Each values are represented subject, session, dtype(or pipeline), step(or results) file_tags, ignores

        self.__path = project_path

        # Set internal objects
        self.__df = methods.DataFrame()

        # Parsing the information from the reference
        self.__ref = [ds_ref, img_format]       #
        ref = Reference(*self.__ref)
        self.img_ext = ref.imgext
        self.ds_type = ref.ref_ds

        # Define default filter values
        self.__dc_idx = 0                       # Dataclass index
        self.__ext_filter = self.img_ext        # File extension

        # Generate folders for dataclasses
        methods.mk_main_folder(self)

        # Scan project folder
        try:
            self.scan_prj()
        except:
            methods.raiseerror(messages.Errors.ProjectScanFailure, 'Error is occurred during a scanning.')

    @property
    def df(self):
        """ Dataframe for handling data structure

        Returns
        -------
        dataset : pandas.Dataframe
        """
        columns = self.__df.columns
        return self.__df.reset_index()[columns]

    @property
    def path(self):
        """ Project path

        Returns
        -------
        path    : str
        """
        return self.__path

    @property
    def dataclass(self):
        """ Dataclass index

        Returns
        -------
        index   : int
        """
        return self.ds_type[self.__dc_idx]

    @dataclass.setter
    def dataclass(self, idx):
        if idx in range(3):
            self.__dc_idx = idx
            self.reset_filters()
            self.__update()
        else:
            methods.raiseerror(messages.Errors.InputDataclassError, 'Wrong dataclass index.')

    @property
    def subjects(self):
        return self.__subjects

    @property
    def sessions(self):
        return self.__sessions

    @property
    def dtypes(self):
        return self.__dtypes

    @property
    def pipelines(self):
        return self.__pipelines

    @property
    def steps(self):
        return self.__steps

    @property
    def results(self):
        return self.__results

    @property
    def filters(self):
        return self.__filters

    @property
    def summary(self):
        return self.__summary()

    @property
    def ext(self):
        return self.__ext_filter

    @ext.setter
    def ext(self, value):
        if type(value) == str:
            self.__ext_filter = [value]
        elif type(value) == list:
            self.__ext_filter = value
        elif not value:
            self.__ext_filter = None
        else:
            methods.raiseerror(messages.Errors.InputTypeError,
                               'Please use correct input type.')
        self.scan_prj()

    def reset_filters(self, ext=None):
        """ Reset filter - Clear all filter information and extension

        Parameters
        ----------
        ext     : str
            Filter parameter for file extension

        Returns
        -------
        None
        """
        self.__filters = [None] * 6
        if not ext:
            self.ext = self.img_ext
        else:
            self.ext = ext
        self.scan_prj()
        self.__update()

    def scan_prj(self):
        """ Reload the Dataframe based on current set data class and extension

        Returns
        -------
        None
        """
        # Parsing command works
        self.__df, self.single_session, empty_prj = methods.parsing(self.path, self.ds_type, self.__dc_idx)
        if not empty_prj:
            self.__df = methods.initial_filter(self.__df, self.ds_type, self.__ext_filter)
            if len(self.__df):
                self.__df = self.__df[methods.reorder_columns(self.__dc_idx, self.single_session)]
            self.__update()
            self.__empty_project = False
        else:
            self.__empty_project = True

    def set_filters(self, *args, **kwargs):
        """ Set filters

        Parameters
        ----------
        args    : str[, ]
            String arguments regarding hierarchical data structures
        kwargs  : key=value pair[, ]
            Key and value pairs for the filtering parameter on filename

            (key) file_tag  : str or list of str
                Keywords of interest for filename
            (key) ignore    : str or list of str
                Keywords of neglect for filename
            (key) keep      : boolean
                True, if you want to keep previous parameter

        Returns
        -------
        None

        """
        if kwargs:
            for key in kwargs.keys():
                if key == 'ext':
                    self.ext = kwargs['ext']
                elif key == 'keep':
                    # This option allows to keep previous filter
                    if kwargs['keep']:
                        self.__update()
                    else:
                        self.reset_filters(self.ext)
                elif key == 'file_tag':
                    if type(kwargs['file_tag']) == str:
                        self.__filters[4] = [kwargs['file_tag']]
                    elif type(kwargs['file_tag']) == list:
                        self.__filters[4] = kwargs['file_tag']
                    else:
                        methods.raiseerror(messages.Errors.InputTypeError,
                                                 'Please use correct input type for FileTag')
                elif key == 'ignore':
                    if type(kwargs['ignore']) == str:
                        self.__filters[5] = [kwargs['ignore']]
                    elif type(kwargs['ignore']) == list:
                        self.__filters[5] = kwargs['ignore']
                    else:
                        methods.raiseerror(messages.Errors.InputTypeError,
                                                 'Please use correct input type for FileTag to ignore')
                else:
                    methods.raiseerror(messages.Errors.KeywordError,
                                             "'{key}' is not correct kwarg")
        else:
            self.reset_filters(self.ext)
        if args or kwargs:
            if args:
                residuals = list(args)
                if self.subjects:
                    subj_filter, residuals = methods.check_arguments(args, residuals, self.subjects)
                    if self.__filters[0]:
                        self.__filters[0].extend(subj_filter)
                    else:
                        self.__filters[0] = subj_filter[:]
                    if not self.single_session:
                        sess_filter, residuals = methods.check_arguments(args, residuals, self.sessions)
                        if self.__filters[1]:
                            self.__filters[1].extend(sess_filter)
                        else:
                            self.__filters[1] = sess_filter[:]
                    else:
                        self.__filters[1] = None
                else:
                    self.__filters[0] = None
                    self.__filters[1] = None
                if self.__dc_idx == 0:
                    if self.dtypes:
                        dtyp_filter, residuals = methods.check_arguments(args, residuals, self.dtypes)
                        if self.__filters[2]:
                            self.__filters[2].extend(dtyp_filter)
                        else:
                            self.__filters[2] = dtyp_filter[:]
                    else:
                        self.__filters[2] = None
                    self.__filters[3] = None
                elif self.__dc_idx == 1:
                    if self.pipelines:
                        pipe_filter, residuals = methods.check_arguments(args, residuals, self.pipelines)
                        if self.__filters[2]:
                            self.__filters[2].extend(pipe_filter)
                        else:
                            self.__filters[2] = pipe_filter[:]
                    else:
                        self.__filters[2] = None
                    if self.steps:
                        step_filter, residuals = methods.check_arguments(args, residuals, self.steps)
                        if self.__filters[3]:
                            self.__filters[3].extend(step_filter)
                        else:
                            self.__filters[3] = step_filter
                    else:
                        self.__filters[3] = None
                else:
                    if self.pipelines:
                        pipe_filter, residuals = methods.check_arguments(args, residuals, self.pipelines)
                        if self.__filters[2]:
                            self.__filters[2].extend(pipe_filter)
                        else:
                            self.__filters[2] = pipe_filter[:]
                    else:
                        self.__filters[2] = None
                    if self.results:
                        rslt_filter, residuals = methods.check_arguments(args, residuals, self.results)
                        if self.__filters[3]:
                            self.__filters[3].extend(rslt_filter)
                        else:
                            self.__filters[3] = rslt_filter[:]
                    else:
                        self.__filters[3] = None
                if len(residuals):
                    methods.raiseerror(messages.Errors.InputValueError,
                                             'Wrong filter input:{residuals}'.format(residuals=residuals))
        self.__df = self.applying_filters(self.__df)
        self.__update()

    def applying_filters(self, df):
        """ Applying current filters to the input dataframe

        Parameters
        ----------
        df      : pandas.DataFrame

        Returns
        -------
        df      : pandas.DataFrame
        """
        if len(df):
            if self.__filters[0]:
                df = df[df.Subject.isin(self.__filters[0])]
            if self.__filters[1]:
                df = df[df.Session.isin(self.__filters[1])]
            if self.__filters[2]:
                if self.__dc_idx == 0:
                    df = df[df.DataType.isin(self.__filters[2])]
                else:
                    df = df[df.Pipeline.isin(self.__filters[2])]
            if self.__filters[3]:
                if self.__dc_idx == 1:
                    df = df[df.Step.isin(self.__filters[3])]
                elif self.__dc_idx == 2:
                    df = df[df.Result.isin(self.__filters[3])]
                else:
                    pass
            if self.__filters[4] is not None:
                file_tag = list(self.__filters[4])
                df = df[df.Filename.str.contains('|'.join(file_tag))]
            if self.__filters[5] is not None:
                ignore = list(self.__filters[5])
                df = df[~df.Filename.str.contains('|'.join(ignore))]
            return df
        else:
            return df

    def run(self, command, *args, **kwargs):
        """Execute processing tools
        """
        if command in dir(Interface):
            try:
                if os.path.exists(args[0]):
                    pass
                else:
                    getattr(Interface, command)(*args, **kwargs)
            except:
                exec('help(Interface.{})'.format(command))
                print(Interface, command, args, kwargs)
                raise messages.CommandExecutionFailure
        else:
            raise messages.NotExistingCommand

    def __summary(self):
        """Print summary of current project
        """
        summary = '** Project summary'
        summary = '{}\nProject: {}'.format(summary, os.path.dirname(self.path).split(os.sep)[-1])
        if self.__empty_project:
            summary = '{}\n[Empty project]'.format(summary)
        else:
            summary = '{}\nSelected DataClass: {}\n'.format(summary, self.dataclass)
            if self.pipelines:
                summary = '{}\nApplied Pipeline(s): {}'.format(summary, self.pipelines)
            if self.steps:
                summary = '{}\nApplied Step(s): {}'.format(summary, self.steps)
            if self.results:
                summary = '{}\nProcessed Result(s): {}'.format(summary, self.results)
            if self.subjects:
                summary = '{}\nSubject(s): {}'.format(summary, self.subjects)
            if self.sessions:
                summary = '{}\nSession(s): {}'.format(summary, self.sessions)
            if self.dtypes:
                summary = '{}\nDataType(s): {}'.format(summary, self.dtypes)
            if self.single_session:
                summary = '{}\nSingle session dataset'.format(summary)
            summary = '{}\n\nApplied filters'.format(summary)
            if self.__filters[0]:
                summary = '{}\nSet subject(s): {}'.format(summary, self.__filters[0])
            if self.__filters[1]:
                summary = '{}\nSet session(s): {}'.format(summary, self.__filters[1])
            if self.__dc_idx == 0:
                if self.__filters[2]:
                    summary = '{}\nSet datatype(s): {}'.format(summary, self.__filters[2])
            else:
                if self.__filters[2]:
                    summary = '{}\nSet Pipeline(s): {}'.format(summary, self.__filters[2])
                if self.__filters[3]:
                    if self.__dc_idx == 1:
                        summary = '{}\nSet Step(s): {}'.format(summary, self.__filters[3])
                    else:
                        summary = '{}\nSet Result(s): {}'.format(summary, self.__filters[3])
            if self.__ext_filter:
                summary = '{}\nSet file extension(s): {}'.format(summary, self.__ext_filter)
            if self.__filters[4]:
                summary = '{}\nSet file tag(s): {}'.format(summary, self.__filters[4])
            if self.__filters[5]:
                summary = '{}\nSet ignore(s): {}'.format(summary, self.__filters[5])
        print(summary)

    def __update(self):
        """Update attributes of Project object based on current set filter information
        """
        if len(self.df):
            try:
                self.__subjects = sorted(list(set(self.df.Subject.tolist())))
                if self.single_session:
                    self.__sessions = None
                else:
                    self.__sessions = sorted(list(set(self.df.Session.tolist())))
                if self.__dc_idx == 0:
                    self.__dtypes = sorted(list(set(self.df.DataType.tolist())))
                    self.__pipelines = None
                    self.__steps = None
                    self.__results = None
                elif self.__dc_idx == 1:
                    self.__dtypes = None
                    self.__pipelines = sorted(list(set(self.df.Pipeline.tolist())))
                    self.__steps = sorted(list(set(self.df.Step.tolist())))
                    self.__results = None
                elif self.__dc_idx == 2:
                    self.__dtypes = None
                    self.__pipelines = sorted(list(set(self.df.Pipeline.tolist())))
                    self.__results = sorted(list(set(self.df.Result.tolist())))
                    self.__steps = None
            except:
                methods.raiseerror(messages.Errors.UpdateAttributesFailed,
                                         "Error occured during update project's attributes")
        else:
            self.__subjects = None
            self.__sessions = None
            self.__dtypes = None
            self.__pipelines = None
            self.__steps = None
            self.__results = None

    def __call__(self, dc_id, *args, **kwargs):
        """Return DataFrame followed applying filters
        """
        prj = copy.copy(self)
        if self.__dc_idx != dc_id:
            prj.dataclass = dc_id
        prj.set_filters(*args, **kwargs)
        return prj


    def __repr__(self):
        """Return absolute path for current filtered dataframe
        """
        if self.__empty_project:
            return str(self.summary)
        else:
            return str(self.df.Abspath)

    def __getitem__(self, index):
        """Return particular data based on input index
        """
        if self.__empty_project:
            return None
        else:
            return self.df.loc[index]

    def __iter__(self):
        """Iterator for dataframe
        """
        if self.__empty_project:
            raise messages.EmptyProject
        else:
            for row in self.df.iterrows():
                yield row

    def __len__(self):
        """Return number of data
        """
        if self.__empty_project:
            return 0
        else:
            return len(self.df)


class Process(object):
    """ Collections of step components for pipelines
    """
    def __init__(self, prjobj, name, parallel=True, logging=True):
        """

        Parameters
        ----------
        prjobj
        name
        """
        prjobj.reset_filters()
        self._prjobj = prjobj
        self._processing = name
        path = os.path.join(self._prjobj.path, self._prjobj.ds_type[1])
        self._path = os.path.join(path, self._processing)
        if logging:
            self.logger = methods.get_logger(path, name)
        self._subjects = None
        self._sessions = None
        self._history = {}
        self._parallel = parallel
        self._tempfiles = []
        self.init_proc()

    def mean_calculation(self, input_path, surfix='func', **kwargs):
        """ BOLD image preparation step

        Parameters
        ----------
        input_path : str
            datatype of absolute_path
        surfix : str

        Returns
        -------
        None
        """
        step = Step(self)
        step.set_input(name='func', input_path=input_path, static=True) # TODO: filter not work properly
        cmd01 = "3dvolreg -prefix {temp_01} -Fourier -verbose -base 0 {func}"
        step.set_command(cmd01)
        cmd02 = "3dTstat -prefix {output} -mean {temp_01}"
        step.set_command(cmd02)
        output_path = step.run('MeanImgCalc', surfix)
        return {'meanfunc':output_path}

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

    def reset(self):
        """ reset subject and session information

        Returns
        -------
        None
        """
        if self._prjobj.subjects:
            self._subjects = sorted(self._prjobj.subjects[:])
            if not self._prjobj.single_session:
                self._sessions = sorted(self._prjobj.sessions[:])
        self.logger.info('Attributes [subjects, sessions] are reset to default value.')

    def init_proc(self):
        """ Initiate process folder

        Returns
        -------
        None
        """
        self.reset()
        methods.mkdir(self._path)
        self.logger.info('Process object is initiated with {0}'.format(self.processing))
        history = os.path.join(self._path, '.proc_hisroty')
        if os.path.exists(history):
            with open(history, 'r') as f:
                self._history = pickle.load(f)
            self.logger.info("History file is loaded".format(history))
        else:
            self.save_history()
        return self._path

    def init_step(self, name):
        """ Initiate step

        Parameters
        ----------
        name : str

        Returns
        -------
        path : str
        """
        if self._processing:
            path = methods.get_step_name(self, name)
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

class Step(object):
    """ Basic processing step template
    """
    dataset = namedtuple('Dataset', ['name', 'input_path', 'static'])  # projectobj handler
    cmdset = namedtuple('Command', ['name', 'command', 'option'])  # commandobj handler

    def __init__(self, procobj):
        self._procobj = procobj
        self._processing = procobj.processing
        self._tempfiles = []
        self._mainset = None
        self._sidesets = []
        self._staticinput = {}
        self._subjects = procobj.subjects[:]
        try:
            self._sessions = procobj.sessions[:]
        except:
            self._sessions = None
        self._commands = []
        self._filters = {'main':[], 'sides':{}}

    def set_input(self, name, input_path, filters=None, static=False, side=False):
        """ Import input dataset

        Parameters
        ----------
        input_path :  str
            datatype or absolute path
        kw_filters : dict
            kw_argment filters
        static : bool
            True, if this object need to be looped
            if not, only use first index
        side : bool
            True, if this object is side prjobj

        Returns
        -------
        None
        """
        dc, ipath = methods.check_dataclass(input_path)
        if side:
            self._sidesets.append(self.dataset(name=name, input_path=ipath, static=static))
            self._filters['sides'][name] = self.get_filtercode(str(dc), ipath, filters)
        else:
            self._mainset = self.dataset(name=name, input_path=input_path, static=static)
            self._filters['main'] = self.get_filtercode(str(dc), ipath, filters)

    def set_staticinput(self, name, input_path):
        """ Import static file
        """
        self._staticinput['name'] = input_path

    def set_command(self, command):
        """

        Parameters
        ----------
        command : str

        Returns
        -------
        command : str
        """
        objs = [obj.strip('{}') for obj in command.split(" ") if obj[0] == '{' and obj[-1] == '}']
        total = dict([(sideobj.name, sideobj.static) for sideobj in self._sidesets])
        total[self._mainset.name] = self._mainset.static
        try:
            totalobjs = total.keys()[:]
        except:
            totalobjs = []

        # Get list of residual inputs
        residuals = [obj for obj in objs if obj not in totalobjs]
        residuals = [obj for obj in residuals if 'temp' not in obj]
        residuals = [obj for obj in residuals if 'output' not in obj]
        residuals = [obj for obj in residuals if obj not in self._staticinput.keys()]

        # Get list of extra inputs
        lacks = [obj for obj in totalobjs if obj not in objs]

        # Check accuracy
        if len(residuals):
            methods.raiseerror(ValueError, 'Too many inputs{0}'.format(str(residuals)))
        output = "'{0}'.format(".format(command)
        str_format = []
        for obj in objs:
            if obj == 'output':
                str_format.append("output={0}".format(obj))
            else:
                if 'temp' in obj:
                    str_format.append("{0}=os.path.join(temppath, '{1}.nii')".format(obj, obj))
                    self._tempfiles.append(obj)
                elif obj in self._staticinput.keys():
                    str_format.append("{0}={1}".format(obj, self._staticinput[obj]))
                else:
                    if total[obj]:
                        str_format.append("{0}={1}.Abspath".format(obj, obj))
                    else:
                        str_format.append("{0}={1}[i].Abspath".format(obj, obj))
        output = "{0}{1})".format(output, ", ".join(str_format))
        self._commands.append(output)
        if self._tempfiles:
            self._tempfiles = sorted(list(set(self._tempfiles)))

    def get_inputcode(self):
        """ Generate datasets to execute step
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
        return inputcode

    def get_filtercode(self, dataclass, input_path, filters):
        """ Generate list of items for filtering based on input

        Returns
        -------
        foldercode : str
        """
        output_filters = [dataclass, '"{0}"'.format(input_path)]
        if self._sessions:
            output_filters.extend(['subj', 'sess'])
        else:
            output_filters.extend(['subj'])
        if isinstance(filters, dict):
            kwargs = ['{key}="{value}"'.format(key=k, value=v) for k, v in filters.items()]
            output_filters.extend(kwargs)
        else:
            pass
        return ', '.join(output_filters)

    def get_executefunc(self, name):
        filter = ['\t{}'.format(input) for input in self.get_inputcode()]
        if self._mainset.static:
            body = ['\toutputs = []',
                    '\toutput = os.path.join(output, {0}.Filename)'.format(self._mainset.name),
                    '\tif os.path.isfile(output):',
                    '\t\tself.logger.info("TheFile[{0}] is already exist.".format(output))',
                    '\telse:']
            body += ['\t\toutputs.append(methods.shell({0}))'.format(cmd) for cmd in self._commands]
            if self._tempfiles:
                temp = ['\ttemppath = mkdtemp()',
                        '\tself.logger.info("TempFolder[{0}] is generated".format(temppath))']
                close = ['\trmtree(temppath)']
                body = temp + body + close
            else:
                pass
        else:
            loop = ['\tfor i in range(len({0})):'.format(self._mainset)]
            body = ['\t\toutputs = []',
                    '\t\toutput = os.path.join(output, {0}.Filename)'.format(self._mainset.name),
                    '\t\tprint(output)',
                    '\t\tif os.path.isfile(output):',
                    '\t\t\tself.logger.info("TheFile[{0}] is already exist.".format(output))',
                    '\t\telse:']
            body += ['\t\t\toutputs.append(methods.shell({0}))'.format(cmd) for cmd in self._commands]
            if self._tempfiles:
                temp = ['\t\ttemppath = mkdtemp()',
                        '\t\tself.logger.info("TempFolder[{0}] is generated".format(temppath))']
                close = ['\t\trmtree(temppath)']
                body = loop + temp + body + close
            else:
                body = loop + body
        if self._sessions:
            header = ['def {0}(self, output, subj, sess):'.format(name),
                      '\toutput = os.path.join(output, subj, sess)',
                      '\tmethods.mkdir(output)']
        else:
            header = ['def {0}(self, output, subj):'.format(name),
                      '\toutput = os.path.join(output, subj, sess)',
                      '\tmethods.mkdir(output)']
        footer = ['\treturn outputs\n']
        output = header+filter+body+footer
        output = '\n'.join(output)
        return output

    def worker(self, args):
        """

        Parameters
        ----------
        func
        args

        Returns
        -------

        """
        funccode = self.get_executefunc('stepexec')
        exec(funccode)
        output = None
        exec('output = stepexec(*args)')
        return output

    def run(self, step_name, surfix):
        """ Generate loop commands for step
        """
        if self._procobj._parallel:
            thread = multiprocessing.cpu_count()
        else:
            thread = 1
        pool = ThreadPool(thread)
        self._procobj.logger.info("Step:[{0}] is executed with Thread({1}).".format(step_name, thread))
        output_path = self._procobj.init_step("{0}-{1}".format(step_name, surfix))
        if self._sessions:
            for subj in progressbar(self._subjects, desc='Subjects'):
                methods.mkdir(os.path.join(output_path, subj))
                iteritem = [(self._procobj, output_path, subj, sess) for sess in self._sessions]
                for output in progressbar(pool.imap_unordered(self.worker, iteritem), desc='Sessions', leave=False):
                    print(output)
        else:
            dirs = [os.path.join(output_path, subj) for subj in self._subjects]
            methods.mkdir(dirs)
            iteritem = [(self._procobj, output_path, subj) for subj in self._subjects]
            for output in progressbar(pool.imap_unordered(self.worker, iteritem), desc='Subjects'):
                print(output)
        # results = ['STDOUT:\n{0}\nError:\n{1}'.format(out, err) for out, err in results] #TODO: save the history and log for all execution.
        # with open(os.path.join(output_path, 'stephistory.log'), 'w') as f:
        #     f.write('\n'.join(results))
        # self._procobj._history[os.path.basename(output_path.split['_'][0])] = output_path
        # self._procobj.save_history()
        return output_path


# Below classes will be deprecated soon
class Preprocess(object):
    """ Preprocessing pipeline
    """
    def __init__(self, prjobj, pipeline):
        prjobj.reset_filters()
        self._subjects = None
        self._sessions = None
        self._processing = None
        self._prjobj = prjobj
        self.reset()
        self.initiate_pipeline(pipeline)

    @property
    def processing(self):
        return self._processing

    @property
    def subjects(self):
        return self._subjects

    @property
    def sessions(self):
        return self._sessions

    def reset(self):
        if self._prjobj.subjects:
            self._subjects = sorted(self._prjobj.subjects[:])
            if not self._prjobj.single_session:
                self._sessions = sorted(self._prjobj.sessions[:])

    def initiate_pipeline(self, pipeline):
        pipe_path = os.path.join(self._prjobj.path, self._prjobj.ds_type[1], pipeline)
        methods.mkdir(os.path.join(self._prjobj.path, self._prjobj.ds_type[1], pipeline))
        self._processing = pipeline

    def init_step(self, stepname):
        if self._processing:
            steppath = methods.get_step_name(self, stepname)
            steppath = os.path.join(self._prjobj.path, self._prjobj.ds_type[1], self._processing, steppath)
            methods.mkdir(steppath)
            return steppath
        else:
            raise messages.PipelineNotSet

    def calculate_seedbased_global_connectivity(self, func, seed, dtype='func', **kwargs):
        """ Seed-based Global Connectivity Analysis

        Parameters
        ----------
        func : str
            Root path for input dataset
        seed : str
            Absolute path for seed image
        Returns
        -------
        path : dict
            Absolute path for output
        """
        dataclass, func = methods.check_dataclass(func)
        print('SeedBaseConnectivityMap-{}'.format(func))
        step01 = self.init_step('SeedBaseConnectivityMap-{}'.format(dtype))
        if not os.path.isfile(seed):
            methods.raiseerror(ValueError, 'Input file does not exist.')
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                epi = self._prjobj(dataclass, self._processing, func, subj, **kwargs)
                for i, finfo in epi:
                    print(" +Filename: {}".format(finfo.Filename))
                    seed_path = os.path.join(step01, subj, "{0}.1D".format(methods.splitnifti(finfo.Filename)))
                    output_path = os.path.join(step01, subj, finfo.Filename)
                    self._prjobj.run('afni_3dmaskave', seed_path, finfo.Abspath, seed)
                    self._prjobj.run('afni_3dTcorr1D', output_path, finfo.Abspath, seed_path)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    epi = self._prjobj(dataclass, self._processing, func, subj, sess, **kwargs)
                    for i, finfo in epi:
                        print("  +Filename: {}".format(finfo.Filename))
                        seed_path = os.path.join(step01, subj, sess, "{0}.1D".format(methods.splitnifti(finfo.Filename)))
                        output_path = os.path.join(step01, subj, sess, finfo.Filename)
                        self._prjobj.run('afni_3dmaskave', seed_path, finfo.Abspath, seed)
                        self._prjobj.run('afni_3dTcorr1D', output_path, finfo.Abspath, seed_path)
        return {'connMap': step01}

    def fisher_transform(self, func, dtype='func', **kwargs):
        dataclass, func = methods.check_dataclass(func)
        print('FisherTransform-{}'.format(func))
        step01 = self.init_step('FisherTransform-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                epi = self._prjobj(dataclass, self._processing, func, subj, **kwargs)
                for i, finfo in epi:
                    print(" +Filename: {}".format(finfo.Filename))
                    output_path = os.path.join(step01, subj, finfo.Filename)
                    self._prjobj.run('afni_3dcalc', output_path, 'atanh(a)', finfo.Abspath)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    epi = self._prjobj(dataclass, self._processing, func, subj, sess, **kwargs)
                    for i, finfo in epi:
                        print("  +Filename: {}".format(finfo.Filename))
                        output_path = os.path.join(step01, subj, sess, finfo.Filename)
                        self._prjobj.run('afni_3dcalc', output_path, 'atanh(a)', finfo.Abspath)
        return {'Zmap': step01}

    def group_average(self, func, subjects, sessions=None, group='groupA', **kwargs):
        """ Calculate group average

        Parameters
        ----------
        func : str
            Root path for input dataset
        subjects : list
            list of the subjects for group
        sessions : list
            list of the sessions that want to calculate average

        Returns
        -------

        """
        dataclass, func = methods.check_dataclass(func)
        print('AverageGroupData-{0}: [{1}]'.format(func, ', '.join(subjects)))
        step01 = self.final_step('AverageGRoupData-{}'.format(group))
        root_output = os.path.join(step01, 'AllSubjects')
        methods.mkdir(root_output)
        if sessions:
            print("*Average image will be calculated for each session listed in: [{0}]".format(', '.join(sessions)))
            for sess in sessions:
                print("-Session: {}".format(sess))
                sess_output = os.path.join(root_output, sess)
                methods.mkdir(sess_output)
                epi = self._prjobj(dataclass, self._processing, func, sess, *subjects, **kwargs).df.Abspath
                grouplist = [path for path in epi.to_dict().values()]
                print(" :List of subjects in {}".format(sess))
                print("\n  ".join(grouplist))
                self._prjobj.run('afni_3dMean', os.path.join(sess_output, '{0}-{1}.nii.gz'.format(group, sess)),
                                 *grouplist)
        else:
            epi = self._prjobj(dataclass, self._processing, func, *subjects, **kwargs).df.Abspath
            grouplist = [path for path in epi.to_dict().values()]
            print(":List of subjects in this group")
            print("\n ".join(grouplist))
            self._prjobj.run('afni_3dMean', os.path.join(root_output, '{0}.nii.gz'.format(group)),
                             *grouplist)
        return {"groupavr": step01}

    def cbv_meancalculation(self, func, **kwargs):
        """ CBV image preparation
        """
        dataclass, func = methods.check_dataclass(func)
        print("MotionCorrection")
        step01 = self.init_step('MotionCorrection-CBVinduction')
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                cbv_img = self._prjobj(dataclass, func, subj, **kwargs)
                for i, finfo in cbv_img:
                    print(" +Filename: {}".format(finfo.Filename))
                    self._prjobj.run('afni_3dvolreg', os.path.join(step01, subj, finfo.Filename), finfo.Abspath)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    cbv_img = self._prjobj(dataclass, func, subj, sess, **kwargs)
                    for i, finfo in cbv_img:
                        print("  +Filename: {}".format(finfo.Filename))
                        self._prjobj.run('afni_3dvolreg', os.path.join(step01, subj, sess, finfo.Filename),
                                         finfo.Abspath)
        step02 = self.init_step('MeanImageCalculation-BOLD')
        step03 = self.init_step('MeanImageCalculation-CBV')
        print("MeanImageCalculation-BOLD&CBV")
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step02, subj), os.path.join(step03, subj))
            if self._prjobj.single_session:
                cbv_img = self._prjobj(1, self._processing, os.path.basename(step01), subj, **kwargs)
                for i, finfo in cbv_img:
                    print(" +Filename: {}".format(finfo.Filename))
                    shape = ImageObj.load(finfo.Abspath).shape
                    self._prjobj.run('afni_3dTstat', os.path.join(step02, subj, finfo.Filename),
                                     "{path}'[{start}..{end}]'".format(path=finfo.Abspath,
                                                                       start=0,
                                                                       end=(int(shape[-1] / 3))))
                    self._prjobj.run('afni_3dTstat', os.path.join(step03, subj, finfo.Filename),
                                     "{path}'[{start}..{end}]'".format(path=finfo.Abspath,
                                                                       start=int(shape[-1] * 2 / 3),
                                                                       end=shape[-1] - 1))
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step02, subj, sess), os.path.join(step03, subj, sess))
                    cbv_img = self._prjobj(1, os.path.basename(step01), subj, sess, **kwargs)
                    for i, finfo in cbv_img:
                        print("  +Filename: {}".format(finfo.Filename))
                        shape = methods.load(finfo.Abspath).shape
                        self._prjobj.run('afni_3dTstat', os.path.join(step02, subj, sess, finfo.Filename),
                                         "{path}'[{start}..{end}]'".format(path=finfo.Abspath,
                                                                           start=0,
                                                                           end=(int(shape[-1] / 3))))
                        self._prjobj.run('afni_3dTstat', os.path.join(step03, subj, sess, finfo.Filename),
                                         "{path}'[{start}..{end}]'".format(path=finfo.Abspath,
                                                                           start=int(shape[-1] * 2 / 3),
                                                                           end=shape[-1] - 1))
        return {'CBVinduction': step01, 'meanBOLD': step02, 'meanCBV': step03}

    def mean_calculation(self, func, dtype='func', **kwargs):
        """ BOLD image preparation

        Parameters
        ----------
        func       : str
            Datatype or absolute path of the input mean functional image
        dtype      : str
            Surfix for step path

        Returns
        -------
        step_paths : dict
        """
        dataclass, func = methods.check_dataclass(func)
        step01 = self.init_step('InitialPreparation-{}'.format(dtype))
        print("MotionCorrection")
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                finfo = self._prjobj(dataclass, func, subj, **kwargs).df.loc[0]
                print(" +Filename: {}".format(finfo.Filename))
                self._prjobj.run('afni_3dvolreg', os.path.join(step01, subj, finfo.Filename), finfo.Abspath)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    finfo = self._prjobj(dataclass, func, subj, sess, **kwargs).df.loc[0]
                    print("  +Filename: {}".format(finfo.Filename))
                    self._prjobj.run('afni_3dvolreg', os.path.join(step01, subj, sess, finfo.Filename), finfo.Abspath)
        step02 = self.init_step('MeanImageCalculation-{}'.format(dtype))
        print("MeanImageCalculation-{}".format(func))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step02, subj))
            if self._prjobj.single_session:
                funcs = self._prjobj(1, self._processing, os.path.basename(step01), subj, **kwargs)
                funcs = funcs.df.loc[0]
                print(" +Filename: {}".format(funcs.Filename))
                self._prjobj.run('afni_3dTstat', os.path.join(step02, subj, funcs.Filename),
                                 funcs.Abspath)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step02, subj, sess))
                    funcs = self._prjobj(1, self._processing, os.path.basename(step01), subj, sess, **kwargs)
                    funcs = funcs.df.loc[0]
                    print(" +Filename: {}".format(funcs.Filename))
                    self._prjobj.run('afni_3dTstat', os.path.join(step02, subj, sess, funcs.Filename),
                                     funcs.Abspath)
        return {'firstfunc': step01, 'meanfunc': step02}

    def slicetiming_correction(self, func, tr=None, tpattern='altplus', dtype='func', **kwargs):
        """ Corrects for slice time differences when individual 2D slices are recorded over a 3D image

        Parameters
        ----------
        func       : str
            Data type or absolute path of the input functional image
        tr         : int
        tpattern   : str
        dtype      : str
            Surfix for the step paths

        Returns
        -------
        step_paths : dict
        """
        dataclass, func = methods.check_dataclass(func)
        # if os.path.exists(func):
        #     dataclass = 1
        #     func = methods.path_splitter(func)[-1]
        # else:
        #     dataclass = 0
        print('SliceTimingCorrection-{}'.format(func))
        step01 = self.init_step('SliceTimingCorrection-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                epi = self._prjobj(dataclass, func, subj, **kwargs)
                for i, finfo in epi:
                    print(" +Filename: {}".format(finfo.Filename))
                    self._prjobj.run('afni_3dTshift', os.path.join(step01, subj, finfo.Filename),
                                     finfo.Abspath, tr=tr, tpattern=tpattern)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    epi = self._prjobj(dataclass, func, subj, sess, **kwargs)
                    for i, finfo in epi:
                        print("  +Filename: {}".format(finfo.Filename))
                        self._prjobj.run('afni_3dTshift', os.path.join(step01, subj, sess, finfo.Filename),
                                         finfo.Abspath, tr=tr, tpattern=tpattern)
        return {'func': step01}

    def motion_correction(self, func, base=None, baseidx=0, meancbv=None, dtype='func', **kwargs):
        """ Corrects for motion artifacts in the  input functional image

        Parameters
        ----------
        func       : str
            Datatype or absolute step path for the input functional image
        base       : str
            Datatype or absolute step path for the mean functional image
        baseidx    :
        meancbv   :
        dtype      : str
            Surfix for the step path


        Returns
        -------
        step_paths : dict
        """
        s0_dataclass, s0_func = methods.check_dataclass(func)
        print('MotionCorrection-{}'.format(func))
        step01 = self.init_step('MotionCorrection-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                epi = self._prjobj(s0_dataclass, s0_func, subj, **kwargs)
                if base:
                    if type(base) == str:
                        meanimg = self._prjobj(1, self._processing, os.path.basename(base), subj, **kwargs)
                        meanimg = meanimg.df.Abspath[baseidx]

                    else:
                        meanimg = base
                else:
                    meanimg = 0
                for i, finfo in epi:
                    print(" +Filename: {}".format(finfo.Filename))
                    self._prjobj.run('afni_3dvolreg', os.path.join(step01, subj, finfo.Filename), finfo.Abspath,
                                     base_slice=meanimg)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    epi = self._prjobj(s0_dataclass, s0_func, subj, sess, **kwargs)
                    if base:
                        if type(base) == str:
                            meanimg = self._prjobj(1, self._processing, os.path.basename(base), subj, sess)
                            meanimg = meanimg.df.Abspath[baseidx]
                        else:
                            meanimg = base
                    else:
                        meanimg = 0
                    for i, finfo in epi:
                        print("  +Filename: {}".format(finfo.Filename))
                        self._prjobj.run('afni_3dvolreg', os.path.join(step01, subj, sess, finfo.Filename),
                                         finfo.Abspath, base_slice=meanimg)
        if meancbv:
            # Calculate mean image for each 3D+time data
            s1_dataclass, s1_func = methods.check_dataclass(step01)
            print('MeanCalculation-{}'.format(s1_func))
            step02 = self.init_step('MeanFunctionalImages-{}'.format(dtype))
            for subj in self.subjects:
                print("-Subject: {}".format(subj))
                methods.mkdir(os.path.join(step02, subj))
                if self._prjobj.single_session:
                    epi = self._prjobj(s1_dataclass, s1_func, subj, **kwargs)
                    for i, finfo in epi:
                        print(" +Filename: {}".format(finfo.Filename))
                        self._prjobj.run('afni_3dTstat', os.path.join(step02, subj, finfo.Filename), finfo.Abspath,
                                         mean=True)
                else:
                    for sess in self.sessions:
                        print(" :Session: {}".format(sess))
                        methods.mkdir(os.path.join(step02, subj, sess))
                        epi = self._prjobj(s1_dataclass, s1_func, subj, sess, **kwargs)

                        for i, finfo in epi:
                            print("  +Filename: {}".format(finfo.Filename))
                            self._prjobj.run('afni_3dTstat', os.path.join(step02, subj, sess, finfo.Filename),
                                             finfo.Abspath, mean=True)
            # Realigning each run of CBV images
            s2_dataclass, s2_func = methods.check_dataclass(step02)
            print('IntersubjectRealign-{}'.format(func))
            step03 = self.init_step('InterSubjectRealign-{}'.format(dtype))
            for subj in self.subjects:
                print("-Subject: {}".format(subj))
                methods.mkdir(os.path.join(step03, subj))
                if self._prjobj.single_session:
                    epi = self._prjobj(s2_dataclass, s2_func, subj, **kwargs)
                    try:
                        baseimg = self._prjobj(1, os.path.basename(meancbv), subj).df.Abspath[0]
                    except:
                        baseimg = 0
                    for i, finfo in epi:
                        print(" +Filename: {}".format(finfo.Filename))
                        output_path = os.path.join(step03, subj, finfo.Filename)
                        self._prjobj.run('afni_3dAllineate', output_path, finfo.Abspath,
                                         base=baseimg, warp='sho',
                                         matrix_save=methods.splitnifti(output_path) + '.aff12.1D')
                else:
                    for sess in self.sessions:
                        print(" :Session: {}".format(sess))
                        methods.mkdir(os.path.join(step03, subj, sess))
                        epi = self._prjobj(s2_dataclass, s2_func, subj, sess, **kwargs)
                        try:
                            baseimg = self._prjobj(1, os.path.basename(meancbv), subj, sess)
                            baseimg = baseimg.df.Abspath[0]
                        except:
                            baseimg = 0
                        for i, finfo in epi:
                            print("  +Filename: {}".format(finfo.Filename))
                            output_path = os.path.join(step03, subj, sess, finfo.Filename)
                            self._prjobj.run('afni_3dAllineate', output_path,
                                             finfo.Abspath, base=baseimg, warp='sho',
                                             matrix_save=methods.splitnifti(output_path) + '.aff12.1D')
            # Realigning each run of CBV images
            s3_dataclass, s3_func = methods.check_dataclass(step03)
            print('InterSubj-ApplyTranform-{}'.format(func))
            step04 = self.init_step('InterSubj-ApplyTransform-{}'.format(dtype))
            for subj in self.subjects:
                print("-Subject: {}".format(subj))
                methods.mkdir(os.path.join(step04, subj))
                if self._prjobj.single_session:
                    param = self._prjobj(s3_dataclass, s3_func, subj).df
                    epi = self._prjobj(s1_dataclass, s1_func, subj, **kwargs)
                    for i, finfo in epi:
                        print(" +Filename: {}".format(finfo.Filename))
                        try:
                            if finfo.Filename != param.Filename[i]:
                                raise messages.ObjectMismatch()
                            else:
                                self._prjobj.run('afni_3dAllineate', os.path.join(step04, subj, finfo.Filename),
                                                 finfo.Abspath, warp='sho',
                                                 matrix_apply=methods.splitnifti(param.Abspath[i])+'.aff12.1D')
                        except:
                            print('  ::Skipped')
                else:
                    for sess in self.sessions:
                        print(" :Session: {}".format(sess))
                        methods.mkdir(os.path.join(step04, subj, sess))
                        param = self._prjobj(s3_dataclass, s3_func, subj, sess).df
                        epi = self._prjobj(s1_dataclass, s1_func, subj, sess, **kwargs)
                        for i, finfo in epi:
                            print(" +Filename: {}".format(finfo.Filename))
                            try:
                                if finfo.Filename != param.Filename[i]:
                                    print finfo.Filename, param.Filename[i]
                                    raise messages.ObjectMismatch()
                                else:
                                    self._prjobj.run('afni_3dAllineate', os.path.join(step04, subj, sess, finfo.Filename),
                                                    finfo.Abspath, warp='sho',
                                                    matrix_apply=methods.splitnifti(param.Abspath[i]) + '.aff12.1D')
                            except:
                                print('  ::Skipped')
            return {'func': step04}
        else:
            return {'func': step01}

    def maskdrawing_preparation(self, meanfunc, anat, padding=False, zaxis=2):
        """

        Parameters
        ----------
        meanfunc
        anat
        padding
        zaxis

        Returns
        -------

        """
        f_dataclass, meanfunc = methods.check_dataclass(meanfunc)
        a_dataclass, anat = methods.check_dataclass(anat)
        print('MaskDrawing-{} & {}'.format(meanfunc, anat))

        step01 = self.init_step('MaskDrwaing-func')
        step02 = self.init_step('MaskDrawing-anat')
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj), os.path.join(step02, subj))
            if self._prjobj.single_session:
                epi = self._prjobj(f_dataclass, meanfunc, subj)
                t2 = self._prjobj(a_dataclass, anat, subj)
                for i, finfo in epi:
                    print(" +Filename: {}".format(finfo.Filename))
                    epiimg = methods.load(finfo.Abspath)
                    if padding:
                        epiimg.padding(low=1, high=1, axis=zaxis)
                    epiimg.save_as(os.path.join(step01, subj, finfo.Filename), quiet=True)
                for i, finfo in t2:
                    print(" +Filename: {}".format(finfo.Filename))
                    t2img = methods.load(finfo.Abspath)
                    if padding:
                        t2img.padding(low=1, high=1, axis=zaxis)
                    t2img.save_as(os.path.join(step02, subj, finfo.Filename), quiet=True)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess), os.path.join(step02, subj, sess))
                    epi = self._prjobj(f_dataclass, meanfunc, subj, sess)
                    t2 = self._prjobj(a_dataclass, anat, subj, sess)
                    for i, finfo in epi:
                        print("  +Filename: {}".format(finfo.Filename))
                        epiimg = methods.load(finfo.Abspath)
                        if padding:
                            epiimg.padding(low=1, high=1, axis=zaxis)
                        epiimg.save_as(os.path.join(step01, subj, sess, finfo.Filename), quiet=True)
                    for i, finfo in t2:
                        print("  +Filename: {}".format(finfo.Filename))
                        print(finfo.Abspath)
                        t2img = methods.load(finfo.Abspath)
                        if padding:
                            t2img.padding(low=1, high=1, axis=zaxis)
                        t2img.save_as(os.path.join(step02, subj, sess, finfo.Filename), quiet=True)
        return {'meanfunc': step01, 'anat': step02}

    def compute_skullstripping(self, meanfunc, anat, padded=False, zaxis=2):
        axis = {0: 'x', 1: 'y', 2: 'z'}
        f_dataclass, meanfunc = methods.check_dataclass(meanfunc)
        a_dataclass, anat = methods.check_dataclass(anat)
        print('SkullStripping-{} & {}'.format(meanfunc, anat))
        step01 = self.init_step('SkullStripped-meanfunc')
        step02 = self.init_step('SkullStripped-anat')
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj), os.path.join(step02, subj))
            if self._prjobj.single_session:
                # Load image paths
                epi = self._prjobj(1, self._processing, meanfunc, subj, ignore='_mask')
                t2 = self._prjobj(1, self._processing, anat, subj, ignore='_mask')
                # Load mask image obj
                epimask = self._prjobj(1, self._processing, meanfunc, subj, file_tag='_mask').df.Abspath[0]
                t2mask = self._prjobj(1, self._processing, anat, subj, file_tag='_mask').df.Abspath[0]
                # Execute process
                for i, finfo in epi:
                    print(" +Filename of meanfunc: {}".format(finfo.Filename))
                    filename = finfo.Filename
                    fpath = os.path.join(step01, subj, '_{}'.format(filename))
                    self._prjobj.run('afni_3dcalc', fpath, 'a*step(b)',
                                     finfo.Abspath, epimask)
                    ss_epi = methods.load(fpath)
                    if padded:
                        exec('ss_epi.crop({}=[1, {}])'.format(axis[zaxis], ss_epi.shape[zaxis]-1))
                    ss_epi.save_as(os.path.join(step01, subj, filename), quiet=True)
                    os.remove(fpath)
                for i, finfo in t2:
                    print(" +Filename of anat: {}".format(finfo.Filename))
                    filename = finfo.Filename
                    fpath = os.path.join(step02, subj, '_{}'.format(filename))
                    self._prjobj.run('afni_3dcalc', fpath, 'a*step(b)',
                                     finfo.Abspath, t2mask)
                    ss_t2 = methods.load(fpath)
                    if padded:
                        exec('ss_t2.crop({}=[1, {}])'.format(axis[zaxis], ss_t2.shape[zaxis] - 1))
                    ss_t2.save_as(os.path.join(step02, subj, filename), quiet=True)
                    os.remove(fpath)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess), os.path.join(step02, subj, sess))
                    # Load image paths
                    epi = self._prjobj(1, self._processing, meanfunc, subj, sess, ignore='_mask')
                    t2 = self._prjobj(1, self._processing, anat, subj, sess, ignore='_mask')
                    # Load mask image obj
                    epimask = self._prjobj(1, self._processing, meanfunc, subj, sess, file_tag='_mask').df.Abspath[0]
                    t2mask = self._prjobj(1, self._processing, anat, subj, sess, file_tag='_mask').df.Abspath[0]
                    # Execute process
                    for i, finfo in epi:
                        print("  +Filename of meanfunc: {}".format(finfo.Filename))
                        filename = finfo.Filename
                        fpath = os.path.join(step01, subj, sess, '_{}'.format(filename))
                        self._prjobj.run('afni_3dcalc', fpath, 'a*step(b)',
                                         finfo.Abspath, epimask)
                        ss_epi = methods.load(fpath)
                        if padded:
                            exec('ss_epi.crop({}=[1, {}])'.format(axis[zaxis], ss_epi.shape[zaxis] - 1))
                        ss_epi.save_as(os.path.join(step01, subj, sess, filename), quiet=True)
                        os.remove(fpath)
                    for i, finfo in t2:
                        print("  +Filename of anat: {}".format(finfo.Filename))
                        filename = finfo.Filename
                        fpath = os.path.join(step02, subj, sess, '_{}'.format(filename))
                        self._prjobj.run('afni_3dcalc', fpath, 'a*step(b)',
                                         finfo.Abspath, t2mask)
                        ss_t2 = methods.load(fpath)
                        if padded:
                            exec('ss_t2.crop({}=[1, {}])'.format(axis[zaxis], ss_t2.shape[zaxis] - 1))
                        ss_t2.save_as(os.path.join(step02, subj, sess, filename), quiet=True)
                        os.remove(fpath)
        return {'meanfunc': step01, 'anat': step02}

    def timecrop(self, func, crop_loc, dtype='func'):
        """

        Parameters
        ----------
        func
        crop_loc
        dtype

        Returns
        -------

        """
        dataclass, func = methods.check_dataclass(func)
        print('TimeCropped-{}'.format(func))
        step01 = self.init_step('CropTimeAxis-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                # Load image paths
                epi = self._prjobj(1, self._processing, func, subj)
                # Execute process
                for i, finfo in epi:
                    print(" +Filename: {}".format(finfo.Filename))
                    output_path = os.path.join(step01, subj, finfo.Filename)
                    if '.gz' not in output_path:
                        output_path += '.gz'
                    self._prjobj.run('afni_3dcalc', output_path, 'a',
                                     "{}'[{}..{}]'".format(finfo.Abspath, crop_loc[0], crop_loc[1]))
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    # Load image paths
                    epi = self._prjobj(1, self._processing, func, subj)
                    # Execute process
                    for i, finfo in epi:
                        print(" +Filename: {}".format(finfo.Filename))
                        output_path = os.path.join(step01, subj, sess, finfo.Filename)
                        if '.gz' not in output_path:
                            output_path += '.gz'
                        self._prjobj.run('afni_3dcalc', output_path, 'a',
                                         "{}'[{}..{}]'".format(finfo.Abspath, crop_loc[0], crop_loc[1]))
        return {'func': step01}

    def coregistration(self, meanfunc, anat, dtype='func', **kwargs):
        """ Method for mean functional image realignment to anatomical image of same subject

        Parameters
        ----------
        meanfunc   : str
            Datatype or absolute path of the input mean functional image
        anat       : str
            Datatype or absolute path of the input anatomical image
        dtype      : str
            Surfix for the step path
        kwargs     :

        Returns
        -------
        step_paths : dict
        """
        f_dataclass, meanfunc = methods.check_dataclass(meanfunc)
        a_dataclass, anat = methods.check_dataclass(anat)
        print('BiasFieldCorrection-{} & {}'.format(meanfunc, anat))
        step01 = self.init_step('BiasFieldCorrection-{}'.format(dtype))
        step02 = self.init_step('BiasFieldCorrection-{}'.format(anat.split('-')[-1]))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj), os.path.join(step02, subj))
            if self._prjobj.single_session:
                epi = self._prjobj(f_dataclass, meanfunc, subj)
                t2 = self._prjobj(a_dataclass, anat, subj)
                for i, finfo in epi:
                    print(" +Filename of func: {}".format(finfo.Filename))
                    self._prjobj.run('ants_BiasFieldCorrection', os.path.join(step01, subj, finfo.Filename),
                                     finfo.Abspath, algorithm='n4')
                for i, finfo in t2:
                    print(" +Filename of anat: {}".format(finfo.Filename))
                    self._prjobj.run('ants_BiasFieldCorrection', os.path.join(step02, subj, finfo.Filename),
                                     finfo.Abspath, algorithm='n4')
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess), os.path.join(step02, subj, sess))
                    epi = self._prjobj(f_dataclass, meanfunc, subj, sess)
                    t2 = self._prjobj(f_dataclass, anat, subj, sess)
                    for i, finfo in epi:
                        print("  +Filename of func: {}".format(finfo.Filename))
                        self._prjobj.run('ants_BiasFieldCorrection', os.path.join(step01, subj, sess, finfo.Filename),
                                         finfo.Abspath, algorithm='n4')
                    for i, finfo in t2:
                        print("  +Filename of anat: {}".format(finfo.Filename))
                        self._prjobj.run('ants_BiasFieldCorrection', os.path.join(step02, subj, sess, finfo.Filename),
                                         finfo.Abspath, algorithm='n4')
        print('Coregistration-{} to {}'.format(meanfunc, anat))
        step03 = self.init_step('Coregistration-{}2{}'.format(dtype, anat.split('-')[-1]))
        num_step = os.path.basename(step03).split('_')[0]
        step04 = self.final_step('{}_CheckRegistraton-{}'.format(num_step, dtype))
        for subj in self.subjects:
            methods.mkdir(os.path.join(step04, 'AllSubjects'))
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step03, subj))
            if self._prjobj.single_session:
                epi = self._prjobj(1, self._processing, os.path.basename(step01), subj)
                t2 = self._prjobj(1, self._processing, os.path.basename(step02), subj)
                for i, finfo in epi:
                    print(" +Filename: {}".format(finfo.Filename))
                    fixed_img = t2.df.Abspath[0]
                    moved_img = os.path.join(step03, subj, finfo.Filename)
                    self._prjobj.run('afni_3dAllineate', moved_img, finfo.Abspath, onepass=True, EPI=True,
                                     base=fixed_img, cmass='+xy', matrix_save=os.path.join(step03, subj, subj))
                    fig1 = Viewer.check_reg(methods.load(fixed_img),
                                            methods.load(moved_img), sigma=2, **kwargs)
                    fig1.suptitle('EPI to T2 for {}'.format(subj), fontsize=12, color='yellow')
                    fig1.savefig(os.path.join(step04, 'AllSubjects', '{}.png'.format('-'.join([subj, 'func2anat']))),
                                 facecolor=fig1.get_facecolor())
                    fig2 = Viewer.check_reg(methods.load(moved_img),
                                            methods.load(fixed_img), sigma=2, **kwargs)
                    fig2.suptitle('T2 to EPI for {}'.format(subj), fontsize=12, color='yellow')
                    fig2.savefig(os.path.join(step04, 'AllSubjects', '{}.png'.format('-'.join([subj, 'anat2func']))),
                                 facecolor=fig2.get_facecolor())
            else:
                methods.mkdir(os.path.join(step04, subj), os.path.join(step04, subj, 'AllSessions'))
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step03, subj, sess))
                    epi = self._prjobj(1, self._processing, os.path.basename(step01), subj, sess)
                    t2 = self._prjobj(1, self._processing, os.path.basename(step02), subj, sess)
                    for i, finfo in epi:
                        print("  +Filename of anat: {}".format(finfo.Filename))
                        fixed_img = t2.df.Abspath[0]
                        moved_img = os.path.join(step03, subj, sess, finfo.Filename)
                        self._prjobj.run('afni_3dAllineate', moved_img, finfo.Abspath, onepass=True, EPI=True,
                                         base=fixed_img, cmass='+xy',
                                         matrix_save=os.path.join(step03, subj, sess, sess))
                        fig1 = Viewer.check_reg(methods.load(fixed_img),
                                                methods.load(moved_img), sigma=2, **kwargs)
                        fig1.suptitle('EPI to T2 for {}'.format(subj), fontsize=12, color='yellow')
                        fig1.savefig(os.path.join(step04, subj, 'AllSessions',
                                                  '{}.png'.format('-'.join([sess, 'func2anat']))),
                                     facecolor=fig1.get_facecolor())
                        fig2 = Viewer.check_reg(methods.load(moved_img),
                                                methods.load(fixed_img), sigma=2, **kwargs)
                        fig2.suptitle('T2 to EPI for {}'.format(subj), fontsize=12, color='yellow')
                        fig2.savefig(os.path.join(step04, subj, 'AllSessions',
                                                  '{}.png'.format('-'.join([sess, 'anat2func']))),
                                     facecolor=fig2.get_facecolor())
        return {'meanfunc': step01, 'anat':step02, 'realigned_func': step03, 'checkreg': step04}

    def apply_brainmask(self, func, mask, padded=False, zaxis=2, dtype='func'):
        """ Method for applying brain mark to individual 3d+t functional images

        Parameters
        ----------
        func       : str
            Datatype or absolute step path of the input functional image
        mask       : str
            Absolute step path which contains the mask of the functional image
        padded     : bool
        zaxis      : int
        dtype      : str
            Surfix for the step path

        Returns
        -------
        step_paths : dict
        """
        axis = {0: 'x', 1: 'y', 2: 'z'}
        dataclass, func = methods.check_dataclass(func)
        print('ApplyingBrainMask-{}'.format(func))
        step01 = self.init_step('ApplyingBrainMask-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                epi = self._prjobj(dataclass, func, subj)
                epimask = self._prjobj(1, self._processing, os.path.basename(mask), subj, file_tag='_mask').df
                maskobj = methods.load(epimask.Abspath[0])
                if padded:
                    exec ('maskobj.crop({}=[1, {}])'.format(axis[zaxis], maskobj.shape[zaxis] - 1))
                temp_epimask = TempFile(maskobj, 'epimask_{}'.format(subj))
                for i, finfo in epi:
                    print(" +Filename: {}".format(finfo.Filename))
                    self._prjobj.run('afni_3dcalc', os.path.join(step01, subj, finfo.Filename), 'a*step(b)',
                                     finfo.Abspath, str(temp_epimask))
                temp_epimask.close()
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    epi = self._prjobj(dataclass, func, subj, sess)
                    epimask = self._prjobj(1, self._processing, os.path.basename(mask), subj, sess, file_tag='_mask').df
                    maskobj = methods.load(epimask.Abspath[0])
                    if padded:
                        exec ('maskobj.crop({}=[1, {}])'.format(axis[zaxis], maskobj.shape[zaxis] - 1))
                    temp_epimask = TempFile(maskobj, 'epimask_{}_{}'.format(subj, sess))
                    for i, finfo in epi:
                        print("  +Filename: {}".format(finfo.Filename))
                        self._prjobj.run('afni_3dcalc', os.path.join(step01, subj, sess, finfo.Filename), 'a*step(b)',
                                         finfo.Abspath, str(temp_epimask))
                    temp_epimask.close()
        return {'func': step01}

    def apply_transformation(self, func, realigned_func, dtype='func'):
        """ Method for applying transformation matrix to individual 3d+t functional images

        Parameters
        ----------
        func           : str
            Datatype or absolute step path for the input functional image
        realigned_func : str
            Absolute step path which contains the realigned functional image
        dtype          : str
            Surfix for the step path

        Returns
        -------
        step_paths     : dict
        """
        dataclass, func = methods.check_dataclass(func)
        print('ApplyingTransformation-{}'.format(func))
        step01 = self.init_step('ApplyingTransformation-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                ref = self._prjobj(1, self._processing, os.path.basename(realigned_func), subj)
                param = self._prjobj(1, self._processing, os.path.basename(realigned_func), subj, ext='.1D')
                funcs = self._prjobj(dataclass, os.path.basename(func), subj)
                for i, finfo in funcs:
                    print(" +Filename: {}".format(finfo.Filename))
                    moved_img = os.path.join(step01, subj, finfo.Filename)
                    self._prjobj.run('afni_3dAllineate', moved_img, finfo.Abspath, master=ref.df.Abspath.loc[0],
                                     matrix_apply=param.df.Abspath.loc[0])
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    ref = self._prjobj(1, self._processing, os.path.basename(realigned_func), subj, sess)
                    param = self._prjobj(1, self._processing, os.path.basename(realigned_func), subj, sess, ext='.1D')
                    funcs = self._prjobj(dataclass, os.path.basename(func), subj, sess)
                    for i, finfo in funcs:
                        print("  +Filename: {}".format(finfo.Filename))
                        moved_img = os.path.join(step01, subj, sess, finfo.Filename)
                        self._prjobj.run('afni_3dAllineate', moved_img, finfo.Abspath, master=ref.df.Abspath.loc[0],
                                         matrix_apply=param.df.Abspath.loc[0])
        return {'func': step01}

    def global_regression(self, func, dtype='func', detrend=-1):
        """ Method for global signal regression of individual functional image

        Parameters
        ----------
        func       : str
            Datatype or absolute step path for the input functional image
        dtype      : str
            Surfix for the step path
        detrend    : int

        Returns
        -------
        step_paths : dict
        """
        dataclass, func = methods.check_dataclass(func)
        print('GlobalRegression-{}'.format(func))
        step01 = self.init_step('GlobalRegression-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                funcs = self._prjobj(dataclass, func, subj)
                for i, finfo in funcs:
                    print(" +Filename: {}".format(finfo.Filename))
                    regressor = os.path.join(step01, subj, "{}.1D".format(os.path.splitext(finfo.Filename)[0]))
                    self._prjobj.run('afni_3dmaskave', regressor, finfo.Abspath, finfo.Abspath)
                    self._prjobj.run('afni_3dDetrend', os.path.join(step01, subj, finfo.Filename), finfo.Abspath,
                                     vector=regressor, polort='-1')
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    funcs = self._prjobj(dataclass, func, subj)
                    for i, finfo in funcs:
                        print("  +Filename: {}".format(finfo.Filename))
                        regressor = os.path.join(func, subj, sess,
                                                 "{}.1D".format(os.path.splitext(finfo.Filename)[0]))
                        self._prjobj.run('afni_3dmaskave', regressor, finfo.Abspath, finfo.Abspath)
                        self._prjobj.run('afni_3dDetrend', os.path.join(step01, subj, sess, finfo.Filename),
                                         finfo.Abspath, vector=regressor, polort=str(detrend))
        return {'func': step01}

    def motion_parameter_regression(self, func, motioncorrected_func, dtype='func', detrend=-1):
        """ Method for motion parameter regression of individual functional image

        Parameters
        ----------
        func                 : str
            Datatype or absolute path of the input functional image
        motioncorrected_func : str
            Absolute step path which contains the motion corrected functional image
        dtype                : str
            Surfix for the step path
        detrend              : int

        Returns
        -------
        step_paths          : dict
        """
        dataclass, func = methods.check_dataclass(func)
        print('MotionRegression-{}'.format(func))
        step01 = self.init_step('MotionRegression-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                funcs = self._prjobj(dataclass, func, subj)
                for i, finfo in funcs:
                    print(" +Filename: {}".format(finfo.Filename))
                    regressor = self._prjobj(dataclass, motioncorrected_func, subj, ext='.1D', ignore='.aff12',
                                             file_tag=os.path.splitext(finfo.Filename)[0]).df.Abspath[i]
                    self._prjobj.run('afni_3dDetrend', os.path.join(step01, subj, finfo.Filename), finfo.Abspath,
                                     vector=regressor, polort=str(detrend))
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    funcs = self._prjobj(dataclass, func, subj, sess)
                    for i, finfo in funcs:
                        print("  +Filename: {}".format(finfo.Filename))
                        regressor = self._prjobj(dataclass, motioncorrected_func, subj, ext='.1D', ignore='.aff12',
                                                 file_tag=os.path.splitext(finfo.Filename)[0]).df.Abspath[i]
                        self._prjobj.run('afni_3dDetrend', os.path.join(step01, subj, sess, finfo.Filename),
                                         finfo.Abspath, vector=regressor, polort=str(detrend))
        return {'func': step01}

    def cbv_calculation(self, func, meanBOLD, mean_range=10, dtype='func', **kwargs):
        """

        Parameters
        ----------
        func
        meanBOLD
        dtype
        kwargs

        Returns
        -------

        """
        dataclass, func = methods.check_dataclass(func)
        mb_dataclass, meanBOLD = methods.check_dataclass(meanBOLD)
        print('CBV_Calculation-{}'.format(func))
        step01 = self.init_step('CBV_Calculation-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                funcs = self._prjobj(dataclass, func, subj)
                szero = self._prjobj(mb_dataclass, meanBOLD, subj).df.loc[0]
                for i, finfo in funcs:
                    imgobj = methods.load(finfo.Abspath)
                    imgobj._dataobj = np.mean(imgobj._dataobj[:, :, :, :mean_range], axis=3)
                    spre = TempFile(imgobj, 'spre_{}'.format(subj))
                    print(" +Filename: {}".format(finfo.Filename))
                    self._prjobj.run('afni_3dcalc', os.path.join(step01, subj, finfo.Filename), 'log(a/b)/log(b/c)',
                                     finfo.Abspath, str(spre), szero.Abspath)
                    spre.close()
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    funcs = self._prjobj(dataclass, func, subj, sess)
                    szero = self._prjobj(mb_dataclass, meanBOLD, subj, sess).df.loc[0]
                    for i, finfo in funcs:
                        imgobj = methods.load(finfo.Abspath)
                        imgobj._dataobj = np.mean(imgobj._dataobj[:, :, :, :mean_range], axis=3)
                        spre = TempFile(imgobj, 'spre_{}_{}'.format(subj, sess))
                        print(" +Filename: {}".format(finfo.Filename))
                        self._prjobj.run('afni_3dcalc', os.path.join(step01, subj, sess, finfo.Filename), 'log(a/b)/log(b/c)',
                                         finfo.Abspath, str(spre), szero.Abspath)
                        spre.close()
        return {'cbv': step01}

    def spatial_smoothing(self, func, mask=False, FWHM=False, quiet=False, dtype='func'):
        """

        Parameters
        ----------
        func
        mask
        FWHM
        quiet

        Returns
        -------

        """
        dataclass, func = methods.check_dataclass(func)
        print('SpatialSmoothing-{}'.format(func))
        step01 = self.init_step('SpatialSmoothing-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                epi = self._prjobj(dataclass, func, subj)
                if mask:
                    epimask = self._prjobj(1, self._processing, os.path.basename(mask), subj, file_tag='_mask').df
                else:
                    epimask = None
                for i, finfo in epi:
                    if mask:
                        mask = epimask[i].Abspath
                    print(" +Filename: {}".format(finfo.Filename))
                    self._prjobj.run('afni_3dBlurInMask', os.path.join(step01, subj, finfo.Filename), finfo.Abspath,
                                     mask=mask, FWHM=FWHM, quiet=quiet)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    epi = self._prjobj(dataclass, func, subj, sess)
                    if mask:
                        epimask = self._prjobj(1, self._processing, os.path.basename(mask), subj, sess, file_tag='_mask').df
                    else:
                        epimask = False
                    for i, finfo in epi:
                        if mask:
                            mask = epimask[i].Abspath
                        print(" +Filename: {}".format(finfo.Filename))
                        self._prjobj.run('afni_3dBlurInMask', os.path.join(step01, subj, sess, finfo.Filename),
                                         finfo.Abspath, mask=mask, FWHM=FWHM, quiet=quiet)
        return {'func': step01}

    def signal_processing(self, func, dt=1, norm=False, despike=False, detrend=False,
                          blur=False, band=False, dtype='func', file_tag=None, ignore=None):
        """ Method for signal processing and spatial smoothing of individual functional image

        Parameters
        ----------
        func        : str
            Datatype or Absolute step path for the input functional image
        dt          : int
        norm        : boolean
        despike     : boolean
        detrend     : int
        blur        : float
        band        : list of float
        dtype       : str
            Surfix for the step path

        Returns
        -------
        step_paths  : dict
        """
        dataclass, func = methods.check_dataclass(func)
        print('SignalProcessing-{}'.format(func))
        step01 = self.init_step('SignalProcessing-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                if not file_tag:
                    if not ignore:
                        funcs = self._prjobj(dataclass, func, subj)
                    else:
                        funcs = self._prjobj(dataclass, func, subj, ignore=ignore)
                else:
                    if not ignore:
                        funcs = self._prjobj(dataclass, func, subj, file_tag=file_tag)
                    else:
                        funcs = self._prjobj(dataclass, func, subj, file_tag=file_tag, ignore=ignore)
                for i, finfo in funcs:
                    print(" +Filename: {}".format(finfo.Filename))
                    self._prjobj.run('afni_3dBandpass', os.path.join(step01, subj, finfo.Filename), finfo.Abspath,
                                     norm=norm, despike=despike, detrend=detrend, blur=blur, band=band, dt=dt)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    if not file_tag:
                        if not ignore:
                            funcs = self._prjobj(dataclass, func, subj, sess)
                        else:
                            funcs = self._prjobj(dataclass, func, subj, sess, ignore=ignore)
                    else:
                        if not ignore:
                            funcs = self._prjobj(dataclass, func, subj, sess, file_tag=file_tag)
                        else:
                            funcs = self._prjobj(dataclass, func, subj, sess, file_tag=file_tag, ignore=ignore)
                    for i, finfo in funcs:
                        print("  +Filename: {}".format(finfo.Filename))
                        self._prjobj.run('afni_3dBandpass', os.path.join(step01, subj, sess, finfo.Filename), finfo.Abspath,
                                         norm=norm, despike=despike, detrend=detrend, blur=blur, band=band, dt=dt)
        return {'func': step01}

    def warp_func(self, func, warped_anat, tempobj, atlas=False, dtype='func', **kwargs):
        """ Method for warping the individual functional image to template space

        Parameters
        ----------
        func        : str
            Datatype or Absolute step path for the input functional image
        warped_anat : str
            Absolute step path which contains diffeomorphic map and transformation matrix
            which is generated by the methods of 'pynit.Preprocessing.warp_anat_to_template'
        tempobj     : pynit.Template
            The template object which contains set of atlas
        dtype       : str
            Surfix for the step path

        Returns
        -------
        step_paths  : dict
        """
        # Check the source of input data
        in_kwargs = {'atlas': atlas}
        dataclass, func = methods.check_dataclass(func)
        print("Warp-{} to Atlas and Check it's registration".format(func))
        step01 = self.init_step('Warp-{}2atlas'.format(dtype))
        num_step = os.path.basename(step01).split('_')[0]
        step02 = self.final_step('{}_CheckAtlasRegistration-{}'.format(num_step, dtype))
        # Loop the subjects
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                methods.mkdir(os.path.join(step02, 'AllSubjects'))
                # Grab the warping map and transform matrix
                mats, warps, warped = methods.get_warp_matrix(self, warped_anat, subj, inverse=False)
                temp_path = os.path.join(step01, subj, "base")
                tempobj.save_as(temp_path, quiet=True)
                funcs = self._prjobj(dataclass, func, subj)
                print(" +Filename of fixed image: {}".format(warped.Filename))
                for i, finfo in funcs:
                    print(" +Filename of moving image: {}".format(finfo.Filename))
                    output_path = os.path.join(step01, subj, finfo.Filename)
                    self._prjobj.run('ants_WarpTimeSeriresImageMultiTransform', output_path,
                                     finfo.Abspath, warped.Abspath, warps, mats, **in_kwargs)
                # subjatlas = methods.load_temp(warped.Abspath, '{}_atlas.nii'.format(temp_path))
                subjatlas = methods.load_temp(output_path, '{}_atlas.nii'.format(temp_path))
                # subjatlas.show()
                fig = subjatlas.show(**kwargs)
                if type(fig) is tuple:
                    fig = fig[0]
                fig.suptitle('Check atlas registration of {}'.format(subj), fontsize=12, color='yellow')
                fig.savefig(os.path.join(step02, 'AllSubjects', '{}.png'.format('-'.join([subj, 'checkatlas']))),
                            facecolor=fig.get_facecolor())
                os.remove('{}_atlas.nii'.format(temp_path))
                os.remove('{}_atlas.label'.format(temp_path))
                os.remove('{}_template.nii'.format(temp_path))
            else:
                methods.mkdir(os.path.join(step02, subj))
                for sess in self.sessions:
                    methods.mkdir(os.path.join(step02, subj, 'AllSessions'), os.path.join(step01, subj, sess))
                    print(" :Session: {}".format(sess))
                    # Grab the warping map and transform matrix
                    mats, warps, warped = methods.get_warp_matrix(self, warped_anat, subj, sess, inverse=False)
                    temp_path = os.path.join(step01, subj, sess, "base")
                    tempobj.save_as(temp_path, quiet=True)
                    funcs = self._prjobj(dataclass, func, subj, sess)
                    print(" +Filename of fixed image: {}".format(warped.Filename))
                    for i, finfo in funcs:
                        print(" +Filename of moving image: {}".format(finfo.Filename))
                        output_path = os.path.join(step01, subj, sess, finfo.Filename)
                        self._prjobj.run('ants_WarpTimeSeriesImageMultiTransform', output_path,
                                         finfo.Abspath, warped.Abspath, warps, mats, **in_kwargs)
                    # subjatlas = methods.load_temp(warped.Abspath, '{}_atlas.nii'.format(temp_path))
                    subjatlas = methods.load_temp(output_path, '{}_atlas.nii'.format(temp_path))
                    fig = subjatlas.show(**kwargs)
                    if type(fig) is tuple:
                        fig = fig[0]
                    fig.suptitle('Check atlas registration of {}'.format(subj), fontsize=12, color='yellow')
                    fig.savefig(os.path.join(step02, subj, 'AllSessions',
                                             '{}.png'.format('-'.join([subj, sess, 'checkatlas']))),
                                facecolor=fig.get_facecolor())
                    os.remove('{}_atlas.nii'.format(temp_path))
                    os.remove('{}_atlas.label'.format(temp_path))
                    os.remove('{}_template.nii'.format(temp_path))
        return {'func': step01, 'checkreg': step02}

    def linear_spatial_normalization(self, anat, tempobj, dtype='anat', **kwargs):
        """

        Parameters
        ----------
        anat
        tempobj
        dtype
        kwargs

        Returns
        -------

        """
        # Check the source of input data
        dataclass, anat = methods.check_dataclass(anat)
        # Print step ans initiate the step
        print('SpatialNormalization-{} to Tempalte'.format(anat))
        step01 = self.init_step('SpatialNormalization-{}2temp'.format(dtype))
        num_step = os.path.basename(step01).split('_')[0]
        step02 = self.final_step('{}_CheckRegistraton-{}2Temp'.format(num_step, dtype))
        # Loop the subjects
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                methods.mkdir(os.path.join(step02, 'AllSubjects'))
                anats = self._prjobj(dataclass, anat, subj)
                methods.mkdir(os.path.join(step01, subj))
                for i, finfo in anats:
                    print(" +Filename: {}".format(finfo.Filename))
                    fixed_img = tempobj.template_path
                    moved_img = os.path.join(step01, subj, finfo.Filename)
                    trans_mat = methods.splitnifti(moved_img)+'.aff12.1D'
                    self._prjobj.run('afni_3dAllineate', moved_img,
                                     finfo.Abspath, base=fixed_img, twopass=True, cmass='xy',
                                     zclip=True, conv='0.01', cost='crM', ckeck='nmi', warp='shr',
                                     matrix_save=trans_mat)
                    fig1 = Viewer.check_reg(methods.load(fixed_img),
                                            methods.load(moved_img), sigma=2, **kwargs)
                    fig1.suptitle('T2 to Temp for {}'.format(subj), fontsize=12, color='yellow')
                    fig1.savefig(os.path.join(step02, 'AllSubjects', '{}.png'.format('-'.join([subj, 'anat2temp']))),
                                 facecolor=fig1.get_facecolor())
                    fig2 = Viewer.check_reg(methods.load(moved_img),
                                            methods.load(fixed_img), sigma=2, **kwargs)
                    fig2.suptitle('Temp to T2 for {}'.format(subj), fontsize=12, color='yellow')
                    fig2.savefig(os.path.join(step02, 'AllSubjects', '{}.png'.format('-'.join([subj, 'temp2anat']))),
                                 facecolor=fig2.get_facecolor())
            else:
                methods.mkdir(os.path.join(step02, subj), os.path.join(step02, subj, 'AllSessions'))
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    anats = self._prjobj(dataclass, anat, subj, sess)
                    methods.mkdir(os.path.join(step01, subj, sess))
                    methods.mkdir(os.path.join(step02, subj, 'AllSessions'))
                    for i, finfo in anats:
                        print("  +Filename: {}".format(finfo.Filename))
                        fixed_img = tempobj.template_path
                        moved_img = os.path.join(step01, subj, sess, finfo.Filename)
                        trans_mat = methods.splitnifti(moved_img) + '.aff12.1D'
                        self._prjobj.run('afni_3dAllineate', moved_img,
                                         finfo.Abspath, base=fixed_img, twopass=True, cmass='xy',
                                         zclip=True, conv='0.01', cost='crM', ckeck='nmi', warp='shr',
                                         matrix_save=trans_mat)
                        fig1 = Viewer.check_reg(methods.load(fixed_img),
                                                methods.load(moved_img), sigma=2, **kwargs)
                        fig1.suptitle('T2 to Temp for {}-{}'.format(subj, sess), fontsize=12, color='yellow')
                        fig1.savefig(
                            os.path.join(step02, subj, 'AllSessions',
                                         '{}.png'.format('-'.join([subj, sess, 'anat2temp']))),
                            facecolor=fig1.get_facecolor())
                        fig2 = Viewer.check_reg(methods.load(moved_img),
                                                methods.load(fixed_img), sigma=2, **kwargs)
                        fig2.suptitle('Temp to T2 for {}-{}'.format(subj, sess), fontsize=12, color='yellow')
                        fig2.savefig(
                            os.path.join(step02, subj, 'AllSessions',
                                         '{}.png'.format('-'.join([subj, sess, 'temp2anat']))),
                            facecolor=fig2.get_facecolor())
        return {'norm_anat': step01, 'checkreg': step02}

    def apply_spatial_normalization(self, func, norm_anat, tempobj, dtype='func', **kwargs):
        """

        Parameters
        ----------
        func
        norm_anat
        dtype

        Returns
        -------

        """
        dataclass, func = methods.check_dataclass(func)
        print('ApplyingSpatialNormalization-{}'.format(func))
        step01 = self.init_step('ApplyingSpatialNormalization-{}'.format(dtype))
        num_step = os.path.basename(step01).split('_')[0]
        step02 = self.final_step('{}_CheckAtlasRegistration-{}'.format(num_step, dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj),
                            os.path.join(step02, 'AllSubjects'))
            if self._prjobj.single_session:
                ref = self._prjobj(1, self._processing, os.path.basename(norm_anat), subj)
                param = self._prjobj(1, self._processing, os.path.basename(norm_anat), subj, ext='.1D')
                temp_path = os.path.join(step01, subj, "base")
                tempobj.save_as(temp_path, quiet=True)
                funcs = self._prjobj(dataclass, os.path.basename(func), subj)
                for i, finfo in funcs:
                    print(" +Filename: {}".format(finfo.Filename))
                    moved_img = os.path.join(step01, subj, finfo.Filename)
                    self._prjobj.run('afni_3dAllineate', moved_img, finfo.Abspath, master=ref.df.Abspath.loc[0],
                                     matrix_apply=param.df.Abspath.loc[0], warp='shr')
                subjatlas = methods.load_temp(moved_img, '{}_atlas.nii'.format(temp_path))
                fig = subjatlas.show(**kwargs)
                if type(fig) is tuple:
                    fig = fig[0]
                fig.suptitle('Check atlas registration of {}'.format(subj), fontsize=12, color='yellow')
                fig.savefig(
                    os.path.join(step02, 'AllSubjects', '{}.png'.format('-'.join([subj, 'checkatlas']))),
                    facecolor=fig.get_facecolor())
                os.remove('{}_atlas.nii'.format(temp_path))
                os.remove('{}_atlas.label'.format(temp_path))
                os.remove('{}_template.nii'.format(temp_path))
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess),
                                    os.path.join(step02, subj),
                                    os.path.join(step02, subj, 'AllSessions'))
                    ref = self._prjobj(1, self._processing, os.path.basename(norm_anat), subj, sess)
                    param = self._prjobj(1, self._processing, os.path.basename(norm_anat), subj, sess, ext='.1D')
                    temp_path = os.path.join(step01, subj, sess, "base")
                    tempobj.save_as(temp_path, quiet=True)
                    funcs = self._prjobj(dataclass, os.path.basename(func), subj, sess)
                    for i, finfo in funcs:
                        print(" +Filename: {}".format(finfo.Filename))
                        moved_img = os.path.join(step01, subj, sess, finfo.Filename)
                        self._prjobj.run('afni_3dAllineate', moved_img, finfo.Abspath, master=ref.df.Abspath.loc[0],
                                         matrix_apply=param.df.Abspath.loc[0], warp='shr')
                    subjatlas = methods.load_temp(moved_img, '{}_atlas.nii'.format(temp_path))
                    fig = subjatlas.show(**kwargs)
                    if type(fig) is tuple:
                        fig = fig[0]
                    fig.suptitle('Check atlas registration of {}-{}'.format(subj, sess), fontsize=12, color='yellow')
                    fig.savefig(
                        os.path.join(step02, subj, 'AllSessions',
                                     '{}.png'.format('-'.join([subj, sess, 'checkatlas']))),
                        facecolor=fig.get_facecolor())
                    os.remove('{}_atlas.nii'.format(temp_path))
                    os.remove('{}_atlas.label'.format(temp_path))
                    os.remove('{}_template.nii'.format(temp_path))

        return {'func': step01}

    def warp_anat_to_template(self, anat, tempobj, dtype='anat', ttype='s', **kwargs): # TODO: This code not work if the template image resolution is different with T2 image
        """ Method for warping the individual anatomical image to template

        Parameters
        ----------
        anat        : str
            Datatype or Absolute step path for the input anatomical image
        tempobj     : pynit.Template
            The template object which contains set of atlas
        dtype       : str
            Surfix for the step path

        ttype       : str
            Type of transformation
            's' : Warping
            'a' : Affine

        Returns
        -------
        step_paths  : dict
        """
        # Check the source of input data
        if os.path.exists(anat):
            dataclass = 1
            anat = methods.path_splitter(anat)[-1]
        else:
            dataclass = 0
        # Print step ans initiate the step
        print('Warp-{} to Tempalte'.format(anat))
        step01 = self.init_step('Warp-{}2temp'.format(dtype))
        num_step = os.path.basename(step01).split('_')[0]
        step02 = self.final_step('{}_CheckRegistraton-{}'.format(num_step, dtype))
        # Loop the subjects
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                methods.mkdir(os.path.join(step02, 'AllSubjects'))
                anats = self._prjobj(dataclass, anat, subj)
                methods.mkdir(os.path.join(step01, subj))
                for i, finfo in anats:
                    print(" +Filename: {}".format(finfo.Filename))
                    output_path = os.path.join(step01, subj, "{}".format(subj))
                    self._prjobj.run('ants_RegistrationSyn', output_path,
                                     finfo.Abspath, base_path=tempobj.template_path, quick=False, ttype=ttype)
                    fig1 = Viewer.check_reg(methods.load(tempobj.template_path),
                                            methods.load("{}_Warped.nii.gz".format(output_path)), sigma=2, **kwargs)
                    fig1.suptitle('T2 to Atlas for {}'.format(subj), fontsize=12, color='yellow')
                    fig1.savefig(os.path.join(step02, 'AllSubjects', '{}.png'.format('-'.join([subj, 'anat2temp']))),
                                 facecolor=fig1.get_facecolor())
                    fig2 = Viewer.check_reg(methods.load("{}_Warped.nii.gz".format(output_path)),
                                            methods.load(tempobj.template_path), sigma=2, **kwargs)
                    fig2.suptitle('Atlas to T2 for {}'.format(subj), fontsize=12, color='yellow')
                    fig2.savefig(os.path.join(step02, 'AllSubjects', '{}.png'.format('-'.join([subj, 'temp2anat']))),
                                 facecolor=fig2.get_facecolor())
            else:
                methods.mkdir(os.path.join(step02, subj), os.path.join(step02, subj, 'AllSessions'))
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    anats = self._prjobj(dataclass, anat, subj, sess)
                    methods.mkdir(os.path.join(step01, subj, sess))
                    for i, finfo in anats:
                        print("  +Filename: {}".format(finfo.Filename))
                        output_path = os.path.join(step01, subj, sess, "{}".format(subj))
                        self._prjobj.run('ants_RegistrationSyn', output_path,
                                         finfo.Abspath, base_path=tempobj.template_path, quick=False, ttype=ttype)
                        fig1 = Viewer.check_reg(methods.load(tempobj.template_path),
                                                methods.load("{}_Warped.nii.gz".format(output_path)),
                                                sigma=2, **kwargs)
                        fig1.suptitle('T2 to Atlas for {}'.format(subj), fontsize=12, color='yellow')
                        fig1.savefig(
                            os.path.join(step02, subj, 'AllSessions', '{}.png'.format('-'.join([subj, 'anat2temp']))),
                            facecolor=fig1.get_facecolor())
                        fig2 = Viewer.check_reg(methods.load("{}_Warped.nii.gz".format(output_path)),
                                                methods.load(tempobj.template_path), sigma=2, **kwargs)
                        fig2.suptitle('Atlas to T2 for {}'.format(subj), fontsize=12, color='yellow')
                        fig2.savefig(
                            os.path.join(step02, subj, 'AllSessions', '{}.png'.format('-'.join([subj, 'temp2anat']))),
                            facecolor=fig2.get_facecolor())
        return {'warped_anat': step01, 'checkreg': step02}

    def warp_atlas_to_anat(self, anat, warped_anat, tempobj, dtype='anat', **kwargs):
        """ Method for warping the atlas to individual anatomical image space

        Parameters
        ----------
        anat        : str
            Datatype or Absolute step path for the input anatomical image
        warped_anat : str
            Absolute step path which contains diffeomorphic map and transformation matrix
            which is generated by the methods of 'pynit.Preprocessing.warp_anat_to_template'
        tempobj     : pynit.Template
            The template object which contains set of atlas
        dtype       : str
            Surfix for the step path

        Returns
        -------
        step_paths  : dict
        """
        dataclass, anat = methods.check_dataclass(anat)
        print("Warp-Atlas to {} and Check it's registration".format(anat))
        step01 = self.init_step('Warp-atlas2{}'.format(dtype))
        num_step = os.path.basename(step01).split('_')[0]
        step02 = self.final_step('{}_CheckAtlasRegistration-{}'.format(num_step, dtype))
        # Loop the subjects
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                # Grab the warping map and transform matrix
                mats, warps, warped = methods.get_warp_matrix(self, warped_anat, subj, inverse=True)
                temp_path = os.path.join(warped_anat, subj, "base")
                tempobj.save_as(temp_path, quiet=True)
                anats = self._prjobj(dataclass, anat, subj)
                output_path = os.path.join(step01, subj, "{}_atlas.nii".format(subj))
                methods.mkdir(os.path.join(step01, subj), os.path.join(step02, 'AllSubjects'))
                print(" +Filename: {}".format(warped.Filename))
                self._prjobj.run('ants_WarpImageMultiTransform', output_path,
                                 '{}_atlas.nii'.format(temp_path), warped.Abspath,
                                 True, '-i', mats, warps)
                tempobj.atlasobj.save_as(os.path.join(step01, subj, "{}_atlas".format(subj)), label_only=True)
                for i, finfo in anats:
                    subjatlas = methods.load_temp(finfo.Abspath, output_path)
                    fig = subjatlas.show(**kwargs)
                    if type(fig) is tuple:
                        fig = fig[0]
                    fig.suptitle('Check atlas registration of {}'.format(subj), fontsize=12, color='yellow')
                    fig.savefig(os.path.join(step02, '{}.png'.format('-'.join([subj, 'checkatlas']))),
                                facecolor=fig.get_facecolor())
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    # Grab the warping map and transform matrix
                    mats, warps, warped = methods.get_warp_matrix(self, warped_anat, subj, sess, inverse=True)
                    temp_path = os.path.join(step01, subj, sess, "base")
                    tempobj.save_as(temp_path, quiet=True)
                    anats = self._prjobj(dataclass, anat, subj, sess)
                    output_path = os.path.join(step01, subj, sess, "{}_atlas.nii".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess), os.path.join(step02, subj, 'AllSessoions'))
                    print(" +Filename: {}".format(warped.Filename))
                    self._prjobj.run('ants_WarpImageMultiTransform', output_path,
                                     '{}_atlas.nii'.format(temp_path), warped.Abspath, True, '-i', mats, warps)
                    tempobj.atlasobj.save_as(os.path.join(step01, subj, sess, "{}_atlas".format(sess)), label_only=True)
                    for i, finfo in anats:
                        subjatlas = methods.load_temp(finfo.Abspath, output_path)
                        fig = subjatlas.show(**kwargs)
                        if type(fig) is tuple:
                            fig = fig[0]
                        fig.suptitle('Check atlas registration of {}'.format(sess), fontsize=12, color='yellow')
                        fig.savefig(os.path.join(step02, subj, 'AllSessions',
                                                 '{}.png'.format('-'.join([sess, 'checkatlas']))),
                                    facecolor=fig.get_facecolor())
        return {'atlas': step01, 'checkreg': step02}

    def get_timetrace(self, func, atlas, dtype='func', file_tag=None, ignore=None, subjects=None, **kwargs):
        """ Method for extracting timecourse from mask

        Parameters
        ----------
        func       : str
            Datatype or absolute path of the input mean functional image
        atlas      : str
            template object which has atlas, or folder name which includes all mask
        dtype      : str
            Surfix for the step path
        kwargs     :

        Returns
        -------
        step_paths : dict

        """
        if not subjects: #TODO: Subject selection testcode, need to apply for all steps
            subjects = self.subjects[:]
        dataclass, func = methods.check_dataclass(func)
        atlas, tempobj = methods.check_atals_datatype(atlas)
        print('ExtractTimeCourseData-{}'.format(func))
        step01 = self.init_step('ExtractTimeCourse-{}'.format(dtype))
        for subj in subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                if not tempobj:
                    # atlas = self._prjobj(1, self._pipeline, atlas, subj).df.Abspath.loc[0]
                    warped = self._prjobj(1, self._processing, subj, file_tag='_InverseWarped').df.Abspath.loc[0]
                    tempobj = methods.load_temp(warped, atlas)
                if not file_tag:
                    if not ignore:
                        funcs = self._prjobj(dataclass, func, subj)
                    else:
                        funcs = self._prjobj(dataclass, func, subj, ignore=ignore)
                else:
                    if not ignore:
                        funcs = self._prjobj(dataclass, func, subj, file_tag=file_tag)
                    else:
                        funcs = self._prjobj(dataclass, func, subj, file_tag=file_tag, ignore=ignore)
                for i, finfo in funcs:
                    print(" +Filename: {}".format(finfo.Filename))
                    df = Analysis.get_timetrace(methods.load(finfo.Abspath), tempobj, afni=True, **kwargs)
                    df.to_excel(os.path.join(step01, subj, "{}.xlsx".format(os.path.splitext(finfo.Filename)[0])))
            else:
                methods.mkdir(os.path.join(step01, subj))
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    if not tempobj:
                        # atlas = self._prjobj(1, self._pipeline, atlas, subj, sess).df.Abspath.loc[0]
                        warped = self._prjobj(1, self._processing, subj, sess,
                                              file_tag='_InverseWarped').df.Abspath.loc[0]
                        tempobj = methods.load_temp(warped, atlas)
                    if not file_tag:
                        if not ignore:
                            funcs = self._prjobj(dataclass, func, subj, sess)
                        else:
                            funcs = self._prjobj(dataclass, func, subj, sess, ignore=ignore)
                    else:
                        if not ignore:
                            funcs = self._prjobj(dataclass, func, subj, sess, file_tag=file_tag)
                        else:
                            funcs = self._prjobj(dataclass, func, subj, sess, file_tag=file_tag, ignore=ignore)
                    methods.mkdir(os.path.join(step01, subj, sess))
                    for i, finfo in funcs:
                        print("  +Filename: {}".format(finfo.Filename))
                        df = Analysis.get_timetrace(methods.load(finfo.Abspath), tempobj, afni=True, **kwargs)
                        df.to_excel(os.path.join(step01, subj, sess, "{}.xlsx".format(
                            os.path.splitext(finfo.Filename)[0])))
        return {'timecourse': step01}

    def get_correlation_matrix(self, func, atlas, dtype='func', file_tag=None, ignore=None, subjects=None, **kwargs):
        """ Method for extracting timecourse, correlation matrix and calculating z-score matrix

        Parameters
        ----------
        func       : str
            Datatype or absolute path of the input mean functional image
        atlas      : str
        dtype      : str
            Surfix for the step path
        kwargs     :

        Returns
        -------
        step_paths : dict

        """
        if not subjects: #TODO: Subject selection testcode, need to apply for all steps
            subjects = self.subjects[:]
        dataclass, func = methods.check_dataclass(func)
        atlas, tempobj = methods.check_atals_datatype(atlas)
        print('ExtractTimeCourseData-{}'.format(func))
        step01 = self.init_step('ExtractTimeCourse-{}'.format(dtype))
        step02 = self.init_step('CC_Matrix-{}'.format(dtype))
        num_step = os.path.basename(step02).split('_')[0]
        step03 = self.final_step('{}_Zscore_Matrix-{}'.format(num_step, dtype))
        for subj in subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj), os.path.join(step02, subj), os.path.join(step03, subj))
            if self._prjobj.single_session:
                if not tempobj:
                    # atlas = self._prjobj(1, self._pipeline, atlas, subj).df.Abspath.loc[0]
                    warped = self._prjobj(1, self._processing, subj, file_tag='_InverseWarped').df.Abspath.loc[0]
                    tempobj = methods.load_temp(warped, atlas)
                if not file_tag:
                    if not ignore:
                        funcs = self._prjobj(dataclass, func, subj)
                    else:
                        funcs = self._prjobj(dataclass, func, subj, ignore=ignore)
                else:
                    if not ignore:
                        funcs = self._prjobj(dataclass, func, subj, file_tag=file_tag)
                    else:
                        funcs = self._prjobj(dataclass, func, subj, file_tag=file_tag, ignore=ignore)
                for i, finfo in funcs:
                    print(" +Filename: {}".format(finfo.Filename))
                    df = Analysis.get_timetrace(methods.load(finfo.Abspath), tempobj, afni=True, **kwargs)
                    df.to_excel(os.path.join(step01, subj, "{}.xlsx".format(os.path.splitext(finfo.Filename)[0])))
                    df.corr().to_excel(os.path.join(step02, subj, "{}.xlsx".format(
                        os.path.splitext(finfo.Filename)[0])))
                    np.arctanh(df.corr()).to_excel(
                        os.path.join(step03, subj, "{}.xlsx").format(os.path.splitext(finfo.Filename)[0]))
            else:
                methods.mkdir(os.path.join(step01, subj), os.path.join(step02, subj),
                                os.path.join(step03, subj))
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    if not tempobj:
                        # atlas = self._prjobj(1, self._pipeline, atlas, subj, sess).df.Abspath.loc[0]
                        warped = self._prjobj(1, self._processing, subj, sess,
                                              file_tag='_InverseWarped').df.Abspath.loc[0]
                        tempobj = methods.load_temp(warped, atlas)
                    if not file_tag:
                        if not ignore:
                            funcs = self._prjobj(dataclass, func, subj, sess)
                        else:
                            funcs = self._prjobj(dataclass, func, subj, sess, ignore=ignore)
                    else:
                        if not ignore:
                            funcs = self._prjobj(dataclass, func, subj, sess, file_tag=file_tag)
                        else:
                            funcs = self._prjobj(dataclass, func, subj, sess, file_tag=file_tag, ignore=ignore)
                    methods.mkdir(os.path.join(step01, subj, sess), os.path.join(step02, subj, sess),
                                    os.path.join(step03, subj, sess))
                    for i, finfo in funcs:
                        print("  +Filename: {}".format(finfo.Filename))
                        df = Analysis.get_timetrace(methods.load(finfo.Abspath), tempobj, afni=True, **kwargs)
                        df.to_excel(os.path.join(step01, subj, sess, "{}.xlsx".format(
                            os.path.splitext(finfo.Filename)[0])))
                        df.corr().to_excel(
                            os.path.join(step02, subj, sess, "{}.xlsx".format(os.path.splitext(finfo.Filename)[0])))
                        np.arctanh(df.corr()).to_excel(
                            os.path.join(step03, subj, sess, "{}.xlsx".format(os.path.splitext(finfo.Filename)[0])))
        return {'timecourse': step01, 'cc_matrix': step02}

    def set_stim_paradigm(self, num_of_time, tr, filename='stim_paradigm', **kwargs):
        onset = []
        num_stimts = 1
        duration = None
        peak = 1
        stim_type = None
        if kwargs:
            for kwarg in kwargs.keys():
                if kwarg is 'onset':
                    if type(kwargs[kwarg]) is not list:
                        raise messages.CommandExecutionFailure
                    else:
                        onset = kwargs[kwarg]
                if kwarg is 'duration':
                    if type(kwargs[kwarg]) is not int:
                        raise messages.CommandExecutionFailure
                    else:
                        duration = str(kwargs[kwarg])
                if kwarg is 'peak':
                    if type(kwargs[kwarg]) is not int:
                        raise messages.CommandExecutionFailure
                    else:
                        peak = str(kwargs[kwarg])
                if kwarg is 'hrf_function':
                    if type(kwargs[kwarg]) is not str:
                        raise messages.CommandExecutionFailure
                    else:
                        if kwargs[kwarg] is 'MION':
                            stim_type = "MIONN({})".format(duration)
                        elif kwargs[kwarg] is 'BLOCK':
                            stim_type = "BLOCK({},{})".format(duration, peak)
                        else:
                            raise messages.CommandExecutionFailure
        output_path = os.path.join('.tmp', '{}.xmat.1D'.format(filename))
        Interface.afni_3dDeconvolve(output_path, None, nodata=[str(num_of_time), str(tr)],
                                    num_stimts=num_stimts, polort=-1,
                                    stim_times=['1', '1D: {}'.format(" ".join(onset)),
                                                "'{}'".format(stim_type)])
        return {'paradigm': output_path}

    # def general_linear_model(self, func, paradigm, dtype='func'):
    #     if os.path.exists(func):
    #         dataclass = 1
    #         func = methods.path_splitter(func)[-1]
    #     else:
    #         dataclass = 0
    #     print('GLM Analysis-{}'.format(func))
    #     step01 = self.init_step('ExtractTimeCourse-{}'.format(dtype))
    #     # num_step = os.path.basename(step02).split('_')[0]
    #     # step02 = self.final_step('{}_ActivityMap-{}'.format(num_step, dtype))
    #     for subj in self.subjects:
    #         print("-Subject: {}".format(subj))
    #         SystemMethods.mkdir(os.path.join(step01, subj))
    #         if self._prjobj.single_session:
    #             funcs = self._prjobj(dataclass, func, subj)
    #             for i, finfo in funcs.iterrows():
    #                 print(" +Filename: {}".format(finfo.Filename))
    #                 output_path = os.path.join(step01, subj, finfo.Filename)
    #                 self._prjobj.run('afni_3dDeconvolve', str(output_path), str(finfo.Abspath),
    #                                  num_stimts='1', nfirst='0', polort='-1', stim_file=['1', "'{}'".format(paradigm)],
    #                                  stim_label=['1', "'STIM'"], num_glt='1', glt_label=['1', "'STIM'"],
    #                                  gltsym='SYM: +STIM')
    #     return {'func': step01}

    def final_step(self, title):
        path = os.path.join(self._prjobj.path, self._prjobj.ds_type[2],
                            self.processing, title)
        methods.mkdir(os.path.join(self._prjobj.path, self._prjobj.ds_type[2],
                                   self.processing), path)
        self._prjobj.scan_prj()
        return path
