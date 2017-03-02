import os
import re
import sys
import json
import copy as ccopy
import pickle
import pandas as pd
import itertools
import messages
import methods
import tools
import multiprocessing
from multiprocessing.pool import ThreadPool
from shutil import rmtree, copy
from collections import namedtuple
from .objects import Reference, ImageObj
from .processors import Analysis, Interface, TempFile
from .methods import np
from .visualizers import Viewer
try:
    if len([key for key in sys.modules.keys() if key == 'ipykernel']):
        from tqdm import tqdm_notebook as progressbar
        from ipywidgets import widgets
        from ipywidgets.widgets import HTML as title
        jupyter_env = True
        from IPython.display import display, display_html
    else:
        from tqdm import tqdm as progressbar
        jupyter_env = False
except:
    pass
# Hidden inside the string
import pipelines
from time import sleep
from StringIO import StringIO
from tempfile import mkdtemp


class Project(object):
    """Project handler
    """

    def __init__(self, project_path, ds_ref='NIRAL', img_format='NifTi-1', **kwargs):
        """Load and initiate the project

        :param project_path: str, Path of particular project
        :param ds_ref: str, Reference of data structure (default: 'NIRAL')
        :param img_format: str, Reference img format (default: 'NifTi-1')
        :param kwargs: dict, key arguments for options
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
        self.__ref = [ds_ref, img_format]       #TODO: Check the develope note for future usage of this part
        self.ref = Reference(*self.__ref)
        self.img_ext = self.ref.imgext
        self.ds_type = self.ref.ref_ds

        # Define default filter values
        self.__dc_idx = 0                       # Dataclass index
        self.__ext_filter = self.img_ext        # File extension

        # Generate folders for dataclasses
        methods.mk_main_folder(self)

        # Scan project folder

        try:
            self.scan_prj()
            self.apply()
        except:
            methods.raiseerror(messages.Errors.ProjectScanFailure, 'Error is occurred during a scanning.')

    @property
    def df(self):
        """Dataframe for handling data structure

        :return: pandas.DataFrame
        """
        columns = self.__df.columns
        return self.__df.reset_index()[columns]

    @property
    def path(self):
        """Project path

        :return: str, path
        """
        return self.__path

    @property
    def dataclass(self):
        """Dataclass index

        :return: int, index
        """
        return self.ds_type[self.__dc_idx]

    @dataclass.setter
    def dataclass(self, idx):
        """Setter method for dataclass

        :param idx: int, index of dataclass
        :return: None
        """
        if idx in range(3):
            self.__dc_idx = idx
            self.reset()
            self.apply()
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
        self.reset()
        self.apply()

    @property
    def ref_exts(self, type='all'):
        """Reference extention handler

        :param type: str, Choose one of 'all', 'img' or 'txt'
        :return: list, list of extensions
        """
        img_ext = self.ref.img.values()
        txt_ext = self.ref.txt.values()
        all_ext = img_ext+txt_ext
        if type in ['all', 'img', 'txt']:
            if type == 'all':
                output = all_ext
            elif type == 'img':
                output = img_ext
            elif type == 'txt':
                output = txt_ext
            else:
                output = None
            return list(itertools.chain.from_iterable(output))
        else:
            methods.raiseerror(messages.Errors.InputTypeError,
                               "only one of the value in ['all'.'img'.'txt'] is available for type.\n")

    def reload(self):
        """Reload dataset

        :return: None
        """
        self.reset(True)
        self.apply()

    def reset(self, rescan=False, verbose=False):
        """Reset DataFrame

        :param rescan: boolean, Choose if you want to re-scan all dataset
        :param verbose: boolean
        :return: None
        """

        if rescan:
            for i in range(2):
                self.__dc_idx = i+1
                self.scan_prj()
                if self.__empty_project:
                    if verbose:
                        print("Dataclass '{}' is Empty".format(self.ds_type[self.__dc_idx]))
            self.__dc_idx = 0
            self.scan_prj()
        else:
            prj_file = os.path.join(self.__path, self.ds_type[self.__dc_idx], '.class_dataframe')
            try:
                with open(prj_file, 'r') as f:
                    self.__df = pickle.load(f)
            except:
                self.scan_prj()
        if len(self.__df):
            self.__empty_project = False

    def save_df(self, dc_idx):
        """Save Dataframe to pickle file

        :param dc_idx: idx, index in range(3)
        :return: None
        """
        dc_df = os.path.join(self.__path, self.ds_type[dc_idx], '.class_dataframe')
        with open(dc_df, 'wb') as f:
            pickle.dump(self.__df, f, protocol=pickle.HIGHEST_PROTOCOL)

    def reset_filters(self, ext=None):
        """Reset filter - Clear all filter information and extension

        :param ext: str, Filter parameter for file extension
        :return: None
        """
        self.__filters = [None] * 6
        if not ext:
            self.ext = self.img_ext
        else:
            self.ext = ext

    def scan_prj(self):
        """Reload the Dataframe based on current set data class and extension

        :return: None
        """
        # Parsing command works
        self.__df, self.single_session, empty_prj = methods.parsing(self.path, self.ds_type, self.__dc_idx)
        if not empty_prj:
            self.__df = methods.initial_filter(self.__df, self.ds_type, self.ref_exts)
            if len(self.__df):
                self.__df = self.__df[methods.reorder_columns(self.__dc_idx, self.single_session)]
            self.__empty_project = False
            self.__update()
        else:
            self.__empty_project = True
        self.save_df(self.__dc_idx)

    def set_filters(self, *args, **kwargs):
        """Set filters

        :param args: str[, ], String arguments regarding hierarchical data structures
        :param kwargs: key=value pair[, ], Key and value pairs for the filtering parameter on filename
            :subparam file_tag: str or list of str, Keywords of interest for filename
            :subparam ignore: str or list of str, Keywords of neglect for filename
        :return: None
        """
        self.reset_filters(self.ext)
        pipe_filter = None
        if kwargs:
            for key in kwargs.keys():
                if key == 'dataclass':
                    self.dataclass = kwargs['dataclass']
                elif key == 'ext':
                    self.ext = kwargs['ext']
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
            pass
        if args:
            residuals = list(set(args))
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
                if self.dataclass == self.ds_type[1]:
                    if len(pipe_filter) == 1:
                        dc_path = os.path.join(self.path, self.dataclass, pipe_filter[0])
                        processed = os.listdir(dc_path)
                        if len([step for step in processed if step in residuals]):
                            methods.raiseerror(messages.Errors.NoFilteredOutput,
                                               'Cannot find any results from [{residuals}]\n'
                                               '\t\t\tPlease take a look if you had applied correct input'
                                               ''.format(residuals=residuals))
                    else:
                        methods.raiseerror(messages.Errors.NoFilteredOutput,
                                           'Uncertain exception occured, please report to Author (shlee@unc.edu)')
                else:
                    methods.raiseerror(messages.Errors.NoFilteredOutput,
                                       'Wrong filter input:{residuals}'.format(residuals=residuals))

    def apply(self):
        """Applying all filters to current dataframe

        :return: None
        """
        self.__df = self.applying_filters(self.__df)
        self.__update()

    def applying_filters(self, df):
        """Applying current filters to the given dataframe

        :param df: pandas.DataFrame
        :return: pandas.DataFrame
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
            if self.ext:
                df = df[df['Filename'].str.contains('|'.join([r"{ext}$".format(ext=ext) for ext in self.ext]))]
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
        self.dataclass = dc_id
        self.reset()
        prj = ccopy.copy(self)
        prj.set_filters(*args, **kwargs)
        prj.apply()
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
    """Collections of step components for pipelines
    """
    def __init__(self, prjobj, name, parallel=True, logging=True):
        """

        :param prjobj:
        :param name:
        :param parallel:
        :param logging:
        """

        # Prepare inputs
        prjobj.reset_filters()
        self._prjobj = prjobj(1, name)
        self._processing = name
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

        # Initiate
        self.init_proc()

    def check_input(self, input_path):
        """Check input_path and return absolute path

        :param input_path: str, name of the Processing step folder
        :return: str, Absolute path of the step
        """
        if input_path in self.executed:
            return self._history[input_path]
        else:
            return input_path

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
            options += " -tr {0}".format(tr)
        if tpattern:
            options += " -tpattern {0}".format(tpattern)
        else:
            options += " -tpattern altplus"
        input_str = " {func}"
        cmd = cmd+options+input_str
        step.set_command(cmd)
        output_path = step.run('SliceTmCorrect', surfix)
        return dict(func=output_path)

    def afni_MotionCorrection(self, func, surfix='func'):
        """

        :param func:
        :param surfix:
        :return:
        """
        display(title(value='** Processing motion correction.....'))
        func = self.check_input(func)
        step = Step(self)

        step.set_input(name='func', input_path=func, static=False)
        try:
            mimg_path = self.steps[0]
            if '-CBV-' in mimg_path:
                mimg_filters = {'file_tag': '_CBV'}
                step.set_input(name='base', input_path=mimg_path, filters=mimg_filters, static=True, side=True)
            else:
                step.set_input(name='base', input_path=mimg_path, static=True, side=True)
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

    def afni_MaskPrep(self, anat, tmpobj):
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
            step.set_staticinput(name='mask', value=tmpobj.mask.get_filename())
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
            step.set_staticinput(name='mask', value=tmpobj.mask.get_filename())
        except:
            methods.raiseerror(messages.InputPathError,
                               'No mask template file!')
        cmd01 = "3dAllineate -prefix {temp1} -NN -onepass -EPI -base {func} -cmass+xy {mask}"
        cmd02 = '3dcalc -prefix {output} -expr "astep(a, 0.5)" -a {temp1}'
        step.set_command(cmd01, idx=0)
        step.set_command(cmd02)
        func_mask = step.run('MaskPrep', 'func')
        if jupyter_env:
            display(widgets.VBox([title(value='-'*43 + ' Anatomical images ' + '-'*43),
                                  tools.itksnap(self, anat_mask, anat),
                                  title(value='<br>' + '-'*43 + ' Functional images ' + '-'*43),
                                  tools.itksnap(self, func_mask, mimg_path)]))
        else:
            return dict(anat_mask=anat_mask, func_mask=func_mask)

    def afni_SkullStrip(self, anat, func):
        """

        :param anat:
        :param func:
        :return:
        """
        display(title(value='** Processing skull stripping.....'))
        anat = self.check_input(anat)
        func = self.check_input(func)
        anat_mask = [self.steps[idx] for idx, step in self.executed.items() if 'MaskPrep-anat' in step][0]
        anat_mask = self.check_input(anat_mask)
        func_mask = [self.steps[idx] for idx, step in self.executed.items() if 'MaskPrep-func' in step][0]
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
        func_path = step.run('SkullStrip', 'func')
        return dict(anat=anat_path, func=func_path)

    def afni_Coreg(self, anat, meanfunc, surfix='func'):
        """

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
        """

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
        """

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
        """

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
        """

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

    def afni_SpatialSmoothing(self, func, fwhm=0.5, tmpobj=None, surfix='func'):
        """

        :param func:
        :param fwhm:
        :param tmpobj:
        :param surfix:
        :return:
        """
        display(title(value='** Processing spatial smoothing.....'))
        func = self.check_input(func)
        step = Step(self)
        step.set_input(name='func', input_path=func)
        if not fwhm:
            methods.raiseerror(messages.Errors.InputValueError, 'the FWHM value have to specified')
        else:
            step.set_staticinput('fwhm', fwhm)
        cmd = '3dBlurInMask -prefix {output} -FWHM {fwhm}'
        if tmpobj:
            step.set_staticinput('mask', value=tmpobj.mask.get_filename())
            cmd += ' -mask {mask}'
        cmd += ' -quiet {func}'
        step.set_command(cmd)
        output_path = step.run('SpatialSmoothing', surfix)
        return dict(func=output_path)

    def afni_GLManalysis(self, func, paradigm, surfix='func'):
        """

        :param func:
        :param paradigm:
        :param surgix:
        :return:
        """
        display(title(value='** Processing General Linear Analysis'))
        func = self.check_input(func)
        step = Step(self)
        step.set_input(name='func', input_path=func)
        step.set_variable(name='paradigm', value=paradigm)
        step.set_staticinput(name='param', value='" ".join(map(str, paradigm[idx][0]))')
        step.set_staticinput(name='model', value='paradigm[idx][1][0]')
        step.set_staticinput(name='mparam', value='" ".join(map(str, paradigm[idx][1][1]))')
        cmd01 = '3dDeconvolve -input {func} -num_stimts 1 -polort 2 -stim_times 1 "1D: {param}" ' \
                '"{model}({mparam})" -stim_label 1 STIM -tout -bucket {output}'
        step.set_command(cmd01)
        glm = step.run('GLMAnalysis', surfix)
        display(title(value='** Estimating the temporal auto-correlation structure'))
        step = Step(self)
        step.set_input(name='func', input_path=func)
        filter = dict(ext='.xmat.1D')
        step.set_input(name='glm', input_path=glm, filters=filter, side=True)
        cmd02 = '3dREMLfit -matrix {glm} -input {func} -tout -Rbuck {output} -verb'
        step.set_command(cmd02)
        output_path = step.run('REMLfit', surfix)
        return dict(GLM=output_path)

    def afni_ClusterMap(self, glm, tmpobj, pval=0.01, cluster_size=40, surfix='func'):
        """"""
        display(title(value='** Generating clustered masks'))
        glm = self.check_input(glm)
        step = Step(self)
        step.set_input(name='glm', input_path=glm)
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
        # step.get_executefunc('test', verbose=True)
        output_path = step.run('ClusteredMask', surfix=surfix)
        if jupyter_env:
            display(tools.itksnap(self, output_path, tmpobj.image.get_filename()))
        else:
            return dict(mask=output_path)


    def afni_SignalProcessing(self, func, mparam, mask=None, fwhm=None, dt=None, surfix='func'):
        """

        :param func:
        :param mparam:
        :param mask:
        :param fwhm:
        :param dt:
        :return:
        """
        cmd = 'afni_3dTproject -prefix {output} -ort {regr} -norm -mask {mask} -blur {fwhm} -dt {dt} {func}'

    def afni_ROIStats(self, func, rois, cbv=None, surfix='func'):
        """

        :param func:
        :param roi:
        :param cbv: list, [echotime, number of TR for averaging]
        :param surfix:
        :return:
        """
        display(title(value='** Extracting time-course data from ROIs'))
        func = self.check_input(func)
        rois = self.check_input(rois)
        step = Step(self)
        step.set_input(name='func', input_path=func)
        if os.path.isfile(rois):
            step.set_staticinput(name='rois', value=rois)
        else:
            step.set_input(name='rois', input_path=rois, side=True)
        if cbv:
            step.set_input(name='cbv', input_path=func, side=True, filters=dict(ext='.json'))
        cmd = '3dROIstats -mask {rois} {func}'
        step.set_command(cmd, stdout='out')
        step.set_execmethod('temp_outputs.append([None, err])')
        step.set_execmethod('pd.read_table(StringIO(out))', var='df')
        step.set_execmethod('df[df.columns[2:]]', var='df')
        if cbv:
            if isinstance(cbv, list) and len(cbv) == 2:
                step.set_variable(name='te', value=cbv[0])
                step.set_variable(name='n_tr', value=cbv[1])
                step.set_execmethod('cbv_path = json.load(open(cbv[i].Abspath))["cbv"]')
                step.set_staticinput(name='cbv_path', value='cbv_path')
                cbv_cmd = '3dROIstats -mask {rois} {cbv_path}'
                step.set_command(cbv_cmd, stdout='cbv_out')
                step.set_execmethod('temp_outputs.append([None, err])')
                step.set_execmethod('pd.read_table(StringIO(cbv_out))', var='cbv_df')
                step.set_execmethod('cbv_df[cbv_df.columns[:]]', var='cbv_df')
            else:
                methods.raiseerror(messages.Errors.InputValueError, 'Please check input CBV parameters')
        step.set_execmethod('if len(df.columns):')
        if cbv:
            step.set_execmethod('\tdR2_mion = (-1 / te) * np.log(df.loc[:n_tr, :].mean(axis=0) / '
                                'cbv_df.loc[:n_tr, :].mean(axis=0))')
            step.set_execmethod('\tdR2_stim = (-1 / te) * np.log(df / df.loc[:n_tr, :].mean(axis=0))')
            step.set_execmethod('\tdf = dR2_stim/dR2_mion')
        step.set_execmethod('\tdf.to_excel(os.path.join(sub_path, methods.splitnifti(func[i].Filename)+".xlsx"), '
                            'index=False)')
        step.set_execmethod('\tpass')
        step.set_execmethod('else:')
        step.set_execmethod('\tpass')
        # step.get_executefunc('test', verbose=True)
        output_path = step.run('ExtractROIs', surfix=surfix)
        return dict(timecourse=output_path)

    def itksnap(self, idx, base_idx=None):
        """Launch ITK-snap

        :param idx:
        :param base_idx:
        :return:
        """
        if base_idx:
            tools.itksnap(self, self.steps[idx], self.steps[base_idx])
        else:
            tools.itksnap(self, self.steps[idx])

    def afni(self, idx, tmpobj=None):
        """Launch AFNI gui

        :param idx:
        :param tmpobj:
        :return:
        """
        tools.afni(self, self.steps[idx], tmpobj=tmpobj)


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
        if self._prjobj.subjects:
            if self._subjects:
                if self._subjects != self._prjobj.subjects:
                    self._subjects = sorted(self._prjobj(1, self.processing).subjects[:])
                else:
                    self._subjects = sorted(self._prjobj.subjects[:])
            else:
                self._subjects = sorted(self._prjobj.subjects[:])
            if not self._prjobj.single_session:
                self._sessions = sorted(self._prjobj.sessions[:])

        self.logger.info('Attributes [subjects, sessions] are reset to default value.')

    def init_proc(self):
        """Initiate process folder

        :return: None
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
        """Initiate step

        :param name: str
        :return: str
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

    def __init__(self, procobj):
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
        self._subjects = procobj.subjects[:]            # load all subject list from process object
        self._outputs = {}                              # output handler
        try:
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
        dc, ipath = methods.check_dataclass(input_path)
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
        :return:
        """
        if isinstance(value, str):
            if os.path.exists(value):
                value = '"{}"'.format(value)
            else:
                value = '{}'.format(value)
        self._staticinput[name] = value

    def set_outparam(self, name, ext, prefix=None):
        """Set parameter output files on out_param handler

        :param name: str, variables for parameter output file
        :param ext: str, for setting extension of the file
        :param prefix:
        :return: None
        """
        self._outparam[name] = (self.oppset(name=name, prefix=prefix, ext=ext))

    def set_execmethod(self, command, var=None, idx=None):
        """Set structured command on command handler

        :param command: str, structured command with input and output variables
        :param var: name of variable
        :param idx: int, Index for replacing the commands on handler

        :return:
        """
        if idx:
            self._commands[idx] = (command, [var])
        else:
            self._commands.append((command, [var]))

    def set_command(self, command, verbose=False, idx=None, stdout=None ):
        """Set structured command on command handler

        :param command: str, Structured command with input and output variables
        :param verbose: boolean
        :param idx: int, Index for replacing the commands on handler
        :param stdout: str or None, if True, the input string can be used the variable for stand output results of a command
        :return:
        """
        objs = [obj.strip('{}') for obj in re.findall(r"[{\w'}]+", command) if obj[0] == '{' and obj[-1] == '}']
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
        residuals = [obj for obj in residuals if obj not in self._staticinput.keys()]
        residuals = [obj for obj in residuals if obj not in self._outparam.keys()]
        residuals = [obj for obj in residuals if obj not in self._cmdstdout]

        # Check accuracy
        if len(residuals):
            methods.raiseerror(ValueError, 'Too many inputs :{0}'.format(str(residuals)))
        output = "'{0}'.format(".format(command)
        str_format = []
        for obj in objs:
            if obj == 'output':
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
            # print(input_path)
            output_filters = [dataclass, '"{0}"'.format(input_path)]
        else:
            output_filters = [dataclass, '"{0}"'.format(self._processing), '"{0}"'.format(input_path)]
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
                    '\tprefix = methods.splitnifti(os.path.basename(output))',
                    '\tflist = [f for f in os.listdir(sub_path)]',
                    '\tif len([f for f in flist if prefix in f]):',
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
                    '\t\tprefix = methods.splitnifti(os.path.basename(output))',
                    '\t\tflist = [f for f in os.listdir(sub_path)]',
                    '\t\tif len([f for f in flist if prefix in f]):',
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
            print(output)
            return None
        else:
            return output

    def run(self, step_name, surfix):
        """Generate loop commands for step

        :param step_name: str
        :param surfix: str
        :return: None
        """
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


class Pipelines(object):
    """ Pipeline handler

    This class is the major features of PyNIT project (for most of general users)
    You can either use default pipeline packages we provide or load custom designed pipelines
    """
    def __init__(self, prj_path, tmpobj, parallel=True, logging=True):
        """Initiate class

        :param prj_path:
        :param tmpobj:
        :param parallel:
        :param logging:
        """
        # Define default attributes
        self._prjobj = Project(prj_path)
        self._avail = self._prjobj.ref.avail
        self._proc = None
        self._tmpobj = tmpobj
        self._parallel = parallel
        self._logging = logging
        self.selected = None

        # Print out project summary
        self._prjobj.summary

        # Print out available pipeline packages
        avails = ["\t{} : {}".format(*item) for item in self._avail.items()]
        output = ["\nList of available packages:"] + avails
        print("\n".join(output))

    @property
    def avail(self):
        return self._avail

    def initiate(self, pipeline, verbose=False, **kwargs):
        """Initiate pipeline

        :param pipeline:
        :param verbose:
        :param kwargs:
        :return:
        """
        if isinstance(pipeline, int):
            pipeline = self.avail[pipeline]
        if pipeline in self.avail.values():
            self._proc = Process(self._prjobj, pipeline, parallel=self._parallel, logging=self._logging)
            command = 'self.selected = pipelines.{}(self._proc, self._tmpobj'.format(pipeline)
            if kwargs:
                command += ', **{})'.format(kwargs)
            else:
                command += ')'
            exec(command)
        else:
            methods.raiseerror(messages.PipelineNotSet, "Incorrect package is selected")
        if verbose:
            print(self.selected.__init__.__doc__)
        avails = ["\t{} : {}".format(*item) for item in self.selected.avail.items()]
        output = ["List of available pipelines:"] + avails
        print("\n".join(output))

    def help(self, pipeline):
        """ Print help function

        :param pipeline:
        :return:
        """
        selected = None
        if isinstance(pipeline, int):
            pipeline = self.avail[pipeline]
        if pipeline in self.avail.values():
            command = 'selected = pipelines.{}(self._proc, self._tmpobj)'.format(pipeline)
            exec(command)
            print(selected.__init__.__doc__)
            avails = ["\t{} : {}".format(*item) for item in selected.avail.items()]
            output = ["List of available pipelines:"] + avails
            print("\n".join(output))

    def run(self, idx):
        """Execute selected pipeline

        :param idx:
        :return:
        """
        display(title(value='---=[[[ Running "{}" pipeline ]]]=---'.format(self.selected.avail[idx])))
        exec('self.selected.pipe_{}()'.format(self.selected.avail[idx]))

    def load(self, pipeline):
        """Load custom pipeline

        :param pipeline:
        :return:
        """
        pass

    def group_organizer(self, group_filters, i_pipe_id, i_step_id, o_pipe_id, cbv=None, **kwargs):
        """Organizing groups for 2nd level analysis

        :param group_filters:
        :param i_pipe_id:
        :param i_step_id:
        :param o_pipe_id:
        :param cbv:
        :param kwargs:
        :return:
        """
        self.initiate(o_pipe_id, **kwargs)
        input_proc = Process(self._prjobj, self.avail[i_pipe_id])
        init_path = self._proc.init_step('GroupOrganizing')
        groups = sorted(group_filters.keys())
        for group in progressbar(sorted(groups), desc='Subjects'):
            grp_path = os.path.join(init_path, group)
            methods.mkdir(grp_path)
            if self._prjobj.single_session:
                if group_filters[group][2]:
                    dset = self._prjobj(1, input_proc.processing, input_proc.executed[i_step_id],
                                        *group_filters[group][0], **group_filters[group][2])
                else:
                    dset = self._prjobj(1, input_proc.processing, input_proc.executed[i_step_id],
                                        *group_filters[group][0])

            else:
                grp_path = os.path.join(init_path, group, 'files')
                methods.mkdir(grp_path)
                if group_filters[group][2]:
                    dset = self._prjobj(1, input_proc.processing, input_proc.executed[i_step_id],
                                        *(group_filters[group][0] + group_filters[group][1]),
                                        **group_filters[group][2])
                else:
                    dset = self._prjobj(1, input_proc.processing, input_proc.executed[i_step_id],
                                        *(group_filters[group][0] + group_filters[group][1]))
            for i, finfo in dset:
                output_path = os.path.join(grp_path, finfo.Filename)
                if os.path.exists(output_path):
                    pass
                else:
                    if self._prjobj.single_session:
                        cbv_file = self._prjobj(1, input_proc.processing, input_proc.executed[cbv], finfo.Subject)
                    else:
                        cbv_file = self._prjobj(1, input_proc.processing, input_proc.executed[cbv],
                                                finfo.Subject, finfo.Session)
                    copy(finfo.Abspath, os.path.join(grp_path, finfo.Filename))
                    with open(methods.splitnifti(output_path)+'.json', 'wb') as f:
                        json.dump(dict(cbv=cbv_file[0].Abspath), f)
        self._proc._subjects = groups[:]
        self._proc._history[os.path.basename(init_path)] = init_path
        self._proc.save_history()
        self._proc._prjobj.reload()

    @property
    def executed(self):
        """Listing out executed steps

        :return:
        """
        return self._proc.executed


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
        methods.mkdir(pipe_path)
        self._processing = pipeline

    def init_step(self, stepname):
        if self._processing:
            steppath = methods.get_step_name(self, stepname)
            steppath = os.path.join(self._prjobj.path, self._prjobj.ds_type[1], self._processing, steppath)
            methods.mkdir(steppath)
            return steppath
        else:
            raise messages.PipelineNotSet

    def seedbased_dynamic_connectivity(self, func, seed, winsize=100, step=1, dtype='func', **kwargs):
        """ Seed-based dynamic connectivity using sliding windows # TODO: use this methods as model

        Parameters
        ----------
        func : str
            Root path for input dataset
        seed : str
            Absolute path for seed image
        winsize : int
            Size of sliding window
        step : int
            Moving step for sliding window
        dtype : str
            Surfix for output folder
        kwargs : dict
            Options (Nothing available)

        Returns
        -------
        path : dict
            output path
        """
        dataclass, func = methods.check_dataclass(func)
        print('SeedBaseDynamicConnectivityMap-{}'.format(func))
        step01 = self.init_step('SeedBaseDynamicConnectivityMap-{}'.format(dtype))
        if not os.path.isfile(seed):
            methods.raiseerror(ValueError, 'Input file does not exist.')

        def processor(result, temppath, finfo, seed_path, winsize, i):
            temp_path = os.path.join(temppath, "{}.nii".format(str(i).zfill(5)))
            self._prjobj.run('afni_3dTcorr1D', temp_path,
                             "{0}'[{1}..{2}]'".format(finfo.Abspath, i, i + winsize - 1),
                             "{0}'{{{1}..{2}}}'".format(seed_path, i, i + winsize - 1))
            result.append(temp_path)

        def worker(args):
            """

            Parameters
            ----------
            func
            args

            Returns
            -------

            """
            return processor(*args)

        def start_process():
            print 'Starting', multiprocessing.current_process().name

        for subj in progressbar(self.subjects, desc='Subjects', leave=False):
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                epi = self._prjobj(dataclass, self._processing, func, subj, **kwargs)
                for i, finfo in progressbar(epi, desc='Files', leave=False):
                    print(" +Filename: {}".format(finfo.Filename))
                    # Check dimension
                    total, err = methods.shell('3dinfo -nv {}'.format(finfo.Abspath))
                    if err:
                        methods.raiseerror(NameError, 'Cannot load: {0}'.format(finfo.Filename))
                    total = int(total)
                    print(total)
                    if total <= winsize:
                        methods.raiseerror(KeyError,
                                           'Please use proper windows size: [less than {}]'.format(str(total)))
                    seed_path = os.path.join(step01, subj, "{0}.1D".format(methods.splitnifti(finfo.Filename)))
                    self._prjobj.run('afni_3dmaskave', seed_path, finfo.Abspath, seed)
                    output_path = os.path.join(step01, subj, finfo.Filename)
                    if not os.path.exists(output_path):
                        methods.mkdir('.dtmp')
                        temppath = os.path.join('.dtmp')
                        list_of_files = []
                        cpu = multiprocessing.cpu_count()
                        cpu = cpu - int(cpu / 4)
                        pool = ThreadPool(cpu, initializer=start_process)
                        iteritem = [(list_of_files, temppath, finfo, seed_path, winsize, i) for i in range(0, total - winsize, step)]
                        for output in progressbar(pool.imap_unordered(worker, iteritem), desc='Window',
                                                  total=len(iteritem) ,leave=False):
                            pass
                        list_of_files = sorted(list_of_files)
                        methods.shell('3dTcat -prefix {0} -tr {1} {2}'.format(output_path, str(step),
                                                                              ' '.join(list_of_files)))
                        rmtree(temppath)
                        pool.close()
                        pool.join()
            else:
                for sess in progressbar(self.sessions, desc='Sessions', leave=False):
                    methods.mkdir(os.path.join(step01, subj, sess))
                    epi = self._prjobj(dataclass, self._processing, func, subj, sess, **kwargs)
                    for i, finfo in progressbar(epi, desc='Files', leave=False):
                        print(" +Filename: {}".format(finfo.Filename))
                        # Check dimension
                        total, err = methods.shell('3dinfo -nv {0}'.format(finfo.Abspath))
                        if err:
                            methods.raiseerror(NameError, 'Cannot load: {0}'.format(finfo.Filename))
                        total = int(total)
                        if total <= winsize:
                            methods.raiseerror(KeyError,
                                               'Please use proper windows size: [less than {}]'.format(str(total)))
                        seed_path = os.path.join(step01, subj, sess,
                                                 "{0}.1D".format(methods.splitnifti(finfo.Filename)))
                        self._prjobj.run('afni_3dmaskave', seed_path, finfo.Abspath, seed)
                        output_path = os.path.join(step01, subj, sess, finfo.Filename)
                        if not os.path.exists(output_path):
                            methods.mkdir('.dtmp')
                            temppath = os.path.join('.dtmp')
                            list_of_files = []
                            cpu = multiprocessing.cpu_count()
                            cpu = cpu - int(cpu/4)
                            pool = ThreadPool(cpu, initializer=start_process)
                            iteritem = [(list_of_files, temppath, finfo, seed_path, winsize, i) for i in
                                        range(0, total - winsize, step)]
                            for output in progressbar(pool.imap_unordered(worker, iteritem), desc='Window',
                                                      total=len(iteritem), leave=False):
                                pass
                            list_of_files = sorted(list_of_files)
                            methods.shell('3dTcat -prefix {0} -tr {1} {2}'.format(output_path, str(step),
                                                                                  ' '.join(list_of_files)))
                            rmtree(temppath)
                            pool.close()
                            pool.join()
        return {'dynamicMap': step01}

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
        self._prjobj.reset(True)
        self._prjobj.apply()
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
        self._prjobj.reset(True)
        self._prjobj.apply()
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
                                                                       # end=20))
                                                                       end=(int(shape[-1] / 3))))
                    self._prjobj.run('afni_3dTstat', os.path.join(step03, subj, finfo.Filename),
                                     "{path}'[{start}..{end}]'".format(path=finfo.Abspath,
                                                                       # start=int(shape[-1]-21),
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
                                                                           # end=20))
                                                                           end=(int(shape[-1] / 3))))
                        self._prjobj.run('afni_3dTstat', os.path.join(step03, subj, sess, finfo.Filename),
                                         "{path}'[{start}..{end}]'".format(path=finfo.Abspath,
                                                                           start=int(shape[-1] * 2 / 3),
                                                                           # start=int(shape[-1]-21),
                                                                           end=shape[-1] - 1))
        self._prjobj.reset(True)
        self._prjobj.apply()
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
        self._prjobj.reset(True)
        self._prjobj.apply()
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
        self._prjobj.reset(True)
        self._prjobj.apply()
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
        self._prjobj.reset(True)
        self._prjobj.apply()
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
        self._prjobj.reset(True)
        self._prjobj.apply()
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
            self._prjobj.reset(True)
            self._prjobj.apply()
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
            self._prjobj.reset(True)
            self._prjobj.apply()
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
            self._prjobj.reset(True)
            self._prjobj.apply()
            return {'func': step04, 'mparam': step01}
        else:
            self._prjobj.reset(True)
            self._prjobj.apply()
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
        self._prjobj.reset(True)
        self._prjobj.apply()
        return {'meanfunc': step01, 'anat': step02}

    def compute_skullstripping(self, meanfunc, anat, padded=False, zaxis=2):
        self._prjobj.reset(True)
        self._prjobj.apply()
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
                    fpath = os.path.join(step01, subj, filename)
                    self._prjobj.run('afni_3dcalc', fpath, 'a*step(b)',
                                     finfo.Abspath, epimask)
                    ss_epi = methods.load(fpath)
                    if padded:
                        exec('ss_epi.crop({}=[1, {}])'.format(axis[zaxis], ss_epi.shape[zaxis]-1))
                        ss_epi.save_as(os.path.join(step01, subj, filename), quiet=True)
                for i, finfo in t2:
                    print(" +Filename of anat: {}".format(finfo.Filename))
                    filename = finfo.Filename
                    fpath = os.path.join(step02, subj, filename)
                    self._prjobj.run('afni_3dcalc', fpath, 'a*step(b)',
                                     finfo.Abspath, t2mask)
                    ss_t2 = methods.load(fpath)
                    if padded:
                        exec('ss_t2.crop({}=[1, {}])'.format(axis[zaxis], ss_t2.shape[zaxis] - 1))
                        ss_t2.save_as(os.path.join(step02, subj, filename), quiet=True)
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
                        tpath = os.path.join(step01, subj, sess)
                        fpath = os.path.join(tpath, finfo.Filename)
                        self._prjobj.run('afni_3dcalc', fpath, 'a*step(b)',
                                         finfo.Abspath, epimask)
                        ss_epi = methods.load(fpath)
                        if padded:
                            exec('ss_epi.crop({}=[1, {}])'.format(axis[zaxis], ss_epi.shape[zaxis] - 1))
                            ss_epi.save_as(os.path.join(step01, subj, sess, filename), quiet=True)
                    for i, finfo in t2:
                        print("  +Filename of anat: {}".format(finfo.Filename))
                        filename = finfo.Filename
                        tpath = os.path.join(step02, subj, sess)
                        fpath = os.path.join(tpath, finfo.Filename)
                        self._prjobj.run('afni_3dcalc', fpath, 'a*step(b)',
                                         finfo.Abspath, t2mask)
                        ss_t2 = methods.load(fpath)
                        if padded:
                            exec('ss_t2.crop({}=[1, {}])'.format(axis[zaxis], ss_t2.shape[zaxis] - 1))
                            ss_t2.save_as(os.path.join(step02, subj, sess, filename), quiet=True)
        self._prjobj.reset(True)
        self._prjobj.apply()
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
        self._prjobj.reset(True)
        self._prjobj.apply()
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
        self._prjobj.reset(True)
        self._prjobj.apply()
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
        self._prjobj.reset(True)
        self._prjobj.apply()
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
        self._prjobj.reset(True)
        self._prjobj.apply()
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
        self._prjobj.reset(True)
        self._prjobj.apply()
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
                    self._prjobj.run('afni_3dcalc', os.path.join(step01, subj, finfo.Filename), 'log(b/a)/log(c/b)',
                                     finfo.Abspath, str(spre), szero.Abspath)
                    # self._prjobj.run('afni_3dcalc', os.path.join(step01, subj, finfo.Filename), '(b-a)/(c-b)*100',
                    #                  finfo.Abspath, str(spre), szero.Abspath)
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
                        self._prjobj.run('afni_3dcalc', os.path.join(step01, subj, sess, finfo.Filename), 'log(b/a)/log(c/b)',
                                         finfo.Abspath, str(spre), szero.Abspath)
                        # self._prjobj.run('afni_3dcalc', os.path.join(step01, subj, sess, finfo.Filename),
                        #                  '(b-a)/(c-b)*100', finfo.Abspath, str(spre), szero.Abspath)
                        spre.close()
        self._prjobj.reset(True)
        self._prjobj.apply()
        return {'cbv': step01}

    def check_cbv_correction(self, func, meanBOLD, meanCBV, mean_range=20, echotime=0.008, dtype='func', **kwargs):
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
        mc_dataclass, meanCBV = methods.check_dataclass(meanCBV)
        print('CBV_Calculation-{}'.format(func))
        step01 = self.init_step('deltaR2forSTIM-{}'.format(dtype))
        step02 = self.init_step('deltaR2forMION-{}'.format(dtype))
        step03 = self.init_step('deltaR2forMIONcorrected-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj), os.path.join(step02, subj), os.path.join(step03, subj))
            if self._prjobj.single_session:
                funcs = self._prjobj(dataclass, func, subj)
                szero = self._prjobj(mb_dataclass, meanBOLD, subj).df.loc[0]
                smion = self._prjobj(mb_dataclass, meanCBV, subj).df.loc[0]
                for i, finfo in funcs:
                    imgobj = methods.load(finfo.Abspath)
                    imgobj._dataobj = np.mean(imgobj._dataobj[:, :, :, :mean_range], axis=3)
                    spre = TempFile(imgobj, 'spre_{}'.format(subj))
                    print(" +Filename: {}".format(finfo.Filename))
                    self._prjobj.run('afni_3dcalc', os.path.join(step01, subj, finfo.Filename),
                                     '(-1/{TE})*log(a/b)'.format(TE=echotime),
                                     finfo.Abspath, str(spre))
                    self._prjobj.run('afni_3dcalc', os.path.join(step02, subj, finfo.Filename),
                                     '(-1/{TE})*log(a/b)'.format(TE=echotime),
                                     smion.Abspath, szero.Abspath)
                    self._prjobj.run('afni_3dcalc', os.path.join(step03, subj, finfo.Filename),
                                     '(-1/{TE})*log(a/b)'.format(TE=echotime),
                                     str(spre), szero.Abspath)
                    # self._prjobj.run('afni_3dcalc', os.path.join(step01, subj, finfo.Filename), '(b-a)/(c-b)*100',
                    #                  finfo.Abspath, str(spre), szero.Abspath)
                    spre.close()
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess), os.path.join(step02, subj, sess),
                                  os.path.join(step03, subj, sess))
                    funcs = self._prjobj(dataclass, func, subj, sess)
                    szero = self._prjobj(mb_dataclass, meanBOLD, subj, sess).df.loc[0]
                    smion = self._prjobj(mb_dataclass, meanCBV, subj, sess).df.loc[0]
                    for i, finfo in funcs:
                        imgobj = methods.load(finfo.Abspath)
                        imgobj._dataobj = np.mean(imgobj._dataobj[:, :, :, :mean_range], axis=3)
                        spre = TempFile(imgobj, 'spre_{}_{}'.format(subj, sess))
                        print(" +Filename: {}".format(finfo.Filename))
                        self._prjobj.run('afni_3dcalc', os.path.join(step01, subj, sess, finfo.Filename),
                                         '(-1/{TE})*log(a/b)'.format(TE=echotime),
                                         finfo.Abspath, str(spre))
                        self._prjobj.run('afni_3dcalc', os.path.join(step02, subj, sess, finfo.Filename),
                                         '(-1/{TE})*log(a/b)'.format(TE=echotime),
                                         smion.Abspath, szero.Abspath)
                        self._prjobj.run('afni_3dcalc', os.path.join(step02, subj, sess, finfo.Filename),
                                         '(-1/{TE})*log(a/b)'.format(TE=echotime),
                                         str(spre), szero.Abspath)
                        # self._prjobj.run('afni_3dcalc', os.path.join(step01, subj, sess, finfo.Filename),
                        #                  '(b-a)/(c-b)*100', finfo.Abspath, str(spre), szero.Abspath)
                        spre.close()
        self._prjobj.reset(True)
        self._prjobj.apply()
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
        self._prjobj.reset(True)
        self._prjobj.apply()
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
        self._prjobj.reset(True)
        self._prjobj.apply()
        return {'func': step01}

    def signal_processing2(self, func, dt=1, mask=None, ort=None, orange=None, norm=False, blur=False, band=False,
                           dtype='func'):
        """ New method for signal processing and spatial smoothing of individual functional image

        Parameters
        ----------
        func
        dt
        mask
        ort
        norm
        blur
        band
        dtype
        file_tag
        ignore

        Returns
        -------

        """
        dataclass, func = methods.check_dataclass(func)
        if ort:
            mdataclass, ort = methods.check_dataclass(ort)
        print('SignalProcessing-{}'.format(func))
        step01 = self.init_step('SignalProcessing-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                funcs = self._prjobj(dataclass, func, subj)
                for i, finfo in funcs:
                    if ort:
                        regressor = self._prjobj(mdataclass, ort, subj, ext='.1D',
                                                 file_tag=methods.splitnifti(finfo.Filename),
                                                 ignore='.aff12').df.Abspath[0]
                    else:
                        regressor = None
                    print(" +Filename: {}".format(finfo.Filename))
                    self._prjobj.run('afni_3dTproject', os.path.join(step01, subj, finfo.Filename), finfo.Abspath,
                                     ort=regressor, mask=mask, orange=orange, norm=norm, blur=blur, band=band, dt=dt)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    funcs = self._prjobj(dataclass, func, subj, sess)
                    for i, finfo in funcs:
                        if ort:
                            regressor = self._prjobj(mdataclass, ort, subj, sess, ext='.1D',
                                                     file_tag=methods.splitnifti(finfo.Filename),
                                                     ignore='.aff12').df.Abspath[0]
                        else:
                            regressor = None
                        print("  +Filename: {}".format(finfo.Filename))
                        self._prjobj.run('afni_3dTproject', os.path.join(step01, subj, sess, finfo.Filename),
                                         finfo.Abspath, ort=regressor, orange=orange, mask=mask, norm=norm,
                                         blur=blur, band=band, dt=dt)
        self._prjobj.reset(True)
        self._prjobj.apply()
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
                # temp_path = os.path.join(step01, subj, "base")
                # tempobj.save_as(temp_path, quiet=True)
                funcs = self._prjobj(dataclass, func, subj)
                print(" +Filename of fixed image: {}".format(warped.Filename))
                for i, finfo in funcs:
                    print(" +Filename of moving image: {}".format(finfo.Filename))
                    output_path = os.path.join(step01, subj, finfo.Filename)
                    self._prjobj.run('ants_WarpTimeSeriresImageMultiTransform', output_path,
                                     finfo.Abspath, warped.Abspath, warps, mats, **in_kwargs)
                subjatlas = methods.load_temp(output_path, tempobj._atlas.path)
                fig = subjatlas.show(**kwargs)
                if type(fig) is tuple:
                    fig = fig[0]
                fig.suptitle('Check atlas registration of {}'.format(subj), fontsize=12, color='yellow')
                fig.savefig(os.path.join(step02, 'AllSubjects', '{}.png'.format('-'.join([subj, 'checkatlas']))),
                            facecolor=fig.get_facecolor())
            else:
                methods.mkdir(os.path.join(step02, subj))
                for sess in self.sessions:
                    methods.mkdir(os.path.join(step02, subj, 'AllSessions'), os.path.join(step01, subj, sess))
                    print(" :Session: {}".format(sess))
                    # Grab the warping map and transform matrix
                    mats, warps, warped = methods.get_warp_matrix(self, warped_anat, subj, sess, inverse=False)
                    # temp_path = os.path.join(step01, subj, sess, "base")
                    # tempobj.save_as(temp_path, quiet=True)
                    funcs = self._prjobj(dataclass, func, subj, sess)
                    print(" +Filename of fixed image: {}".format(warped.Filename))
                    for i, finfo in funcs:
                        print(" +Filename of moving image: {}".format(finfo.Filename))
                        output_path = os.path.join(step01, subj, sess, finfo.Filename)
                        self._prjobj.run('ants_WarpTimeSeriesImageMultiTransform', output_path,
                                         finfo.Abspath, warped.Abspath, warps, mats, **in_kwargs)
                    subjatlas = methods.load_temp(output_path, tempobj._atlas.path)
                    fig = subjatlas.show(**kwargs)
                    if type(fig) is tuple:
                        fig = fig[0]
                    fig.suptitle('Check atlas registration of {}'.format(subj), fontsize=12, color='yellow')
                    fig.savefig(os.path.join(step02, subj, 'AllSessions',
                                             '{}.png'.format('-'.join([subj, sess, 'checkatlas']))),
                                facecolor=fig.get_facecolor())
        self._prjobj.reset(True)
        self._prjobj.apply()
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
        self._prjobj.reset(True)
        self._prjobj.apply()
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
                try:
                    subjatlas = methods.load_temp(moved_img, '{}_atlas.nii'.format(temp_path))
                    fig = subjatlas.show(**kwargs)
                    if type(fig) is tuple:
                        fig = fig[0]
                    fig.suptitle('Check atlas registration of {}'.format(subj), fontsize=12, color='yellow')
                    fig.savefig(
                        os.path.join(step02, 'AllSubjects', '{}.png'.format('-'.join([subj, 'checkatlas']))),
                        facecolor=fig.get_facecolor())
                except:
                    pass
                try:
                    os.remove('{}_atlas.nii'.format(temp_path))
                    os.remove('{}_atlas.label'.format(temp_path))
                    os.remove('{}_template.nii'.format(temp_path))
                except:
                    pass
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
                    try:
                        subjatlas = methods.load_temp(moved_img, '{}_atlas.nii'.format(temp_path))
                        fig = subjatlas.show(**kwargs)
                        if type(fig) is tuple:
                            fig = fig[0]
                        fig.suptitle('Check atlas registration of {}-{}'.format(subj, sess), fontsize=12, color='yellow')
                        fig.savefig(
                            os.path.join(step02, subj, 'AllSessions',
                                         '{}.png'.format('-'.join([subj, sess, 'checkatlas']))),
                            facecolor=fig.get_facecolor())
                    except:
                        pass
                    try:
                        os.remove('{}_atlas.nii'.format(temp_path))
                        os.remove('{}_atlas.label'.format(temp_path))
                        os.remove('{}_template.nii'.format(temp_path))
                    except:
                        pass
        self._prjobj.reset(True)
        self._prjobj.apply()
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
        self._prjobj.reset(True)
        self._prjobj.apply()
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
        self._prjobj.reset(True)
        self._prjobj.apply()
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
        self._prjobj.reset(True)
        self._prjobj.apply()
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
        self._prjobj.reset(True)
        self._prjobj.apply()
        return {'timecourse': step01, 'cc_matrix': step02}

    def final_step(self, title):
        path = os.path.join(self._prjobj.path, self._prjobj.ds_type[2],
                            self.processing, title)
        methods.mkdir(os.path.join(self._prjobj.path, self._prjobj.ds_type[2],
                                   self.processing), path)
        self._prjobj.scan_prj()
        return path
