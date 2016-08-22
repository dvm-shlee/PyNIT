import os
import pandas as pd
from .objects import Reference, GroupImages, ImageObj
from .process import Analysis, Interface
from .utility import InternalMethods#, Interface
import error


class Project(object):
    """Project Handler for Neuroimage data
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
        # Variables for attributes
        max_rows = 100
        max_colwidth = 100
        if kwargs:
            if 'max_rows' in kwargs.keys():
                max_rows = kwargs['max_rows']
            if 'max_colwidth' in kwargs.keys():
                max_colwidth = kwargs['max_colwidth']
        pd.options.display.max_rows = max_rows
        pd.options.display.max_colwidth = max_colwidth
        self.single_session = False
        self.__path = project_path
        self.__filters = [None] * 6
        # Each values are represented subject, session, dtype(or pipeline), step(or results) file_tags, ignores
        self.__df = pd.DataFrame()
        # Parsing the information from the reference
        self.__ref = [ds_ref, img_format]
        self.__empty_project = False
        ref = Reference(*self.__ref)
        self.img_ext = ref.imgext
        self.ds_type = ref.ref_ds
        # Define basic variables for initiating instance
        self.__dc_idx = 0           # Data class index
        self.__ext_filter = self.img_ext
        InternalMethods.mk_main_folder(self)
        self.__pipeline = None
        self.interface = Interface()
        try:
            self.reload()
        except:
            raise error.ReloadFailure

    @property
    def df(self):
        columns = self.__df.columns
        return self.__df.reset_index()[columns]

    @property
    def path(self):
        return self.__path

    @property
    def dataclass(self):
        return self.ds_type[self.__dc_idx]

    @dataclass.setter
    def dataclass(self, idx):
        if idx in range(3):
            self.__dc_idx = idx
            self.reset()
            self.__update()
        else:
            raise error.NotExistingDataclass

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
    def pipeline(self):
        return self.__pipeline

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
    def executed_steps(self):
        if self.__pipeline:
            df = self(1, self.__pipeline)
            return sorted(list(set(df.Step)))
        else:
            raise error.PipelineNotSet

    def initiate_pipeline(self, pipeline):
        InternalMethods.mkdir(os.path.join(self.path, self.ds_type[1], pipeline))
        self.__pipeline = pipeline

    def initiate_step(self, stepname):
        if self.__pipeline:
            steppath = InternalMethods.get_step_name(self, stepname)
            steppath = os.path.join(self.path, self.ds_type[1], self.__pipeline, steppath)
            InternalMethods.mkdir(steppath)
            return steppath
        else:
            raise error.PipelineNotSet

    def get_objs(self):
        groupobj = Group()
        for i, subject in self:
            groupobj[InternalMethods.remove_nifti_ext(subject.Filename)] = ImageObj.load(subject.Abspath)

    def reset(self):
        """Reset filter - Clear all filter information and extension
        """
        self.__filters = [None] * 6
        self.__ext_filter = self.img_ext
        self.reload()
        self.__update()

    def reload(self):
        """Reload the dataframe based on current set data class and extension

        :return:
        """
        # Parsing command works
        self.__df, self.single_session, empty_prj = InternalMethods.parsing(self.path, self.ds_type, self.__dc_idx)
        if not empty_prj:
            self.__df = InternalMethods.initial_filter(self.__df, self.ds_type, self.__ext_filter)
            if len(self.__df):
                self.__df = self.__df[InternalMethods.reorder_columns(self.__dc_idx, self.single_session)]
            self.__update()
        else:
            print('Empty project')
            self.__empty_project = True

    def copy(self):
        """Make copy of current project

        :return: niph.Project instance
        """
        return Project(self.__path, *self.__ref)

    def set_filters(self, *args, **kwargs):
        """Set filters

        :param args:    str[, ]
            String arguments regarding hierarchical data structures
        :param kwargs:  key=value pair[, ]
            Key and value pairs regarding the filename
            :key file_tag:  str or list of str
                Keywords of interest for filename
            :key ignore:    str of list of str
                Keywords of neglect for filename
            :key addition:    boolean
                If this argument is exist and True, keep previous filter information
        :return:
        """
        if 'addition' in kwargs.keys():
            # This option allows to keep previous filter
            if kwargs['addition']:
                self.__update()
            else:
                self.reset()
        else:
            self.reset()
        if args or kwargs:
            if args:
                if self.subjects:
                    if self.__filters[0]:
                        self.__filters[0].extend([arg for arg in args if arg in self.subjects])
                    else:
                        self.__filters[0] = [arg for arg in args if arg in self.subjects]
                    if not self.single_session:
                        if self.__filters[1]:
                            self.__filters[1].extend([arg for arg in args if arg in self.sessions])
                        else:
                            self.__filters[1] = [arg for arg in args if arg in self.sessions]
                    else:
                        self.__filters[1] = None
                else:
                    self.__filters[0] = None
                    self.__filters[1] = None
                if self.__dc_idx == 0:
                    if self.dtypes:
                        if self.__filters[2]:
                            self.__filters[2].extend([arg for arg in args if arg in self.dtypes])
                        else:
                            self.__filters[2] = [arg for arg in args if arg in self.dtypes]
                    else:
                        self.__filters[2] = None
                    self.__filters[3] = None
                elif self.__dc_idx == 1:
                    if self.pipelines:
                        if self.__filters[2]:
                            self.__filters[2].extend([arg for arg in args if arg in self.pipelines])
                        else:
                            self.__filters[2] = [arg for arg in args if arg in self.pipelines]
                    else:
                        self.__filters[2] = None
                    if self.steps:
                        if self.__filters[3]:
                            self.__filters[3].extend([arg for arg in args if arg in self.steps])
                        else:
                            self.__filters[3] = [arg for arg in args if arg in self.steps]
                    else:
                        self.__filters[3] = None
                else:
                    if self.pipelines:
                        if self.__filters[2]:
                            self.__filters[2].extend([arg for arg in args if arg in self.pipelines])
                        else:
                            self.__filters[2] = [arg for arg in args if arg in self.pipelines]
                    else:
                        self.__filters[2] = None
                    if self.results:
                        if self.__filters[3]:
                            self.__filters[3].extend([arg for arg in args if arg in self.results])
                        else:
                            self.__filters[3] = [arg for arg in args if arg in self.results]
                    else:
                        self.__filters[3] = None
            if kwargs:
                if 'file_tag' in kwargs.keys():
                    if type(kwargs['file_tag']) == str:
                        self.__filters[4] = [kwargs['file_tag']]
                    elif type(kwargs['file_tag']) == list:
                        self.__filters[4] = kwargs['file_tag']
                    else:
                        raise error.FilterInputTypeError
                if 'ignore' in kwargs.keys():
                    if type(kwargs['ignore']) == str:
                        self.__filters[5] = [kwargs['ignore']]
                    elif type(kwargs['ignore']) == list:
                        self.__filters[5] = kwargs['ignore']
                    else:
                        raise error.FilterInputTypeError
        self.__df = self.applying_filters(self.__df)
        self.__update()

    def applying_filters(self, df):
        """Applying current filters to the input dataframe

        :param df: pandas.DataFrame
        :return: pandas.DataFrame
        """
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
        if self.__filters[4]:
            df = df[df.Filename.str.contains('|'.join(self.__filters[4]))]
        if self.__filters[5]:
            df = df[~df.Filename.str.contains('|'.join(self.__filters[5]))]
        if self.__ext_filter:
            df = df[df.Filename.str.contains('|'.join(self.__ext_filter))]
        return df

    def help(self, command=None):
        """Print doc string for command or pipeline

        :param command:
        :return:
        """
        if command:
            if command in dir(Interface):
                exec 'help(Interface.{})'.format(command)
            elif command in dir(Analysis):
                exec 'help(Analysis.{})'.format(command)
            else:
                raise error.UnableInterfaceCommand

    def run(self, command, *args, **kwargs): # TODO: put more function for filtering error case
        """Execute processing tools

        :param command: str
            Name of the tool (Use 'pynit.tools.avail' for check available tools)
        :param args: str[, ]
            Arguments for input tool
        :param kwargs: key=value[, ]
            Keyword arguments for input tool
        """
        if command in dir(Interface):
            try:
                if os.path.exists(args[0]):
                    pass
                else:
                    getattr(Interface, command)(*args, **kwargs)
            except:
                exec('help(Interface.{})'.format(command))
                raise error.CommandExecutionFailure
        else:
            raise error.NotExistingCommand

    def __summary(self):
        """Print summary of current project
        """
        summary = 'Project summary'
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
            if self.__pipeline:
                summary = '{}\nInitiated pipeline: {}'.format(summary, self.__pipeline)
        print(summary)

    def __update(self):
        """Update sub variables based on current set filter information
        """
        if len(self.df):
            try:
                self.__subjects = list(set(self.df.Subject.tolist()))
                if self.single_session:
                    self.__sessions = None
                else:
                    self.__sessions = list(set(self.df.Session.tolist()))
                if self.__dc_idx == 0:
                    self.__dtypes = list(set(self.df.DataType.tolist()))
                    self.__pipelines = None
                    self.__steps = None
                    self.__results = None
                elif self.__dc_idx == 1:
                    self.__dtypes = None
                    self.__pipelines = list(set(self.df.Pipeline.tolist()))
                    self.__steps = list(set(self.df.Step.tolist()))
                    self.__results = None
                elif self.__dc_idx == 2:
                    self.__dtypes = None
                    self.__pipelines = list(set(self.df.Pipeline.tolist()))
                    self.__results = list(set(self.df.Result.tolist()))
                    self.__steps = None
            except:
                raise error.UpdateFailed
        else:
            self.__subjects = None
            self.__sessions = None
            self.__dtypes = None
            self.__pipelines = None
            self.__steps = None
            self.__results = None

    def __call__(self, dc_id, *args, **kwargs):
        """Return DataFrame followed applying filters
        :param args:    str[, ]
            String arguments regarding hierarchical data structures

        :param kwargs:  key=value pair[, ]
            Key and value pairs regarding the filename
            :key file_tag:  str or list of str
                Keywords of interest for filename
            :key ignore:    str of list of str
                Keywords of neglect for filename
        :return:
        """
        if self.__empty_project:
            return None
        else:
            copy = self.copy()
            copy.dataclass = dc_id
            copy.set_filters(*args, **kwargs)
            return copy.df

    def __repr__(self):
        """Return absolute path for current filtered dataframe
        :return: str
            Absolute path for filtered data
        """
        if self.__empty_project:
            return str(self.summary)
        else:
            return str(self.df.Abspath)

    def __getitem__(self, index):
        """Return particular data based on input index

        :param index: int
            Index of data
        :return: pandas.DataFrame
        """
        if self.__empty_project:
            return None
        else:
            return self.df.loc[index]

    def __iter__(self):
        """Iterator for dataframe

        :return: pandas.DataFrame
            One row of dataframe
        """
        if self.__empty_project:
            raise error.EmptyProject
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