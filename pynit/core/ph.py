# Standard library
import os
import time
import inspect
from os.path import join

import pandas as pd

from .tool import Commands, ImageObject
from .statics import InternalMethods, ErrorHandler


class ImageObject(ImageObject):
    pass

class Reference(object):
    """Class of reference informations for image processing and data analysis
    """
    img = {'NifTi-1':           ['.nii', '.nii.gz'],
           'ANALYZE7.5':        ['.img', '.hdr'],
           'AFNI':              ['.BRIK', '.HEAD'],
           'Shihlab':           ['.sdt', '.spr']
           }
    txt = {'Common':            ['.txt', '.cvs', '.tvs'],
           'AFNI':              ['.1D'],
           'MATLAB':            ['.mat'],
           'Slicer_Transform':  ['.tfm'],
           'JSON':              ['.json']
           }
    data_structure = {'NIRAL': ['Data', 'Processing', 'Results'],
                      'BIDS': ['sourcedata', 'derivatives']
                      }

    def __init__(self, *args):
        try:
            self._img = [arg for arg in args if arg in self.img.keys()]
            self._txt = [arg for arg in args if arg in self.txt.keys()]
            self._ds = [arg for arg in args if arg in self.data_structure.keys()]
            if (len(self._img) or len(self._txt) or len(self._ds)) > 1:
                raise AttributeError
        except:
            raise AttributeError

    def __repr__(self):
        title = 'Predefined values'
        img = 'Image format:\t{}'.format(self.img.keys())
        txt = 'Text format:\t{}'.format(self.txt.keys())
        ds = 'Data structure:\t{}'.format(self.data_structure.keys())
        output = '{}\n{}\n{}\n{}\n{}'.format(title,'-'*len(title), img, txt, ds)
        return output

    @property
    def imgext(self):
        return self.img[self._img[0]]

    @property
    def txtext(self):
        return self.txt[self._txt[0]]

    @property
    def ref_ds(self):
        return self.data_structure[self._ds[0]]

    def set_img_format(self, img_format):
        if img_format in self.img.keys():
            raise AttributeError
        else:
            self._img = img_format

    def set_txt_format(self, txt_format):
        if txt_format in self.txt.keys():
            raise AttributeError
        else:
            self._txt = txt_format

    def set_ref_data_structure(self, ds_ref):
        if ds_ref in self.data_structure.keys():
            raise AttributeError
        else:
            self._ds = ds_ref


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
        ref = Reference(*self.__ref)
        self.img_ext = ref.imgext
        self.ds_type = ref.ref_ds
        # Define basic variables for initiating instance
        self.__dc_idx = 0           # Data class index
        self.__ext_filter = self.img_ext
        InternalMethods.mk_main_folder(self)
        try:
            self.reload()
        except Exception as e:
            print(e.message, e.args)

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
            # self.reload()
            self.reset()
            self.__update()
        else:
            raise IndexError

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
        self.__df, self.single_session = InternalMethods.parsing(self.path, self.ds_type, self.__dc_idx)
        self.__df = InternalMethods.initial_filter(self.__df, self.ds_type, self.__ext_filter)
        if len(self.__df):
            self.__df = self.__df[InternalMethods.reorder_columns(self.__dc_idx, self.single_session)]
        self.__update()

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
            :key extend:    boolean
                If this argument is exist and True, keep pervious filter information
        :return:
        """
        if 'extend' in kwargs.keys():
            # This oprion allows to keep previous filter
            if kwargs['extend']:
                self.__update()
            else:
                self.reset()
                # self.reload()
        else:
            self.reset()
            # self.reload()
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
                        raise TypeError
                if 'ignore' in kwargs.keys():
                    if type(kwargs['ignore']) == str:
                        self.__filters[5] = [kwargs['ignore']]
                    elif type(kwargs['ignore']) == list:
                        self.__filters[5] = kwargs['ignore']
                    else:
                        raise TypeError
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

    def __summary(self):
        """Print summary of current project
        """
        summary = 'Project summary'
        summary = '{}\nProject: {}'.format(summary, os.path.dirname(self.path).split(os.sep)[-1])
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
                    if self.__filters[2]:
                        self.__steps = list(set(self.df.Step.tolist()))
                    else:
                        self.__steps = None
                    self.__results = None
                elif self.__dc_idx == 2:
                    self.__dtypes = None
                    self.__pipelines = list(set(self.df.Pipeline.tolist()))
                    if self.__filters[2]:
                        self.__results = list(set(self.df.Result.tolist()))
                    else:
                        self__results = None
                    self.__steps = None
            except:
                raise AttributeError
        else:
            self.__subjects = None
            self.__sessions = None
            self.__dtypes = None
            self.__pipelines = None
            self.__steps = None
            self.__results = None

    def __call__(self, *args, **kwargs):
        """Return DataFrame followed applying filters
        :param args:    str[, ]
            String arguments regarding hierarchical data structures
        :param kwargs:  key=value pair[, ]
            Key and value pairs regarding the filename
            :key file_tag:  str or list of str
                Keywords of interest for filename
            :key ignore:    str of list of str
                Keywords of neglect for filename
            :key extend:    boolean
                If this argument is exist and True, keep pervious filter information
        :return:
        """
        self.dataclass = self.__dc_idx
        self.set_filters(*args, **kwargs)
        return self.df

    def __repr__(self):
        """Return absolute path for current filtered dataframe
        :return: str
            Absolute path for filtered data
        """
        return str(self.df.Abspath)

    def __getitem__(self, index):
        """Return particular data based on input index

        :param index: int
            Index of data
        :return: pandas.DataFrame
        """
        return self.df.loc[index]

    def __iter__(self):
        """Iterator for dataframe

        :return: pandas.DataFrame
            One row of dataframe
        """
        for row in self.df.iterrows():
            yield row

    def __len__(self):
        """Return number of data
        """
        return len(self.df)


class Processing(object):
    """DataClass for applying image processing
    """
    def __init__(self, obj):
        """Initiate processing object.
        This class is base class of Pipeline

        :param obj:
        """
        try:
            if type(obj) is str:
                obj = Project(obj)
            self.prj = obj
        except:
            raise TypeError
        self.__path = join(obj.path, self.prj.ds_type[1])
        self.commands = Commands()

    @property
    def path(self):
        return self.__path

    @staticmethod
    def check_kwargs(kwargs, command):
        """Validate input arguments for input command

        :param kwargs: dict
        :param command: str
        :return:
        """
        args, defaults, varargs, keywords = Processing.check_args(command)
        # check kwargs
        output = dict()
        for key in kwargs.keys():
            if key not in args:
                if defaults and key in defaults.keys():
                    output[key] = kwargs[key]
                elif varargs and key in varargs:
                    if type(kwargs[key]) != list:
                        raise TypeError("'{}' keyword must be list".format(key))
                    else:
                        output[args] = kwargs[key]
                elif keywords and key in keywords:
                    if type(kwargs[key]) != dict:
                        raise TypeError("'{}' keyword must be dictionary".format(key))
                    else:
                        output[kwargs] = kwargs[key]
                else:
                    raise KeyError("'{}' is not fitted for the command '{}'".format(key, command))
            else:
                output[key] = kwargs[key]
        return output

    @staticmethod
    def check_args(command):
        """Check arguments of input command

        :param command:
        :return:
            args
            defaults
            varargs
            keywords
        """
        argspec = dict(inspect.getargspec(getattr(Commands, command)).__dict__)
        if argspec['defaults'] is None:
            def_len = 0
            defaults = None
        else:
            def_len = len(argspec['defaults'])
            defaults = dict(zip(argspec['args'][len(argspec['args']) - def_len:], argspec['defaults']))
        args = argspec['args'][1:(len(argspec['args']) - def_len)]
        varargs = argspec['varargs']
        kwargs = argspec['keywords']
        return args, defaults, varargs, kwargs

    def run_cmd(self, command, *args, **kwargs): # TODO: put more function for filtering error case
        """Execute processing tools

        :param command: str
            Name of the tool (Use 'pynit.tools.avail' for check available tools)
        :param args: str[, ]
            Arguments for input tool
        :param kwargs: key=value[, ]
            Keyword arguments for input tool
        """
        if command in dir(self.commands):
            try:
                if os.path.exists(args[0]):
                    pass
                else:
                    getattr(self.commands, command)(*args, **kwargs)
            except:
                raise SyntaxError("Given argument is not fitted for the tool '{}'".format(command))
        else:
            raise AttributeError("{} is not exist. Please check available tools.".format(command))


class Pipeline(Processing):
    """Class for executing pipelines
    """
    def __init__(self, obj):
        super(Pipeline, self).__init__(obj)
        self.prj.reset()
        self.prj(1)
        self.pipeline = None
        self.__path = super(Pipeline, self).path

    @property
    def path(self):
        if self.pipeline:
            return join(self.__path, self.pipeline)
        else:
            return self.__path

    @property
    def done(self):
        """Check processed steps

        :return:
        """
        if self.pipeline:
            return [f for f in os.listdir(self.path) if os.path.isdir(join(self.path, f))]
        else:
            return []

    @property
    def avail(self):
        """Check available pipeline to execute

        :return:
        """
        return [pipeline[6:] for pipeline in dir(self) if '_pipe_' in pipeline]

    @property
    def steps(self):
        if self.pipeline:
            return len(self.done) + 1
        else:
            return 0

    @property
    def processed(self):
        return [pipeline for pipeline in self.avail if pipeline in os.listdir(self.path)]

    @property
    def summary(self):
        return self.prj.summary

    def init_step(self, step_name):
        """Initiating step for set

        :param step_name: str
            Name of step for pipeline
        :return:
        """
        if self.pipeline:
            step_name = InternalMethods.get_step_name(self, step_name)
            step_path = join(self.path, step_name)
            InternalMethods.mkdir(step_path)
        else:
            raise ErrorHandler.no_pipeline
        return step_name, step_path

    def set_filters(self, dc_idx=1, *args, **kwargs):
        """Set filters
        :param dc_idx:      int
            Index for dataclass (Default is 1, which is corresponded with niph.ds_method[1])
        :param args:        str[, ]
            String arguments regarding hierarchical data structures
        :param kwargs:      key=value pair[, ]
            Key and value pairs regarding the filename
            :key file_tag:  str or list of str
                Keywords of interest for filename
            :key ignore:    str of list of str
                Keywords of neglect for filename
            :key extend:    boolean
                If this argument is exist and True, keep pervious filter information
        """
        self.prj(dc_idx, *args, **kwargs)

    def init(self, name):
        """Initiate custom pipeline

        :param name: str
            Name of pipeline
        :return:
        """
        self.pipeline = name
        InternalMethods.mkdir(self.path)
        self.prj.filters[2] = self.pipeline

    def set(self, pipeline = None):
        """Set pipeline for execution

        :param pipeline:
        :return:
        """
        if pipeline in self.avail:
            self.pipeline = pipeline
            InternalMethods.mkdir(self.path)
            if not len(self.done):
                print("Pipeline '{}' is initiated first time for this project.".format(self.pipeline))
            else:
                print("This pipeline already had been processed.")
            self.prj.filters[2] = pipeline
        else:
            print("Warning:The input pipeline not exist in this package.\n")
            print("If you want to initiate custom pipeline, use 'init' method instead.")
            print("-*-Available pipelines-*-")
            for i, pipeline in enumerate(self.avail):
                print('{}. {}'.format(i+1, pipeline))
            raise ErrorHandler.no_pipeline

    def run(self, *args, **kwargs):
        """Execute current pipeline

        :param args:
        :param kwargs:
        :return:
        """
        if self.pipeline:
            print("Executing pipeline {}".format(self.pipeline))
            getattr(self, '_pipe_{}'.format(self.pipeline))(*args, **kwargs)
        else:
            raise ErrorHandler.no_pipeline

    def run_step(self, step_name, command, *args, **kwargs):
        """Method to execute step for processing input command with input arguments

        :param step_name: str
        :param command: str
        :param args: str[, ]
            'merged_output'
        :param kwargs:
            :key 'dc_id': int
                Data class index
            :key 'inputs': dict

            :key 'filters': list
                filter arguments for set_filters() method
                str for *args
                dict for **kwargs
            :key 'output_level': int
            :key 'file_index': int
            :key 'output_prefix': str
            etc: keyword arguments for command
        :return: str
            Step name
        """
        # Set default parameters
        merged_output = False
        input_kwargs = dict()
        # Count processing time
        start_time = time.time()
        # Parsing arguments
        if args:
            for arg in args:
                if arg is 'multi_inputs':
                    multi_inputs = True
                if arg is 'merged_output':
                    merged_output = True
        # Parsing keyword arguments
        if kwargs:
            if 'dc_id' in kwargs.keys():
                # Index of DataClass of interest for setting point of departure of main inputs
                input_kwargs['dc_id'] = kwargs['dc_id']
                del kwargs['dc_id']
            else:
                input_kwargs['dc_id'] = 1
            if 'filters' in kwargs.keys():
                # input arguments for 'set_filters' method, list
                input_kwargs['filters'] = kwargs['filters']
                del kwargs['filters']
            else:
                input_kwargs['filters'] = None
            if 'output_level' in kwargs.keys():
                # 'output_level' allowing to select hierarchy level for output folder
                # integer, [0-2]: higher number represent more order parent folder
                if kwargs['output_level'] in range(3):
                    input_kwargs['output_level'] = kwargs['output_level']
                    del kwargs['output_level']
                else:
                    raise IndexError
            else:
                input_kwargs['output_level'] = 0
            if 'output_prefix' in kwargs.keys():
                # If output is merged, set output_filename here
                if merged_output:
                    input_kwargs['output_prefix'] = kwargs['output_prefix']
                    del kwargs['output_prefix']
                else:
                    raise KeyError("Only merged_output type of execution can import 'output_prefix'")
            if 'file_index' in kwargs.keys():
                # You can select particular index for dataset here
                if type(kwargs['file_index']) == int:
                    input_kwargs['file_index'] = [kwargs['file_index']]
                elif type(kwargs['file_index']) == list:
                    for i in kwargs['file_index']:
                        if type(i) != int:
                            raise KeyError('Index value need to be integer')
                        else:
                            pass
                    input_kwargs['file_index'] = kwargs['file_index']
                else:
                    raise KeyError('Index value need to be integer')
                del kwargs['file_index']
            else:
                input_kwargs['file_index'] = None
        else:
            raise KeyError('kwargs need to be imported')

        # Initiate Step environment
        step_name, step_path = self.init_step(step_name)
        input_kwargs['step_name'] = step_name
        input_kwargs['command'] = command
        # Validate input arguments, input_args instance is dictionary which contains all required parameter to
        # run given command
        input_args = Processing.check_kwargs(kwargs, command)
        # Processing the command based on type of execution
        self.__regular_execution(input_args, merged_output, **input_kwargs)
        end_time = time.time() - start_time
        print('\nStep{} takes {} sec'.format(step_name, round(end_time, 2)))
        return step_name

    def __parsing_args_exec(self, input_args, **input_kwargs):
        """Parsing arguments for command execution through dataset

        :param input_args:
        :param merged_output:
        :param input_kwargs:
        :return:
        """
        # Parsing arguments for setting the execution of the command
        try:
            dc_id = input_kwargs['dc_id']
            filters = input_kwargs['filters']
            filter_kw = dict()
            output_level = input_kwargs['output_level']
            step_name = input_kwargs['step_name']
            command = input_kwargs['command']
            file_index = input_kwargs['file_index']
        except:
            raise KeyError("The components of 'input_kwargs' is not valid")
        if filters:
            for arg in filters:
                # If filters contain dictionary, separate it to kwargs for set_filters method
                if type(arg) == dict:
                    filter_kw = arg
                    filters.remove(arg)
                else:
                    pass
        else:
            filters = []
        if 'args' in input_args.keys():
            # Parsing *args parameters for the command execution
            option = input_args['args']
            del input_args['args']
        else:
            option = []
        if 'kwargs' in input_args.keys():
            # Parsing **kwargs parameters for the command execution
            kwoption = input_args['kwargs']
            del input_args['kwargs']
        else:
            kwoption = dict()
        # Set dataclass for input data
        self.prj.dataclass = dc_id
        # If subject or session names are included in filters, applying it
        subjects = []
        sessions = []
        if filters:
            for arg in filters:
                if arg in self.prj.subjects:
                    subjects.append(arg)
                if self.prj.single_session:
                    pass
                else:
                    if arg in self.prj.sessions:
                        sessions.append(arg)
        if not subjects:
            subjects = self.prj.subjects
        if not sessions:
            if not self.prj.single_session:
                sessions = self.prj.sessions
        return dc_id, filters, filter_kw, output_level, step_name,\
               command, file_index, subjects, sessions, option, kwoption

    def __regular_execution(self, input_args, merged_output, **input_kwargs):
        """Execution with single input dataset

        :param input_args:

        :param merged_output: boolean
            check output type
        :param input_kwargs:
        :return:
        """
        dc_id, filters, filter_kw, output_level, step_name, command, file_index, subjects, \
        sessions, option, kwoption = self.__parsing_args_exec(input_args, **input_kwargs)
        step_path = join(self.path, step_name)
        # For the regular execution, there are two parameter which can apply
        # First is output_level, and second is merged output
        option.insert(0, merged_output)
        if merged_output:
            # If the output of the command need to be merged. the command usually takes multiple inputs
            # In this case, parameter 'output_prefix' need to be defined.
            try:
                output_prefix = input_kwargs['output_prefix']
            except:
                output_prefix = 'merged'
            # Based on the parameter output_level, level of merging can be vary
            if output_level == 0:
                # output will be merged at subject or session level.
                for subj in subjects:
                    # loop through subjects
                    if pd.isnull(subj):
                        pass
                    else:
                        output_path = join(step_path, subj)
                        if filters:
                            subgroup = filters[:]
                        else:
                            subgroup = []
                        InternalMethods.mkdir(output_path)
                        if self.prj.single_session:
                            # If the project is single session, run the command at the subject level
                            subgroup.append(subj)
                            self.prj(dc_id, *subgroup, **filter_kw)
                            output_filename = '{}_{}{}'.format(output_prefix, subj, self.prj.img_ext[0])
                            option = InternalMethods.filter_file_index(option, self.prj, file_index)
                            self.run_cmd(command, join(output_path, output_filename), *option, **kwoption)
                        else:
                            # If the project has multi session, run the command through session level
                            for sess in sessions:
                                if pd.isnull(sess):
                                    pass
                                else:
                                    output_path = join(step_path, subj, sess)
                                    InternalMethods.mkdir(output_path)
                                    subgroup.extend([subj, sess])
                                    self.prj(dc_id, *subgroup, **filter_kw)
                                    output_filename = '{}_{}_{}{}'.format(output_prefix, subj, sess, self.prj.img_ext[0])
                                    option = InternalMethods.filter_file_index(option, self.prj, file_index)
                                    self.run_cmd(command, join(output_path, output_filename), *option, **kwoption)
            elif output_level == 1:
                # output will be merged at step folder or subject level
                if self.prj.single_session:
                    # If single session, output file will be merged at step path
                    output_path = step_path
                    InternalMethods.mkdir(output_path)
                    self.prj(dc_id, *filters, **filter_kw)
                    output_filename = '{}_{}{}'.format(output_prefix, step_name, self.prj.img_ext[0])
                    option = InternalMethods.filter_file_index(option, self.prj, file_index)
                    self.run_cmd(command, join(output_path, output_filename), *option, **kwoption)
                else:
                    for subj in subjects:
                        if pd.isnull(subj):
                            pass
                        else:
                            output_path = join(step_path, subj)
                            if filters:
                                subgroup = filters[:]
                            else:
                                subgroup = []
                            output_filename = '{}_{}{}'.format(output_prefix, subj, self.prj.img_ext[0])
                            subgroup.append(subj)
                            InternalMethods.mkdir(output_path)
                            self.prj(dc_id, *subgroup, **filter_kw)
                            option = InternalMethods.filter_file_index(option, self.prj, file_index)
                            self.run_cmd(command, join(output_path, output_filename), *option, **kwoption)
            elif output_level == 2:
                # output will be merged at step folder if project has multi sessions
                if self.prj.single_session:
                    raise ValueError('Maximum value of output level for single session is 1.')
                else:
                    output_path = step_path
                    InternalMethods.mkdir(output_path)
                    self.prj(dc_id, *filters, **filter_kw)
                    output_filename = '{}_{}{}'.format(output_prefix, step_name, self.prj.img_ext[0])
                    option = InternalMethods.filter_file_index(option, self.prj, file_index)
                    self.run_cmd(command, join(output_path, output_filename), *option, **kwoption)
            else:
                raise ValueError('Value range of output level is range(3).')
        else:
            # If output is not merged
            for subj in subjects:
                if pd.isnull(subj):
                    pass
                else:
                    if filters:
                        subgroup = filters[:]
                    else:
                        subgroup = []
                    if output_level == 0:
                        output_path = join(step_path, subj)
                    else:
                        output_path = step_path
                    InternalMethods.mkdir(output_path)
                    if self.prj.single_session:
                        # Check single session
                        subgroup.append(subj)
                        self.prj(dc_id, *subgroup, **filter_kw)
                        if file_index:
                            for i in file_index:
                                finfo = self.prj[i]
                                self.run_cmd(command, join(output_path, finfo.Filename), finfo.Abspath,
                                             *option, **kwoption)
                        else:
                            for i, finfo in self.prj.df.iterrows():
                                self.run_cmd(command, join(output_path, finfo.Filename), finfo.Abspath,
                                             *option, **kwoption)
                    else:
                        for sess in sessions:
                            if pd.isnull(sess):
                                pass
                            else:
                                subgroup.extend([subj, sess])
                                if output_level == 1:
                                    output_path = join(step_path, subj)
                                elif output_level == 2:
                                    pass
                                else:
                                    output_path = join(step_path, subj, sess)
                                InternalMethods.mkdir(output_path)
                                self.prj(dc_id, *subgroup, **filter_kw)
                                if file_index:
                                    for i in file_index:
                                        finfo = self.prj[i]
                                        self.run_cmd(command, join(output_path, finfo.Filename), finfo.Abspath,
                                                     *option, **kwoption)
                                for i, finfo in self.prj.df.iterrows():
                                    self.run_cmd(command, join(output_path, finfo.Filename), finfo.Abspath,
                                                 *option, **kwoption)

    def help(self, command=None):
        """Print doc string for command or pipeline

        :param command:
        :return:
        """
        if command:
            if command in self.commands.avail:
                exec 'help(self.commands.{})'.format(command)
            else:
                raise NameError('Given command is not exist.')
        else:
            if self.pipeline in self.avail:
                exec 'help(self._pipe_{})'.format(self.pipeline)
            else:
                raise TypeError('Custom pipeline do not have docstring to print out.')

    def reset(self, pipeline=True):
        """Reset filters

        :param pipeline: boolean
            if True, discard selected pipeline
        :return:
        """
        self.prj.reset()
        if pipeline:
            if self.pipeline:
                self.pipeline = None
        else:
            pass

    def __getitem__(self, idx):
        return self.prj[idx]

    def __call__(self, step_name, command, *args, **kwargs):
        """Initiate the step for pipeline
        """
        step_name = self.run_step(step_name, command, *args, **kwargs)
        return step_name

    def __repr__(self):
        output = 'Available pipelines: {}'.format(self.avail)
        if self.pipeline:
            output = 'Selected pipeline: {}'.format(self.pipeline)
            if self.pipeline not in self.avail:
                output = '\n{} - Custom pipeline\n'.format(output)
        output = '{}\nProcessed step folders:'.format(output)
        if not len(self.done):
            output = '{}\n {}'.format(output, 'None')
        else:
            for step in self.done:
                output = '{}\n {}'.format(output, step)
            output = '{}\n'.format(output)
        return output

