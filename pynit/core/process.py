# Standard library
import os
import time
import inspect
from os.path import join

import pandas as pd

from .utility import Internal, Interface
from .project import Project
import error


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
        self.commands = Interface()

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
        if command in dir(Interface):
            argspec = dict(inspect.getargspec(getattr(Interface, command)).__dict__)
        elif command in dir(Internal):
            argspec = dict(inspect.getargspec(getattr(Internal, command)).__dict__)
        else:
            raise error.CommandExecutionFailure
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
                raise error.CommandExecutionFailure
        elif command in dir(Internal):
            # try:
            if os.path.exists(args[0]):
                pass
            else:
                print args, kwargs
                getattr(Internal, command)(*args, **kwargs)
            # except:
            #     raise error.CommandExecutionFailure
        else:
            raise error.CommandExecutionFailure


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
            step_name = Internal.get_step_name(self, step_name)
            step_path = join(self.path, step_name)
            Internal.mkdir(step_path)
        else:
            print('Error')
            return None
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
        Internal.mkdir(self.path)
        self.prj.filters[2] = self.pipeline

    def set(self, pipeline = None):
        """Set pipeline for execution

        :param pipeline:
        :return:
        """
        if pipeline in self.avail:
            self.pipeline = pipeline
            Internal.mkdir(self.path)
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
            return None

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
            return None

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
        # option.insert(0, merged_output)
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
                        Internal.mkdir(output_path)
                        if self.prj.single_session:
                            # If the project is single session, run the command at the subject level
                            subgroup.append(subj)
                            self.prj(dc_id, *subgroup, **filter_kw)
                            output_filename = '{}_{}{}'.format(output_prefix, subj, self.prj.img_ext[0])
                            option = Internal.filter_file_index(option, self.prj, file_index)
                            self.run_cmd(command, join(output_path, output_filename), *option, **kwoption)
                        else:
                            # If the project has multi session, run the command through session level
                            for sess in sessions:
                                if pd.isnull(sess):
                                    pass
                                else:
                                    output_path = join(step_path, subj, sess)
                                    Internal.mkdir(output_path)
                                    subgroup.extend([subj, sess])
                                    self.prj(dc_id, *subgroup, **filter_kw)
                                    output_filename = '{}_{}_{}{}'.format(output_prefix, subj, sess, self.prj.img_ext[0])
                                    option = Internal.filter_file_index(option, self.prj, file_index)
                                    self.run_cmd(command, join(output_path, output_filename), *option, **kwoption)
            elif output_level == 1:
                # output will be merged at step folder or subject level
                if self.prj.single_session:
                    # If single session, output file will be merged at step path
                    output_path = step_path
                    Internal.mkdir(output_path)
                    self.prj(dc_id, *filters, **filter_kw)
                    output_filename = '{}_{}{}'.format(output_prefix, step_name, self.prj.img_ext[0])
                    option = Internal.filter_file_index(option, self.prj, file_index)
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
                            Internal.mkdir(output_path)
                            self.prj(dc_id, *subgroup, **filter_kw)
                            option = Internal.filter_file_index(option, self.prj, file_index)
                            self.run_cmd(command, join(output_path, output_filename), *option, **kwoption)
            elif output_level == 2:
                # output will be merged at step folder if project has multi sessions
                if self.prj.single_session:
                    raise ValueError('Maximum value of output level for single session is 1.')
                else:
                    output_path = step_path
                    Internal.mkdir(output_path)
                    self.prj(dc_id, *filters, **filter_kw)
                    output_filename = '{}_{}{}'.format(output_prefix, step_name, self.prj.img_ext[0])
                    option = Internal.filter_file_index(option, self.prj, file_index)
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
                    Internal.mkdir(output_path)
                    if self.prj.single_session:
                        # Check single session
                        subgroup.append(subj)
                        print dc_id
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
                                Internal.mkdir(output_path)
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