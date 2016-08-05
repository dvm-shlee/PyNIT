from __future__ import print_function
import os
from os.path import join

import pandas as pd


class InternalMethods(object):
    @staticmethod
    def path_splitter(path):
        """Split path structure into list

        Parameters
        ----------
        path:   str
            Absolute path

        Returns
        -------
        list
        """
        return path.strip(os.sep).split(os.sep)

    @staticmethod
    def parsing(path, ds_type, idx):
        """Parsing the data information based on input data class

        :param path: str
            Project main path
        :param ds_type: list
            Project.ds_type instance
        :param idx: int
            Index for data class
        :return: pandas.DataFrame, boolean
            Return DataFrame instance of the project and
            Whether the project is single session or not
        """
        single_session = False
        df = pd.DataFrame()
        for f in os.walk(join(path, ds_type[idx])):
            if f[2]:
                for filename in f[2]:
                    row = pd.Series(InternalMethods.path_splitter(os.path.relpath(f[0], path)))
                    row['Filename'] = filename
                    row['Abspath'] = join(f[0], filename)
                    df = df.append(pd.DataFrame([row]), ignore_index=True)
        if idx == 0:
            if len(df.columns) is 5:
                single_session = True
        else:
            if len(df.columns) is 6:
                single_session = True
        columns = InternalMethods.update_columns(idx, single_session)
        return df.rename(columns=columns), single_session

    @staticmethod
    def update_columns(idx, single_session):
        """Update columns information based on data class

        :param single_session: boolean
            True, if the project has only single session for each subject
        :param idx: int
            Index of the data class
        :return: dict
            Updated columns
        """
        if idx == 0:
            if single_session:
                subject, session, dtype = (1, 3, 2)
            else:
                subject, session, dtype = (1, 2, 3)
            columns = {0: 'DataClass', subject: 'Subject', session: 'Session', dtype: 'DataType'}
        elif idx == 1:
            columns = {0: 'DataClass', 1: 'Pipeline', 2: 'Step', 3: 'Subject', 4: 'Session'}
        elif idx == 2:
            columns = {0: 'DataClass', 1: 'Pipeline', 2: 'Result', 3: 'Subject', 4: 'Session'}
        else:
            columns = {0: 'DataClass'}
        return columns

    @staticmethod
    def reorder_columns(idx, single_session):
        if idx == 0:
            if single_session:
                return ['Subject', 'DataType', 'Filename', 'Abspath']
            else:
                return ['Subject', 'Session', 'DataType', 'Filename', 'Abspath']
        elif idx == 1:
            if single_session:
                return ['Pipeline', 'Step', 'Subject', 'Filename', 'Abspath']
            else:
                return ['Pipeline', 'Step', 'Subject', 'Session', 'Filename', 'Abspath']
        elif idx == 2:
            if single_session:
                return ['Pipeline', 'Result', 'Subject', 'Filename', 'Abspath']
            else:
                print('Ho')
                return ['Pipeline', 'Result', 'Subject', 'Session', 'Filename', 'Abspath']
        else:
            return None

    @staticmethod
    def initial_filter(df, data_class, ext):
        """Filtering out only selected file type in the project folder

        :param df: pandas.DataFrame
            Project dataframe
        :param data_class: list
            Interested data class of the project
            e.g.) ['Data', 'Processing', 'Results'] for NIRAL method
        :param ext: list
            Interested extension for particular file type
        :return: pandas.DataFrame
            Filtered dataframe
        """
        if data_class:
            if not type(data_class) is list:
                data_class = [data_class]
            df = df[df['DataClass'].isin(data_class)]
        if ext:
            df = df[df['Filename'].str.contains('|'.join(ext))]
        columns = df.columns
        return df.reset_index()[columns]

    @staticmethod
    def isnull(df):
        """Check missing value

        :param df: pandas.DataFrame
        :return:
        """
        return pd.isnull(df)

    @staticmethod
    def mk_main_folder(prj):
        """Make processing and results folders
        """
        InternalMethods.mkdir(join(prj.path, prj.ds_type[1]), join(prj.path, prj.ds_type[2]))

    @staticmethod
    def check_merged_output(args):
        if True in args:
            return True, args[1:]
        else:
            return False, args[1:]

    @staticmethod
    def filter_file_index(option, prj, file_index):
        if file_index:
            option.extend(prj.df.Abspath.tolist()[min(file_index):max(file_index) + 1])
        else:
            option.extend(prj.df.Abspath.tolist())
        return option

    @staticmethod
    def get_step_name(pipeline_inst, step):
        """ Generate step name with step index

        :param pipeline_inst:
        :param step:
        :return:
        """
        if pipeline_inst.pipeline:
            if len(pipeline_inst.done):
                last_step = []
                # Check the folder of last step if the step has been processed or not
                for f in os.walk(join(pipeline_inst.path, pipeline_inst.done[-1])):
                    last_step.extend(f[2])
                fin_list = [s for s in pipeline_inst.done if step in s]
                # Check if the step name is overlapped or not
                if len(fin_list):
                    return fin_list[0]
                else:
                    if not len([f for f in last_step if '.nii' in f]):
                        print('Last step folder returned instead, it is empty.')
                        return pipeline_inst.done[-1]
                    else:
                        return "_".join([str(pipeline_inst.steps).zfill(3), step])
            else:
                return "_".join([str(pipeline_inst.steps).zfill(3), step])
        else:
            raise ErrorHandler.no_pipeline

    @staticmethod
    def mkdir(*paths):
        for path in paths:
            try:
                os.mkdir(path)
            except:
                pass


class ErrorHandler(object):
    @property
    def no_pipeline(self):
        return AttributeError('No pipeline is attached with this project')

    @property
    def not_merged(self):
        return TypeError('Output of this command need to be merged')

    @property
    def no_merge(self):
        return TypeError('Output of this command cannot be merged')

    @property
    def no_inputs(self):
        return KeyError('')
