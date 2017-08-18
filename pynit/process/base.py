import os
import sys
import pickle
from pynit.tools import messages
from pynit.tools import methods
from pynit.tools import viewer

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


def get_step_name(procobj, step, verbose=None):
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
    processing_path = os.path.join(procobj.__prj.path,
                                   procobj.__prj.ds_type[1],
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


class BaseProcess(object):
    """Collections of step components for pipelines
    """
    def __init__(self, prjobj, name, logging=True, viewer='itksnap'):
        """

        :param prjobj:
        :param name:
        :param logging:
        :param viewer:
        """

        # Prepare inputs
        prjobj.reset_filters()
        self._processing = name
        self.__prj = prjobj
        path = os.path.join(self.__prj.path, self.__prj.ds_type[1])
        self._path = os.path.join(path, self._processing)

        # Initiate logger
        if logging:
            self.logger = methods.get_logger(path, name)

        # Define default arguments
        self._subjects = None
        self._sessions = None
        self._history = {}
        self._tempfiles = []
        self._viewer = viewer

        # Initiate
        self.init_proc()

    @property
    def prj(self):
        return self.__prj

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

    def itksnap(self, idx, base_idx=None):
        """Launch ITK-snap

        :param idx:
        :param base_idx:
        :return:
        """
        if base_idx:
            display(viewer.itksnap(self, self.steps[idx], self.steps[base_idx]))
        else:
            display(viewer.itksnap(self, self.steps[idx]))

    def afni(self, idx, tmpobj=None):
        """Launch AFNI gui

        :param idx:
        :param tmpobj:
        :return:
        """
        viewer.afni(self, self.steps[idx], tmpobj=tmpobj)

    def fslview(self, idx, base_idx=None):
        """Launch fslview

        :param idx:
        :param base_idx:
        :return:
        """
        if base_idx:
            viewer.fslview(self, self.steps[idx], self.steps[base_idx])
        else:
            viewer.fslview(self, self.steps[idx])


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
        if self.__prj(1).subjects:
            if self.__prj(0).subjects:
                datasubj = set(list(self.__prj(0).subjects))
                procsubj = set(list(self.__prj(1).subjects))
                if datasubj.issubset(procsubj):
                    if not procsubj.issubset(datasubj):
                        try:
                            self._subjects = sorted(self.__prj(1, self.processing).subjects[:])
                            if not self.__prj.single_session:
                                self._sessions = sorted(self.__prj(1, self.processing).sessions[:])
                        except:
                            self._subjects = sorted(self.__prj(0).subjects[:])
                            if not self.__prj.single_session:
                                self._sessions = sorted(self.__prj(0).sessions[:])
                    else:
                        self._subjects = sorted(self.__prj(0).subjects[:])
                        if not self.__prj.single_session:
                            self._sessions = sorted(self.__prj(0).sessions[:])
                else:
                    self._subjects = sorted(self.__prj(0).subjects[:])
                    if not self.__prj.single_session:
                        self._sessions = sorted(self.__prj(0).sessions[:])
            else:
                self._subjects = sorted(self.__prj(1).subjects[:])
        else:
            self._subjects = sorted(self.__prj(0).subjects[:])
            if not self.__prj.single_session:
                self._sessions = sorted(self.__prj(0).sessions[:])

        self.logger.info('Proc::Attributes [subjects, sessions] are reset to default value.')
        self.logger.info('Proc::Subject is defined as [{}]'.format(",".join(self._subjects)))
        if self._sessions:
            self.logger.info('Proc::Session is defined as [{}]'.format(",".join(self._sessions)))

    def init_proc(self):
        """Initiate process folder

        :return: None
        """
        methods.mkdir(self._path)
        self.logger.info('Proc::Initiating instance {0}'.format(self.processing))
        history = os.path.join(self._path, '.proc_hisroty')
        if os.path.exists(history):
            with open(history, 'r') as f:
                self._history = pickle.load(f)
            self.logger.info("Proc::History file is loaded".format(history))
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
            path = os.path.join(self.__prj.path, self.__prj.ds_type[1], self._processing, path)
            methods.mkdir(path)
            return path
        else:
            methods.raiseerror(messages.Errors.InitiationFailure, 'Error on initiating step')

    def save_history(self):
        history = os.path.join(self._path, '.proc_hisroty')
        with open(history, 'w') as f:
            pickle.dump(self._history, f)
        self.logger.info("Proc::History file '{0}' is saved".format(history))
