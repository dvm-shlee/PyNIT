import os
import pickle
from pynit.tools import methods, messages, HTML as title, widgets
from pynit.tools import gui, display, notebook_env


class BaseProcess(object):
    """Collections of step components for pipelines
    """
    def __init__(self, prjobj, name, tag, logging=True, viewer='itksnap'):
        """

        :param prjobj:
        :param name:
        :param logging:
        :param viewer:
        """

        # Prepare inputs
        prjobj.reset_filters()
        self.__prj = prjobj
        self._ext = prjobj.img_ext
        if tag:
            name = "{}-{}".format(name, tag)
        self._path = os.path.join(self.prj.path, self.prj.ds_type[1], name)
        self._rpath = os.path.join(self.prj.path, self.prj.ds_type[2], name)
        self._processing = name

        # Initiate logger
        if logging:
            self.logger = methods.get_logger(os.path.dirname(self._path), '{}'.format(name))
            self.rlogger = methods.get_logger(os.path.dirname(self._rpath), '{}'.format(name))

        # Define default arguments
        self._subjects = None
        self._sessions = None
        self._history = {}
        self._rhistory = {}
        self._tempfiles = []
        self._viewer = viewer

        # Update information
        self.init_proc()
        if len(self._history.keys()) or len(self._rhistory.keys()):
            count_incorrect = 0
            for path in self._history.values():
                if not os.path.exists(path):
                    count_incorrect += 1
            for path in self._rhistory.values():
                if not os.path.exists(path):
                    count_incorrect += 1
            if count_incorrect:
                self.logger.debug('Incorrect subpathes are detected ({})'.format(count_incorrect))
                self.update()
        else:
            self.update()

    @property
    def prj(self):
        return self.__prj

    def check_input(self, input_path, dc=0):
        """Check input_path and return absolute path

        :param input_path: str, name of the Processing step folder
        :param dc:  0-Processing
                    1-Results
        :return: str, Absolute path of the step
        """
        if not dc:
            source_idx = self.steps
            source_exe = self.executed
            source_hst = self._history
        else:
            source_idx = self.results
            source_exe = self.reported
            source_hst = self._rhistory
        if isinstance(input_path, int):
            input_path = source_idx[input_path]
        if input_path in source_exe:
            return source_hst[input_path]
        else:
            return input_path

    def _get_subpath(self, path):
        import re
        pattern = r'^\d{3}_.*'
        list_subpath = sorted([f for f in os.listdir(path) if re.match(pattern, f)])
        return list_subpath

    def update(self):
        list_steps = self._get_subpath(self._path)
        list_results = self._get_subpath(self._rpath)
        self.logger.debug("Executed steps are updated as [{}]".format(", ".join(list_steps)))
        self.rlogger.debug("Reported results are updated as [{}]".format(", ".join(list_results)))

        # Update processing history
        for step in list_steps:
            self._history[step] = os.path.join(self._path, step)

        # Update results history
        for result in list_results:
            self._rhistory[result] = os.path.join(self._rpath, result)

        # Save all history
        self._save_history(self._path, self._history)
        self._save_history(self._rpath, self._rhistory)

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

    def image_viewer(self, idx, base_idx=None, viewer=None):
        """Launch image viewer

        :param idx:
        :param base_idx:
        :return:
        """
        if notebook_env:
            if base_idx:
                display(gui.image_viewer(self, self.steps[idx], self.steps[base_idx], viewer=viewer))
            else:
                display(gui.image_viewer(self, self.steps[idx], viewer=viewer))
        else:
            methods.raiseerror(messages.Errors.InsufficientEnv, 'This method only works on Jupyter Notebook')

    def afni(self, idx, tmpobj=None, dc=0):
        """Launch AFNI gui

        :param idx:
        :param tmpobj:
        :return:
        """
        if dc==0:
            self.update()
            gui.afni(self, self.steps[idx], tmpobj)
        elif dc==1:
            self.update()
            gui.afni(self, self.results[idx], tmpobj)
        else:
            self.logger.debug('Wrong dataclass value [dc = 0 or 1 but {} is given]'.format(dc))
            methods.raiseerror(messages.Errors.InputValueError, '')

    @property
    def path(self):
        return self._path

    def ext(self):
        return self._ext

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
        """
        try:
            list_steps = self._get_subpath(self._path)
            for step in self._history.keys():
                if step not in list_steps:
                    del self._history[step]
            n_hist = len(self._history.keys())
            output = zip(range(n_hist), sorted(self._history.keys()))
            return dict(output)
        except:
            self.logger.debug('No subfolder founds...')
            pass

    @property
    def reported(self):
        """Listing out reported results
        """
        try:
            list_results = self._get_subpath(self._rpath)
            for step in self._rhistory.keys():
                if step not in list_results:
                    del self._rhistory[step]
            n_hist = len(self._rhistory.keys())
            output = zip(range(n_hist), sorted(self._rhistory.keys()))
            return dict(output)
        except:
            self.rlogger.debug('No subfolder founds...')
            pass

    @property
    def results(self):
        return [self._rhistory[result] for result in self.reported.values()]

    @property
    def steps(self):
        return [self._history[step] for step in self.executed.values()]

    def reset(self):
        """reset subject and session information
        """
        if self.__prj(1).subjects:
            idx_source = 1
            if self.__prj(0).subjects:
                idx_source = 0
                datasubj = set(list(self.__prj(0).subjects))
                procsubj = set(list(self.__prj(1).subjects))
                if datasubj.issubset(procsubj):
                    if not procsubj.issubset(datasubj):
                        try:
                            self._subjects = sorted(self.__prj(1, self.processing).subjects[:])
                            if not self.__prj.single_session:
                                self._sessions = sorted(self.__prj(1, self.processing).sessions[:])
                            idx_source = 1
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
                try:
                    self._subjects = sorted(self.__prj(1, self.processing).subjects[:])
                    if not self.__prj.single_session:
                        self._sessions = sorted(self.__prj(1, self.processing).sessions[:])
                except:
                    self._subjects = sorted(self.__prj(1).subjects[:])
                    if not self.__prj.single_session:
                        self._sessions = sorted(self.__prj(1).sessions[:])
        else:
            idx_source = 0
            self._subjects = sorted(self.__prj(0).subjects[:])
            if not self.__prj.single_session:
                self._sessions = sorted(self.__prj(0).sessions[:])
        self.logger.debug('Proc::Attributes [subjects, sessions] '
                          'are reset to default value [source dataclass index={}].'.format(idx_source))
        self.logger.debug('Proc::Subject is defined as [{}]'.format(",".join(self._subjects)))
        if self._sessions:
            self.logger.debug('Proc::Session is defined as [{}]'.format(",".join(self._sessions)))
        else:
            self.logger.debug('Proc::SingleSession')

    def init_proc(self):
        """Initiate process folder
        """
        self.reset() # correct subject and session information
        methods.mkdir(self._path, self._rpath)
        self.logger.debug('Proc::Initiating instance {0}'.format(self.processing))
        self._history = self._check_history(self._path, self._history)
        self._rhistory = self._check_history(self._rpath, self._rhistory)
        return self._path

    def _check_history(self, path, history_obj, name='.history'):
        history = os.path.join(path, name)
        if os.path.exists(history):
            with open(history, 'rb') as f:
                history_obj = pickle.load(f)
            self.logger.debug("Subfolder history file is loaded".format(history))
        else:
            self._save_history(path, history_obj)
        return history_obj

    def _save_history(self, path, history_obj, name='.history'):
        history = os.path.join(path, name)
        with open(history, 'wb') as f:
            pickle.dump(history_obj, f)
        self.logger.debug("Subfolder history file '{0}' is saved".format(history))
