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
        if tag:
            name = "{}-{}".format(name, tag)
        self._path = os.path.join(self.prj.path, self.prj.ds_type[1], name)
        self._rpath = os.path.join(self.prj.path, self.prj.ds_type[2], name)
        self._processing = name

        # Initiate logger
        if logging:
            self.logger = methods.get_logger(os.path.dirname(self._path), name)

        # Define default arguments
        self._subjects = None
        self._sessions = None
        self._history = {}
        self._rhisroty = {}
        self._tempfiles = []
        self._viewer = viewer

        # Initiate
        self.init_proc()
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
            source_hst = self._rhisroty
        if isinstance(input_path, int):
            input_path = source_idx[input_path]
        if input_path in source_exe:
            return source_hst[input_path]
        else:
            return input_path

    def update(self):
        processing_path = os.path.join(self.prj.path,
                                       self.prj.ds_type[1],
                                       self.processing)
        for f in os.listdir(processing_path):
            if f not in self.executed.values():
                self._history[f] = os.path.join(processing_path, f)
        self.save_history()

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

    # def itksnap(self, idx, base_idx=None):
    #     """Launch ITK-snap
    #
    #     :param idx:
    #     :param base_idx:
    #     :return:
    #     """
    #     if notebook_env:
    #         if base_idx:
    #             display(gui.itksnap(self, self.steps[idx], self.steps[base_idx]))
    #         else:
    #             display(gui.itksnap(self, self.steps[idx]))
    #     else:
    #         methods.raiseerror(messages.Errors.InsufficientEnv, 'This method only works on Jupyter Notebook')

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
            methods.raiseerror(messages.Errors.InputValueError, '')

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
        try:
            exists = [d for d in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, d))]
            for step in self._history.keys():
                if step not in exists:
                    del self._history[step]
            n_hist = len(self._history.keys())
            output = zip(range(n_hist), sorted(self._history.keys()))
            return dict(output)
        except:
            pass

    @property
    def reported(self):
        """Listing out reported results

        :return:
        """
        try:
            exists = dict([(d, os.path.join(self._rpath, d)) for d in os.listdir(self._rpath) \
                           if os.path.isdir(os.path.join(self._rpath, d))])
            self._rhisroty = exists
            output = [(i, e) for i, e in enumerate(sorted(exists.keys()))]
            return dict(output)
        except:
            pass

    @property
    def results(self):
        return [self._rhisroty[result] for result in self.reported.values()]

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
                try:
                    self._subjects = sorted(self.__prj(1, self.processing).subjects[:])
                    if not self.__prj.single_session:
                        self._sessions = sorted(self.__prj(1, self.processing).sessions[:])
                except:
                    self._subjects = sorted(self.__prj(1).subjects[:])
                    if not self.__prj.single_session:
                        self._sessions = sorted(self.__prj(1).sessions[:])
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

    def save_history(self):
        history = os.path.join(self._path, '.proc_hisroty')
        with open(history, 'w') as f:
            pickle.dump(self._history, f)
        self.logger.info("Proc::History file '{0}' is saved".format(history))
