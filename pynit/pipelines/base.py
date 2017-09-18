import json
import os
import sys
from shutil import copy

from pynit.tools import messages
from pynit.tools import methods
from pynit.handler.project import Project
from pynit.pipelines import pipelines
from pynit.process import Process

jupyter_env = False
try:
    if len([key for key in sys.modules.keys() if 'ipykernel' in key]):
        from ipywidgets.widgets import HTML as title
        from IPython.display import display, clear_output
        from tqdm import tqdm_notebook as progressbar
        jupyter_env = True
    else:
        from tqdm import tqdm as progressbar
except:
    pass


class Pipelines(object):
    """ Pipeline handler

    This class is the major features of PyNIT project (for most of general users)
    You can either use default pipeline packages we provide or load custom designed pipelines
    """
    def __init__(self, prj_path, tmpobj, logging=True, viewer='itksnap'):
        """Initiate class

        :param prj_path:
        :param tmpobj:
        :param logging:
        """

        # Define default attributes
        self.__prj = Project(prj_path)
        self._proc = None
        self._tmpobj = tmpobj
        self._logging = logging
        self.selected = None
        self.preprocessed = None
        self._viewer = viewer

        # Print out project summary
        print(self.__prj.summary)

        # Print out available pipeline packages
        avails = ["\t{} : {}".format(*item) for item in self.avail.items()]
        output = ["\nList of available packages:"] + avails
        print("\n".join(output))

    @property
    def avail(self):
        pipes = [pipe for pipe in dir(pipelines) if 'PipeTemplate' not in pipe if '__' not in pipe]
        n_pipe = len(pipes)
        output = dict(zip(range(n_pipe), pipes))
        return output

    def initiate(self, pipeline, verbose=False, listing=True, **kwargs):
        """Initiate pipeline

        :param pipeline:
        :param verbose:
        :param kwargs:
        :return:
        """
        self.__prj.reload()
        if isinstance(pipeline, int):
            pipeline = self.avail[pipeline]
        if pipeline in self.avail.values():
            self._proc = Process(self.__prj, pipeline, logging=self._logging, viewer=self._viewer)
            command = 'self.selected = pipelines.{}(self._proc, self._tmpobj'.format(pipeline)
            if kwargs:
                command += ', **{})'.format('kwargs')
            else:
                command += ')'
            exec(command)
        else:
            methods.raiseerror(messages.PipelineNotSet, "Incorrect package is selected")
        if verbose:
            print(self.selected.__init__.__doc__)
        if listing:
            avails = ["\t{} : {}".format(*item) for item in self.selected.avail.items()]
            output = ["List of available pipelines:"] + avails
            print("\n".join(output))

    def set_param(self, **kwargs):
        """Set additional parameters

        :param kwargs:
        :return:
        """
        if self.selected:
            for key, value in kwargs.items():
                if hasattr(self.selected, key):
                    if isinstance(value, str):
                        command = 'self.selected.{} = "{}"'.format(key, value)
                    else:
                        command = 'self.selected.{} = {}'.format(key, value)
                    exec(command)
                else:
                    methods.raiseerror(messages.Errors.KeywordError, '{} is not available keyword for this project')
        else:
            methods.raiseerror(messages.Errors.InitiationFailure, 'Pipeline package is not specified')

    def afni(self, idx):
        self._proc.afni(idx, self._tmpobj)

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

    def run(self, idx, **kwargs):
        """Execute selected pipeline

        :param idx:
        :return:
        """
        display(title(value='---=[[[ Running "{}" pipeline ]]]=---'.format(self.selected.avail[idx])))
        exec('self.selected.pipe_{}(**kwargs)'.format(self.selected.avail[idx]))

    def update(self):
        proc = self._proc
        processing_path = os.path.join(proc.prj.path,
                                       proc.prj.ds_type[1],
                                       proc.processing)
        for f in os.listdir(processing_path):
            if f not in self.executed.values():
                self._proc._history[f] = os.path.join(processing_path, f)
        self._proc.save_history()

    # def load(self, pipeline):
    #     """Load custom pipeline
    #
    #     :param pipeline:
    #     :return:
    #     """
    #     pass

    def __init_path(self, name):
        """Initiate path

        :param name: str
        :return: str
        """
        proc = self._proc


        def get_step_name(proc, step, verbose=None):
            processing_path = os.path.join(proc.prj.path,
                                           proc.prj.ds_type[1],
                                           proc.processing)
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
                    print('The pipeline [{pipeline}] is initiated'.format(pipeline=proc.processing))
                return "_".join([str(1).zfill(3), step])

        if proc._processing:
            path = get_step_name(proc, name)
            path = os.path.join(proc.prj.path, proc.prj.ds_type[1], proc._processing, path)
            methods.mkdir(path)
            return path
        else:
            methods.raiseerror(messages.Errors.InitiationFailure, 'Error on initiating step')

    def group_organizer(self, origin, target, step_id, group_filters, option_filters=None, cbv=None,
                        **kwargs):
        """Organizing groups for 2nd level analysis

        :param origin:      Pipeline id of original location
        :param target:      Pipeline id of destination
        :param step_id:     step ID of processed folder need to be use for organizing group
        :param group_filters: Group filters, (e.g. dict(group=[[subj_id],[sess_id],filters], ...))
        :param option_filters:  dict(step_id=filters, ...)
        :param cbv:         If CBV image, put step ID
        :param kwargs:      ADditional option for initiating pipeline
        :return:
        """
        display(title(value='---=[[[ Move subject to group folder ]]]=---'))
        self.initiate(target, listing=False, **kwargs)
        input_proc = Process(self.__prj, self.avail[origin])
        init_path = self.__init_path('GroupOrganizing')
        groups = sorted(group_filters.keys())
        oset = dict()
        for group in progressbar(sorted(groups), desc='Subjects'):
            grp_path = os.path.join(init_path, group)
            methods.mkdir(grp_path)
            if self.__prj.single_session:
                if group_filters[group][2]:
                    dset = self.__prj(1, input_proc.processing, input_proc.executed[step_id],
                                      *group_filters[group][0], **group_filters[group][2])
                else:
                    dset = self.__prj(1, input_proc.processing, input_proc.executed[step_id],
                                      *group_filters[group][0])
                if option_filters:
                    for i, id in enumerate(option_filters.keys()):
                        oset[i] = self.__prj(1, input_proc.processing, input_proc.executed[id],
                                             *group_filters[group][0], **option_filters[id])
            else:
                grp_path = os.path.join(init_path, group, 'Files')
                methods.mkdir(grp_path)
                if group_filters[group][2]:
                    dset = self.__prj(1, input_proc.processing, input_proc.executed[step_id],
                                      *(group_filters[group][0] + group_filters[group][1]),
                                      **group_filters[group][2])
                else:
                    dset = self.__prj(1, input_proc.processing, input_proc.executed[step_id],
                                      *(group_filters[group][0] + group_filters[group][1]))
                if option_filters:
                    oset = dict()
                    for i, id in enumerate(option_filters.keys()):
                        oset[i] = self.__prj(1, input_proc.processing, input_proc.executed[id],
                                             *(group_filters[group][0] + group_filters[group][1]),
                                             **option_filters[id])
            for i, finfo in dset:
                output_path = os.path.join(grp_path, finfo.Filename)
                if os.path.exists(output_path):
                    pass
                else:
                    copy(finfo.Abspath, os.path.join(grp_path, finfo.Filename))
                    if cbv:
                        if self.__prj.single_session:
                            cbv_file = self.__prj(1, input_proc.processing, input_proc.executed[cbv], finfo.Subject)
                        else:
                            cbv_file = self.__prj(1, input_proc.processing, input_proc.executed[cbv],
                                                  finfo.Subject, finfo.Session)
                        with open(methods.splitnifti(output_path)+'.json', 'wb') as f:
                            json.dump(dict(cbv=cbv_file[0].Abspath), f)
            if option_filters:
                for prj in oset.values():
                    for i, finfo in prj:
                        output_path = os.path.join(grp_path, finfo.Filename)
                        if os.path.exists(output_path):
                            pass
                        else:
                            copy(finfo.Abspath, os.path.join(grp_path, finfo.Filename))

        self._proc._subjects = groups[:]
        self._proc._history[os.path.basename(init_path)] = init_path
        self._proc.save_history()
        self._proc.prj.reload()
        clear_output()
        self.help(target)

    @property
    def executed(self):
        """Listing out executed steps

        :return:
        """
        return self._proc.executed