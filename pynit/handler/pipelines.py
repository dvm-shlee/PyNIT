import sys
import os
import json
from ..core import methods
from ..core import messages
from .project import Project
from .process import Process
from ..pipelines import pipelines
from shutil import copy

jupyter_env = False
try:
    if len([key for key in sys.modules.keys() if 'ipykernel' in key]):
        from ipywidgets.widgets import HTML as title
        from IPython.display import display
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
    def __init__(self, prj_path, tmpobj, parallel=True, logging=True, viewer='itksnap'):
        """Initiate class

        :param prj_path:
        :param tmpobj:
        :param parallel:
        :param logging:
        """

        # Define default attributes
        self._prjobj = Project(prj_path)
        self._proc = None
        self._tmpobj = tmpobj
        self._parallel = parallel
        self._logging = logging
        self.selected = None
        self.preprocessed = None
        self._viewer = viewer

        # Print out project summary
        print(self._prjobj.summary)

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
        self._prjobj.reload()
        if isinstance(pipeline, int):
            pipeline = self.avail[pipeline]
        if pipeline in self.avail.values():
            self._proc = Process(self._prjobj, pipeline, parallel=self._parallel,
                                 logging=self._logging, viewer=self._viewer)
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
        display(title(value='---=[[[ Move subject to group folder ]]]=---'))
        self.initiate(o_pipe_id, listing=False, **kwargs)
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
        self.help(o_pipe_id)

    @property
    def executed(self):
        """Listing out executed steps

        :return:
        """
        return self._proc.executed