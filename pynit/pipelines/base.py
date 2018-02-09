import json
import os
from shutil import copy
from pynit.tools import messages
from pynit.tools import methods
from pynit.handler.project import Project
from pynit.pipelines import pipelines
from pynit.process import Process
from ..tools import progressbar, display, clear_output, HTML as title, display_html


#TODO: option for logging need to be subdivided, error messages need to be more clear
class Pipelines(object):
    """ Pipeline handler

    This class is the major features of PyNIT project (for most of general users)
    You can either use default pipeline packages we provide or load custom designed pipelines
    """
    def __init__(self, prj_path, tmpobj, logging=True, viewer='itksnap', **kwargs):
        """Initiate class

        :param prj_path:    Project path
        :param tmpobj:      Template image
        :param logging:     generate log file (default=True)
        :param viewer:      viewer tool to edit ROI, this feature only tested on Mac (default='itksnap')
        :param ds_ref:      Data class reference (default='NIRAL')
        :param img_format:  Image format reference (default='NifTi-1')
        :type prj_path:     str
        :type tmpobj:       pynit.Template
        :type logging:      bool
        :type viewer:       str
        :type ds_ref:       str
        :type img_format:   str
        """

        # Define default attributes
        self.__prj = Project(prj_path, **kwargs)
        self._procobj = Process
        self._pipeobj = pipelines
        self._tmpobj = tmpobj
        self._logging = logging
        self._reset()
        # self.preprocessed = None
        self._viewer = viewer

        # Print out project summary
        print(self.__prj.summary)

        # Print out available pipeline packages
        avails = ["\t{} : {}".format(*item) for item in self.avail.items()]
        output = ["\nList of available packages:"] + avails
        print("\n".join(output))

    @property
    def avail(self):
        pipes = [pipe for pipe in dir(self._pipeobj) if 'PipeTemplate' not in pipe if '__' not in pipe if pipe[0] != '_']
        n_pipe = len(pipes)
        output = dict(zip(range(n_pipe), pipes))
        return output

    #TODO: method to clean pipeline steps folder need to be provided

    def unload(self):
        """ Unload all plugin, and use default pipelines and processes
        """
        self._reset()
        self._procobj = Process
        self._pipeobj = pipelines
        print('The plugins are unloaded.\n')
        # Print out available pipeline packages
        avails = ["\t{} : {}".format(*item) for item in self.avail.items()]
        output = ["List of available packages:"] + avails
        print("\n".join(output))

    def _reset(self):
        """ Reset pipeline
        """
        self._proc = None
        self.selected = None

    def load(self, proc=None, pipe=None, reset=False):
        """ Load plugin for custom-coded pipelines and processes.
        If you want more detail information about this plugin feature,
        please visit out documentation.

        :param proc:    custom-coded processes (python script)
        :param pipe:    custom-coded pipelines (python script)
        :type proc:     str
        :type pipe:     str
        """
        self._reset()
        import imp
        if proc:
            self._procobj = imp.load_source('Process', proc).Process
            print('The process plugin has been successfully loaded.\n')
        if pipe:
            self._pipeobj = imp.load_source('', pipe)
            print('The pipeline plugin has been successfully loaded.\n')
        del imp
        # Print out available pipeline packages
        avails = ["\t{} : {}".format(*item) for item in self.avail.items()]
        output = ["List of available packages:"] + avails
        print("\n".join(output))

    def initiate(self, package_id, verbose=False, listing=True, tag=None, **kwargs):
        """Initiate package

        :param package_id:  Id code for package to initiate
        :param verbose:     Printing out the help of initiating package
        :param kwargs:      Input parameters for initiating package
        :type package_id:   int
        :type verbose:      bool
        :type kwargs:       key=value pairs
        """
        self.__prj.reload()
        if isinstance(package_id, int):
            package_id = self.avail[package_id]
        if package_id in self.avail.values():
            self._proc = self._procobj(self.__prj, package_id, tag=tag, logging=self._logging, viewer=self._viewer)
            command = 'self.selected = self._pipeobj.{}(self._proc, self._tmpobj'.format(package_id)
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
            display_html("The package '{}' is initiated.<br>"
                         "Please double check if all parameters are "
                         "correctly provided before run this pipline".format(package_id))
            avails = ["\t{} : {}".format(*item) for item in self.selected.avail.items()]
            output = ["List of available pipelines:"] + avails
            print("\n".join(output))

    def set_param(self, **kwargs):
        """Set parameters

        :param kwargs:      Input parameters for current initiated package
        """
        if self.selected:
            for key, value in kwargs.items():
                if hasattr(self.selected, key):
                    setattr(self.selected, key, value)
                    self.update()
                else:
                    print(key)
                    methods.raiseerror(messages.Errors.KeywordError, '{} is not available keyword for this project')
        else:
            methods.raiseerror(messages.Errors.InitiationFailure, 'Pipeline package is not specified')

    def get_param(self):
        if self.selected:
            return dict([(param, getattr(self.selected, param)) for param in dir(self.selected) if param[0] != '_'
                         if 'pipe_' not in param if param not in ['avail', 'proc', 'tmpobj', 'update']])
        else:
            return None

    def afni(self, idx, dc=0):
        """

        :param idx:
        :param dc:
        :return:
        """
        self._proc.afni(idx, tmpobj=self._tmpobj, dc=dc)

    def viewer(self, idx, viewer=None):
        """ Launch default viewer

        :param idx:
        :param dc:
        :return:
        """
        if not viewer:
            viewer = self._viewer
        self._proc.image_viewer(idx, viewer=viewer)

    def help(self, idx):
        """ Print help function

        :param idx: index of available pipeline package
        :type idx: int
        :return:
        """
        selected = None
        if isinstance(idx, int):
            idx = self.avail[idx]
        if idx in self.avail.values():
            command = 'selected = self._pipeobj.{}(self._proc, self._tmpobj)'.format(idx)
            exec(command)
            print(selected.__init__.__doc__)

    def inspect(self, idx): #TODO: Test on linux environment
        """  Inspect pipeline packages

        :param idx: index of available pipeline package
        :type idx: int
        :return:
        """
        selected = []
        if isinstance(idx, int):
            idx = self.avail[idx]
        if idx in self.avail.values():
            import inspect
            command = 'selected = inspect.getsourcelines(self._pipeobj.{})[0]'.format(idx)
            exec(command)
            i = 0
            for line in selected:
                if '"""' in line:
                    i += 1
                else:
                    if i%2 == 0:
                        if '_display' in line:
                            pass
                        else:
                            print(line.split('\n')[0])
                    else:
                        pass
            del inspect

    def run(self, idx, **kwargs):
        """Execute selected pipeline

        :param idx: index of available pipeline
        :type idx: int
        :return:
        """
        self.set_param(**kwargs)
        display(title('---=[[[ Running "{}" pipeline ]]]=---'.format(self.selected.avail[idx])))
        exec('self.selected.pipe_{}()'.format(self.selected.avail[idx]))

    def update(self):
        """ Update history
        """
        proc = self._proc
        processing_path = os.path.join(proc.prj.path,
                                       proc.prj.ds_type[1],
                                       proc.processing)
        for f in os.listdir(processing_path):
            if f not in self.executed.values():
                self._proc._history[f] = os.path.join(processing_path, f)
        self._proc.save_history()

    def get_proc(self):
        if self._proc:
            return self._proc
        else:
            methods.raiseerror(messages.Errors.PackageUpdateFailure, 'Pipeline package is not initiated')

    def get_prj(self):
        return self.__prj

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
                        listing=True, help=False, **kwargs):
        """Organizing groups using given filter for applying 2nd level analysis
        In terms of the 'filters', here is two types that you have to distinguish,
        First type is group filter to classify data into group, it can be defined as below
        :example:
        case 1. Assume that you have a subjects of 'sub-e01', 'sub-e02' as an experimental group
                and 'sub-c01', 'sub-c02' as an control group, and have two sessions for each saline injection
                and drug injection ('ses-sal' and 'ses-drg' respectively). Finally, the files you want to
                analyze has 'task-pharm' tag as filename.
                To design filter, you you can define group filters as below

        >> exp_group = ['sub-e01', 'sub-e02']
        >> con_group = ['sub-c01', 'sub-c02']
        >> sess = ['ses-sal', 'ses-drg']
        >> filename_filters = dict(file_tag=['task-pharm'])
        >> group_filters = dict(exp_sal= [ exp_group, sess[0], filename_filters ],
                                exp_drg= [ exp_group, sess[1], filename_filters ],
                                con_sal= [ con_group, sess[0], filename_filters ],
                                con_drg= [ con_group, sess[1], filename_filters ],)

        Second type is filename filter which is widely used in PyNIT. This filter, you

        :param origin:          index of package that subjects data are located
        :param target:          index of package that groups need to be organized
        :param step_id:         step ID that contains preprocessed subjects data
        :param group_filters:   Filters to provide components of group. Please provide as below shape
                                e.g. dict(group1=[list(subj_id,..),
                                                  list(sess_id,..),
                                                  dict(file_tag=.., ignore=..)],
                                          group2=[list(...), list(...), filename_filter]))
        :param cbv:             if CBV correction needed, put step ID of preprocessed MION infusion image
                                (Default=None)
        :param option_filters:  While running pipeline, only the files inside the package folder are used.
                                (e.g. 'A_fMRI_preprocess'). in case you want to use other derived data
                                or meta data such as physiological records, estimated motion paradigm,
                                you should provide option_filters to move together with your image data.
                                In case you want to use this filter, please provide as below shape
                                e.g. dict(step_id=filename_filters,
                                          step_id=filename_filters,...)
                                step id can be 1) DataType is source folder, or 2) index of pipeline step
                                (Default=None)
        :param kwargs:          Additional option to initiate pipeline package
                                To provide this options please check help document on the package
        :type origin:           int
        :type target:           int
        :type step_id:          int
        :type group_filters:    dict(key=list(list, list, dict), key=list(list, list, dict),...)
        :type cbv:              int
        :type option_filters:   dict(key=dict(), key=dict(),...)
        :type kwargs:           key=value, key=value, ...
        """
        display(title('---=[[[ Move subject to group folder ]]]=---'))
        self.initiate(target, listing=False, **kwargs)
        input_proc = Process(self.__prj, self.avail[origin])
        init_path = self.__init_path('GroupOrganizing')
        groups = sorted(group_filters.keys())
        oset = dict()

        def merge_filters(g_filter, o_filter):
            output = dict()
            try:
                for k, v in g_filter.items():
                    # Check if the same filters are used for option
                    if k in o_filter.keys():
                        pass
                    else:
                        output[k] = v
            except:
                pass
            for k, v in o_filter.items():
                output[k] = v
            return output

        for group in progressbar(sorted(groups), desc='Subjects'):
            grp_path = os.path.join(init_path, group)
            methods.mkdir(grp_path)

            if self.__prj.single_session: # If dataset is single session
                if group_filters[group][2]:
                    dset = self.__prj(1, input_proc.processing, input_proc.executed[step_id],
                                      *group_filters[group][0], **group_filters[group][2])
                else:
                    dset = self.__prj(1, input_proc.processing, input_proc.executed[step_id],
                                      *group_filters[group][0])

                if option_filters: # If option filters are provided
                    for i, id in enumerate(option_filters.keys()):
                        # Option filter integration
                        updated_filters = merge_filters(group_filters[group][2], option_filters[id])
                        if isinstance(id, int):
                            oset[i] = self.__prj(1, input_proc.processing, input_proc.executed[id],
                                                 *group_filters[group][0], **updated_filters)
                        elif isinstance(id, str):
                            if id in list(set(self.__prj.df.DataType)):
                                oset[i] = self.__prj(1, input_proc.processing, id,
                                                     *group_filters[group][0], **updated_filters)
            else: # multi-session data
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
                        updated_filters = merge_filters(group_filters[group][2], option_filters[id])
                        if isinstance(id, int):
                            oset[i] = self.__prj(1, input_proc.processing, input_proc.executed[id],
                                                 *(group_filters[group][0] + group_filters[group][1]),
                                                 **updated_filters)
                        elif isinstance(id, str):
                            if id in list(set(self.__prj.df.DataType)):
                                oset[i] = self.__prj(0, input_proc.processing, id,
                                                     *(group_filters[group][0] + group_filters[group][1]),
                                                     **updated_filters)
                            else:
                                pass #TODO: error message, and log something
                        else:
                            pass #TODO: error message, and log something

            # Copy selected files into the group folder
            for i, finfo in dset: # Preprocessed dataset
                output_path = os.path.join(grp_path, finfo.Filename)
                if os.path.exists(output_path):
                    pass
                else:
                    copy(finfo.Abspath, os.path.join(grp_path, finfo.Filename))
                    if cbv: # CBV infusion files
                        if self.__prj.single_session:
                            cbv_file = self.__prj(1, input_proc.processing, input_proc.executed[cbv], finfo.Subject)
                        else:
                            cbv_file = self.__prj(1, input_proc.processing, input_proc.executed[cbv],
                                                  finfo.Subject, finfo.Session)
                        with open(methods.splitnifti(output_path)+'.json', 'wb') as f:
                            json.dump(dict(cbv=cbv_file[0].Abspath), f)
            if option_filters: # Regressors or other metadata
                for prj in oset.values():
                    for i, finfo in prj:
                        output_path = os.path.join(grp_path, finfo.Filename)
                        if os.path.exists(output_path):
                            pass #TODO: log something
                        else:
                            copy(finfo.Abspath, os.path.join(grp_path, finfo.Filename))

        self._proc._subjects = groups[:]
        self._proc._history[os.path.basename(init_path)] = init_path
        self._proc.save_history()
        self._proc.prj.reload()
        clear_output()
        display_html("The package '{}' is initiated.<br>"
                     "Please double check if all parameters are "
                     "correctly provided before run this pipline".format(self.avail[target]))
        if help:
            self.help(target)
        if listing:
            avails = ["\t{} : {}".format(*item) for item in self.selected.avail.items()]
            output = ["List of available pipelines:"] + avails
            print("\n".join(output))

    @property
    def executed(self):
        """Listing out executed steps

        :return:
        """
        try:
            return self._proc.executed
        except:
            return None