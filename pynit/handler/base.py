import os
import re
import sys
import numpy as np
import json
import multiprocessing
from multiprocessing.pool import ThreadPool
from nibabel import Nifti1Image, affines
from ..tools import methods, messages
from ..tools.visualizers import check_invert, apply_invert
from ..tools.visualizers import BrainPlot
from collections import namedtuple
import datetime
from time import sleep
from ..tools import display, clear_output, progressbar


########################################################################################################################
# Reference for PyNIT
########################################################################################################################
class Reference(object):
    """Class of reference informations for image processing and data analysis
    """
    img = {'NifTi-1':           ['.nii', '.nii.gz'],
           'ANALYZE7.5':        ['.img', '.hdr'],
           'AFNI':              ['.BRIK', '.HEAD'],
           'Shihlab':           ['.sdt', '.spr'],
           'Nrrd':              ['.nrrd', '.nrdh'],
           'PNG':               ['.png'],
           }
    txt = {'Common':            ['.txt', '.cvs', '.tvs'],
           'Mictosoft':         ['.xlsx', '.xls'],
           'AFNI':              ['.1D'],
           'MATLAB':            ['.mat'],
           'Slicer_Transform':  ['.tfm'],
           'JSON':              ['.json'],
           }
    data_structure = {'NIRAL': ['Data', 'Processing', 'Results'],
                      }

    def __init__(self, *args):
        if not len(args):
            raise AttributeError
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
        output = '{}\n{}\n{}\n{}\n{}'.format(title, '-'*len(title), img, txt, ds)
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


########################################################################################################################
# Base function and class for Image object
########################################################################################################################
def reset_orient(imageobj, affine):
    """ Reset to the original scanner space

    :param imageobj:
    :param affine:
    :return:
    """
    imageobj.set_qform(affine)
    imageobj.set_sform(affine)
    imageobj.header['sform_code'] = 0
    imageobj.header['qform_code'] = 1
    imageobj._affine = affine


def swap_axis(imageobj, axis1, axis2):
    """ Swap axis of image object

    :param imageobj:
    :param axis1:
    :param axis2:
    :return:
    """
    resol, origin = affines.to_matvec(imageobj.get_affine())
    resol = np.diag(resol).copy()
    origin = origin
    imageobj._dataobj = np.swapaxes(imageobj._dataobj, axis1, axis2)
    resol[axis1], resol[axis2] = resol[axis2], resol[axis1]
    origin[axis1], origin[axis2] = origin[axis2], origin[axis1]
    affine = affines.from_matvec(np.diag(resol), origin)
    reset_orient(imageobj, affine)


def down_reslice(imageobj, ac_slice, ac_loc, slice_thickness, total_slice, axis=2):
    """ Reslicing

    :param imageobj:
    :param ac_slice:
    :param ac_loc:
    :param slice_thickness:
    :param total_slice:
    :param axis:
    :return:
    """
    data = np.asarray(imageobj.dataobj)
    resol, origin = affines.to_matvec(imageobj.affine)
    resol = np.diag(resol).copy()
    scale = float(slice_thickness) / resol[axis]
    resol[axis] = slice_thickness
    idx = []
    for i in range(ac_loc):
        idx.append(ac_slice - int((ac_loc - i) * scale))
    for i in range(total_slice - ac_loc):
        idx.append(ac_slice + int(i * scale))
    imageobj._dataobj = data[:, :, idx]
    affine, origin = affines.to_matvec(imageobj.affine[:, :])
    affine = np.array(np.diag(affine))
    affine[axis] = slice_thickness
    affine_mat = affines.from_matvec(np.diag(affine), origin)
    imageobj._affine = affine_mat
    imageobj.set_qform(affine_mat)
    imageobj.set_sform(affine_mat)
    imageobj.header['sform_code'] = 0
    imageobj.header['qform_code'] = 1


def crop(imageobj, **kwargs):
    """ Crop

    :param imageobj:
    :param kwargs:
    :return:
    """
    x = None
    y = None
    z = None
    t = None
    for arg in kwargs.keys():
        if arg == 'x':
            x = kwargs[arg]
        if arg == 'y':
            y = kwargs[arg]
        if arg == 'z':
            z = kwargs[arg]
        if arg == 't':
            t = kwargs[arg]
        else:
            pass
    if x:
        if (type(x) != list) and (len(x) != 2):
            raise TypeError
    else:
        x = [None, None]
    if y:
        if (type(y) != list) and (len(y) != 2):
            raise TypeError
    else:
        y = [None, None]
    if z:
        if (type(z) != list) and (len(z) != 2):
            raise TypeError
    else:
        z = [None, None]
    if t:
        if (type(t) != list) and (len(t) != 2):
            raise TypeError
    else:
        t = [None, None]
    if len(imageobj.shape) == 3:
        imageobj._dataobj = imageobj._dataobj[x[0]:x[1], y[0]:y[1], z[0]:z[1]]
    if len(imageobj.shape) == 4:
        imageobj._dataobj = imageobj._dataobj[x[0]:x[1], y[0]:y[1], z[0]:z[1], t[0]:t[1]]


def set_center(imageobj, corr):
    """ Applying center corrdinate to the object
    """
    resol, origin = affines.to_matvec(imageobj.affine[:, :])
    affine = affines.from_matvec(resol, corr)
    reset_orient(imageobj, affine)



class ImageObj(Nifti1Image):
    """ ImageObject for PyNIT
    """
    # def __init__(self):
    #     super(ImageObj, self).__init__()

    def show(self, **kwargs):
        """ Plotting slice of the object
        """
        BrainPlot.slice(self, **kwargs)

    def mosaic(self, *args, **kwargs):                          #TODO: update needed
        """ Mosaic view for the object
        """
        fig = BrainPlot.mosaic(self, *args, **kwargs)

    def swap_axis(self, axis1, axis2):
        """ Swap input axis with given axis of the object
        """
        swap_axis(self, axis1, axis2)

    def flip(self, **kwargs):
        invert = check_invert(kwargs)
        self._dataobj = apply_invert(self._dataobj, *invert)

    def crop(self, **kwargs):
        crop(self, **kwargs)

    def reslice(self, ac_slice, ac_loc, slice_thickness, total_slice, axis=2):
        """ Reslice the image with given number of slice and slice thinkness

        :param ac_slice: int
            Slice location of anterior commissure in original image
        :param ac_loc: int
            The slice number of anterior commissure want to be in resliced image
        :param slice_thickness:
            Desired slice thickness for re-sliced image
        :param total_slice:
            Desired total number of slice for  re-sliced image
        :param axis:
            Axis want to be re-sliced
        :return:
        """
        down_reslice(self, ac_slice, ac_loc, slice_thickness, total_slice, axis)

    def save_as(self, filename, quiet=False):
        """ Save as a new file with current affine information
        """
        self.header['sform_code'] = 0
        self.header['qform_code'] = 1
        self.to_filename(filename)
        if not quiet:
            print("NifTi1 format image is saved to '{}'".format(filename))

    def padding(self, low, high, axis):
        dataobj = self._dataobj[...]
        dataobj = np.swapaxes(dataobj, axis, 2)
        shape = list(dataobj.shape[:])
        shape[2] = low
        lower_pad = np.zeros(shape)
        shape[2] = high
        higher_pad = np.zeros(shape)
        dataobj = np.concatenate((lower_pad, dataobj, higher_pad), axis=2)
        self._dataobj = np.swapaxes(dataobj, axis, 2)

    def check_reg(self, imageobj, scale=10, **kwargs):          #TODO: update needed
        fig = BrainPlot.check_reg(imageobj, self, scale=scale, norm=True, **kwargs)

    def check_mask(self, maskobj, scale=15, **kwargs):          #TODO: update needed
        fig = BrainPlot.check_mask(self, maskobj, scale=scale, **kwargs)

    @property
    def affine(self):
        return self._affine


########################################################################################################################
# Base function and class for Processor
########################################################################################################################
_dset = namedtuple('Dataset', ['name', 'idx'])
_vset = namedtuple('Variable', ['name', 'value', 'type'])
_gset = namedtuple('Group', ['name', 'args', 'kwargs'])
_oset = namedtuple('OutputParam', ['name', 'code', 'type', 'ext', 'prefix'])
_fltr = namedtuple('Filters', ['name', 'code'])
_cmds = namedtuple('Command', ['name', 'command', 'nscode', 'type', 'level'])


class BaseProcessor(object):
    """ This class is designed to be used as template of Step and Report handler

    --following namespaces are pre-assigned
     title: folder name of basedir for this processor
     subj: folder name of subject
     sess: folder name of sessions

    --following variable are pre-assinged and can used when you set customizing python code
     i: this variable is used for iterating internally
     idx: this variable is used for indexing input at subject (or session in case of multi-session project) internally
     output_checker: this variable is used to check if the output file generated previously
     flist_checker: this variable is used to check list of file already processed
     stdout_collector:
     stdout_basket:
     out:
     err:

    """
    def __init__(self, procobj, n_thread='max'):
        """ Initiating class

        :param procobj: Activated Process instance
        """
        self.__init_container(procobj)
        self.set_parallel(n_thread=n_thread)

    # for debuging purpose, uncomment belows
    # @property
    # def inputs(self):
    #     return [self.__mainset] + self.__sideset
    #
    # @property
    # def cmds(self):
    #     return self.__cmd
    #
    # @property
    # def vars(self):
    #     return self.__var
    #
    # @property
    # def outputs(self):
    #     return self.__output
    #
    # @property
    # def groups(self):
    #     return self.__group
    #
    # @property
    # def imported(self):
    #     return self.__import

    def __init_container(self, procobj):
        """ Initiating containers for objects

        :param procobj: Activated Process instance
        :return:
        """
        # Environment related containers
        self.__proc = procobj
        self._parallel = 1
        self.__prj = procobj.prj
        self.__pipeline = procobj.processing
        self.__import = list()
        self.__message = None
        # Input containers
        self.__mainset = None
        self.__sideset = list()
        self.__group = list()
        self.__multi = list()
        self.__filters = dict(main=None, side=dict(), group=dict(), multi=dict())
        self.__var = list()
        self.__output = list()
        self.__assigned_namespace = ['title', 'subj', 'sess', 'i', 'idx',
                                     'output_checker', 'flist_checker']
        self.__mkdir = list()
        self.__opened_temps = list()
        # Command containers
        self.__cmd = list()
        self.__dc = None
        self.__input_dc = None

    def init_path(self, title, dc=0, verbose=False):
        """ This method checks if the input step had been executed or not.
        If the step already executed, return the name of existing folder,
        if not, the number of order is attached as prefix and will be returned

        :param title:       title of this step
        :param dc:          0-processing step class
                            1-resulting step class
        :param verbose:     print information
        :type title:        str
        :type dc:           inc
        :return:            name of the step
        :rtype:             str
        """
        processing_path = os.path.join(self.__prj.path,
                                       self.__prj.ds_type[dc + 1],
                                       self.__pipeline)
        executed_steps = [f for f in os.listdir(processing_path) if os.path.isdir(os.path.join(processing_path, f))]
        if len(executed_steps):
            overlapped = [old_step for old_step in executed_steps if title in old_step]
            if len(overlapped):
                if verbose:
                    print('Notice: existing path')
                checked_files = []
                for f in os.walk(os.path.join(processing_path, overlapped[0])):
                    checked_files.extend(f[2])
                if len(checked_files):
                    if verbose:
                        print('Notice: Last step path is not empty')
                return os.path.join(processing_path, overlapped[0])
            else:
                return os.path.join(processing_path, "_".join([str(len(executed_steps) + 1).zfill(3), title]))
        else:
            if verbose:
                print('The pipeline [{pipeline}] is initiated'.format(pipeline=self.__pipeline))
            return os.path.join(processing_path, "_".join([str(1).zfill(3), title]))

    def set_logging(self, message):
        return 'self.logger.info({0})'.format(message)

    def set_message(self, message):
        self.__message = message

    def set_parallel(self, n_thread):
        """ Method to initiate parallel computing

        :param n_thread:    Number of thread for parallel computing
        """
        self.__proc.logger.info("Step::n_thread is setted as {}".format(n_thread))
        if isinstance(n_thread, int):
            if n_thread >= 1:
                if n_thread > multiprocessing.cpu_count():
                    self._parallel = multiprocessing.cpu_count()
                else:
                    self._parallel = n_thread
        elif isinstance(n_thread, str):
            if n_thread == 'max':
                self._parallel = multiprocessing.cpu_count()
            else:
                methods.raiseerror(messages.Errors.InputTypeError, 'Wrong parameter')
        else:
            methods.raiseerror(messages.Errors.InputTypeError, 'Wrong parameter')

    def set_input(self, name, path, filters=None, idx=None, type=0, args=None, kwargs=None):
        """ Method to assign namespace of inputs

        :param name:    namespace for input object
        :param path:    input main path
        :param filters: set of filters for input instance
        :param idx:     file index, if this is given, input became static mode
        :param type:    0-main input:       this input type take each file as input
                        1-side input:       this input type take each file as input, but
                                            main input needs to be specified prior
                        2-multi inputs:     this input type takes all files in the subject/session level as inputs
                                            can be used for one-sample t-test
                        3-groups inputs:    this input type takes all files in the step folder
                                            can be used for group-level analysis (ANOVA, so on..)
        """
        # path = self.
        dc, path = self.__check_dataclass(path)
        self.__input_dc = dc
        if type in [0,1,2,3]:
            self.__check_namespace(name)
            if type == 0:
                if self.__mainset:
                    methods.raiseerror(messages.InputObjectError, 'Main input had been assigned already')
                self.__mainset = _dset(name=name, idx=idx)
                self.__filters['main'] = _fltr(name=name,
                                               code=self.__convert_filtercode(dc, path, filters, type))
            elif type == 1:
                self.__sideset.append(_dset(name=name, idx=idx))
                self.__filters['side'][name] = _fltr(name=name,
                                                     code=self.__convert_filtercode(dc, path, filters, type))
            elif type == 2:
                self.__multi.append(_gset(name=name, args=args, kwargs=kwargs))
                self.__filters['multi'][name] = _fltr(name=name,
                                                      code=self.__convert_filtercode(dc, path, filters, type))
            elif type == 3:
                self.__group.append(_gset(name=name, args=args, kwargs=kwargs))
                self.__filters['group'][name] = _fltr(name=name,
                                                      code=self.__convert_filtercode(dc, path, filters, type))
        else:
            methods.raiseerror(messages.Errors.InputTypeError, 'Wrong input type')

    def set_module(self, module, sub=None, rename=None):
        """ Method to import module during steps
        Only one module can be imported, for importing multiple modules
        Use this method multiple time.

        :param module:  any python module installed in your environment
        :param sub:     submodule you want to import from the parent module
                        if you want to import multiple submodules, use list instead
        :param rename:  new name for imported module.
                        if you want to rename submodules, use list instead
        :type module: str
        :type sub: str or list of str
        :type rename: str or list of str
        """
        self.__configure_module(module, sub=sub, rename=rename)

    def set_var(self, name, value, type=0):
        """ Mehtod to assign namespace of variable for the processor function

        :param name:    namespace for input object
        :param value:   input value
        :param type:    0-set variable only works on custom code (defined before enter the loop)
                        1-set variable able to works on command line (defined inside loop)
        """
        if type not in [0, 1]:
            methods.raiseerror(messages.Errors.InputTypeError,
                               'Wrong variable type')
        else:
            self.__check_namespace(name)
            if isinstance(value, str):
                if os.path.exists(value):
                    value = '"{}"'.format(value)
                else:
                    value = "{}".format(value)
            else:
                value = "{}".format(value)
            self.__var.append(_vset(name=name, value=value, type=type))

    def set_output(self, name, level=0, dc=0, ext=None, prefix=None, type=0):
        """ Method to assign output namespace

        :param name:    namespace for output path
        :param level:   0-same level as input,
                        1-one step upper level,
                        2-process root level
        :param dc:      0-processing step class,
                        1-resulting step class
        :param ext:     file extension without dot, this parameter available when type is 0 or 2,
                        if ext is 'remove', filename without extension can be use as output
        :param prefix:  add prefix when type is 0,
                        use this as the name of new folder when type is 2
        :param type:    0-file: use input filename,
                                cannot use this type if input type is group,
                        1-folder: use basedir, prefix parameter cannot be used for this type,
                                if input type is group, prefix need to be used to specify filename
                        2-sub-folder or new-file: generate new folder using prefix,
                                if ext parameter is assigned, then generate new output file instead,
                                if ext is 'remove' then use prefix as prefix like type 0
                                if input type is group, new folder will be generate under process roots
                        3-temporary: generate temporary file, prefix parameter cannot be used for this type,
                                if input type is group, cannot use this type as output
                        4-no output: output will not be defined
        :type name: str
        :type level: int
        :type dc: int
        :type ext: str
        :type prefix: str
        :type type: int
        """
        if self.__dc and dc != self.__dc:
            methods.raiseerror(messages.Errors.InputTypeError,
                               'Each step only allow one output dataclass')
        else:
            self.__check_output_dc(dc, level, type)
            self.__check_output(prefix, ext, type)
            self.__check_namespace(name)
            self.__convert_outputcode(name, level, dc, ext, prefix, type)

    def set_cmd(self, command, name=None, type=0, level=0):
        """ Method to set command for inputs

        :param name:        namespace for output of command
        :param command:     command want to execute
        :param type:        0=local shell
                            1=python methods
                            2=scheduler
        :return:
        """
        for cmd in self.__cmd:
            if command == cmd.command:
                methods.raiseerror(messages.Errors.InputValueError, 'Already assigned in container')
        # if type==0:
        nspace = self.__retreive_namespaces_from_command(command)
        nspace, nscode = self.__convert_namespace(nspace)
        if any(ns not in self.__assigned_namespace for ns in nspace):
            print(nspace, self.__assigned_namespace)
            methods.raiseerror(messages.Errors.InputValueError, 'Namespace are not correctly assigned')
        # else:
        #     nscode=None
        if type not in [0, 1, 2]:
            methods.raiseerror(messages.Errors.InputTypeError,
                               'Wrong command type')
        else:
            self.__proc.logger.info('CMD:{}'.format(command))
            self.__cmd.append(_cmds(name=name, command=command, nscode=nscode, type=type, level=level))

    def reset(self):
        """ Method to reset all containers
        """
        self.__init_container(self.__proc)

    def __check_dataclass(self, path):
        """ Methods to check dataclass of given path

        :param path: datatype or step level path
        :return: dataclass and path
        """
        if os.path.exists(path):
            if self.__prj.ds_type[2] in path:
                dataclass = 2
            else:
                dataclass = 1
            path = methods.path_splitter(path)[-1]
        else:
            dataclass = 0
        return dataclass, path

    def __check_namespace(self, name):
        """ To check if given name is assigned already or not
        If the name already assigned, raise error

        :param name: name for inputs, outputs, variables, and commands for the processor
        """
        if name not in self.__assigned_namespace:
            self.__assigned_namespace.append(name)
        else:
            methods.raiseerror(messages.Errors.InputValueError,
                               '{} is already used in namespace'.format(name))

    def __output_sorter(self):
        """ Methods to classify assigned output based on their type

        :return:
        """
        outputs = dict()
        if self.__output:
            for op in self.__output:
                if op.type not in outputs.keys():
                    outputs[op.type] = dict()
                outputs[op.type][op.name] = [op.ext, op.prefix]
            return outputs
        else:
            return None

    def __check_output(self, prefix, ext, type):
        """ Method to check if the type of assigned output is correct or not

        - Why this method needed?
        In this class, only one major output is available to set, except side output.
        major output can be either file or dir, which is the result file processed from correspond input file
        side output is the output file that generated additionally, such as transform matrix,
        multiple side output can be assigned.

        :param prefix: demanded output prefix
        :param ext: demanded output extension
        :param type: demanded output type
        :return:
        """
        def raise_error():
            methods.raiseerror(messages.Errors.InputTypeError, 'Only one major output can be assigned')
        if self.__group or self.__multi:
            if type == 0:
                if not prefix:
                    methods.raiseerror(messages.Errors.InputTypeError, 'prefix parameter must be provided for this type')
            elif type == 3:
                methods.raiseerror(messages.Errors.InputTypeError, 'cannot use temporary output for this type')
        outputs = self.__output_sorter()
        if outputs:
            if type == 0:
                if (not ext and not prefix): # this case will be indicated as major output
                    # check if the major output previously assigned
                    if len([op for op in outputs[0].keys() if not (outputs[0][op][0] and outputs[0][op][1])]):
                        raise_error()
                    if len([op for op in outputs[0].keys() if (outputs[0][op][0] == 'remove' and not outputs[0][op][1])]):
                        raise_error()
                if any(i in [1,2] for i in outputs.keys()):
                    raise_error()
            elif type == 3: # type 3 is allowable any types except 4
                pass
            elif type == 4: # type 4 only allowed to use it's own (since it not generate output)
                raise_error()
            else:
                if 0 in outputs.keys():
                    if any(prefix is v[1] for k, v in outputs[0].items() if ext == v[0]):
                        # check if there are any overlap with parameter of pre-assigned output
                        raise_error()
                if (type in [1,2]):
                    if len(outputs.keys()): # if any type output was assigned already
                        raise_error()

    def __check_output_dc(self, dc, level, type):
        if type==4:
            self.__dc = 0
        else:
            if dc == 0:
                if level != 0 or (type != 0 and type != 3):
                    methods.raiseerror(messages.Errors.InputTypeError,
                                       'Cannot set level or types during pre-processing step,'+
                                       'Please use dc=1 instead')
            self.__dc = dc

    def __check_output_level(self, level):
        if not self.__multi:
            if not self.__group:
                if self.__proc.sessions:
                    if level == 0:
                        leveldir = 'subj, sess'
                    elif level == 1:
                        leveldir = 'subj'
                    else:
                        leveldir = None
                else:
                    if level == 0:
                        leveldir = 'subj'
                    else:
                        leveldir = None
            else:
                leveldir = None
        else:
            leveldir = 'subj'
        return leveldir

    def __convert_filtercode(self, dataclass, path, filters, type):
        """ Method to convert filter code feasible for Project handler

        :param dataclass: dataclass
        :param path: input path
        :param filters: filters
        :return: filter code for Project handler
        """
        filter_ref = ['ext', 'file_tag', 'ignore']
        group_filter_ref = ['subj', 'sess', 'group']
        if dataclass == 0:
            output_filters = [dataclass, '"{0}"'.format(path)]
        else:
            output_filters = [dataclass, '"{0}"'.format(self.__pipeline), '"{0}"'.format(path)]
        if type in [0, 1]:
            if self.__proc.sessions:
                output_filters.extend(['subj', 'sess'])
            else:
                output_filters.extend(['subj'])
        else:
            if type == 2:
                output_filters.extend(['subj'])
            elif type == 3:
                if isinstance(filters, dict):
                    for k, vs in filters.items():
                        if k in group_filter_ref:
                            if isinstance(vs, list):
                                output_filters.extend(['"{0}"'.format(v) for v in vs])
                            elif isinstance(vs, str):
                                output_filters.append('"{0}"'.format(vs))
                            else:
                                methods.raiseerror(messages.Errors.InputTypeError, 'Value type must be str or list')
                            del(filters[k])
        if isinstance(filters, dict):
            if any(k not in filter_ref for k in filters.keys()):
                methods.raiseerror(messages.Errors.KeywordError, 'Unable filter keyword')
            else:
                kwargs = ['{key}="{value}"'.format(key=k, value=v) for k, v in filters.items() if isinstance(v, str)]
                output_filters.extend(kwargs)
                kwargs = ['{key}={value}'.format(key=k, value=v) for k, v in filters.items() if isinstance(v, list)]
                output_filters.extend(kwargs)
        return ', '.join(map(str, output_filters))

    def __convert_inputcode(self):
        """ Method to convert input code for building custom code

        :return:
        """
        inputcodes = []
        if self.__mainset:
            main_input = self.__mainset
            inputcode = '{0} = self.prj({1})'.format(main_input.name, self.__filters['main'].code)
            if main_input.idx:
                inputcode += '[{}]'.format(main_input.idx)
            inputcodes.append(inputcode)
            if self.__sideset:
                for sip in self.__sideset:
                    inputcode = '{0} = self.prj({1})'.format(sip.name,
                                                               self.__filters['side'][sip.name].code)
                    if sip.idx:
                        inputcode += '[{}]'.format(sip.idx)
                    inputcodes.append(inputcode)
        else:
            if self.__multi:
                inputcodes = []
                for grp in self.__multi:
                    inputcodes.append('{0} = self.prj({1})'.format(grp.name,
                                                                   self.__filters['multi'][grp.name].code))
            else:
                if self.__group:
                    inputcodes = []
                    for grp in self.__group:
                        inputcodes.append('{0} = self.prj({1})'.format(grp.name,
                                                                         self.__filters['group'][grp.name].code))
                else:
                    methods.raiseerror(messages.Errors.InputTypeError, 'Inputs are not sufficiently assigned')
        if self.__var:
            for var in self.__var:
                if var.type == 0:
                    inputcodes.append('{0} = {1}'.format(var.name, var.value))
        return inputcodes


    def __convert_mkdir(self):
        """ Method to convert code for making directory for output
        """
        mk_pathes = []
        if self.__mkdir:
            for op in self.__mkdir:
                if op.type == 0:
                    mk_pathes.append('os.path.dirname({0})'.format(op.name))
                elif op.type == 1:
                    if self.__group or self.__multi:
                        mk_pathes.append('os.path.dirname({0})'.format(op.name))
                    else:
                        mk_pathes.append('{0}'.format(op.name))
                elif op.type == 2:
                    if op.ext:
                        mk_pathes.append('os.path.dirname({0})'.format(op.name))
                    else:
                        mk_pathes.append(op.name)
                else:
                    mk_pathes.append(op.name)
            return ['methods.mkdir({0})'.format(', '.join(mk_pathes))]
        else:
            return None

    def __convert_outputcode(self, name, level, dc, ext, prefix, type):
        """ Method to convert output information to sufficient code for input

        :param level: output level
        :param dc:  output datatype
        :param prefix: output prefix
        :param type: output type
        """
        path = os.path.join(self.__prj.path, self.__prj.ds_type[dc + 1], self.__pipeline)
        if not os.path.exists(path):
            methods.mkdir(path)
        filename = None
        code = None
        leveldir = self.__check_output_level(level)

        if type == 0:
            if self.__mainset:
                if isinstance(self.__mainset.idx, int):
                    filename = '{0}[{1}].Filename'.format(self.__mainset.name, self.__mainset.idx)
                else:
                    filename = '{0}[i].Filename'.format(self.__mainset.name)
            else:
                methods.raiseerror(messages.Errors.InputTypeError, 'Main input was not assigned')
            if prefix:
                filename = '"{0}_"+{1}'.format(prefix, filename)
            if ext:
                if ext == 'remove':
                    filename = 'methods.splitnifti({0})'.format(filename)
                else:
                    filename = 'methods.splitnifti({0})+".{1}"'.format(filename, ext)
            if leveldir:
                code = 'os.path.join("{0}", title, {1}, {2})'.format(path, leveldir, filename)
            else:
                code = 'os.path.join("{0}", title, {1})'.format(path, filename)
        elif type == 1:
            if self.__mainset:
                if prefix:
                    methods.raiseerror(messages.Errors.InputValueError, 'Cannot use prefix on this type of output')
                if leveldir:
                    code = 'os.path.join("{0}", title, {1})'.format(path, leveldir)
                else:
                    code = 'os.path.join("{0}", title)'.format(path)
            else:
                if self.__group or self.__multi:
                    if not ext:
                        ext = 'nii.gz'
                    if prefix:
                        if ext == 'remove':
                            if leveldir:
                                code = 'os.path.join("{0}", title, {1}, "{2}")'.format(path, leveldir, prefix)
                            else:
                                code = 'os.path.join("{0}", title, "{1}")'.format(path, prefix)
                        else:
                            if leveldir:
                                code = 'os.path.join("{0}", title, {1}, "{2}"+".{3}")'.format(path, leveldir,
                                                                                              prefix, ext)
                            else:
                                code = 'os.path.join("{0}", title, "{1}"+".{2}")'.format(path, prefix, ext)
                    else:
                        if leveldir:
                            if ext == 'remove':
                                code = 'os.path.join("{0}", title, {1}, {2})'.format(path, leveldir, leveldir)
                            else:
                                code = 'os.path.join("{0}", title, {1}, {2}+".{3}")'.format(path, leveldir,
                                                                                            leveldir, ext)
                        else:
                            methods.raiseerror(messages.Errors.InputTypeError,
                                               'incorrect dir')
                else:
                    methods.raiseerror(messages.Errors.InputTypeError, 'Main, multi or group input was not assigned')
        elif type == 2:
            if not prefix:
                methods.raiseerror(messages.Errors.InputValueError, 'Please assign folder name using prefix argument')
            if self.__mainset:
                if ext:
                    if ext == 'remove':
                        prefix = '{0}'.format(prefix)
                    else:
                        prefix = '{0}.{1}'.format(prefix, ext)
                if leveldir:
                    code = 'os.path.join("{0}", title, {1}, "{2}")'.format(path, leveldir, prefix)
                else:
                    code = 'os.path.join("{0}", title, "{1}")'.format(path, prefix)
            elif self.__group or self.__multi:
                if ext:
                    if ext == 'remove':
                        filename = '{0}'.format(prefix)
                    else:
                        filename = '{0}.{1}'.format(prefix, ext)
                else:
                    filename = '{0}'.format(prefix)
                code = 'os.path.join("{0}", title, "{1}")'.format(path, filename)
            else:
                methods.raiseerror(messages.Errors.InputTypeError, 'Main, multi or group input was not assigned')
        elif type == 3:
            if self.__mainset:
                if prefix:
                    methods.raiseerror(messages.Errors.InputValueError, 'Cannot use prefix on this type of output')
                self.__configure_module('tempfile', sub='mkdtemp')
                self.__configure_module('shutil', sub='rmtree')
                code = 'mkdtemp()'
            else:
                methods.raiseerror(messages.Errors.InputTypeError, 'Main input was not assigned')
        elif type == 4:
            if self.__mainset:
                code = 'None'
            else:
                methods.raiseerror(messages.Errors.InputTypeError, 'Main input was not assigned')
        else:
            methods.raiseerror(messages.Errors.InputTypeError, 'Wrong output type')
        self.__output.append(_oset(name=name, code=code, type=type, ext=ext, prefix=prefix))
        self.__assigned_namespace.append(name)

    def __convert_namespace(self, nspace):
        nspace = list(set(nspace))
        ns_code = []
        if self.__group:
            for grp in self.__group:
                if grp.args:
                    for i, arg in enumerate(grp.args):
                        arg_ns = '{0}_arg{1}'.format(grp.name, str(i).zfill(2))
                        ns_code.append('{0} = "{1}"'.format(arg_ns, arg))
                        self.__assigned_namespace.append(arg_ns)
                        nspace.append(arg_ns)
                if grp.kwargs:
                    for key, value in grp.kwargs.items():
                        kwarg_ns = '{0}_{1}'.format(grp.name, key)
                        ns_code.append('{0} = "{1}"'.format(kwarg_ns, value))
                        self.__assigned_namespace.append(kwarg_ns)
                        nspace.append(kwarg_ns)
        elif self.__multi:
            for grp in self.__multi:
                if grp.args:
                    for i, arg in enumerate(grp.args):
                        arg_ns = '{0}_arg{1}'.format(grp.name, str(i).zfill(2))
                        ns_code.append('{0} = "{1}"'.format(arg_ns, arg))
                        self.__assigned_namespace.append(arg_ns)
                        nspace.append(arg_ns)
                if grp.kwargs:
                    for key, value in grp.kwargs.items():
                        kwarg_ns = '{0}_{1}'.format(grp.name, key)
                        ns_code.append('{0} = "{1}"'.format(kwarg_ns, value))
                        self.__assigned_namespace.append(kwarg_ns)
                        nspace.append(kwarg_ns)
        for ns in nspace:
            if self.__mainset:
                if ns in [self.__mainset.name]:
                    if isinstance(self.__mainset.idx, int):
                        ns_code.append('{0}={0}[{1}].Abspath'.format(ns, self.__mainset.idx))
                    else:
                        ns_code.append('{0}={0}[i].Abspath'.format(ns))
                if self.__sideset:
                    for ss in self.__sideset:
                        if ns in [ss.name]:
                            if isinstance(ss.idx, int):
                                ns_code.append('{0}={0}[{1}].Abspath'.format(ns, ss.idx))
                            else:
                                ns_code.append('{0}={0}[i].Abspath'.format(ns))
            else:
                if not self.__multi:
                    if not self.__group:
                        methods.raiseerror(messages.InputObjectError, 'No sufficient input was assigned')
                    else:
                        for grp in self.__group:
                            if ns in [grp.name]:
                                ns_code.append('{0} = " ".join(list({0}.df.Abspath))'.format(ns))
                else:
                    for grp in self.__multi:
                        if ns in [grp.name]:
                            ns_code.append('{0} = " ".join(list({0}.df.Abspath))'.format(ns))
            if self.__var:
                for var in self.__var:
                    if ns in [var.name]:
                        if var.type:
                            if os.path.exists(var.value):
                                ns_code.append('{0}="{1}"'.format(var.name, var.value))
                            else:
                                ns_code.append('{0}={1}'.format(var.name, var.value))
                        else:
                            ns_code.append('{0}={0}'.format(var.name))
            if self.__output:
                for op in self.__output:
                    if ns in [op.name]:
                        if op.type == 3:
                            ns_code.append('{0}=os.path.join({0}, "{0}{1}")'.format(op.name, self.__prj.ext[0]))
                        else:
                            ns_code.append('{0}={0}'.format(op.name))
        return nspace, ns_code

    def __convert_cmdcode(self):
        list_cmd = []
        for i, cmd in enumerate(self.__cmd):

            if cmd.type == 0: #command line tool
                if cmd.name:
                    list_cmd.append("{0}, err = methods.shell('{1}'.format({2}))".format(cmd.name,
                                                                                         cmd.command,
                                                                                         ', '.join(cmd.nscode)))
                    list_cmd.append('stdout_collector.append(({0}, err))'.format(cmd.name))
                else:
                    list_cmd.append("out, err = methods.shell('{0}'.format({1}))".format(cmd.command,
                                                                                         ', '.join(cmd.nscode)))
                    list_cmd.append('stdout_collector.append((out, err))')

            elif cmd.type == 1: #python methods
                if cmd.name:
                    adtn_cmd = list()
                    if cmd.nscode:
                        updated_cmd = str(cmd.command)
                        for nsc in cmd.nscode:
                            ns_key, ns_value = nsc.split('=')
                            updated_cmd = updated_cmd.replace('{'+ns_key+'}', ns_value)
                        adtn_cmd.append('{0} = {1}'.format(cmd.name, updated_cmd))
                    else:
                        adtn_cmd.append('{0} = {1}'.format(cmd.name, cmd.command))
                    adtn_cmd.append('stdout_collector.append(({0}, None))'.format(cmd.command))
                    list_cmd.extend(self.__indent(adtn_cmd, level=cmd.level))
                else:
                    if cmd.nscode:
                        updated_cmd = str(cmd.command)
                        for nsc in cmd.nscode:
                            ns_key, ns_value = nsc.split('=')
                            updated_cmd = updated_cmd.replace('{'+ns_key+'}', ns_value)
                        adtn_cmd = ['try:', self.__indent(updated_cmd),
                                    'except:',
                                    self.__indent("methods.raiseerror(messages.CommandExecutionFailure, 'Error')")]
                        list_cmd.extend(self.__indent(adtn_cmd, level=cmd.level))
                    else:
                        adtn_cmd = self.__indent('{0}'.format(cmd.command), level=cmd.level)
                        list_cmd.append(adtn_cmd)

            elif cmd.type == 2: #TODO: need to integrate scheduler function for cluster computing
                methods.raiseerror(messages.Errors.InputTypeError, 'Scheduler is not available yet')

            else:
                methods.raiseerror(messages.Errors.InputTypeError, 'Wrong command type')
        return list_cmd

    def __retreive_namespaces_from_command(self, command):
        """ Retreive namespaces from the command

        :param command:
        :return:
        """
        def find(s, ch):
            return [i for i, ltr in enumerate(s) if ltr == ch]
        # tmpns = [obj.strip('{}') for obj in re.findall(r"[{\w'}]+", command) if obj[0] == '{' and obj[-1] == '}']
        tmpns = [obj[find(obj,'{')[0]+1:find(obj, '}')[-1]]
                 for obj in re.findall(r"[{\w'}]+", command) if '{' in obj and '}' in obj]
        nss = []
        for ns in tmpns:
            if "}{" in ns:
                nss.extend(ns.split('}{'))
            else:
                nss.append(ns)
        return nss

    def __indent(self, objs, size=4, level=1, chr=' '):
        """ Add indent to str or list of str objs

        :param objs:        list object has string contents
        :param size:        size of intent
        :param level:       level of intent
        :param chr:         character of intent
        :return:            indented objs
        """
        if level:
            if isinstance(objs, list):
                return map(str.__add__, [chr*size*level]*len(objs), objs)
            elif isinstance(objs, str):
                return chr*size*level + objs
            elif not objs:
                pass
            else:
                methods.raiseerror(messages.Errors.InputTypeError, 'input object must be str or list')
        else:
            return objs

    def __configure_function_header(self, name):
        """ Method to configure platform of function,

        :param name: namespace for built function
        :return:
        """
        if self.__multi:
            header = ['def {0}(self, title, idx, subj):'.format(name)]
        else:
            if self.__group:
                header = ['def {0}(self, title):'.format(name)]
            else:
                if self.__proc.sessions:
                    args = 'subj, sess'
                else:
                    args = 'subj'
                header = ['def {0}(self, title, idx, {1}):'.format(name, args)]
        header += self.__indent(self.__configure_environment() + ['stdout_basket = []'], level=1)
        return header

    def __configure_module(self, module, sub=None, rename=None):
        """ Method to import module during steps
        Only one module can be imported, for importing multiple modules
        Use this method multiple time.

        :param module:  any python module installed in your environment
        :param sub:     submodule you want to import from the parent module
                        if you want to import multiple submodules, use list instead
        :param rename:  new name for imported module.
                        if you want to rename submodules, use list instead
        :type module: str
        :type sub: str or list of str
        :type rename: str or list of str
        """
        prefix = None
        surfix = None
        if not sub:
            prefix = 'import'
            if isinstance(rename, str):
                surfix = " ".join(['as', rename])
        else:
            if isinstance(sub, (str, list)):
                prefix = 'from'
            else:
                methods.raiseerror(messages.Errors.InputTypeError,
                                   'Wrong type on module, use str or list of str')
            if rename:
                if isinstance(rename, (str, list)):
                    if isinstance(sub, str):
                        if isinstance(rename, str):
                            surfix = " ".join([sub, 'as', rename])
                        else:
                            methods.raiseerror(messages.Errors.InputTypeError,
                                               'Datatype of sub and rename must be same')
                    if isinstance(sub, list):
                        if isinstance(rename, list):
                            if len(sub) != len(rename):
                                methods.raiseerror(messages.Errors.InputValueError,
                                                   'The number of object in list between sub and rename must be same')
                            else:
                                surfix = ", ".join([" ".join([s, 'as', rename[i]]) for i, s in enumerate(sub)])
            else:
                surfix = sub
        if surfix:
            if prefix == 'from':
                package = " ".join([prefix, module, 'import', surfix])
            else:
                package = " ".join([prefix, module, surfix])

        else:
            package = " ".join([prefix, module])
        if package not in self.__import:
            self.__import.append(package)
        else:
            pass

    def __configure_environment(self):
        """ Method to configure module and input code
        :return: list of codes to build function
        """
        return self.__import + self.__convert_inputcode()

    def __configure_output(self):
        code = []
        for output in self.__output:
            code.append('{0} = {1}'.format(output.name, output.code))
            if output.type in [3, 4]:
                pass
            else:
                # if self.__multi:
                #     pass
                # else:
                if output not in self.__mkdir:
                    self.__mkdir.append(output)
        return code

    def __configure_loop_contents(self, level=2):
        output = self.__configure_output()
        if self.__mkdir:
            output.extend(self.__convert_mkdir())
        return self.__indent(output, level=level)

    def __configure_tempobj(self, level):
        tmps = [op.name for op in self.__output if op.type == 3]
        if tmps:
            output = []
            for tmp in tmps:
                output.append(self.set_logging('"SYS::TempFolder[{0}] is generated"'.format(tmp)))
                if tmp not in self.__opened_temps:
                    self.__opened_temps.append(tmp)
            return self.__indent(output, level=level)
        else:
            return []

    def __configure_tempobj_closure(self):
        output = []
        for tmp in self.__opened_temps:
                output.append('rmtree({0})'.format(tmp))
                output.append(self.set_logging('"SYS::TempFolder[{0}] is closed"'.format(tmp)))
        return output

    def __configure_tail(self, level):
        output = []
        output.append(self.__indent('stdout_basket.append(stdout_collector)', level=level))
        output.append(self.__indent('return stdout_basket', level=1))
        return output

    def build_func(self, name):
        """ Method to build processor function

        :param name:    namespace of processor function
        :return:
        """
        level = None
        func = []
        func += self.__configure_function_header(name)
        outputs = self.__output_sorter()
        pad = 0
        if self.__mainset:
            name = self.__mainset.name
            if isinstance(self.__mainset.idx, int):
                name_checker = '{0}[{1}].Filename'.format(name, self.__mainset.idx)
                level = 1
            else:
                name_checker = '{0}[i].Filename'.format(name)
                level = 2
                func += self.__indent(
                    ['for i in progressbar(range(len({0})), desc="Files", leave=False):'.format(name)])
            if 0 in outputs.keys():     # output name is same as input's
                file_output = outputs[0].keys()[0]
                func += self.__configure_loop_contents(level=level)
                func += self.__indent(
                    ['output_checker = methods.splitnifti({0})'.format(name_checker),
                     'flist_checker = [f for f in os.listdir(os.path.dirname({0}))]'.format(file_output),
                     'stdout_collector = []',
                     'if any(output_checker in f for f in flist_checker):'], level=level)
                func += self.__indent(
                    [self.set_logging('"Step::Skipped because the file[{0}] is exist".format(output_checker)')],
                    level=level+1)
                func += self.__indent(['else:'], level=level)
                pad = 1
                func += self.__configure_tempobj(level=level+pad)
                func += self.__indent(self.__convert_cmdcode(), level=level+pad)
                if self.__opened_temps:
                    func += self.__indent(self.__configure_tempobj_closure(), level=level+pad)
            else:
                if 4 in outputs.keys():
                    func += self.__configure_loop_contents(level=level)
                    func += self.__indent(['stdout_collector = []'], level=level)
                    func += self.__indent(self.__convert_cmdcode(), level=level)
                else: # if type is 1 or 2
                    func += self.__configure_loop_contents(level=level)
                    func += self.__indent(['stdout_collector = []'], level=level)
                    func += self.__configure_tempobj(level=level)
                    func += self.__indent(self.__convert_cmdcode(), level=level)
                    if self.__opened_temps:
                        func += self.__indent(self.__configure_tempobj_closure(), level=level)
        else:
            if self.__multi:
                level = 1
                func += self.__configure_loop_contents(level=level)
                func += self.__indent(['stdout_collector = []'], level=level)
                func += self.__indent(self.__convert_cmdcode(), level=level)
            else:
                if self.__group:
                    level = 1
                    func += self.__configure_loop_contents(level=level)
                    func += self.__indent(['stdout_collector = []'], level=level)
                    func += self.__indent(self.__convert_cmdcode(), level=level)
                else:
                    methods.raiseerror(messages.Errors.InputTypeError, 'Main or group input was not assigned')
        func += self.__configure_tail(level+pad)

        return '\n'.join(func)

    def run(self, title, surfix=None, debug=False):
        """Generate loop commands for step

        :param title:
        :param surfix:
        :param debug:
        :return: None
        """
        if self.__message:
            display(self.__message)
        def output_writer(outputs, output_path):
            if outputs:
                if isinstance(outputs[0], list):
                    all_outputs = []
                    for output in outputs:
                        all_outputs.extend(
                            ['STDOUT:\n{0}\nMessage:\n{1}\n\n'.format(out, err) for out, err in output])
                    outputs = all_outputs[:]
                else:
                    outputs = ['STDOUT:\n{0}\nMessage:\n{1}\n\n'.format(out, err) for out, err in outputs]
                today = "".join(str(datetime.date.today()).split('-'))

                if os.path.exists(os.path.join(self.__proc.path, output_path)):
                    path = self.__proc.path
                else:
                    path = self.__proc._rpath
                with open(os.path.join(path, output_path, 'stephistory-{}.log'.format(today)), 'a') as f:
                    f.write('\n\n'.join(outputs))
            else:
                pass
        output_path = self.init_path("{0}-{1}".format(title, surfix), dc=self.__dc)

        if debug:
            print("-="*30)
            print(self.build_func('debug'))
            return output_path
        else:
            self.__prj.reload()
            thread = self._parallel
            pool = ThreadPool(thread)
            self.__proc.logger.info("Step::[{0}] is executed with {1} thread(s).".format(title, thread))
            if self.__multi:
                for idx, subj in enumerate(progressbar(self.__proc.subjects, desc='Subjects')):
                    self.__proc.logger.info("Step::The inputs are identified as multi type")
                    outputs = self.worker([self.__proc, output_path, idx, subj])
                    output_writer(outputs, output_path)
            else:
                if self.__group:
                    self.__proc.logger.info("Step::The inputs are identified as groups type")
                    outputs = self.worker([self.__proc, output_path])
                    output_writer(outputs, output_path)
                else:
                    if self.__proc.sessions:
                        self.__proc.logger.info("Step::The inputs are identified as multi-session scans")
                        for idx, subj in enumerate(progressbar(self.__proc.subjects, desc='Subjects')):
                            iteritem = [(self.__proc, output_path, idx, subj, sess) for sess in self.__proc.sessions]
                            for outputs in progressbar(pool.imap_unordered(self.worker, iteritem), desc='Sessions',
                                                       leave=False, total=len(iteritem)):
                                if 4 not in [o.type for o in self.__output]:
                                    output_writer(outputs, output_path)
                                else:
                                    # self.__proc.logger.info("Step::This step don't generate output folder")
                                    pass
                    else:
                        self.__proc.logger.info("Step::The inputs are identified as single-session scans")
                        iteritem = [(self.__proc, output_path, idx, subj) for idx, subj in enumerate(self.__proc.subjects)]
                        for outputs in progressbar(pool.imap_unordered(self.worker, iteritem), desc='Subjects',
                                                   total=len(iteritem)):
                            if 4 not in [o.type for o in self.__output]:
                                output_writer(outputs, output_path)
                            else:
                                # self.__proc.logger.info("Step::This step don't generate output folder")
                                pass
            self.__proc._history[os.path.basename(output_path)] = output_path
            self.__proc.save_history()
            self.__prj.reload()
            clear_output()
            display('Done.....')
            sleep(0.5)
            clear_output()
            return output_path

    def worker(self, args, name='built_func'):
        """The worker for parallel computing

        :param args: list, Arguments for step execution
        :return: str
        """
        funccode = self.build_func(name)
        output = None
        exec (funccode)  # load step function on memory
        try:
            exec ('output = {0}(*args)'.format(name))  # execute function
        except Exception as e:
                print(e)
        return output