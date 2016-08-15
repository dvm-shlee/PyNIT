from __future__ import print_function

# Command execution
import os
import re
import inspect

import shlex as shl
import shutil
from string import ascii_lowercase as lc
from subprocess import call as call
from subprocess import list2cmdline
from subprocess import check_output

try:
    import SimpleITK as sitk
except ImportError:
    pass

import pandas as pd
import nibabel as nib
import nibabel.affines as affns
import numpy as np
from skimage import exposure

import objects
import error


class Internal(object):
    """ Internal utility for PyNIT package
    """
    # ImageObject handler collection
    @staticmethod
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

    @staticmethod
    def swap_axis(imageobj, axis1, axis2):
        """ Swap axis of image object

        :param imageobj:
        :param axis1:
        :param axis2:
        :return:
        """
        resol, origin = affns.to_matvec(imageobj.get_affine())
        resol = np.diag(resol).copy()
        origin = origin
        imageobj._dataobj = np.swapaxes(imageobj._dataobj, axis1, axis2)
        resol[axis1], resol[axis2] = resol[axis2], resol[axis1]
        origin[axis1], origin[axis2] = origin[axis2], origin[axis1]
        affine = affns.from_matvec(np.diag(resol), origin)
        Internal.reset_orient(imageobj, affine)

    @staticmethod
    def load(filename):
        """ Load imagefile

        :param filename:
        :return:
        """
        if '.nii' in filename:
            img = objects.Image.load(filename)
        elif '.mha' in filename:
            try:
                mha = sitk.ReadImage(filename)
            except:
                raise error.ImportItkFailure
            data = sitk.GetArrayFromImage(mha)
            resol = mha.GetSpacing()
            origin = mha.GetOrigin()
            affine = affns.from_matvec(np.diag(resol), origin)
            img = objects.Image(data, affine)
        else:
            raise error.InputPathError
        return img

    @staticmethod
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
        resol, origin = affns.to_matvec(imageobj.affine)
        resol = np.diag(resol).copy()
        scale = float(slice_thickness) / resol[axis]
        resol[axis] = slice_thickness
        idx = []
        for i in range(ac_loc):
            idx.append(ac_slice - int((ac_loc - i) * scale))
        for i in range(total_slice - ac_loc):
            idx.append(ac_slice + int(i * scale))
        imageobj._dataobj = data[:, :, idx]
        affine, origin = affns.to_matvec(imageobj.affine)
        affine = np.diag(affine)
        affine[axis] = slice_thickness
        affine_mat = affns.from_matvec(np.diag(affine), origin)
        imageobj._affine = affine_mat
        imageobj.set_qform(affine_mat)
        imageobj.set_sform(affine_mat)
        imageobj.header['sform_code'] = 0
        imageobj.header['qform_code'] = 1

    @staticmethod
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

    @staticmethod
    def set_center(imageobj, corr):
        """ Applying center corrdinate to the object
        """
        resol, origin = affns.to_matvec(imageobj.affine)
        affine = affns.from_matvec(resol, corr)
        Internal.reset_orient(imageobj, affine)

    # TemplateObject handler collection
    @staticmethod
    def remove_nifti_ext(path):
        """ Remove extension

        :param path:
        :return:
        """
        filename = os.path.splitext(path)[0]
        if '.nii' in filename:
            filename = os.path.splitext(filename)[0]
        return filename

    @staticmethod
    def parsing_atlas(path):
        """ Parsing atlas imageobj and label

        :param path:
        :return:
        """
        label = dict()
        affine = list()
        if os.path.isdir(path):
            atlasdata = None
            list_of_rois = [img for img in os.listdir(path) if '.nii' in img]
            rgbs = np.random.rand(len(list_of_rois), 3)
            label[0] = 'Clear Label', [.0, .0, .0]

            for idx, img in enumerate(list_of_rois):
                imageobj = objects.Image.load(os.path.join(path, img))
                affine.append(imageobj.affine)
                if not idx:
                    atlasdata = np.asarray(imageobj.dataobj)
                else:
                    atlasdata += np.asarray(imageobj.dataobj) * (idx + 1)
                label[idx+1] = Internal.remove_nifti_ext(img), rgbs[idx]
            atlas = objects.Image(atlasdata, affine[0])
        elif os.path.isfile(path):
            atlas = objects.Image.load(path)
            if '.nii' in path:
                filepath = os.path.basename(Internal.remove_nifti_ext(path))
                dirname = os.path.dirname(path)
                for f in os.listdir(dirname):
                    if '.lbl' in f:
                        filepath = os.path.join(dirname, "{}.lbl".format(filepath))
                    elif '.label' in f:
                        filepath = os.path.join(dirname, "{}.label".format(filepath))
                    else:
                        filepath = filepath
                if filepath == os.path.basename(Internal.remove_nifti_ext(path)):
                    raise error.NoLabelFile
            else:
                raise error.NoLabelFile
            pattern = r'^\s+(?P<idx>\d+)\s+(?P<R>\d+)\s+(?P<G>\d+)\s+(?P<B>\d+)\s+' \
                      r'(\d+|\d+\.\d+)\s+\d+\s+\d+\s+"(?P<roi>.*)$'
            with open(filepath, 'r') as labelfile:
                for line in labelfile:
                    if re.match(pattern, line):
                        idx = int(re.sub(pattern, r'\g<idx>', line))
                        roi = re.sub(pattern, r'\g<roi>', line)
                        roi = roi.split('"')[0]
                        rgb = re.sub(pattern, r'\g<R>\s\g<G>\s\g<B>', line)
                        rgb = rgb.split(r'\s')
                        rgb = np.array(map(float, rgb))/255
                        label[idx] = roi, rgb
        else:
            raise error.InputPathError
        return atlas, label

    @staticmethod
    def save_label(label, filename):
        """ Save label instance to file

        :param label:
        :param filename:
        :return:
        """
        with open(filename, 'w') as f:
            line = list()
            for idx in label.keys():
                roi, rgb = label[idx]
                rgb = np.array(rgb) * 255
                rgb = rgb.astype(int)
                if idx == 0:
                    line = '{:>5}   {:>3}  {:>3}  {:>3}        0  0  0    "{}"\n'.format(idx, rgb[0], rgb[1], rgb[2],
                                                                                         roi)
                else:
                    line = '{}{:>5}   {:>3}  {:>3}  {:>3}        1  1  0    "{}"\n'.format(line, idx, rgb[0], rgb[1],
                                                                                           rgb[2], roi)
            f.write(line)

    # Viewer handler collection
    @staticmethod
    def set_viewaxes(axes):
        """ Set View Axes

        :param axes:
        :return:
        """
        ylim = axes.get_ylim()
        xlim = axes.get_xlim()
        axes.set_ylabel('L', rotation=0, fontsize=20)
        axes.set_xlabel('I', fontsize=20)
        axes.tick_params(labeltop=True, labelright=True, labelsize=8)
        axes.grid(False)
        axes.text(xlim[1]/2, ylim[1] * 1.1, 'P', fontsize=20)
        axes.text(xlim[1]*1.1, sum(ylim)/2*1.05, 'R', fontsize=20)
        return axes

    @staticmethod
    def check_invert(kwargs):
        """ Check image invertion

        :param kwargs:
        :return:
        """
        invertx = False
        inverty = False
        invertz = False
        if kwargs:
            for arg in kwargs.keys():
                if arg == 'invertx':
                    invertx = kwargs[arg]
                if arg == 'inverty':
                    inverty = kwargs[arg]
                if arg == 'invertz':
                    invertz = kwargs[arg]
        return invertx, inverty, invertz

    @staticmethod
    def apply_invert(data, *invert):
        """ Apply image invertion

        :param data:
        :param invert:
        :return:
        """
        if invert[0]:
            data = nib.orientations.flip_axis(data, axis=0)
        if invert[1]:
            data = nib.orientations.flip_axis(data, axis=1)
        if invert[2]:
            data = nib.orientations.flip_axis(data, axis=2)
        return data

    @staticmethod
    def convert_to_3d(imageobj):
        """ Reduce demension to 3D

        :param imageobj:
        :return:
        """
        dim = len(imageobj.shape)
        if dim == 4:
            data = np.asarray(imageobj.dataobj)[..., 0]
        elif dim == 3:
            data = np.asarray(imageobj.dataobj)
        else:
            raise error.ImageDimentionMismatched
        return data

    @staticmethod
    def apply_p2_98(data):
        """ Image normalization

        :param data:
        :return:
        """
        p2 = np.percentile(data, 2)
        p98 = np.percentile(data, 98)
        data = exposure.rescale_intensity(data, in_range=(p2, p98))
        return data

    @staticmethod
    def set_mosaic_fig(data, dim, resol, slice_axis, scale):
        """ Set environment for mosaic figure

        :param data:
        :param dim:
        :param resol:
        :param slice_axis:
        :param scale:
        :return:
        """
        num_of_slice = dim[slice_axis]
        num_height = int(np.sqrt(num_of_slice))
        num_width = int(round(num_of_slice / num_height))
        # Swap axis
        data = np.swapaxes(data, slice_axis, 2)
        resol[2], resol[slice_axis] = resol[slice_axis], resol[2]
        dim[2], dim[slice_axis] = dim[slice_axis], dim[2]
        # Check the size of each slice
        size_height = num_height * dim[1] * resol[1] * scale / max(dim)
        size_width = num_width * dim[0] * resol[0] * scale / max(dim)
        # Figure generation
        slice_grid = [num_of_slice, num_height, num_width]
        size = [size_width, size_height]
        return data, slice_grid, size

    @staticmethod
    def check_sliceaxis_cmap(imageobj, kwargs):
        """ Check sliceaxis (minimal number os slice) and cmap

        :param imageobj:
        :param kwargs:
        :return:
        """
        slice_axis = int(np.argmin(imageobj.shape))
        cmap = 'gray'
        for arg in kwargs.keys():
            if arg == 'slice_axis':
                slice_axis = kwargs[arg]
            if arg == 'cmap':
                cmap = kwargs[arg]
        return slice_axis, cmap

    @staticmethod
    def check_slice(dataobj, axis, slice_num):
        """ Check initial slice number to show

        :param dataobj:
        :param axis:
        :param slice_num:
        :return:
        """
        if slice_num:
            slice_num = slice_num
        else:
            slice_num = dataobj.shape[axis]/2
        return slice_num

    @staticmethod
    def linear_norm(data, new_min, new_max):
        """Linear normalization of the grayscale digital image
        """
        return (data - np.min(data)) * (new_max - new_min) / (np.max(data) - np.min(data)) - new_min

    @staticmethod
    def path_splitter(path):
        """Split path structure into list
        """
        return path.strip(os.sep).split(os.sep)

    # Analysis handler collection
    @staticmethod
    def mask_average(imageobj, maskobj, **kwargs):
        """ Mask average timeseries

        :param imageobj:
        :param maskobj:
        :return:
        """
        contra = None
        merged = None
        # Check kwargs
        if kwargs:
            for arg in kwargs.keys():
                if arg == 'contra':
                    contra = kwargs[arg]
                if arg == 'merge':
                    merged = kwargs[arg]
        newshape = reduce(lambda x, y: x*y, imageobj.shape[:3])
        data = imageobj.get_data()
        mask = maskobj.get_data()
        if contra:
            mask = mask[::-1, :, :]
        if merged:
            mask += mask[::-1, :, :]
        data = data.reshape(newshape, data.shape[3])
        mask = mask.reshape(newshape)
        mask = np.expand_dims(mask, axis=1)
        mask = np.repeat(mask, data.shape[1], axis=1)
        output = np.ma.masked_where(mask == 0, data)
        return pd.Series(np.ma.average(output, axis=0))

    @staticmethod
    def parsing_timetrace(imageobj, tempobj, **kwargs):
        """ Parsing timetrace from imageobj, with multiple rois

        :param imageobj:
        :param tempobj:
        :param kwargs:
        :return:
        """
        contra = None
        bilateral = None
        merged = None
        # Check kwargs
        if kwargs:
            for arg in kwargs.keys():
                if arg == 'contra':
                    contra = kwargs[arg]
                if arg == 'bilateral':
                    bilateral = kwargs[arg]
                if arg == 'merge':
                    merged = kwargs[arg]
        # Initiate dataframe
        df = pd.DataFrame()
        # Check each labels
        for idx in tempobj.label.keys():
            if idx:
                roi, maskobj = tempobj[idx]
                if merged:
                    col = Internal.mask_average(imageobj, maskobj, merged=True)
                else:
                    if contra:
                        col = Internal.mask_average(imageobj, maskobj, contra=True)
                    else:
                        col = Internal.mask_average(imageobj, maskobj)
                df[roi] = col
        if bilateral:
            for idx in tempobj.label.keys():
                if idx:
                    roi, maskobj = tempobj[idx]
                    if merged:
                        pass
                    else:
                        if contra:
                            col = Internal.mask_average(imageobj, maskobj)
                        else:
                            col = Internal.mask_average(imageobj, maskobj, contra=True)
                        df["Cont_{}".format(roi)] = col
        return df

    @staticmethod
    def cal_mean_cbv(output_path, input_path, postfix_bold='BOLD', postfix_cbv='CBV', *args):
        """ Calculate cbv

        :param output_path:
        :param input_path:
        :param postfix_bold:
        :param postfix_cbv:
        :return:
        """
        # Get average images from MION injection scan
        fname_bold = '{}_{}.nii'.format(os.path.splitext(output_path)[0], postfix_bold)
        fname_cbv = '{}_{}.nii'.format(os.path.splitext(output_path)[0], postfix_cbv)
        if os.path.exists(fname_bold) and os.path.exists(fname_cbv):
            pass
        else:
            img = nib.load(input_path)
            affn = img.get_affine()
            img = img.get_data()
            total_tr = img.shape[3]

            epi_bold = np.average(img[:, :, :, :int(total_tr / 3)], 3)
            epi_cbv = np.average(img[:, :, :, total_tr - int(total_tr / 3):], 3)

            nii_bold = nib.Nifti1Image(epi_bold, affn)
            nii_cbv = nib.Nifti1Image(epi_cbv, affn)

            nii_bold.to_filename(fname_bold)
            nii_cbv.to_filename(fname_cbv)

    @staticmethod
    def cal_mean(output_path, input_path, *args):
        """ Calculate average

        :param output_path:
        :param input_path:
        :return:
        """
        img = nib.load(input_path)
        affn = img.get_affine()
        mean = np.average(img.get_data(), axis=3)
        nii_mean = nib.Nifti1Image(mean, affn)
        nii_mean.to_filename(output_path)

    # Project Handler collection
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
        empty_prj = False
        df = pd.DataFrame()
        for f in os.walk(os.path.join(path, ds_type[idx])):
            if f[2]:
                for filename in f[2]:
                    row = pd.Series(Internal.path_splitter(os.path.relpath(f[0], path)))
                    row['Filename'] = filename
                    row['Abspath'] = os.path.join(f[0], filename)
                    df = df.append(pd.DataFrame([row]), ignore_index=True)
        if idx == 0:
            if len(df.columns) is 5:
                single_session = True
        else:
            if len(df.columns) is 6:
                single_session = True
        columns = Internal.update_columns(idx, single_session)
        if not len(df):
            empty_prj = True
        return df.rename(columns=columns), single_session, empty_prj

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
        """ reorder the project columns

        :param idx:
        :param single_session:
        :return:
        """
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
            try:
                df = df[df['DataClass'].isin(data_class)]
            except:
                print('Error')
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
        Internal.mkdir(os.path.join(prj.path, prj.ds_type[0]),
                       os.path.join(prj.path, prj.ds_type[1]),
                       os.path.join(prj.path, prj.ds_type[2]))

    @staticmethod
    def check_kwargs(kwargs, command):
        """Validate input arguments for input command

        :param kwargs: dict
        :param command: str
        :return:
        """
        args, defaults, varargs, keywords = Internal.check_args(command)
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
                for f in os.walk(os.path.join(pipeline_inst.path, pipeline_inst.done[-1])):
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
            return None

    @staticmethod
    def mkdir(*paths):
        for path in paths:
            try:
                os.mkdir(path)
            except:
                pass

    @staticmethod
    def copyfile(output_path, input_path, *args):
        """ Copy File

        :param output_path:
        :param input_path:
        :return:
        """
        shutil.copyfile(input_path, output_path)


class Interface(object):
    """Class for wrapping the commands to run external image processing software packages
    including AFNI, ANTs, FSL
    """
    def __init__(self):
        self.__avail = [f for f in dir(self) if '__' not in f and 'avail' not in f]

    @property
    def avail(self):
        return self.__avail

    # Afni commands
    @staticmethod
    def afni_3dTshift(output_path, input_path, tr=None, tpattern=None, *args):
        """
        Handler for 3dTshift (Slice timing correction tool of AFNI package)

        Parameters
        ----------
        output_path : str
            explanation
        input_path : str
        tr : int
        tpattern : str
            'altplus'

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        cmd = ['3dTshift', '-prefix', output_path]
        if tr:
            cmd.extend(['-TR', str(tr)])
        if tpattern:
            cmd.extend(['-tpattern', tpattern])
        cmd.append(input_path)
        cmd = list2cmdline(cmd)
        call(shl.split(cmd))

    @staticmethod
    def afni_3dvolreg(output_path, input_path, base_slice=0, *args):
        """Wrapper for 3dvolreg (Motion correction tool of AFNI package)

        Parameters
        ----------
        output_path : str
            explanation
        input_path : str
        base_slice : str
        """
        mpfile = os.path.splitext(output_path)[0] + '.1D'
        tfmfile = os.path.splitext(output_path)[0]
        cmd = ['3dvolreg', '-prefix', output_path, '-1Dfile', mpfile, '-1Dmatrix_save', tfmfile,
               '-Fourier', '-verbose', '-base', '{}'.format(int(base_slice)), input_path]
        cmd = list2cmdline(cmd)
        call(shl.split(cmd))

    @staticmethod
    def afni_3dAllineate(output_path, input_path, **kwargs):
        cmd = ['3dAllineate', '-prefix', output_path]
        if kwargs:
            for arg in kwargs.keys():
                if arg == 'matrix_apply':
                    cmd.append('-1D{}'.format(arg))
                    cmd.append(kwargs[arg])
                if arg == 'master':
                    if arg == 'base':
                        raise error.ArgumentsOverlapped
                    else:
                        cmd.append('-{}'.format(arg))
                        cmd.append(kwargs[arg])
                if arg == 'base':
                    if arg == 'master':
                        raise error.ArgumentsOverlapped
                    else:
                        cmd.append('-{}'.format(arg))
                        cmd.append(kwargs[arg])
                if arg == 'warp':
                    cmd.append('-{}'.format(arg))
                    cmd.append(kwargs[arg])
        else:
            raise KeyError("")
        cmd.append(input_path)
        cmd = list2cmdline(cmd)
        call(shl.split(cmd))

    @staticmethod
    def afni_3dcalc(output_path, expr, *inputs):
        # AFNI image calculation (3dcalc)
        cmd = ['3dcalc', '-prefix', output_path]
        if inputs:
            atoz = lc[:len(inputs)]
            data = zip(atoz, inputs)
            for abc, path in data:
                cmd.append('-' + abc)
                cmd.append(path)
        else:
            raise AttributeError("input data are not defined.")
        cmd.append('-expr')
        cmd.append("'{}'".format(expr))
        cmd = list2cmdline(cmd)
        call(shl.split(cmd))

    @staticmethod
    def afni_3dMean(output_path, *inputs):
        # AFNI 3dMean objects.Image calculator
        cmd = ['3dMean', '-prefix', output_path]
        cmd.extend(inputs)
        cmd = list2cmdline(cmd)
        call(shl.split(cmd))

    @staticmethod
    def afni_3dBandpass(output_path, input_path, norm=False, despike=False, mask=None, blur=False,
                        band=False, dt='1', *args):
        # AFNI signal processing for resting state (3dBandpass)
        cmd = ['3dBandpass', '-input', input_path, '-prefix', output_path]
        if 'dt':
            cmd.append('-dt')
            cmd.append(dt)
        if norm:
            cmd.append('-norm')
        if despike:
            cmd.append('-despike')
        if mask:
            cmd.extend(['-mask', mask])
        if blur:
            cmd.extend(['-blur', blur])
        if band:
            cmd.append('-band')
            cmd.extend(band)
        cmd = list2cmdline(cmd)
        call(shl.split(cmd))

    @staticmethod
    def afni_3dmaskave(output_path, input_path, mask_path, *args):
        """ AFNI 3dmaskave command wrapper

        :return: list
            Average timeseries data from given ROI

        """
        cmd = ['3dmaskave', '-mask']
        cmd.append("'{}'".format(mask_path))
        cmd.append('-q')
        cmd.append("'{}'".format(input_path))
        cmd = list2cmdline(cmd)
        if output_path:
            cmd = '{} > {}'.format(cmd, output_path)
        stdout = check_output(shl.split(cmd))
        stdout = stdout.split('\n')
        try:
            stdout = map(float, stdout)
        except:
            stdout.pop()
            stdout = map(float, stdout)
        finally:
            return stdout

    # ANTs commands
    @staticmethod
    def ants_BiasFieldCorrection(output_path, input_path, algorithm='n3', *args):
        """
        Execute the BiasFieldCorrection in the ANTs package

        :param output_path: absolute output path
        :param input_path: absolute input path
        :param algorithm: 'n3' or 'n4'
        """
        if algorithm == 'n3' or 'N3':
            cmd = ['N3BiasFieldCorrection', '3', input_path, output_path]
        elif algorithm == 'n4' or 'N4':
            cmd = ['N4BiasFieldCorrection', '-i', input_path, '-o', output_path]
        else:
            raise BaseException("One of 'n3' or 'n4' must be chosen for last argument")
        cmd = list2cmdline(cmd)
        call(shl.split(cmd))

    @staticmethod
    def ants_RegistrationSyn(output_path, input_path, base_path, quick=True, ttype='s', *args):
        # ANTs SyN registration
        output_path = '{}_'.format(os.path.splitext(output_path)[0])
        if os.path.exists(str(output_path)+'Warped.nii.gz'):
            pass
        else:
            if not quick:
                script = 'antsRegistrationSyN.sh'
            else:
                script = 'antsRegistrationSyNQuick.sh'

            cmd = [script, '-t', ttype, '-f', base_path, '-m', input_path, '-o', output_path]
            cmd = list2cmdline(cmd)
            call(shl.split(cmd))

    @staticmethod
    def ants_WarpImageMultiTransform(output_path, input_path, base_path, *args):
        # ANTs applying transform
        cmd = ['Warp.ImageMultiTransform', '3', input_path, output_path, '-R', base_path]
        for arg in args:
            cmd.append(arg)
        cmd = list2cmdline(cmd)
        call(shl.split(cmd))
