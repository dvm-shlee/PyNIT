from __future__ import print_function

import os
import re
import shutil

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


class InternalMethods(object):
    """ Internal utility for PyNIT package
    """
    # Method collection for ImageObject handler
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
        InternalMethods.reset_orient(imageobj, affine)

    @staticmethod
    def load(filename):
        """ Load imagefile

        :param filename:
        :return:
        """
        if '.nii' in filename:
            img = objects.ImageObj.load(filename)
        elif '.mha' in filename:
            try:
                mha = sitk.ReadImage(filename)
            except:
                raise error.ImportItkFailure
            data = sitk.GetArrayFromImage(mha)
            resol = mha.GetSpacing()
            origin = mha.GetOrigin()
            affine = affns.from_matvec(np.diag(resol), origin)
            img = objects.ImageObj(data, affine)
        else:
            raise error.InputPathError
        return img

    @staticmethod
    def load_temp(path=None, atlas=None):
        """ Load imagefile

        :param filename:
        :return:
        """
        tempobj = objects.Template(path, atlas)
        return tempobj

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
        affine, origin = affns.to_matvec(imageobj.affine[:, :])
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
        resol, origin = affns.to_matvec(imageobj.affine[:, :])
        affine = affns.from_matvec(resol, corr)
        InternalMethods.reset_orient(imageobj, affine)

    # Method collection for templateObject handler
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
                imageobj = objects.ImageObj.load(os.path.join(path, img))
                affine.append(imageobj.affine)
                if not idx:
                    atlasdata = np.asarray(imageobj.dataobj)
                else:
                    atlasdata += np.asarray(imageobj.dataobj) * (idx + 1)
                label[idx+1] = InternalMethods.remove_nifti_ext(img), rgbs[idx]
            atlas = objects.ImageObj(atlasdata, affine[0])
        elif os.path.isfile(path):
            atlas = objects.ImageObj.load(path)
            if '.nii' in path:
                filepath = os.path.basename(InternalMethods.remove_nifti_ext(path))
                dirname = os.path.dirname(path)
                for f in os.listdir(dirname):
                    if '.lbl' in f:
                        filepath = os.path.join(dirname, "{}.lbl".format(filepath))
                    elif '.label' in f:
                        filepath = os.path.join(dirname, "{}.label".format(filepath))
                    else:
                        filepath = filepath
                if filepath == os.path.basename(InternalMethods.remove_nifti_ext(path)):
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

    # Method collection for viewer handler
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
        """
        p2 = np.percentile(data, 2)
        p98 = np.percentile(data, 98)
        data = exposure.rescale_intensity(data, in_range=(p2, p98))
        return data

    @staticmethod
    def set_mosaic_fig(data, dim, resol, slice_axis, scale):
        """ Set environment for mosaic figure
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
        """
        if slice_num:
            slice_num = slice_num
        else:
            slice_num = dataobj.shape[axis]/2
        return slice_num

    # Method collection for project handler
    @staticmethod
    def path_splitter(path):
        """Split path structure into list
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
        empty_prj = False
        df = pd.DataFrame()
        for f in os.walk(os.path.join(path, ds_type[idx])):
            if f[2]:
                for filename in f[2]:
                    row = pd.Series(InternalMethods.path_splitter(os.path.relpath(f[0], path)))
                    row['Filename'] = filename
                    row['Abspath'] = os.path.join(f[0], filename)
                    df = df.append(pd.DataFrame([row]), ignore_index=True)
        if idx == 0:
            if len(df.columns) is 5:
                single_session = True
        else:
            if len(df.columns) is 6:
                single_session = True
        columns = InternalMethods.update_columns(idx, single_session)
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

    # @staticmethod
    # def isnull(df):
    #     """Check missing value
    #
    #     :param df: pandas.DataFrame
    #     :return:
    #     """
    #     return pd.isnull(df)

    @staticmethod
    def mk_main_folder(prj):
        """Make processing and results folders
        """
        InternalMethods.mkdir(os.path.join(prj.path, prj.ds_type[0]),
                              os.path.join(prj.path, prj.ds_type[1]),
                              os.path.join(prj.path, prj.ds_type[2]))

    # @staticmethod
    # def check_args(command):
    #     """Check arguments of input command
    #
    #     :param command:
    #     :return:
    #         args
    #         defaults
    #         varargs
    #         keywords
    #     """
    #     if command in dir(Interface):
    #         argspec = dict(inspect.getargspec(getattr(Interface, command)).__dict__)
    #     else:
    #         raise error.CommandExecutionFailure
    #     if argspec['defaults'] is None:
    #         def_len = 0
    #         defaults = None
    #     else:
    #         def_len = len(argspec['defaults'])
    #         defaults = dict(zip(argspec['args'][len(argspec['args']) - def_len:], argspec['defaults']))
    #     args = argspec['args'][1:(len(argspec['args']) - def_len)]
    #     varargs = argspec['varargs']
    #     kwargs = argspec['keywords']
    #     return args, defaults, varargs, kwargs

    # @staticmethod
    # def filter_file_index(option, prj, file_index):
    #     if file_index:
    #         option.extend(prj.df.Abspath.tolist()[min(file_index):max(file_index) + 1])
    #     else:
    #         option.extend(prj.df.Abspath.tolist())
    #     return option

    @staticmethod
    def get_step_name(prjobj, step):
        pipeline_path = os.path.join(prjobj.path, prjobj.ds_type[1], prjobj.pipeline)
        executed_steps = [f for f in os.listdir(pipeline_path) if os.path.isdir(os.path.join(pipeline_path, f))]
        if len(executed_steps):
            overlapped = [old_step for old_step in executed_steps if step in old_step]
            if len(overlapped):
                print('Notice: existing path')
                checked_files = []
                for f in os.walk(os.path.join(pipeline_path, overlapped[0])):
                    checked_files.extend(f[2])
                if len(checked_files):
                    print('Notice: Last step path is not empty')
                return overlapped[0]
            else:
                return "_".join([str(len(executed_steps)+1).zfill(3), step])
        else:
            print('First step for the pipeline{pipeline} is initiated'.format(pipeline=prjobj.pipeline))
            return "_".join([str(1).zfill(3), step])

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
        """
        shutil.copyfile(input_path, output_path)

    # Method collection for dynamic analysis
    @staticmethod
    def seed_coords(tractobj, start_point, end_point):
        data = tractobj.dataobj
        coords = np.argwhere(data == 1)
        coords = map(list, coords)
        coords.remove(start_point)
        seed_coords = []

        x, y, z = start_point
        cubic = data[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]
        cubic_corrds = np.argwhere(cubic == 1)
        cubic_results = []

        for dx, dy, dz in cubic_corrds:
            cubic_results.append([x + dx - 1, y + dy - 1, z + dz - 1])
        for cubic_corrd in cubic_results:
            if list(cubic_corrd) in (start_point, end_point):
                pass
            else:
                seed_coords.append(cubic_corrd)

        for i in range(len(coords)):
            x, y, z = seed_coords[-1]
            cubic = data[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]
            cubic_corrds = np.argwhere(cubic == 1)
            cubic_results = []

            for dx, dy, dz in cubic_corrds:
                cubic_results.append([x + dx - 1, y + dy - 1, z + dz - 1])
            for cubic_corrd in cubic_results:
                if list(cubic_corrd) in seed_coords:
                    pass
                else:
                    seed_coords.append(cubic_corrd)
        return seed_coords

    @staticmethod
    def gen_travel_seed(tractobj, start_point, end_point, filename=None):
        seed_coords = InternalMethods.seed_coords(tractobj, start_point, end_point)
        shape = list(tractobj.shape[:])
        shape.append(len(seed_coords))
        data = np.zeros(shape, np.int16)
        for i, coord in enumerate(seed_coords):
            x, y, z = seed_coords
            data[x, y, z, i] = 1
            data[x, y, z + 1, i] = 1
            data[x + 1, y, z, i] = 1
            data[x + 1, y, z + 1, i] = 1
        travelseed_obj = nib.Nifti1Image(data, tractobj.affine)
        if filename:
            travelseed_obj.to_filename(filename)
        return travelseed_obj


# class Usage(object):
#     @staticmethod
#     def project_run():
#         output = """ Usage: <prj_instance>.run(command)
#         """
#         return output