from __future__ import print_function

# Command execution
import os
from os.path import join

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
import numpy as np

import objects


class Internal(object):
    @staticmethod
    def gen_affine(resol=[1, 1, 1], coord=[0, 0, 0]):
        affine = nib.affines.from_matvec(np.diag(resol), coord)
        return affine

    @staticmethod
    def load(filename):
        if '.nii' in filename:
            img = objects.Image.load(filename)
        elif '.mha' in filename:
            try:
                mha = sitk.ReadImage(filename)
            except:
                raise ImportError('SimpleITK package is not imported.')
            data = sitk.GetArrayFromImage(mha)
            resol = mha.GetSpacing()
            origin = mha.GetOrigin()
            affine = nib.affines.from_matvec(np.diag(resol), origin)
            img = objects.Image(data, affine)
        else:
            raise IOError('File cannot be loaded')
        return img

    @staticmethod
    def set_viewaxes(axes):
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
    def down_reslice(obj, ac_slice, ac_loc, slice_thickness, total_slice, axis=2):
        data = np.asarray(obj.dataobj)
        resol, origin = nib.affines.to_matvec(obj.affine)
        resol = np.diag(resol).copy()
        scale = float(slice_thickness) / resol[axis]
        resol[axis] = slice_thickness
        idx = []
        for i in range(ac_loc):
            idx.append(ac_slice - int((ac_loc - i) * scale))
        for i in range(total_slice - ac_loc):
            idx.append(ac_slice + int(i * scale))
        print(idx)
        return data[:, :, idx]

    @staticmethod
    def get_timetrace(obj, maskobj, idx=1):
        data = np.asarray(obj.dataobj)
        data.flatten()
        mask = np.asarray(maskobj.dataobj)
        mask[mask!=idx] = False
        data = data[mask]
        return data

    @staticmethod
    def crop(obj, **kwargs):
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
        if len(obj.shape) == 3:
            obj._dataobj = obj._dataobj[x[0]:x[1], y[0]:y[1], z[0]:z[1]]
        if len(obj.shape) == 4:
            obj._dataobj = obj._dataobj[x[0]:x[1], y[0]:y[1], z[0]:z[1], t[0]:t[1]]

    @staticmethod
    def check_invert(kwargs):
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
        if invert[0]:
            data = nib.orientations.flip_axis(data, axis=0)
        if invert[1]:
            data = nib.orientations.flip_axis(data, axis=1)
        if invert[2]:
            data = nib.orientations.flip_axis(data, axis=2)
        return data

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
        empty_prj = False
        df = pd.DataFrame()
        for f in os.walk(join(path, ds_type[idx])):
            if f[2]:
                for filename in f[2]:
                    row = pd.Series(Internal.path_splitter(os.path.relpath(f[0], path)))
                    row['Filename'] = filename
                    row['Abspath'] = join(f[0], filename)
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
        Internal.mkdir(join(prj.path, prj.ds_type[0]), join(prj.path, prj.ds_type[1]), join(prj.path, prj.ds_type[2]))

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
            return None

    @staticmethod
    def mkdir(*paths):
        for path in paths:
            try:
                os.mkdir(path)
            except:
                pass

    @staticmethod
    def shihlab_cal_mean_cbv(output_path, input_path, postfix_bold='BOLD', postfix_cbv='CBV', *args, **kwargs):
        # Get average images from MION injection scan
        merged_output, args = Internal.check_merged_output(args)
        if merged_output:
            return None
        else:
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
    def shihlab_cal_mean(output_path, input_path, *args, **kwargs):
        merged_output, args = Internal.check_merged_output(args)
        if merged_output:
            return None
        else:
            img = nib.load(input_path)
            affn = img.get_affine()
            mean = np.average(img.get_data(), axis=3)
            nii_mean = nib.Nifti1Image(mean, affn)
            nii_mean.to_filename(output_path)

    @staticmethod
    def shihlab_copyfile(output_path, input_path, *args, **kwargs):
        merged_output, args = Internal.check_merged_output(args)
        if merged_output:
            return None
        else:
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
    def afni_3dTshift(output_path, input_path, tr=None, tpattern=None, *args, **kwargs):
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
        merged_output, args = Internal.check_merged_output(args)
        if merged_output:
            return None
        else:
            cmd = ['3dTshift', '-prefix', output_path]
            if tr:
                cmd.extend(['-TR', str(tr)])
            if tpattern:
                cmd.extend(['-tpattern', tpattern])
            cmd.append(input_path)
            cmd = list2cmdline(cmd)
            call(shl.split(cmd))

    @staticmethod
    def afni_3dvolreg(output_path, input_path, base_slice=0, *args, **kwargs):
        """Wrapper for 3dvolreg (Motion correction tool of AFNI package)

        Parameters
        ----------
        output_path : str
            explanation
        input_path : str
        base_slice : str
        """
        merged_output, args = Internal.check_merged_output(args)
        if merged_output:
            return None
        else:
            mpfile = os.path.splitext(output_path)[0] + '.1D'
            tfmfile = os.path.splitext(output_path)[0]
            cmd = ['3dvolreg', '-prefix', output_path, '-1Dfile', mpfile, '-1Dmatrix_save', tfmfile,
                   '-Fourier', '-verbose', '-base', str(base_slice), input_path]
            cmd = list2cmdline(cmd)
            call(shl.split(cmd))

    @staticmethod
    def afni_3dAllineate(output_path, input_path, *args, **kwargs):
        merged_output, args = Internal.check_merged_output(args)
        if merged_output:
            return None
        else:
            cmd = ['3dAllineate', '-prefix', output_path]
            if kwargs:
                for arg in kwargs.keys():
                    if arg == 'matrix_apply':
                        cmd.append('-1D{}'.format(arg))
                        cmd.append(kwargs[arg])
                    if arg == 'master':
                        if arg == 'base':
                            raise KeyError("'master' and 'base' arguments cannot be overlaped.")
                        else:
                            cmd.append('-{}'.format(arg))
                            cmd.append(kwargs[arg])
                    if arg == 'base':
                        if arg == 'master':
                            raise KeyError("'master' and 'base' arguments cannot be overlaped.")
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
    def afni_3dcalc(output_path, expr, *args, **kwargs):
        # AFNI image calculation (3dcalc)
        merged_output, inputs = Internal.check_merged_output(args)
        if merged_output:
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
        else:
            return None

    @staticmethod
    def afni_3dMean(output_path, *args, **kwargs):
        # AFNI 3dMean objects.Image calculator
        merged_output, inputs = Internal.check_merged_output(args)
        if merged_output:
            cmd = ['3dMean', '-prefix', output_path]
            cmd.extend(inputs)
            cmd = list2cmdline(cmd)
            call(shl.split(cmd))
        else:
            return None

    @staticmethod
    def afni_3dBandpass(output_path, input_path, norm=False, despike=False, mask=None, blur=False,
                        band=False, dt='1', *args, **kwargs):
        # AFNI signal processing for resting state (3dBandpass)
        merged_output, args = Internal.check_merged_output(args)
        if merged_output:
            return None
        else:
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
    def afni_3dmaskave(output_path, input_path, mask_path, *args, **kwargs):
        """ AFNI 3dmaskave command wrapper

        :return: list
            Average timeseries data from given ROI

        """
        merged_output, args = Internal.check_merged_output(args)
        if merged_output:
            return None
        else:
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
    def ants_BiasFieldCorrection(output_path, input_path, algorithm='n3', *args, **kwargs):
        """
        Execute the BiasFieldCorrection in the ANTs package

        :param output_path: absolute output path
        :param input_path: absolute input path
        :param algorithm: 'n3' or 'n4'
        """
        merged_output, args = Internal.check_merged_output(args)
        if merged_output:
            return None
        else:
            if algorithm == 'n3' or 'N3':
                cmd = ['N3BiasFieldCorrection', '3', input_path, output_path]
            elif algorithm == 'n4' or 'N4':
                cmd = ['N4BiasFieldCorrection', '-i', input_path, '-o', output_path]
            else:
                raise BaseException("One of 'n3' or 'n4' must be chosen for last argument")
            cmd = list2cmdline(cmd)
            call(shl.split(cmd))

    @staticmethod
    def ants_RegistrationSyn(output_path, input_path, base_path, quick=True, ttype='s', *args, **kwargs):
        # ANTs SyN registration
        merged_output, args = Internal.check_merged_output(args)
        if merged_output:
            return None
        else:
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
    def ants_WarpImageMultiTransform(output_path, input_path, base_path, *args, **kwargs):
        # ANTs applying transform
        merged_output, args = Internal.check_merged_output(args)
        if merged_output:
            return None
        else:
            cmd = ['Warp.ImageMultiTransform', '3', input_path, output_path, '-R', base_path]
            for arg in args:
                cmd.append(arg)
            cmd = list2cmdline(cmd)
            call(shl.split(cmd))
