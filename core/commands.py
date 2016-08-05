# -*- coding: utf-8 -*-

# Command execution
import os
import shlex as shl
import shutil
from string import ascii_lowercase as lc
from subprocess import call as call
from subprocess import list2cmdline
from subprocess import check_output

# Handling Nifti images and matrix
import nibabel as nib
import numpy as np
from .statics import InternalMethods, ErrorHandler


class Commands(object):
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
        merged_output, args = InternalMethods.check_merged_output(args)
        if merged_output:
            raise ErrorHandler.no_merge
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
        merged_output, args = InternalMethods.check_merged_output(args)
        if merged_output:
            raise ErrorHandler.no_merge
        else:
            mpfile = os.path.splitext(output_path)[0] + '.1D'
            tfmfile = os.path.splitext(output_path)[0]
            cmd = ['3dvolreg', '-prefix', output_path, '-1Dfile', mpfile, '-1Dmatrix_save', tfmfile,
                   '-Fourier', '-verbose', '-base', str(base_slice), input_path]
            cmd = list2cmdline(cmd)
            call(shl.split(cmd))

    @staticmethod
    def afni_3dAllineate(output_path, input_path, *args, **kwargs):
        merged_output, args = InternalMethods.check_merged_output(args)
        if merged_output:
            raise ErrorHandler.no_merge
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
        merged_output, inputs = InternalMethods.check_merged_output(args)
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
            raise ErrorHandler.not_merged

    @staticmethod
    def afni_3dMean(output_path, *args, **kwargs):
        # AFNI 3dMean image calculator
        merged_output, inputs = InternalMethods.check_merged_output(args)
        if merged_output:
            cmd = ['3dMean', '-prefix', output_path]
            cmd.extend(inputs)
            cmd = list2cmdline(cmd)
            call(shl.split(cmd))
        else:
            raise ErrorHandler.not_merged

    @staticmethod
    def afni_3dBandpass(output_path, input_path, norm=False, despike=False, mask=None, blur=False,
                        band=False, dt='1', *args, **kwargs):
        # AFNI signal processing for resting state (3dBandpass)
        merged_output, args = InternalMethods.check_merged_output(args)
        if merged_output:
            raise ErrorHandler.no_merge
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
        merged_output, args = InternalMethods.check_merged_output(args)
        if merged_output:
            raise ErrorHandler.no_merge
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
    def ants_BiasFieldCorrection(output_path, input_path, algorithm='n3'):
        """
        Execute the BiasFieldCorrection in the ANTs package

        :param output_path: absolute output path
        :param input_path: absolute input path
        :param algorithm: 'n3' or 'n4'
        """
        merged_output, args = InternalMethods.check_merged_output(args)
        if merged_output:
            raise ErrorHandler.no_merge
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
    def ants_RegistrationSyn(output_path, input_path, base_path, quick=True, ttype='s'):
        # ANTs SyN registration
        merged_output, args = InternalMethods.check_merged_output(args)
        if merged_output:
            raise ErrorHandler.no_merge
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
        merged_output, args = InternalMethods.check_merged_output(args)
        if merged_output:
            raise ErrorHandler.no_merge
        else:
            cmd = ['WarpImageMultiTransform', '3', input_path, output_path, '-R', base_path]
            for arg in args:
                cmd.append(arg)
            cmd = list2cmdline(cmd)
            call(shl.split(cmd))

    @staticmethod
    def shihlab_cal_mean_cbv(output_path, input_path, postfix_bold='BOLD', postfix_cbv='CBV', *args, **kwargs):
        # Get average images from MION injection scan
        merged_output, args = InternalMethods.check_merged_output(args)
        if merged_output:
            raise ErrorHandler.no_merge
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
        merged_output, args = InternalMethods.check_merged_output(args)
        if merged_output:
            raise ErrorHandler.no_merge
        else:
            img = nib.load(input_path)
            affn = img.get_affine()
            mean = np.average(img.get_data(), axis=3)
            nii_mean = nib.Nifti1Image(mean, affn)
            nii_mean.to_filename(output_path)

    @staticmethod
    def shihlab_copyfile(output_path, input_path, *args, **kwargs):
        merged_output, args = InternalMethods.check_merged_output(args)
        if merged_output:
            raise ErrorHandler.no_merge
        else:
            shutil.copyfile(input_path, output_path)
