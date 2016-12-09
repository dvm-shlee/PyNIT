# Standard library
import shlex as shl
from string import ascii_lowercase as lc
from subprocess import list2cmdline, check_output, call
import error
from shutil import rmtree
from .methods import InternalMethods, np, pd, os


class Analysis(object):
    """ Analysis tools for PyNIT objects
    """
    def __init__(self):
        self.__avail = [f for f in dir(self) if '__' not in f and 'avail' not in f]

    @property
    def avail(self):
        return self.__avail

    @staticmethod
    def linear_norm(imgobj, new_min, new_max):
        """Linear normalization of the grayscale digital image
        """
        data = np.asarray(imgobj.dataobj)
        data = (data - np.min(data)) * (new_max - new_min) / (np.max(data) - np.min(data)) - new_min
        imgobj._dataobj = data
        return imgobj

    @staticmethod
    def mask_average(imageobj, maskobj, **kwargs):
        """ Mask average timeseries
        """
        contra = None
        merged = None
        afni = None
        # Check kwargs
        if kwargs:
            for arg in kwargs.keys():
                if arg == 'contra':
                    contra = kwargs[arg]
                if arg == 'merge':
                    merged = kwargs[arg]
                if arg == 'afni':
                    afni = kwargs[arg]
        newshape = reduce(lambda x, y: x * y, imageobj.shape[:3])
        if afni:
            if contra:
                maskobj._dataobj = maskobj._dataobj[::-1, :, :]
            if merged:
                maskobj._dataobj += maskobj._dataobj[::-1, :, :]
            input_file = TempFile(imageobj, filename='input')
            mask_file = TempFile(maskobj, filename='mask')
            output = Interface.afni_3dmaskave(None, input_file, mask_file)
            input_file.close()
            mask_file.close()
            return pd.Series(output)
        else:
            data = np.asarray(imageobj.dataobj)
            mask = np.asarray(maskobj.dataobj)
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
    def get_timetrace(imageobj, tempobj, **kwargs):
        """ Parsing timetrace from imageobj, with multiple rois
        """
        contra = None
        bilateral = None
        merged = None
        afni = None
        quiet = None
        # Check kwargs
        if kwargs:
            for arg in kwargs.keys():
                if arg == 'contra':
                    contra = kwargs[arg]
                if arg == 'bilateral':
                    bilateral = kwargs[arg]
                if arg == 'merge':
                    merged = kwargs[arg]
                if arg == 'afni':
                    afni = kwargs[arg]
                if arg == 'quiet':
                    quiet = kwargs[arg]
        # Initiate dataframe
        df = pd.DataFrame()
        # Check each labels
        for idx in tempobj.label.keys():
            if idx:
                roi, maskobj = tempobj[idx]
                if merged:
                    col = Analysis.mask_average(imageobj, maskobj, merge=True, afni=afni)
                    roi = 'Bilateral_{}'.format(roi)
                else:
                    if contra:
                        col = Analysis.mask_average(imageobj, maskobj, contra=True, afni=afni)
                    else:
                        col = Analysis.mask_average(imageobj, maskobj, afni=afni)
                df[roi] = col
                if not quiet:
                    print("  * Time trace is extracted using the mask '{}'".format(roi))
        if bilateral:
            for idx in tempobj.label.keys():
                if idx:
                    roi, maskobj = tempobj[idx]
                    if merged:
                        pass
                    else:
                        if contra:
                            col = Analysis.mask_average(imageobj, maskobj, afni=afni)
                        else:
                            col = Analysis.mask_average(imageobj, maskobj, contra=True, afni=afni)
                        df["Cont_{}".format(roi)] = col
                        if not quiet:
                            print("  * Time trace is extracted using the mask 'Cont_{}'".format(roi))
        return df

    @staticmethod
    def cal_mean(imgobj, start=None, end=None):
        """ Calculate average
        """
        if not start:
            start = 0
        if not end:
            end = -1
        return np.average(imgobj.dataobj[..., start:end], axis=3)


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
        motion_parameter = InternalMethods.splitnifti(output_path) + '.1D'
        template_file = InternalMethods.splitnifti(output_path)
        cmd = ['3dvolreg', '-prefix', output_path, '-1Dfile', motion_parameter, '-1Dmatrix_save', template_file,
               '-Fourier', '-verbose', '-base']
        if type(base_slice) is int:
            cmd.append('{}'.format(int(base_slice)))
        elif type(base_slice) is str:
            cmd.append('{}'.format(base_slice))
        cmd.append(input_path)
        cmd = list2cmdline(cmd)
        call(shl.split(cmd))

    @staticmethod
    def afni_3dAllineate(output_path, input_path, **kwargs):
        cmd = ['3dAllineate', '-prefix', output_path]
        if kwargs:
            for arg in kwargs.keys():
                if arg == 'param_save':
                    cmd.append('-1D{}'.format(arg))
                    cmd.append(kwargs[arg])
                if arg == 'param_apply':
                    cmd.append('-1D{}'.format(arg))
                    cmd.append(kwargs[arg])
                if arg == 'matrix_save':
                    cmd.append('-1D{}'.format(arg))
                    cmd.append(kwargs[arg])
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
                if arg == 'cost':
                    cmd.append('-{}'.format(arg))
                    cmd.append(kwargs[arg])
                if arg == 'float':
                    if kwargs[arg]:
                        cmd.append('-{}'.format(arg))
                if arg == 'interp':
                    cmd.append('-{}'.format(arg))
                    cmd.append(kwargs[arg])
                if arg == 'final':
                    cmd.append('-{}'.format(arg))
                    cmd.append(kwargs[arg])
                # Technical option
                if arg == 'nmatch':
                    cmd.append('-{}'.format(arg))
                    cmd.append(kwargs[arg])
                if arg == 'nopad':
                    if kwargs[arg]:
                        cmd.append('-{}'.format(arg))
                if arg == 'zclip':
                    if kwargs[arg]:
                        cmd.append('-{}'.format(arg))
                if arg == 'conv':
                    cmd.append('-{}'.format(arg))
                    cmd.append(kwargs[arg])
                if arg == 'verb':
                    if kwargs[arg]:
                        cmd.append('-{}'.format(arg))
                if arg == 'quiet':
                    if kwargs[arg]:
                        cmd.append('-{}'.format(arg))
                if arg == 'usetemp':
                    if kwargs[arg]:
                        cmd.append('-{}'.format(arg))
                if arg == 'nousetemp':
                    if kwargs[arg]:
                        cmd.append('-{}'.format(arg))
                if arg == 'check':
                    cmd.append('-{}'.format(arg))
                    cmd.append(kwargs[arg])
                # PARAMETERS THAT AFFECT THE COST OPTIMIZATION STRATEGY
                if arg == 'onepass':
                    if kwargs[arg]:
                        cmd.append('-{}'.format(arg))
                if arg == 'twopass':
                    if kwargs[arg]:
                        cmd.append('-{}'.format(arg))
                if arg == 'twoblur':
                    cmd.append('-{}'.format(arg))
                    cmd.append(kwargs[arg])
                if arg == 'twofirst':
                    if kwargs[arg]:
                        cmd.append('-{}'.format(arg))
                if arg == 'twobest':
                    cmd.append('-{}'.format(arg))
                    cmd.append(kwargs[arg])
                if arg == 'fineblur':
                    cmd.append('-{}'.format(arg))
                    cmd.append(kwargs[arg])
                if arg == 'cmass':
                    if type(kwargs['cmass']) is str():
                        cmd.append('-{}{}'.format(arg, kwargs['cmass']))
                    else:
                        cmd.append('-{}'.format(arg))
                if arg == 'nocmass':
                    if kwargs[arg]:
                        cmd.append('-{}'.format(arg))
                if arg == 'autoweight':
                    if kwargs[arg]:
                        cmd.append('-{}'.format(arg))
                if arg == 'automask':
                    if kwargs[arg]:
                        cmd.append('-{}'.format(arg))
                if arg == 'autobox':
                    if kwargs[arg]:
                        cmd.append('-{}'.format(arg))
                if arg == 'nomask':
                    if kwargs[arg]:
                        cmd.append('-{}'.format(arg))
                if arg == 'EPI':
                    if kwargs[arg]:
                        cmd.append('-{}'.format(arg))
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
    def afni_3dTstat(output_path, input_path, **kwargs):
        # AFNI 3dTstat
        cmd = ['3dTstat', '-prefix', output_path]
        for kwarg in kwargs.keys():
            if kwargs[kwarg]:
                if 'mean' in kwarg:
                    cmd.append('-mean')
                if 'median' in kwarg:
                    cmd.append('-median')
                if 'nzmedian' in kwargs:
                    cmd.append('-nzmedian')
        cmd.append(input_path)
        cmd = list2cmdline(cmd)
        call(shl.split(cmd))

    @staticmethod
    def afni_3dBlurInMask(output_path, input_path, **kwargs):
        cmd = ['3dBlurInMask', '-prefix']
        cmd.append("'{}'".format(output_path))
        if kwargs:
            if 'FWHM' in kwargs.keys():
                cmd.append("-FWHM")
                cmd.append(kwargs['FWHM'])
            if 'mask' in kwargs.keys():
                if kwargs['mask']:
                    cmd.append("-mask")
                    cmd.append(kwargs['mask'])
                else:
                    pass
            else:
                cmd.append("-automask")
            if 'quiet' in kwargs.keys():
                cmd.append("-quiet")
        cmd.append("'{}'".format(input_path))
        print(cmd)
        cmd = list2cmdline(cmd)
        call(shl.split(cmd))

    @staticmethod
    def afni_3dBandpass(output_path, input_path, norm=False, detrend=False, despike=False, mask=None, blur=False,
                        band=False, dt='1'):
        # AFNI signal processing for resting state (3dBandpass)
        cmd = ['3dBandpass', '-input', input_path, '-prefix', output_path]
        if 'dt':
            if type(dt) is not str:
                dt = str(dt)
            cmd.extend(['-dt', dt])
        if norm:
            cmd.append('-norm')
        if despike:
            cmd.append('-despike')
        if not detrend:
            cmd.append('-nodetrend')
        if mask:
            cmd.extend(['-mask', mask])
        if blur:
            if type(blur) is not str:
                blur = str(blur)
            cmd.extend(['-blur', blur])
        if band:
            cmd.append('-band')
            band = map(str, band)
            cmd.extend(band)
        cmd = list2cmdline(cmd)
        call(shl.split(cmd))

    @staticmethod
    def afni_3dmaskave(output_path, input_path, mask_path):
        """ AFNI 3dmaskave command wrapper

        :return: list
            Average timeseries data from given ROI

        """
        cmd = ['3dmaskave', '-mask']
        cmd.append("'{}'".format(mask_path))
        cmd.append('-q')
        cmd.append("'{}'".format(input_path))
        cmd = list2cmdline(cmd)
        try:
            stdout = check_output(shl.split(cmd))
        except:
            print("Error: Empty mask.")
        else:
            if not output_path:
                stdout = stdout.split('\n')
                try:
                    stdout = map(float, stdout)
                except:
                    stdout.pop()
                    stdout = map(float, stdout)
                finally:
                    return stdout
            else:
                with open(output_path, 'w') as f:
                    f.write(stdout)

    @staticmethod
    def afni_3dDetrend(output_path, input_path, **kwargs):
        """ AFNI 3dDetrend command wrapper
        """
        cmd = ['3dDetrend', '-prefix']
        cmd.append("'{}'".format(output_path))
        if kwargs:
            if 'vector' in kwargs.keys():
                cmd.append("-vector")
                cmd.append(kwargs['vector'])
            if 'expr' in kwargs.keys():
                cmd.append("-expr")
                cmd.append(kwargs['expr'])
            if 'polort' in kwargs.keys():
                cmd.append("-polort")
                cmd.append(str(kwargs['polort']))
        cmd.append("'{}'".format(input_path))
        cmd = list2cmdline(cmd)
        call(shl.split(cmd))

    @staticmethod
    def afni_3dDeconvolve(output_path, input_path, **kwargs): # TODO: Not working for GLM analysis, need to fix
        """ AFNI 3Ddeconvolve command wrapper
        """
        cmd = ['3dDeconvolve']
        if input_path:
            cmd.append('-input')
            cmd.append("'{}'".format(str(input_path)))
            cmd.append('-num_stimts')
            cmd.append('1')
        if kwargs:
            for kwarg in kwargs.keys():
                cmd.append("-{}".format(kwarg))
                if type(kwargs[kwarg]) is list:
                    cmd.extend(kwargs[kwarg])
                elif type(kwargs[kwarg]) is str:
                    cmd.append(kwargs[kwarg])
                else:
                    cmd.append(str(kwargs[kwarg]))
        if output_path:
            if '.nii' in output_path:
                cmd.append('-tout')
                cmd.append('-bucket')
                cmd.append(str(output_path))
            elif '.1D' in output_path:
                cmd.append('-x1D')
                cmd.append(str(output_path))
            else:
                raise error.CommandExecutionFailure
        else:
            cmd.append('-x1D')
            cmd.append('stdout:')
            cmd = list2cmdline(cmd)
            stdout = check_output(shl.split(cmd))
            return stdout
        cmd = list2cmdline(cmd)
        # print(cmd)
        call(shl.split(cmd))

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
    def ants_RegistrationSyn(output_path, input_path, base_path, quick=True, ttype='s', thread=1):
        # ANTs SyN registration
        output_path = '{}_'.format(os.path.splitext(output_path)[0])
        if os.path.exists(str(output_path)+'Warped.nii.gz'):
            pass
        else:
            if not quick:
                script = 'antsRegistrationSyN.sh'
            else:
                script = 'antsRegistrationSyNQuick.sh'

            cmd = [script, '-t', ttype, '-f', base_path, '-m', input_path, '-o', output_path, '-n', str(thread)]
            cmd = list2cmdline(cmd)
            call(shl.split(cmd))

    @staticmethod
    def ants_WarpImageMultiTransform(output_path, input_path, base_path, *args, **kwargs):
        # ANTs applying transform
        cmd = ['WarpImageMultiTransform', '3', str(input_path), str(output_path), '-R', str(base_path)]
        if 'atlas' in kwargs.keys():
            if kwargs['atlas']:
                cmd.append('--use-NN')
            else:
                cmd.append('--use-BSpline')
        for arg in args:
            cmd.append(arg)
        cmd = list2cmdline(cmd)
        call(shl.split(cmd))
        # print(cmd)

    @staticmethod
    def ants_WarpTimeSeriesImageMultiTransform(output_path, input_path, base_path, *args, **kwargs):
        # ANTs applying transform
        cmd = ['WarpTimeSeriesImageMultiTransform', '4', str(input_path), str(output_path), '-R', str(base_path)]
        if 'atlas' in kwargs.keys():
            if kwargs['atlas']:
                cmd.append('--use-NN')
            else:
                cmd.append('--use-BSpline')
        for arg in args:
            cmd.append(arg)
        cmd = list2cmdline(cmd)
        call(shl.split(cmd))
        # print(cmd)


class TempFile(object):
    """Object for temporary file
    """
    def __init__(self, obj, filename='image_cache', atlas=False):
        if atlas:
            self._image = None
            self._atlas = obj
            self._atlas.extract('./.atlas_tmp')
            self._listdir = [ f for f in os.listdir('./.atlas_tmp') if '.nii' in f ]
        else:
            self._image = obj
            self._fname = filename
            InternalMethods.mkdir('./.tmp')
            self._image.save_as(os.path.join('./.tmp', filename), quiet=True)
            self._atlas = None

    def __getitem__(self, idx):
        if self._image:
            raise IndexError
        else:
            if self._atlas:
                return os.path.abspath(os.path.join('.atlas_tmp', self._listdir[idx]))
            else:
                return None

    def __repr__(self):
        if self._image:
            return os.path.abspath(os.path.join('.tmp', "{}.nii".format(self._fname)))
        else:
            if self._atlas:
                output = []
                for i, roi in enumerate(self._listdir):
                    output.append('{:>3}  {:>100}'.format(i, os.path.abspath(os.path.join('.atlas_tmp', roi))))
                return str('\n'.join(output))

    def close(self):
        if self._image:
            os.remove(os.path.join('.tmp', "{}.nii".format(self._fname)))
        if self._atlas:
            rmtree('.atlas_tmp', ignore_errors=True)
        self._atlas = None
        self._image = None
