import os
import sys

# import pprint
from shutil import rmtree
from subprocess import list2cmdline, check_output, call
from string import ascii_lowercase as lc
import shlex
from StringIO import StringIO
from methods import read_table, objects, np

# Multiprocessing module
import multiprocessing
from multiprocessing.pool import ThreadPool

# Import internal modules
from .objects import ImageObj
from .processors import TempFile
from .visualizers import Viewer

import messages
import methods

# Import modules for interfacing with jupyter notebook
jupyter_env = False
try:
    if len([key for key in sys.modules.keys() if 'ipykernel' in key]):
        from tqdm import tqdm_notebook as progressbar
        from ipywidgets import widgets
        from ipywidgets.widgets import HTML as title
        from IPython.display import display, display_html
        jupyter_env = True
    else:
        from tqdm import tqdm as progressbar
except:
    pass


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
            return methods.Series(output)
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
            return methods.Series(np.ma.average(output, axis=0))

    @staticmethod
    def get_timetrace(imageobj, tempobj, **kwargs):
        """ Parsing timetrace from imageobj, with multiple rois
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
                if arg == 'merged':
                    merged = kwargs[arg]
        # Initiate dataframe

        input_file = TempFile(imageobj, filename='input')
        if contra:
            mask_file = TempFile(tempobj.atlas_obj, filename='mask', flip=True)
        if bilateral:
            list_of_rois = [roi[0] for roi in tempobj.label.itervalues()][1:]
            mask_file = TempFile(tempobj.atlas_obj, filename='mask')
            df = Interface.afni_3dROIstats(None, input_file, mask_file)
            df.columns = list_of_rois

            list_of_rois = ['contra_'+roi[0] for roi in tempobj.label.itervalues()][1:]
            mask2_file = TempFile(tempobj.atlas_obj, filename='mask2', flip=True)
            cont_df = Interface.afni_3dROIstats(None, input_file, mask2_file)
            cont_df.columns = list_of_rois
            df = df.join(cont_df)
            # mask_file.close()
        else:
            if merged:
                new_atlas = tempobj.atlas_obj._dataobj + tempobj.atlas_obj._dataobj[::-1, :, :]
                list_of_rois = ['bilateral_' + roi[0] for roi in tempobj.label.itervalues()][1:]
            else:
                list_of_rois = [roi[0] for roi in tempobj.label.itervalues()][1:]
            nii = objects.ImageObj(new_atlas, tempobj.atlas_obj.affine)
            mask_file = TempFile(nii, filename='mask')
            df = Interface.afni_3dROIstats(None, input_file, mask_file)
            df.columns = list_of_rois
            mask_file.close()
        # Check each labels
        input_file.close()
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
        call(shlex.split(cmd))

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
        motion_parameter = methods.splitnifti(output_path) + '.1D'
        template_file = methods.splitnifti(output_path)
        cmd = ['3dvolreg', '-prefix', output_path, '-1Dfile', motion_parameter, '-1Dmatrix_save', template_file,
               '-Fourier', '-verbose', '-base']
        if type(base_slice) is int:
            cmd.append('{}'.format(int(base_slice)))
        elif type(base_slice) is str:
            cmd.append('{}'.format(base_slice))
        cmd.append(input_path)
        cmd = list2cmdline(cmd)
        call(shlex.split(cmd))

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
                        raise messages.ArgumentsOverlapped
                    else:
                        cmd.append('-{}'.format(arg))
                        cmd.append(kwargs[arg])
                if arg == 'base':
                    if arg == 'master':
                        raise messages.ArgumentsOverlapped
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
        call(shlex.split(cmd))

    @staticmethod
    def afni_3dTcorr1D(output_path, input_path, mask):
        cmd = ['3dTcorr1D', '-prefix', output_path, input_path, mask]
        cmd = list2cmdline(cmd)
        call(shlex.split(cmd))

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
        call(shlex.split(cmd))

    @staticmethod
    def afni_3dMean(output_path, *inputs):
        # AFNI 3dMean objects.Image calculator
        cmd = ['3dMean', '-prefix', output_path]
        cmd.extend(inputs)
        cmd = list2cmdline(cmd)
        call(shlex.split(cmd))

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
        call(shlex.split(cmd))

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
        # print(cmd)
        cmd = list2cmdline(cmd)
        call(shlex.split(cmd))

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
        call(shlex.split(cmd))

    @staticmethod
    def afni_3dTproject(output_path, input_path, ort=None, orange=None, pca=None, mask=None, norm=False, blur=False,
                        band=False, dt='1'):
        # AFNI signal processing for resting state (3dBandpass)
        cmd = ['3dTproject', '-input', input_path, '-prefix', output_path]
        if 'dt':
            if type(dt) is not str:
                dt = str(dt)
            cmd.extend(['-dt', dt])
        if ort:
            if orange:
                if isinstance(orange, list):
                    if len(orange) == 2:
                        orange = "'{"+"{}..{}".format(orange[0], orange[1])+"}'"
                        cmd.extend(['-ort', ort+orange])
            else:
                cmd.extend(['-ort', ort])
        if pca:
            cmd.extend(['-ort', pca])
        if mask:
            cmd.extend(['-mask', mask])
        if blur:
            if type(blur) is not str:
                blur = str(blur)
            cmd.extend(['-blur', blur])
        if band:
            cmd.append('-passband')
            band = map(str, band)
            cmd.extend(band)
        if norm:
            cmd.append('-norm')
        cmd = list2cmdline(cmd)
        # print(cmd)
        call(shlex.split(cmd))

    @staticmethod
    def afni_3dROIstats(output_path, input_path, mask_path):
        """ AFNI 3dROIstats command wrapper

        Parameters
        ----------
        output_path
        input_path
        mask_path

        Returns
        -------

        """
        cmd = ['3dROIstats', '-mask']
        cmd.append("'{}'".format(mask_path))
        cmd.append("'{}'".format(input_path))
        cmd = list2cmdline(cmd)
        out, err = methods.shell(cmd)
        if output_path == None:
            df = read_table(StringIO(out))
            df = df[df.columns[2:]]
            return df
        else:
            with open(output_path, 'w') as f:
                f.write(out)

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
            stdout = check_output(shlex.split(cmd))
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
        call(shlex.split(cmd))

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
                raise messages.CommandExecutionFailure
        else:
            cmd.append('-x1D')
            cmd.append('stdout:')
            cmd = list2cmdline(cmd)
            stdout = check_output(shlex.split(cmd))
            return stdout
        cmd = list2cmdline(cmd)
        # print(cmd)
        call(shlex.split(cmd))

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
        call(shlex.split(cmd))

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
            call(shlex.split(cmd))

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
        call(shlex.split(cmd))
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
        call(shlex.split(cmd))
        # print(cmd)


class Preprocess(object):
    """ Preprocessing pipeline
    """
    def __init__(self, prjobj, pipeline):
        prjobj.reset_filters()
        self._subjects = None
        self._sessions = None
        self._processing = None
        self._prjobj = prjobj
        self.reset()
        self.initiate_pipeline(pipeline)

    @property
    def processing(self):
        return self._processing

    @property
    def subjects(self):
        return self._subjects

    @property
    def sessions(self):
        return self._sessions

    def reset(self):
        if self._prjobj.subjects:
            self._subjects = sorted(self._prjobj.subjects[:])
            if not self._prjobj.single_session:
                self._sessions = sorted(self._prjobj.sessions[:])

    def initiate_pipeline(self, pipeline):
        pipe_path = os.path.join(self._prjobj.path, self._prjobj.ds_type[1], pipeline)
        methods.mkdir(pipe_path)
        self._processing = pipeline

    def init_step(self, stepname):
        if self._processing:
            steppath = methods.get_step_name(self, stepname)
            steppath = os.path.join(self._prjobj.path, self._prjobj.ds_type[1], self._processing, steppath)
            methods.mkdir(steppath)
            return steppath
        else:
            raise messages.PipelineNotSet

    def seedbased_dynamic_connectivity(self, func, seed, winsize=100, step=1, dtype='func', **kwargs):
        """ Seed-based dynamic connectivity using sliding windows # TODO: use this methods as model

        Parameters
        ----------
        func : str
            Root path for input dataset
        seed : str
            Absolute path for seed image
        winsize : int
            Size of sliding window
        step : int
            Moving step for sliding window
        dtype : str
            Surfix for output folder
        kwargs : dict
            Options (Nothing available)

        Returns
        -------
        path : dict
            output path
        """
        dataclass, func = methods.check_dataclass(func)
        print('SeedBaseDynamicConnectivityMap-{}'.format(func))
        step01 = self.init_step('SeedBaseDynamicConnectivityMap-{}'.format(dtype))
        if not os.path.isfile(seed):
            methods.raiseerror(ValueError, 'Input file does not exist.')

        def processor(result, temppath, finfo, seed_path, winsize, i):
            temp_path = os.path.join(temppath, "{}.nii".format(str(i).zfill(5)))
            self._prjobj.run('afni_3dTcorr1D', temp_path,
                             "{0}'[{1}..{2}]'".format(finfo.Abspath, i, i + winsize - 1),
                             "{0}'{{{1}..{2}}}'".format(seed_path, i, i + winsize - 1))
            result.append(temp_path)

        def worker(args):
            """

            Parameters
            ----------
            func
            args

            Returns
            -------

            """
            return processor(*args)

        def start_process():
            print 'Starting', multiprocessing.current_process().name

        for subj in progressbar(self.subjects, desc='Subjects', leave=False):
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                epi = self._prjobj(dataclass, self._processing, func, subj, **kwargs)
                for i, finfo in progressbar(epi, desc='Files', leave=False):
                    print(" +Filename: {}".format(finfo.Filename))
                    # Check dimension
                    total, err = methods.shell('3dinfo -nv {}'.format(finfo.Abspath))
                    if err:
                        methods.raiseerror(NameError, 'Cannot load: {0}'.format(finfo.Filename))
                    total = int(total)
                    print(total)
                    if total <= winsize:
                        methods.raiseerror(KeyError,
                                           'Please use proper windows size: [less than {}]'.format(str(total)))
                    seed_path = os.path.join(step01, subj, "{0}.1D".format(methods.splitnifti(finfo.Filename)))
                    self._prjobj.run('afni_3dmaskave', seed_path, finfo.Abspath, seed)
                    output_path = os.path.join(step01, subj, finfo.Filename)
                    if not os.path.exists(output_path):
                        methods.mkdir('.dtmp')
                        temppath = os.path.join('.dtmp')
                        list_of_files = []
                        cpu = multiprocessing.cpu_count()
                        cpu = cpu - int(cpu / 4)
                        pool = ThreadPool(cpu, initializer=start_process)
                        iteritem = [(list_of_files, temppath, finfo, seed_path, winsize, i) for i in range(0, total - winsize, step)]
                        for output in progressbar(pool.imap_unordered(worker, iteritem), desc='Window',
                                                  total=len(iteritem) ,leave=False):
                            pass
                        list_of_files = sorted(list_of_files)
                        methods.shell('3dTcat -prefix {0} -tr {1} {2}'.format(output_path, str(step),
                                                                              ' '.join(list_of_files)))
                        rmtree(temppath)
                        pool.close()
                        pool.join()
            else:
                for sess in progressbar(self.sessions, desc='Sessions', leave=False):
                    methods.mkdir(os.path.join(step01, subj, sess))
                    epi = self._prjobj(dataclass, self._processing, func, subj, sess, **kwargs)
                    for i, finfo in progressbar(epi, desc='Files', leave=False):
                        print(" +Filename: {}".format(finfo.Filename))
                        # Check dimension
                        total, err = methods.shell('3dinfo -nv {0}'.format(finfo.Abspath))
                        if err:
                            methods.raiseerror(NameError, 'Cannot load: {0}'.format(finfo.Filename))
                        total = int(total)
                        if total <= winsize:
                            methods.raiseerror(KeyError,
                                               'Please use proper windows size: [less than {}]'.format(str(total)))
                        seed_path = os.path.join(step01, subj, sess,
                                                 "{0}.1D".format(methods.splitnifti(finfo.Filename)))
                        self._prjobj.run('afni_3dmaskave', seed_path, finfo.Abspath, seed)
                        output_path = os.path.join(step01, subj, sess, finfo.Filename)
                        if not os.path.exists(output_path):
                            methods.mkdir('.dtmp')
                            temppath = os.path.join('.dtmp')
                            list_of_files = []
                            cpu = multiprocessing.cpu_count()
                            cpu = cpu - int(cpu/4)
                            pool = ThreadPool(cpu, initializer=start_process)
                            iteritem = [(list_of_files, temppath, finfo, seed_path, winsize, i) for i in
                                        range(0, total - winsize, step)]
                            for output in progressbar(pool.imap_unordered(worker, iteritem), desc='Window',
                                                      total=len(iteritem), leave=False):
                                pass
                            list_of_files = sorted(list_of_files)
                            methods.shell('3dTcat -prefix {0} -tr {1} {2}'.format(output_path, str(step),
                                                                                  ' '.join(list_of_files)))
                            rmtree(temppath)
                            pool.close()
                            pool.join()
        return {'dynamicMap': step01}

    def calculate_seedbased_global_connectivity(self, func, seed, dtype='func', **kwargs):
        """ Seed-based Global Connectivity Analysis

        Parameters
        ----------
        func : str
            Root path for input dataset
        seed : str
            Absolute path for seed image
        Returns
        -------
        path : dict
            Absolute path for output
        """
        dataclass, func = methods.check_dataclass(func)
        print('SeedBaseConnectivityMap-{}'.format(func))
        step01 = self.init_step('SeedBaseConnectivityMap-{}'.format(dtype))
        if not os.path.isfile(seed):
            methods.raiseerror(ValueError, 'Input file does not exist.')
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                epi = self._prjobj(dataclass, self._processing, func, subj, **kwargs)
                for i, finfo in epi:
                    print(" +Filename: {}".format(finfo.Filename))
                    seed_path = os.path.join(step01, subj, "{0}.1D".format(methods.splitnifti(finfo.Filename)))
                    output_path = os.path.join(step01, subj, finfo.Filename)
                    self._prjobj.run('afni_3dmaskave', seed_path, finfo.Abspath, seed)
                    self._prjobj.run('afni_3dTcorr1D', output_path, finfo.Abspath, seed_path)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    epi = self._prjobj(dataclass, self._processing, func, subj, sess, **kwargs)
                    for i, finfo in epi:
                        print("  +Filename: {}".format(finfo.Filename))
                        seed_path = os.path.join(step01, subj, sess, "{0}.1D".format(methods.splitnifti(finfo.Filename)))
                        output_path = os.path.join(step01, subj, sess, finfo.Filename)
                        self._prjobj.run('afni_3dmaskave', seed_path, finfo.Abspath, seed)
                        self._prjobj.run('afni_3dTcorr1D', output_path, finfo.Abspath, seed_path)
        return {'connMap': step01}

    def fisher_transform(self, func, dtype='func', **kwargs):
        dataclass, func = methods.check_dataclass(func)
        print('FisherTransform-{}'.format(func))
        step01 = self.init_step('FisherTransform-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                epi = self._prjobj(dataclass, self._processing, func, subj, **kwargs)
                for i, finfo in epi:
                    print(" +Filename: {}".format(finfo.Filename))
                    output_path = os.path.join(step01, subj, finfo.Filename)
                    self._prjobj.run('afni_3dcalc', output_path, 'atanh(a)', finfo.Abspath)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    epi = self._prjobj(dataclass, self._processing, func, subj, sess, **kwargs)
                    for i, finfo in epi:
                        print("  +Filename: {}".format(finfo.Filename))
                        output_path = os.path.join(step01, subj, sess, finfo.Filename)
                        self._prjobj.run('afni_3dcalc', output_path, 'atanh(a)', finfo.Abspath)
        self._prjobj.reset(True)
        self._prjobj.apply()
        return {'Zmap': step01}

    def group_average(self, func, subjects, sessions=None, group='groupA', **kwargs):
        """ Calculate group average

        Parameters
        ----------
        func : str
            Root path for input dataset
        subjects : list
            list of the subjects for group
        sessions : list
            list of the sessions that want to calculate average

        Returns
        -------

        """
        dataclass, func = methods.check_dataclass(func)
        print('AverageGroupData-{0}: [{1}]'.format(func, ', '.join(subjects)))
        step01 = self.final_step('AverageGRoupData-{}'.format(group))
        root_output = os.path.join(step01, 'AllSubjects')
        methods.mkdir(root_output)
        if sessions:
            print("*Average image will be calculated for each session listed in: [{0}]".format(', '.join(sessions)))
            for sess in sessions:
                print("-Session: {}".format(sess))
                sess_output = os.path.join(root_output, sess)
                methods.mkdir(sess_output)
                epi = self._prjobj(dataclass, self._processing, func, sess, *subjects, **kwargs).df.Abspath
                grouplist = [path for path in epi.to_dict().values()]
                print(" :List of subjects in {}".format(sess))
                print("\n  ".join(grouplist))
                self._prjobj.run('afni_3dMean', os.path.join(sess_output, '{0}-{1}.nii.gz'.format(group, sess)),
                                 *grouplist)
        else:
            epi = self._prjobj(dataclass, self._processing, func, *subjects, **kwargs).df.Abspath
            grouplist = [path for path in epi.to_dict().values()]
            print(":List of subjects in this group")
            print("\n ".join(grouplist))
            self._prjobj.run('afni_3dMean', os.path.join(root_output, '{0}.nii.gz'.format(group)),
                             *grouplist)
        return {"groupavr": step01}

    def cbv_meancalculation(self, func, **kwargs):
        """ CBV image preparation
        """
        dataclass, func = methods.check_dataclass(func)
        print("MotionCorrection")
        step01 = self.init_step('MotionCorrection-CBVinduction')
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                cbv_img = self._prjobj(dataclass, func, subj, **kwargs)
                for i, finfo in cbv_img:
                    print(" +Filename: {}".format(finfo.Filename))
                    self._prjobj.run('afni_3dvolreg', os.path.join(step01, subj, finfo.Filename), finfo.Abspath)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    cbv_img = self._prjobj(dataclass, func, subj, sess, **kwargs)
                    for i, finfo in cbv_img:
                        print("  +Filename: {}".format(finfo.Filename))
                        self._prjobj.run('afni_3dvolreg', os.path.join(step01, subj, sess, finfo.Filename),
                                         finfo.Abspath)
        self._prjobj.reset(True)
        self._prjobj.apply()
        step02 = self.init_step('MeanImageCalculation-BOLD')
        step03 = self.init_step('MeanImageCalculation-CBV')
        print("MeanImageCalculation-BOLD&CBV")
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step02, subj), os.path.join(step03, subj))
            if self._prjobj.single_session:
                cbv_img = self._prjobj(1, self._processing, os.path.basename(step01), subj, **kwargs)
                for i, finfo in cbv_img:
                    print(" +Filename: {}".format(finfo.Filename))
                    shape = ImageObj.load(finfo.Abspath).shape
                    self._prjobj.run('afni_3dTstat', os.path.join(step02, subj, finfo.Filename),
                                     "{path}'[{start}..{end}]'".format(path=finfo.Abspath,
                                                                       start=0,
                                                                       # end=20))
                                                                       end=(int(shape[-1] / 3))))
                    self._prjobj.run('afni_3dTstat', os.path.join(step03, subj, finfo.Filename),
                                     "{path}'[{start}..{end}]'".format(path=finfo.Abspath,
                                                                       # start=int(shape[-1]-21),
                                                                       start=int(shape[-1] * 2 / 3),
                                                                       end=shape[-1] - 1))
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step02, subj, sess), os.path.join(step03, subj, sess))
                    cbv_img = self._prjobj(1, os.path.basename(step01), subj, sess, **kwargs)
                    for i, finfo in cbv_img:
                        print("  +Filename: {}".format(finfo.Filename))
                        shape = methods.load(finfo.Abspath).shape
                        self._prjobj.run('afni_3dTstat', os.path.join(step02, subj, sess, finfo.Filename),
                                         "{path}'[{start}..{end}]'".format(path=finfo.Abspath,
                                                                           start=0,
                                                                           # end=20))
                                                                           end=(int(shape[-1] / 3))))
                        self._prjobj.run('afni_3dTstat', os.path.join(step03, subj, sess, finfo.Filename),
                                         "{path}'[{start}..{end}]'".format(path=finfo.Abspath,
                                                                           start=int(shape[-1] * 2 / 3),
                                                                           # start=int(shape[-1]-21),
                                                                           end=shape[-1] - 1))
        self._prjobj.reset(True)
        self._prjobj.apply()
        return {'CBVinduction': step01, 'meanBOLD': step02, 'meanCBV': step03}

    def mean_calculation(self, func, dtype='func', **kwargs):
        """ BOLD image preparation

        Parameters
        ----------
        func       : str
            Datatype or absolute path of the input mean functional image
        dtype      : str
            Surfix for step path

        Returns
        -------
        step_paths : dict
        """
        dataclass, func = methods.check_dataclass(func)
        step01 = self.init_step('InitialPreparation-{}'.format(dtype))
        print("MotionCorrection")
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                finfo = self._prjobj(dataclass, func, subj, **kwargs).df.loc[0]
                print(" +Filename: {}".format(finfo.Filename))
                self._prjobj.run('afni_3dvolreg', os.path.join(step01, subj, finfo.Filename), finfo.Abspath)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    finfo = self._prjobj(dataclass, func, subj, sess, **kwargs).df.loc[0]
                    print("  +Filename: {}".format(finfo.Filename))
                    self._prjobj.run('afni_3dvolreg', os.path.join(step01, subj, sess, finfo.Filename), finfo.Abspath)
        self._prjobj.reset(True)
        self._prjobj.apply()
        step02 = self.init_step('MeanImageCalculation-{}'.format(dtype))
        print("MeanImageCalculation-{}".format(func))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step02, subj))
            if self._prjobj.single_session:
                funcs = self._prjobj(1, self._processing, os.path.basename(step01), subj, **kwargs)
                funcs = funcs.df.loc[0]
                print(" +Filename: {}".format(funcs.Filename))
                self._prjobj.run('afni_3dTstat', os.path.join(step02, subj, funcs.Filename),
                                 funcs.Abspath)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step02, subj, sess))
                    funcs = self._prjobj(1, self._processing, os.path.basename(step01), subj, sess, **kwargs)
                    funcs = funcs.df.loc[0]
                    print(" +Filename: {}".format(funcs.Filename))
                    self._prjobj.run('afni_3dTstat', os.path.join(step02, subj, sess, funcs.Filename),
                                     funcs.Abspath)
        self._prjobj.reset(True)
        self._prjobj.apply()
        return {'firstfunc': step01, 'meanfunc': step02}

    def slicetiming_correction(self, func, tr=None, tpattern='altplus', dtype='func', **kwargs):
        """ Corrects for slice time differences when individual 2D slices are recorded over a 3D image

        Parameters
        ----------
        func       : str
            Data type or absolute path of the input functional image
        tr         : int
        tpattern   : str
        dtype      : str
            Surfix for the step paths

        Returns
        -------
        step_paths : dict
        """
        dataclass, func = methods.check_dataclass(func)
        # if os.path.exists(func):
        #     dataclass = 1
        #     func = methods.path_splitter(func)[-1]
        # else:
        #     dataclass = 0
        print('SliceTimingCorrection-{}'.format(func))
        step01 = self.init_step('SliceTimingCorrection-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                epi = self._prjobj(dataclass, func, subj, **kwargs)
                for i, finfo in epi:
                    print(" +Filename: {}".format(finfo.Filename))
                    self._prjobj.run('afni_3dTshift', os.path.join(step01, subj, finfo.Filename),
                                     finfo.Abspath, tr=tr, tpattern=tpattern)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    epi = self._prjobj(dataclass, func, subj, sess, **kwargs)
                    for i, finfo in epi:
                        print("  +Filename: {}".format(finfo.Filename))
                        self._prjobj.run('afni_3dTshift', os.path.join(step01, subj, sess, finfo.Filename),
                                         finfo.Abspath, tr=tr, tpattern=tpattern)
        self._prjobj.reset(True)
        self._prjobj.apply()
        return {'func': step01}

    def motion_correction(self, func, base=None, baseidx=0, meancbv=None, dtype='func', **kwargs):
        """ Corrects for motion artifacts in the  input functional image

        Parameters
        ----------
        func       : str
            Datatype or absolute step path for the input functional image
        base       : str
            Datatype or absolute step path for the mean functional image
        baseidx    :
        meancbv   :
        dtype      : str
            Surfix for the step path


        Returns
        -------
        step_paths : dict
        """
        s0_dataclass, s0_func = methods.check_dataclass(func)
        print('MotionCorrection-{}'.format(func))
        step01 = self.init_step('MotionCorrection-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                epi = self._prjobj(s0_dataclass, s0_func, subj, **kwargs)
                if base:
                    if type(base) == str:
                        meanimg = self._prjobj(1, self._processing, os.path.basename(base), subj, **kwargs)
                        meanimg = meanimg.df.Abspath[baseidx]

                    else:
                        meanimg = base
                else:
                    meanimg = 0
                for i, finfo in epi:
                    print(" +Filename: {}".format(finfo.Filename))
                    self._prjobj.run('afni_3dvolreg', os.path.join(step01, subj, finfo.Filename), finfo.Abspath,
                                     base_slice=meanimg)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    epi = self._prjobj(s0_dataclass, s0_func, subj, sess, **kwargs)
                    if base:
                        if type(base) == str:
                            meanimg = self._prjobj(1, self._processing, os.path.basename(base), subj, sess)
                            meanimg = meanimg.df.Abspath[baseidx]
                        else:
                            meanimg = base
                    else:
                        meanimg = 0
                    for i, finfo in epi:
                        print("  +Filename: {}".format(finfo.Filename))
                        self._prjobj.run('afni_3dvolreg', os.path.join(step01, subj, sess, finfo.Filename),
                                         finfo.Abspath, base_slice=meanimg)
        self._prjobj.reset(True)
        self._prjobj.apply()
        if meancbv:
            # Calculate mean image for each 3D+time data
            s1_dataclass, s1_func = methods.check_dataclass(step01)
            print('MeanCalculation-{}'.format(s1_func))
            step02 = self.init_step('MeanFunctionalImages-{}'.format(dtype))
            for subj in self.subjects:
                print("-Subject: {}".format(subj))
                methods.mkdir(os.path.join(step02, subj))
                if self._prjobj.single_session:
                    epi = self._prjobj(s1_dataclass, s1_func, subj, **kwargs)
                    for i, finfo in epi:
                        print(" +Filename: {}".format(finfo.Filename))
                        self._prjobj.run('afni_3dTstat', os.path.join(step02, subj, finfo.Filename), finfo.Abspath,
                                         mean=True)
                else:
                    for sess in self.sessions:
                        print(" :Session: {}".format(sess))
                        methods.mkdir(os.path.join(step02, subj, sess))
                        epi = self._prjobj(s1_dataclass, s1_func, subj, sess, **kwargs)

                        for i, finfo in epi:
                            print("  +Filename: {}".format(finfo.Filename))
                            self._prjobj.run('afni_3dTstat', os.path.join(step02, subj, sess, finfo.Filename),
                                             finfo.Abspath, mean=True)
            # Realigning each run of CBV images
            s2_dataclass, s2_func = methods.check_dataclass(step02)
            self._prjobj.reset(True)
            self._prjobj.apply()
            print('IntersubjectRealign-{}'.format(func))
            step03 = self.init_step('InterSubjectRealign-{}'.format(dtype))
            for subj in self.subjects:
                print("-Subject: {}".format(subj))
                methods.mkdir(os.path.join(step03, subj))
                if self._prjobj.single_session:
                    epi = self._prjobj(s2_dataclass, s2_func, subj, **kwargs)
                    try:
                        baseimg = self._prjobj(1, os.path.basename(meancbv), subj).df.Abspath[0]
                    except:
                        baseimg = 0
                    for i, finfo in epi:
                        print(" +Filename: {}".format(finfo.Filename))
                        output_path = os.path.join(step03, subj, finfo.Filename)
                        self._prjobj.run('afni_3dAllineate', output_path, finfo.Abspath,
                                         base=baseimg, warp='sho',
                                         matrix_save=methods.splitnifti(output_path) + '.aff12.1D')
                else:
                    for sess in self.sessions:
                        print(" :Session: {}".format(sess))
                        methods.mkdir(os.path.join(step03, subj, sess))
                        epi = self._prjobj(s2_dataclass, s2_func, subj, sess, **kwargs)
                        try:
                            baseimg = self._prjobj(1, os.path.basename(meancbv), subj, sess)
                            baseimg = baseimg.df.Abspath[0]
                        except:
                            baseimg = 0
                        for i, finfo in epi:
                            print("  +Filename: {}".format(finfo.Filename))
                            output_path = os.path.join(step03, subj, sess, finfo.Filename)
                            self._prjobj.run('afni_3dAllineate', output_path,
                                             finfo.Abspath, base=baseimg, warp='sho',
                                             matrix_save=methods.splitnifti(output_path) + '.aff12.1D')
            # Realigning each run of CBV images
            s3_dataclass, s3_func = methods.check_dataclass(step03)
            self._prjobj.reset(True)
            self._prjobj.apply()
            print('InterSubj-ApplyTranform-{}'.format(func))
            step04 = self.init_step('InterSubj-ApplyTransform-{}'.format(dtype))
            for subj in self.subjects:
                print("-Subject: {}".format(subj))
                methods.mkdir(os.path.join(step04, subj))
                if self._prjobj.single_session:
                    param = self._prjobj(s3_dataclass, s3_func, subj).df
                    epi = self._prjobj(s1_dataclass, s1_func, subj, **kwargs)
                    for i, finfo in epi:
                        print(" +Filename: {}".format(finfo.Filename))
                        try:
                            if finfo.Filename != param.Filename[i]:
                                raise messages.ObjectMismatch()
                            else:
                                self._prjobj.run('afni_3dAllineate', os.path.join(step04, subj, finfo.Filename),
                                                 finfo.Abspath, warp='sho',
                                                 matrix_apply=methods.splitnifti(param.Abspath[i])+'.aff12.1D')
                        except:
                            print('  ::Skipped')
                else:
                    for sess in self.sessions:
                        print(" :Session: {}".format(sess))
                        methods.mkdir(os.path.join(step04, subj, sess))
                        param = self._prjobj(s3_dataclass, s3_func, subj, sess).df
                        epi = self._prjobj(s1_dataclass, s1_func, subj, sess, **kwargs)
                        for i, finfo in epi:
                            print(" +Filename: {}".format(finfo.Filename))
                            try:
                                if finfo.Filename != param.Filename[i]:
                                    print finfo.Filename, param.Filename[i]
                                    raise messages.ObjectMismatch()
                                else:
                                    self._prjobj.run('afni_3dAllineate', os.path.join(step04, subj, sess, finfo.Filename),
                                                    finfo.Abspath, warp='sho',
                                                    matrix_apply=methods.splitnifti(param.Abspath[i]) + '.aff12.1D')
                            except:
                                print('  ::Skipped')
            self._prjobj.reset(True)
            self._prjobj.apply()
            return {'func': step04, 'mparam': step01}
        else:
            self._prjobj.reset(True)
            self._prjobj.apply()
            return {'func': step01}

    def maskdrawing_preparation(self, meanfunc, anat, padding=False, zaxis=2):
        """

        Parameters
        ----------
        meanfunc
        anat
        padding
        zaxis

        Returns
        -------

        """
        f_dataclass, meanfunc = methods.check_dataclass(meanfunc)
        a_dataclass, anat = methods.check_dataclass(anat)
        print('MaskDrawing-{} & {}'.format(meanfunc, anat))

        step01 = self.init_step('MaskDrwaing-func')
        step02 = self.init_step('MaskDrawing-anat')
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj), os.path.join(step02, subj))
            if self._prjobj.single_session:
                epi = self._prjobj(f_dataclass, meanfunc, subj)
                t2 = self._prjobj(a_dataclass, anat, subj)
                for i, finfo in epi:
                    print(" +Filename: {}".format(finfo.Filename))
                    epiimg = methods.load(finfo.Abspath)
                    if padding:
                        epiimg.padding(low=1, high=1, axis=zaxis)
                    epiimg.save_as(os.path.join(step01, subj, finfo.Filename), quiet=True)
                for i, finfo in t2:
                    print(" +Filename: {}".format(finfo.Filename))
                    t2img = methods.load(finfo.Abspath)
                    if padding:
                        t2img.padding(low=1, high=1, axis=zaxis)
                    t2img.save_as(os.path.join(step02, subj, finfo.Filename), quiet=True)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess), os.path.join(step02, subj, sess))
                    epi = self._prjobj(f_dataclass, meanfunc, subj, sess)
                    t2 = self._prjobj(a_dataclass, anat, subj, sess)
                    for i, finfo in epi:
                        print("  +Filename: {}".format(finfo.Filename))
                        epiimg = methods.load(finfo.Abspath)
                        if padding:
                            epiimg.padding(low=1, high=1, axis=zaxis)
                        epiimg.save_as(os.path.join(step01, subj, sess, finfo.Filename), quiet=True)
                    for i, finfo in t2:
                        print("  +Filename: {}".format(finfo.Filename))
                        print(finfo.Abspath)
                        t2img = methods.load(finfo.Abspath)
                        if padding:
                            t2img.padding(low=1, high=1, axis=zaxis)
                        t2img.save_as(os.path.join(step02, subj, sess, finfo.Filename), quiet=True)
        self._prjobj.reset(True)
        self._prjobj.apply()
        return {'meanfunc': step01, 'anat': step02}

    def compute_skullstripping(self, meanfunc, anat, padded=False, zaxis=2):
        self._prjobj.reset(True)
        self._prjobj.apply()
        axis = {0: 'x', 1: 'y', 2: 'z'}
        f_dataclass, meanfunc = methods.check_dataclass(meanfunc)
        a_dataclass, anat = methods.check_dataclass(anat)
        print('SkullStripping-{} & {}'.format(meanfunc, anat))
        step01 = self.init_step('SkullStripped-meanfunc')
        step02 = self.init_step('SkullStripped-anat')
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj), os.path.join(step02, subj))
            if self._prjobj.single_session:
                # Load image paths
                epi = self._prjobj(1, self._processing, meanfunc, subj, ignore='_mask')
                t2 = self._prjobj(1, self._processing, anat, subj, ignore='_mask')
                # Load mask image obj
                epimask = self._prjobj(1, self._processing, meanfunc, subj, file_tag='_mask').df.Abspath[0]
                t2mask = self._prjobj(1, self._processing, anat, subj, file_tag='_mask').df.Abspath[0]
                # Execute process
                for i, finfo in epi:
                    print(" +Filename of meanfunc: {}".format(finfo.Filename))
                    filename = finfo.Filename
                    fpath = os.path.join(step01, subj, filename)
                    self._prjobj.run('afni_3dcalc', fpath, 'a*step(b)',
                                     finfo.Abspath, epimask)
                    ss_epi = methods.load(fpath)
                    if padded:
                        exec('ss_epi.crop({}=[1, {}])'.format(axis[zaxis], ss_epi.shape[zaxis]-1))
                        ss_epi.save_as(os.path.join(step01, subj, filename), quiet=True)
                for i, finfo in t2:
                    print(" +Filename of anat: {}".format(finfo.Filename))
                    filename = finfo.Filename
                    fpath = os.path.join(step02, subj, filename)
                    self._prjobj.run('afni_3dcalc', fpath, 'a*step(b)',
                                     finfo.Abspath, t2mask)
                    ss_t2 = methods.load(fpath)
                    if padded:
                        exec('ss_t2.crop({}=[1, {}])'.format(axis[zaxis], ss_t2.shape[zaxis] - 1))
                        ss_t2.save_as(os.path.join(step02, subj, filename), quiet=True)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess), os.path.join(step02, subj, sess))
                    # Load image paths
                    epi = self._prjobj(1, self._processing, meanfunc, subj, sess, ignore='_mask')
                    t2 = self._prjobj(1, self._processing, anat, subj, sess, ignore='_mask')
                    # Load mask image obj
                    epimask = self._prjobj(1, self._processing, meanfunc, subj, sess, file_tag='_mask').df.Abspath[0]
                    t2mask = self._prjobj(1, self._processing, anat, subj, sess, file_tag='_mask').df.Abspath[0]
                    # Execute process
                    for i, finfo in epi:
                        print("  +Filename of meanfunc: {}".format(finfo.Filename))
                        filename = finfo.Filename
                        tpath = os.path.join(step01, subj, sess)
                        fpath = os.path.join(tpath, finfo.Filename)
                        self._prjobj.run('afni_3dcalc', fpath, 'a*step(b)',
                                         finfo.Abspath, epimask)
                        ss_epi = methods.load(fpath)
                        if padded:
                            exec('ss_epi.crop({}=[1, {}])'.format(axis[zaxis], ss_epi.shape[zaxis] - 1))
                            ss_epi.save_as(os.path.join(step01, subj, sess, filename), quiet=True)
                    for i, finfo in t2:
                        print("  +Filename of anat: {}".format(finfo.Filename))
                        filename = finfo.Filename
                        tpath = os.path.join(step02, subj, sess)
                        fpath = os.path.join(tpath, finfo.Filename)
                        self._prjobj.run('afni_3dcalc', fpath, 'a*step(b)',
                                         finfo.Abspath, t2mask)
                        ss_t2 = methods.load(fpath)
                        if padded:
                            exec('ss_t2.crop({}=[1, {}])'.format(axis[zaxis], ss_t2.shape[zaxis] - 1))
                            ss_t2.save_as(os.path.join(step02, subj, sess, filename), quiet=True)
        self._prjobj.reset(True)
        self._prjobj.apply()
        return {'meanfunc': step01, 'anat': step02}

    def timecrop(self, func, crop_loc, dtype='func'):
        """

        Parameters
        ----------
        func
        crop_loc
        dtype

        Returns
        -------

        """
        dataclass, func = methods.check_dataclass(func)
        print('TimeCropped-{}'.format(func))
        step01 = self.init_step('CropTimeAxis-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                # Load image paths
                epi = self._prjobj(1, self._processing, func, subj)
                # Execute process
                for i, finfo in epi:
                    print(" +Filename: {}".format(finfo.Filename))
                    output_path = os.path.join(step01, subj, finfo.Filename)
                    if '.gz' not in output_path:
                        output_path += '.gz'
                    self._prjobj.run('afni_3dcalc', output_path, 'a',
                                     "{}'[{}..{}]'".format(finfo.Abspath, crop_loc[0], crop_loc[1]))
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    # Load image paths
                    epi = self._prjobj(1, self._processing, func, subj)
                    # Execute process
                    for i, finfo in epi:
                        print(" +Filename: {}".format(finfo.Filename))
                        output_path = os.path.join(step01, subj, sess, finfo.Filename)
                        if '.gz' not in output_path:
                            output_path += '.gz'
                        self._prjobj.run('afni_3dcalc', output_path, 'a',
                                         "{}'[{}..{}]'".format(finfo.Abspath, crop_loc[0], crop_loc[1]))
        self._prjobj.reset(True)
        self._prjobj.apply()
        return {'func': step01}

    def coregistration(self, meanfunc, anat, dtype='func', **kwargs):
        """ Method for mean functional image realignment to anatomical image of same subject

        Parameters
        ----------
        meanfunc   : str
            Datatype or absolute path of the input mean functional image
        anat       : str
            Datatype or absolute path of the input anatomical image
        dtype      : str
            Surfix for the step path
        kwargs     :

        Returns
        -------
        step_paths : dict
        """
        f_dataclass, meanfunc = methods.check_dataclass(meanfunc)
        a_dataclass, anat = methods.check_dataclass(anat)
        print('BiasFieldCorrection-{} & {}'.format(meanfunc, anat))
        step01 = self.init_step('BiasFieldCorrection-{}'.format(dtype))
        step02 = self.init_step('BiasFieldCorrection-{}'.format(anat.split('-')[-1]))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj), os.path.join(step02, subj))
            if self._prjobj.single_session:
                epi = self._prjobj(f_dataclass, meanfunc, subj)
                t2 = self._prjobj(a_dataclass, anat, subj)
                for i, finfo in epi:
                    print(" +Filename of func: {}".format(finfo.Filename))
                    self._prjobj.run('ants_BiasFieldCorrection', os.path.join(step01, subj, finfo.Filename),
                                     finfo.Abspath, algorithm='n4')
                for i, finfo in t2:
                    print(" +Filename of anat: {}".format(finfo.Filename))
                    self._prjobj.run('ants_BiasFieldCorrection', os.path.join(step02, subj, finfo.Filename),
                                     finfo.Abspath, algorithm='n4')
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess), os.path.join(step02, subj, sess))
                    epi = self._prjobj(f_dataclass, meanfunc, subj, sess)
                    t2 = self._prjobj(f_dataclass, anat, subj, sess)
                    for i, finfo in epi:
                        print("  +Filename of func: {}".format(finfo.Filename))
                        self._prjobj.run('ants_BiasFieldCorrection', os.path.join(step01, subj, sess, finfo.Filename),
                                         finfo.Abspath, algorithm='n4')
                    for i, finfo in t2:
                        print("  +Filename of anat: {}".format(finfo.Filename))
                        self._prjobj.run('ants_BiasFieldCorrection', os.path.join(step02, subj, sess, finfo.Filename),
                                         finfo.Abspath, algorithm='n4')
        self._prjobj.reset(True)
        self._prjobj.apply()
        print('Coregistration-{} to {}'.format(meanfunc, anat))
        step03 = self.init_step('Coregistration-{}2{}'.format(dtype, anat.split('-')[-1]))
        num_step = os.path.basename(step03).split('_')[0]
        step04 = self.final_step('{}_CheckRegistraton-{}'.format(num_step, dtype))
        for subj in self.subjects:
            methods.mkdir(os.path.join(step04, 'AllSubjects'))
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step03, subj))
            if self._prjobj.single_session:
                epi = self._prjobj(1, self._processing, os.path.basename(step01), subj)
                t2 = self._prjobj(1, self._processing, os.path.basename(step02), subj)
                for i, finfo in epi:
                    print(" +Filename: {}".format(finfo.Filename))
                    fixed_img = t2.df.Abspath[0]
                    moved_img = os.path.join(step03, subj, finfo.Filename)
                    self._prjobj.run('afni_3dAllineate', moved_img, finfo.Abspath, onepass=True, EPI=True,
                                     base=fixed_img, cmass='+xy', matrix_save=os.path.join(step03, subj, subj))
                    fig1 = Viewer.check_reg(methods.load(fixed_img),
                                            methods.load(moved_img), sigma=2, **kwargs)
                    fig1.suptitle('EPI to T2 for {}'.format(subj), fontsize=12, color='yellow')
                    fig1.savefig(os.path.join(step04, 'AllSubjects', '{}.png'.format('-'.join([subj, 'func2anat']))),
                                 facecolor=fig1.get_facecolor())
                    fig2 = Viewer.check_reg(methods.load(moved_img),
                                            methods.load(fixed_img), sigma=2, **kwargs)
                    fig2.suptitle('T2 to EPI for {}'.format(subj), fontsize=12, color='yellow')
                    fig2.savefig(os.path.join(step04, 'AllSubjects', '{}.png'.format('-'.join([subj, 'anat2func']))),
                                 facecolor=fig2.get_facecolor())
            else:
                methods.mkdir(os.path.join(step04, subj), os.path.join(step04, subj, 'AllSessions'))
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step03, subj, sess))
                    epi = self._prjobj(1, self._processing, os.path.basename(step01), subj, sess)
                    t2 = self._prjobj(1, self._processing, os.path.basename(step02), subj, sess)
                    for i, finfo in epi:
                        print("  +Filename of anat: {}".format(finfo.Filename))
                        fixed_img = t2.df.Abspath[0]
                        moved_img = os.path.join(step03, subj, sess, finfo.Filename)
                        self._prjobj.run('afni_3dAllineate', moved_img, finfo.Abspath, onepass=True, EPI=True,
                                         base=fixed_img, cmass='+xy',
                                         matrix_save=os.path.join(step03, subj, sess, sess))
                        fig1 = Viewer.check_reg(methods.load(fixed_img),
                                                methods.load(moved_img), sigma=2, **kwargs)
                        fig1.suptitle('EPI to T2 for {}'.format(subj), fontsize=12, color='yellow')
                        fig1.savefig(os.path.join(step04, subj, 'AllSessions',
                                                  '{}.png'.format('-'.join([sess, 'func2anat']))),
                                     facecolor=fig1.get_facecolor())
                        fig2 = Viewer.check_reg(methods.load(moved_img),
                                                methods.load(fixed_img), sigma=2, **kwargs)
                        fig2.suptitle('T2 to EPI for {}'.format(subj), fontsize=12, color='yellow')
                        fig2.savefig(os.path.join(step04, subj, 'AllSessions',
                                                  '{}.png'.format('-'.join([sess, 'anat2func']))),
                                     facecolor=fig2.get_facecolor())
        self._prjobj.reset(True)
        self._prjobj.apply()
        return {'meanfunc': step01, 'anat':step02, 'realigned_func': step03, 'checkreg': step04}

    def apply_brainmask(self, func, mask, padded=False, zaxis=2, dtype='func'):
        """ Method for applying brain mark to individual 3d+t functional images

        Parameters
        ----------
        func       : str
            Datatype or absolute step path of the input functional image
        mask       : str
            Absolute step path which contains the mask of the functional image
        padded     : bool
        zaxis      : int
        dtype      : str
            Surfix for the step path

        Returns
        -------
        step_paths : dict
        """
        axis = {0: 'x', 1: 'y', 2: 'z'}
        dataclass, func = methods.check_dataclass(func)
        print('ApplyingBrainMask-{}'.format(func))
        step01 = self.init_step('ApplyingBrainMask-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                epi = self._prjobj(dataclass, func, subj)
                epimask = self._prjobj(1, self._processing, os.path.basename(mask), subj, file_tag='_mask').df
                maskobj = methods.load(epimask.Abspath[0])
                if padded:
                    exec ('maskobj.crop({}=[1, {}])'.format(axis[zaxis], maskobj.shape[zaxis] - 1))
                temp_epimask = TempFile(maskobj, 'epimask_{}'.format(subj))
                for i, finfo in epi:
                    print(" +Filename: {}".format(finfo.Filename))
                    self._prjobj.run('afni_3dcalc', os.path.join(step01, subj, finfo.Filename), 'a*step(b)',
                                     finfo.Abspath, str(temp_epimask))
                temp_epimask.close()
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    epi = self._prjobj(dataclass, func, subj, sess)
                    epimask = self._prjobj(1, self._processing, os.path.basename(mask), subj, sess, file_tag='_mask').df
                    maskobj = methods.load(epimask.Abspath[0])
                    if padded:
                        exec ('maskobj.crop({}=[1, {}])'.format(axis[zaxis], maskobj.shape[zaxis] - 1))
                    temp_epimask = TempFile(maskobj, 'epimask_{}_{}'.format(subj, sess))
                    for i, finfo in epi:
                        print("  +Filename: {}".format(finfo.Filename))
                        self._prjobj.run('afni_3dcalc', os.path.join(step01, subj, sess, finfo.Filename), 'a*step(b)',
                                         finfo.Abspath, str(temp_epimask))
                    temp_epimask.close()
        self._prjobj.reset(True)
        self._prjobj.apply()
        return {'func': step01}

    def apply_transformation(self, func, realigned_func, dtype='func'):
        """ Method for applying transformation matrix to individual 3d+t functional images

        Parameters
        ----------
        func           : str
            Datatype or absolute step path for the input functional image
        realigned_func : str
            Absolute step path which contains the realigned functional image
        dtype          : str
            Surfix for the step path

        Returns
        -------
        step_paths     : dict
        """
        dataclass, func = methods.check_dataclass(func)
        print('ApplyingTransformation-{}'.format(func))
        step01 = self.init_step('ApplyingTransformation-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                ref = self._prjobj(1, self._processing, os.path.basename(realigned_func), subj)
                param = self._prjobj(1, self._processing, os.path.basename(realigned_func), subj, ext='.1D')
                funcs = self._prjobj(dataclass, os.path.basename(func), subj)
                for i, finfo in funcs:
                    print(" +Filename: {}".format(finfo.Filename))
                    moved_img = os.path.join(step01, subj, finfo.Filename)
                    self._prjobj.run('afni_3dAllineate', moved_img, finfo.Abspath, master=ref.df.Abspath.loc[0],
                                     matrix_apply=param.df.Abspath.loc[0])
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    ref = self._prjobj(1, self._processing, os.path.basename(realigned_func), subj, sess)
                    param = self._prjobj(1, self._processing, os.path.basename(realigned_func), subj, sess, ext='.1D')
                    funcs = self._prjobj(dataclass, os.path.basename(func), subj, sess)
                    for i, finfo in funcs:
                        print("  +Filename: {}".format(finfo.Filename))
                        moved_img = os.path.join(step01, subj, sess, finfo.Filename)
                        self._prjobj.run('afni_3dAllineate', moved_img, finfo.Abspath, master=ref.df.Abspath.loc[0],
                                         matrix_apply=param.df.Abspath.loc[0])
        self._prjobj.reset(True)
        self._prjobj.apply()
        return {'func': step01}

    def cbv_calculation(self, func, meanBOLD, mean_range=10, dtype='func', **kwargs):
        """

        Parameters
        ----------
        func
        meanBOLD
        dtype
        kwargs

        Returns
        -------

        """
        dataclass, func = methods.check_dataclass(func)
        mb_dataclass, meanBOLD = methods.check_dataclass(meanBOLD)
        print('CBV_Calculation-{}'.format(func))
        step01 = self.init_step('CBV_Calculation-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                funcs = self._prjobj(dataclass, func, subj)
                szero = self._prjobj(mb_dataclass, meanBOLD, subj).df.loc[0]
                for i, finfo in funcs:
                    imgobj = methods.load(finfo.Abspath)
                    imgobj._dataobj = np.mean(imgobj._dataobj[:, :, :, :mean_range], axis=3)
                    spre = TempFile(imgobj, 'spre_{}'.format(subj))
                    print(" +Filename: {}".format(finfo.Filename))
                    self._prjobj.run('afni_3dcalc', os.path.join(step01, subj, finfo.Filename), 'log(b/a)/log(c/b)',
                                     finfo.Abspath, str(spre), szero.Abspath)
                    # self._prjobj.run('afni_3dcalc', os.path.join(step01, subj, finfo.Filename), '(b-a)/(c-b)*100',
                    #                  finfo.Abspath, str(spre), szero.Abspath)
                    spre.close()
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    funcs = self._prjobj(dataclass, func, subj, sess)
                    szero = self._prjobj(mb_dataclass, meanBOLD, subj, sess).df.loc[0]
                    for i, finfo in funcs:
                        imgobj = methods.load(finfo.Abspath)
                        imgobj._dataobj = np.mean(imgobj._dataobj[:, :, :, :mean_range], axis=3)
                        spre = TempFile(imgobj, 'spre_{}_{}'.format(subj, sess))
                        print(" +Filename: {}".format(finfo.Filename))
                        self._prjobj.run('afni_3dcalc', os.path.join(step01, subj, sess, finfo.Filename), 'log(b/a)/log(c/b)',
                                         finfo.Abspath, str(spre), szero.Abspath)
                        # self._prjobj.run('afni_3dcalc', os.path.join(step01, subj, sess, finfo.Filename),
                        #                  '(b-a)/(c-b)*100', finfo.Abspath, str(spre), szero.Abspath)
                        spre.close()
        self._prjobj.reset(True)
        self._prjobj.apply()
        return {'cbv': step01}

    def check_cbv_correction(self, func, meanBOLD, meanCBV, mean_range=20, echotime=0.008, dtype='func', **kwargs):
        """

        Parameters
        ----------
        func
        meanBOLD
        dtype
        kwargs

        Returns
        -------

        """
        dataclass, func = methods.check_dataclass(func)
        mb_dataclass, meanBOLD = methods.check_dataclass(meanBOLD)
        mc_dataclass, meanCBV = methods.check_dataclass(meanCBV)
        print('CBV_Calculation-{}'.format(func))
        step01 = self.init_step('deltaR2forSTIM-{}'.format(dtype))
        step02 = self.init_step('deltaR2forMION-{}'.format(dtype))
        step03 = self.init_step('deltaR2forMIONcorrected-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj), os.path.join(step02, subj), os.path.join(step03, subj))
            if self._prjobj.single_session:
                funcs = self._prjobj(dataclass, func, subj)
                szero = self._prjobj(mb_dataclass, meanBOLD, subj).df.loc[0]
                smion = self._prjobj(mb_dataclass, meanCBV, subj).df.loc[0]
                for i, finfo in funcs:
                    imgobj = methods.load(finfo.Abspath)
                    imgobj._dataobj = np.mean(imgobj._dataobj[:, :, :, :mean_range], axis=3)
                    spre = TempFile(imgobj, 'spre_{}'.format(subj))
                    print(" +Filename: {}".format(finfo.Filename))
                    self._prjobj.run('afni_3dcalc', os.path.join(step01, subj, finfo.Filename),
                                     '(-1/{TE})*log(a/b)'.format(TE=echotime),
                                     finfo.Abspath, str(spre))
                    self._prjobj.run('afni_3dcalc', os.path.join(step02, subj, finfo.Filename),
                                     '(-1/{TE})*log(a/b)'.format(TE=echotime),
                                     smion.Abspath, szero.Abspath)
                    self._prjobj.run('afni_3dcalc', os.path.join(step03, subj, finfo.Filename),
                                     '(-1/{TE})*log(a/b)'.format(TE=echotime),
                                     str(spre), szero.Abspath)
                    # self._prjobj.run('afni_3dcalc', os.path.join(step01, subj, finfo.Filename), '(b-a)/(c-b)*100',
                    #                  finfo.Abspath, str(spre), szero.Abspath)
                    spre.close()
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess), os.path.join(step02, subj, sess),
                                  os.path.join(step03, subj, sess))
                    funcs = self._prjobj(dataclass, func, subj, sess)
                    szero = self._prjobj(mb_dataclass, meanBOLD, subj, sess).df.loc[0]
                    smion = self._prjobj(mb_dataclass, meanCBV, subj, sess).df.loc[0]
                    for i, finfo in funcs:
                        imgobj = methods.load(finfo.Abspath)
                        imgobj._dataobj = np.mean(imgobj._dataobj[:, :, :, :mean_range], axis=3)
                        spre = TempFile(imgobj, 'spre_{}_{}'.format(subj, sess))
                        print(" +Filename: {}".format(finfo.Filename))
                        self._prjobj.run('afni_3dcalc', os.path.join(step01, subj, sess, finfo.Filename),
                                         '(-1/{TE})*log(a/b)'.format(TE=echotime),
                                         finfo.Abspath, str(spre))
                        self._prjobj.run('afni_3dcalc', os.path.join(step02, subj, sess, finfo.Filename),
                                         '(-1/{TE})*log(a/b)'.format(TE=echotime),
                                         smion.Abspath, szero.Abspath)
                        self._prjobj.run('afni_3dcalc', os.path.join(step02, subj, sess, finfo.Filename),
                                         '(-1/{TE})*log(a/b)'.format(TE=echotime),
                                         str(spre), szero.Abspath)
                        # self._prjobj.run('afni_3dcalc', os.path.join(step01, subj, sess, finfo.Filename),
                        #                  '(b-a)/(c-b)*100', finfo.Abspath, str(spre), szero.Abspath)
                        spre.close()
        self._prjobj.reset(True)
        self._prjobj.apply()
        return {'cbv': step01}

    def spatial_smoothing(self, func, mask=False, FWHM=False, quiet=False, dtype='func'):
        """

        Parameters
        ----------
        func
        mask
        FWHM
        quiet

        Returns
        -------

        """
        dataclass, func = methods.check_dataclass(func)
        print('SpatialSmoothing-{}'.format(func))
        step01 = self.init_step('SpatialSmoothing-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                epi = self._prjobj(dataclass, func, subj)
                if mask:
                    epimask = self._prjobj(1, self._processing, os.path.basename(mask), subj, file_tag='_mask').df
                else:
                    epimask = None
                for i, finfo in epi:
                    if mask:
                        mask = epimask[i].Abspath
                    print(" +Filename: {}".format(finfo.Filename))
                    self._prjobj.run('afni_3dBlurInMask', os.path.join(step01, subj, finfo.Filename), finfo.Abspath,
                                     mask=mask, FWHM=FWHM, quiet=quiet)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    epi = self._prjobj(dataclass, func, subj, sess)
                    if mask:
                        epimask = self._prjobj(1, self._processing, os.path.basename(mask), subj, sess, file_tag='_mask').df
                    else:
                        epimask = False
                    for i, finfo in epi:
                        if mask:
                            mask = epimask[i].Abspath
                        print(" +Filename: {}".format(finfo.Filename))
                        self._prjobj.run('afni_3dBlurInMask', os.path.join(step01, subj, sess, finfo.Filename),
                                         finfo.Abspath, mask=mask, FWHM=FWHM, quiet=quiet)
        self._prjobj.reset(True)
        self._prjobj.apply()
        return {'func': step01}

    def signal_processing(self, func, dt=1, norm=False, despike=False, detrend=False,
                          blur=False, band=False, dtype='func', file_tag=None, ignore=None):
        """ Method for signal processing and spatial smoothing of individual functional image

        Parameters
        ----------
        func        : str
            Datatype or Absolute step path for the input functional image
        dt          : int
        norm        : boolean
        despike     : boolean
        detrend     : int
        blur        : float
        band        : list of float
        dtype       : str
            Surfix for the step path

        Returns
        -------
        step_paths  : dict
        """
        dataclass, func = methods.check_dataclass(func)
        print('SignalProcessing-{}'.format(func))
        step01 = self.init_step('SignalProcessing-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                if not file_tag:
                    if not ignore:
                        funcs = self._prjobj(dataclass, func, subj)
                    else:
                        funcs = self._prjobj(dataclass, func, subj, ignore=ignore)
                else:
                    if not ignore:
                        funcs = self._prjobj(dataclass, func, subj, file_tag=file_tag)
                    else:
                        funcs = self._prjobj(dataclass, func, subj, file_tag=file_tag, ignore=ignore)
                for i, finfo in funcs:
                    print(" +Filename: {}".format(finfo.Filename))
                    self._prjobj.run('afni_3dBandpass', os.path.join(step01, subj, finfo.Filename), finfo.Abspath,
                                     norm=norm, despike=despike, detrend=detrend, blur=blur, band=band, dt=dt)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    if not file_tag:
                        if not ignore:
                            funcs = self._prjobj(dataclass, func, subj, sess)
                        else:
                            funcs = self._prjobj(dataclass, func, subj, sess, ignore=ignore)
                    else:
                        if not ignore:
                            funcs = self._prjobj(dataclass, func, subj, sess, file_tag=file_tag)
                        else:
                            funcs = self._prjobj(dataclass, func, subj, sess, file_tag=file_tag, ignore=ignore)
                    for i, finfo in funcs:
                        print("  +Filename: {}".format(finfo.Filename))
                        self._prjobj.run('afni_3dBandpass', os.path.join(step01, subj, sess, finfo.Filename), finfo.Abspath,
                                         norm=norm, despike=despike, detrend=detrend, blur=blur, band=band, dt=dt)
        self._prjobj.reset(True)
        self._prjobj.apply()
        return {'func': step01}

    def signal_processing2(self, func, dt=1, mask=None, ort=None, orange=None, norm=False, blur=False, band=False,
                           dtype='func'):
        """ New method for signal processing and spatial smoothing of individual functional image

        Parameters
        ----------
        func
        dt
        mask
        ort
        norm
        blur
        band
        dtype
        file_tag
        ignore

        Returns
        -------

        """
        dataclass, func = methods.check_dataclass(func)
        if ort:
            mdataclass, ort = methods.check_dataclass(ort)
        print('SignalProcessing-{}'.format(func))
        step01 = self.init_step('SignalProcessing-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                funcs = self._prjobj(dataclass, func, subj)
                for i, finfo in funcs:
                    if ort:
                        regressor = self._prjobj(mdataclass, ort, subj, ext='.1D',
                                                 file_tag=methods.splitnifti(finfo.Filename),
                                                 ignore='.aff12').df.Abspath[0]
                    else:
                        regressor = None
                    print(" +Filename: {}".format(finfo.Filename))
                    self._prjobj.run('afni_3dTproject', os.path.join(step01, subj, finfo.Filename), finfo.Abspath,
                                     ort=regressor, mask=mask, orange=orange, norm=norm, blur=blur, band=band, dt=dt)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess))
                    funcs = self._prjobj(dataclass, func, subj, sess)
                    for i, finfo in funcs:
                        if ort:
                            regressor = self._prjobj(mdataclass, ort, subj, sess, ext='.1D',
                                                     file_tag=methods.splitnifti(finfo.Filename),
                                                     ignore='.aff12').df.Abspath[0]
                        else:
                            regressor = None
                        print("  +Filename: {}".format(finfo.Filename))
                        self._prjobj.run('afni_3dTproject', os.path.join(step01, subj, sess, finfo.Filename),
                                         finfo.Abspath, ort=regressor, orange=orange, mask=mask, norm=norm,
                                         blur=blur, band=band, dt=dt)
        self._prjobj.reset(True)
        self._prjobj.apply()
        return {'func': step01}

    def warp_func(self, func, warped_anat, tempobj, atlas=False, dtype='func', **kwargs):
        """ Method for warping the individual functional image to template space

        Parameters
        ----------
        func        : str
            Datatype or Absolute step path for the input functional image
        warped_anat : str
            Absolute step path which contains diffeomorphic map and transformation matrix
            which is generated by the methods of 'pynit.Preprocessing.warp_anat_to_template'
        tempobj     : pynit.Template
            The template object which contains set of atlas
        dtype       : str
            Surfix for the step path

        Returns
        -------
        step_paths  : dict
        """
        # Check the source of input data
        in_kwargs = {'atlas': atlas}
        dataclass, func = methods.check_dataclass(func)
        print("Warp-{} to Atlas and Check it's registration".format(func))
        step01 = self.init_step('Warp-{}2atlas'.format(dtype))
        num_step = os.path.basename(step01).split('_')[0]
        step02 = self.final_step('{}_CheckAtlasRegistration-{}'.format(num_step, dtype))
        # Loop the subjects
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                methods.mkdir(os.path.join(step02, 'AllSubjects'))
                # Grab the warping map and transform matrix
                mats, warps, warped = methods.get_warp_matrix(self, warped_anat, subj, inverse=False)
                # temp_path = os.path.join(step01, subj, "base")
                # tempobj.save_as(temp_path, quiet=True)
                funcs = self._prjobj(dataclass, func, subj)
                print(" +Filename of fixed image: {}".format(warped.Filename))
                for i, finfo in funcs:
                    print(" +Filename of moving image: {}".format(finfo.Filename))
                    output_path = os.path.join(step01, subj, finfo.Filename)
                    self._prjobj.run('ants_WarpTimeSeriresImageMultiTransform', output_path,
                                     finfo.Abspath, warped.Abspath, warps, mats, **in_kwargs)
                subjatlas = methods.load_temp(output_path, tempobj._atlas.path)
                fig = subjatlas.show(**kwargs)
                if type(fig) is tuple:
                    fig = fig[0]
                fig.suptitle('Check atlas registration of {}'.format(subj), fontsize=12, color='yellow')
                fig.savefig(os.path.join(step02, 'AllSubjects', '{}.png'.format('-'.join([subj, 'checkatlas']))),
                            facecolor=fig.get_facecolor())
            else:
                methods.mkdir(os.path.join(step02, subj))
                for sess in self.sessions:
                    methods.mkdir(os.path.join(step02, subj, 'AllSessions'), os.path.join(step01, subj, sess))
                    print(" :Session: {}".format(sess))
                    # Grab the warping map and transform matrix
                    mats, warps, warped = methods.get_warp_matrix(self, warped_anat, subj, sess, inverse=False)
                    # temp_path = os.path.join(step01, subj, sess, "base")
                    # tempobj.save_as(temp_path, quiet=True)
                    funcs = self._prjobj(dataclass, func, subj, sess)
                    print(" +Filename of fixed image: {}".format(warped.Filename))
                    for i, finfo in funcs:
                        print(" +Filename of moving image: {}".format(finfo.Filename))
                        output_path = os.path.join(step01, subj, sess, finfo.Filename)
                        self._prjobj.run('ants_WarpTimeSeriesImageMultiTransform', output_path,
                                         finfo.Abspath, warped.Abspath, warps, mats, **in_kwargs)
                    subjatlas = methods.load_temp(output_path, tempobj._atlas.path)
                    fig = subjatlas.show(**kwargs)
                    if type(fig) is tuple:
                        fig = fig[0]
                    fig.suptitle('Check atlas registration of {}'.format(subj), fontsize=12, color='yellow')
                    fig.savefig(os.path.join(step02, subj, 'AllSessions',
                                             '{}.png'.format('-'.join([subj, sess, 'checkatlas']))),
                                facecolor=fig.get_facecolor())
        self._prjobj.reset(True)
        self._prjobj.apply()
        return {'func': step01, 'checkreg': step02}

    def linear_spatial_normalization(self, anat, tempobj, dtype='anat', **kwargs):
        """

        Parameters
        ----------
        anat
        tempobj
        dtype
        kwargs

        Returns
        -------

        """
        # Check the source of input data
        dataclass, anat = methods.check_dataclass(anat)
        # Print step ans initiate the step
        print('SpatialNormalization-{} to Tempalte'.format(anat))
        step01 = self.init_step('SpatialNormalization-{}2temp'.format(dtype))
        num_step = os.path.basename(step01).split('_')[0]
        step02 = self.final_step('{}_CheckRegistraton-{}2Temp'.format(num_step, dtype))
        # Loop the subjects
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                methods.mkdir(os.path.join(step02, 'AllSubjects'))
                anats = self._prjobj(dataclass, anat, subj)
                methods.mkdir(os.path.join(step01, subj))
                for i, finfo in anats:
                    print(" +Filename: {}".format(finfo.Filename))
                    fixed_img = tempobj.template_path
                    moved_img = os.path.join(step01, subj, finfo.Filename)
                    trans_mat = methods.splitnifti(moved_img)+'.aff12.1D'
                    self._prjobj.run('afni_3dAllineate', moved_img,
                                     finfo.Abspath, base=fixed_img, twopass=True, cmass='xy',
                                     zclip=True, conv='0.01', cost='crM', ckeck='nmi', warp='shr',
                                     matrix_save=trans_mat)
                    fig1 = Viewer.check_reg(methods.load(fixed_img),
                                            methods.load(moved_img), sigma=2, **kwargs)
                    fig1.suptitle('T2 to Temp for {}'.format(subj), fontsize=12, color='yellow')
                    fig1.savefig(os.path.join(step02, 'AllSubjects', '{}.png'.format('-'.join([subj, 'anat2temp']))),
                                 facecolor=fig1.get_facecolor())
                    fig2 = Viewer.check_reg(methods.load(moved_img),
                                            methods.load(fixed_img), sigma=2, **kwargs)
                    fig2.suptitle('Temp to T2 for {}'.format(subj), fontsize=12, color='yellow')
                    fig2.savefig(os.path.join(step02, 'AllSubjects', '{}.png'.format('-'.join([subj, 'temp2anat']))),
                                 facecolor=fig2.get_facecolor())
            else:
                methods.mkdir(os.path.join(step02, subj), os.path.join(step02, subj, 'AllSessions'))
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    anats = self._prjobj(dataclass, anat, subj, sess)
                    methods.mkdir(os.path.join(step01, subj, sess))
                    methods.mkdir(os.path.join(step02, subj, 'AllSessions'))
                    for i, finfo in anats:
                        print("  +Filename: {}".format(finfo.Filename))
                        fixed_img = tempobj.template_path
                        moved_img = os.path.join(step01, subj, sess, finfo.Filename)
                        trans_mat = methods.splitnifti(moved_img) + '.aff12.1D'
                        self._prjobj.run('afni_3dAllineate', moved_img,
                                         finfo.Abspath, base=fixed_img, twopass=True, cmass='xy',
                                         zclip=True, conv='0.01', cost='crM', ckeck='nmi', warp='shr',
                                         matrix_save=trans_mat)
                        fig1 = Viewer.check_reg(methods.load(fixed_img),
                                                methods.load(moved_img), sigma=2, **kwargs)
                        fig1.suptitle('T2 to Temp for {}-{}'.format(subj, sess), fontsize=12, color='yellow')
                        fig1.savefig(
                            os.path.join(step02, subj, 'AllSessions',
                                         '{}.png'.format('-'.join([subj, sess, 'anat2temp']))),
                            facecolor=fig1.get_facecolor())
                        fig2 = Viewer.check_reg(methods.load(moved_img),
                                                methods.load(fixed_img), sigma=2, **kwargs)
                        fig2.suptitle('Temp to T2 for {}-{}'.format(subj, sess), fontsize=12, color='yellow')
                        fig2.savefig(
                            os.path.join(step02, subj, 'AllSessions',
                                         '{}.png'.format('-'.join([subj, sess, 'temp2anat']))),
                            facecolor=fig2.get_facecolor())
        self._prjobj.reset(True)
        self._prjobj.apply()
        return {'norm_anat': step01, 'checkreg': step02}

    def apply_spatial_normalization(self, func, norm_anat, tempobj, dtype='func', **kwargs):
        """

        Parameters
        ----------
        func
        norm_anat
        dtype

        Returns
        -------

        """
        dataclass, func = methods.check_dataclass(func)
        print('ApplyingSpatialNormalization-{}'.format(func))
        step01 = self.init_step('ApplyingSpatialNormalization-{}'.format(dtype))
        num_step = os.path.basename(step01).split('_')[0]
        step02 = self.final_step('{}_CheckAtlasRegistration-{}'.format(num_step, dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj),
                            os.path.join(step02, 'AllSubjects'))
            if self._prjobj.single_session:
                ref = self._prjobj(1, self._processing, os.path.basename(norm_anat), subj)
                param = self._prjobj(1, self._processing, os.path.basename(norm_anat), subj, ext='.1D')
                temp_path = os.path.join(step01, subj, "base")
                tempobj.save_as(temp_path, quiet=True)
                funcs = self._prjobj(dataclass, os.path.basename(func), subj)
                for i, finfo in funcs:
                    print(" +Filename: {}".format(finfo.Filename))
                    moved_img = os.path.join(step01, subj, finfo.Filename)
                    self._prjobj.run('afni_3dAllineate', moved_img, finfo.Abspath, master=ref.df.Abspath.loc[0],
                                     matrix_apply=param.df.Abspath.loc[0], warp='shr')
                try:
                    subjatlas = methods.load_temp(moved_img, '{}_atlas.nii'.format(temp_path))
                    fig = subjatlas.show(**kwargs)
                    if type(fig) is tuple:
                        fig = fig[0]
                    fig.suptitle('Check atlas registration of {}'.format(subj), fontsize=12, color='yellow')
                    fig.savefig(
                        os.path.join(step02, 'AllSubjects', '{}.png'.format('-'.join([subj, 'checkatlas']))),
                        facecolor=fig.get_facecolor())
                except:
                    pass
                try:
                    os.remove('{}_atlas.nii'.format(temp_path))
                    os.remove('{}_atlas.label'.format(temp_path))
                    os.remove('{}_template.nii'.format(temp_path))
                except:
                    pass
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess),
                                    os.path.join(step02, subj),
                                    os.path.join(step02, subj, 'AllSessions'))
                    ref = self._prjobj(1, self._processing, os.path.basename(norm_anat), subj, sess)
                    param = self._prjobj(1, self._processing, os.path.basename(norm_anat), subj, sess, ext='.1D')
                    temp_path = os.path.join(step01, subj, sess, "base")
                    tempobj.save_as(temp_path, quiet=True)
                    funcs = self._prjobj(dataclass, os.path.basename(func), subj, sess)
                    for i, finfo in funcs:
                        print(" +Filename: {}".format(finfo.Filename))
                        moved_img = os.path.join(step01, subj, sess, finfo.Filename)
                        self._prjobj.run('afni_3dAllineate', moved_img, finfo.Abspath, master=ref.df.Abspath.loc[0],
                                         matrix_apply=param.df.Abspath.loc[0], warp='shr')
                    try:
                        subjatlas = methods.load_temp(moved_img, '{}_atlas.nii'.format(temp_path))
                        fig = subjatlas.show(**kwargs)
                        if type(fig) is tuple:
                            fig = fig[0]
                        fig.suptitle('Check atlas registration of {}-{}'.format(subj, sess), fontsize=12, color='yellow')
                        fig.savefig(
                            os.path.join(step02, subj, 'AllSessions',
                                         '{}.png'.format('-'.join([subj, sess, 'checkatlas']))),
                            facecolor=fig.get_facecolor())
                    except:
                        pass
                    try:
                        os.remove('{}_atlas.nii'.format(temp_path))
                        os.remove('{}_atlas.label'.format(temp_path))
                        os.remove('{}_template.nii'.format(temp_path))
                    except:
                        pass
        self._prjobj.reset(True)
        self._prjobj.apply()
        return {'func': step01}

    def warp_anat_to_template(self, anat, tempobj, dtype='anat', ttype='s', **kwargs): # TODO: This code not work if the template image resolution is different with T2 image
        """ Method for warping the individual anatomical image to template

        Parameters
        ----------
        anat        : str
            Datatype or Absolute step path for the input anatomical image
        tempobj     : pynit.Template
            The template object which contains set of atlas
        dtype       : str
            Surfix for the step path

        ttype       : str
            Type of transformation
            's' : Warping
            'a' : Affine

        Returns
        -------
        step_paths  : dict
        """
        # Check the source of input data
        if os.path.exists(anat):
            dataclass = 1
            anat = methods.path_splitter(anat)[-1]
        else:
            dataclass = 0
        # Print step ans initiate the step
        print('Warp-{} to Tempalte'.format(anat))
        step01 = self.init_step('Warp-{}2temp'.format(dtype))
        num_step = os.path.basename(step01).split('_')[0]
        step02 = self.final_step('{}_CheckRegistraton-{}'.format(num_step, dtype))
        # Loop the subjects
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                methods.mkdir(os.path.join(step02, 'AllSubjects'))
                anats = self._prjobj(dataclass, anat, subj)
                methods.mkdir(os.path.join(step01, subj))
                for i, finfo in anats:
                    print(" +Filename: {}".format(finfo.Filename))
                    output_path = os.path.join(step01, subj, "{}".format(subj))
                    self._prjobj.run('ants_RegistrationSyn', output_path,
                                     finfo.Abspath, base_path=tempobj.template_path, quick=False, ttype=ttype)
                    fig1 = Viewer.check_reg(methods.load(tempobj.template_path),
                                            methods.load("{}_Warped.nii.gz".format(output_path)), sigma=2, **kwargs)
                    fig1.suptitle('T2 to Atlas for {}'.format(subj), fontsize=12, color='yellow')
                    fig1.savefig(os.path.join(step02, 'AllSubjects', '{}.png'.format('-'.join([subj, 'anat2temp']))),
                                 facecolor=fig1.get_facecolor())
                    fig2 = Viewer.check_reg(methods.load("{}_Warped.nii.gz".format(output_path)),
                                            methods.load(tempobj.template_path), sigma=2, **kwargs)
                    fig2.suptitle('Atlas to T2 for {}'.format(subj), fontsize=12, color='yellow')
                    fig2.savefig(os.path.join(step02, 'AllSubjects', '{}.png'.format('-'.join([subj, 'temp2anat']))),
                                 facecolor=fig2.get_facecolor())
            else:
                methods.mkdir(os.path.join(step02, subj), os.path.join(step02, subj, 'AllSessions'))
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    anats = self._prjobj(dataclass, anat, subj, sess)
                    methods.mkdir(os.path.join(step01, subj, sess))
                    for i, finfo in anats:
                        print("  +Filename: {}".format(finfo.Filename))
                        output_path = os.path.join(step01, subj, sess, "{}".format(subj))
                        self._prjobj.run('ants_RegistrationSyn', output_path,
                                         finfo.Abspath, base_path=tempobj.template_path, quick=False, ttype=ttype)
                        fig1 = Viewer.check_reg(methods.load(tempobj.template_path),
                                                methods.load("{}_Warped.nii.gz".format(output_path)),
                                                sigma=2, **kwargs)
                        fig1.suptitle('T2 to Atlas for {}'.format(subj), fontsize=12, color='yellow')
                        fig1.savefig(
                            os.path.join(step02, subj, 'AllSessions', '{}.png'.format('-'.join([subj, 'anat2temp']))),
                            facecolor=fig1.get_facecolor())
                        fig2 = Viewer.check_reg(methods.load("{}_Warped.nii.gz".format(output_path)),
                                                methods.load(tempobj.template_path), sigma=2, **kwargs)
                        fig2.suptitle('Atlas to T2 for {}'.format(subj), fontsize=12, color='yellow')
                        fig2.savefig(
                            os.path.join(step02, subj, 'AllSessions', '{}.png'.format('-'.join([subj, 'temp2anat']))),
                            facecolor=fig2.get_facecolor())
        self._prjobj.reset(True)
        self._prjobj.apply()
        return {'warped_anat': step01, 'checkreg': step02}

    def warp_atlas_to_anat(self, anat, warped_anat, tempobj, dtype='anat', **kwargs):
        """ Method for warping the atlas to individual anatomical image space

        Parameters
        ----------
        anat        : str
            Datatype or Absolute step path for the input anatomical image
        warped_anat : str
            Absolute step path which contains diffeomorphic map and transformation matrix
            which is generated by the methods of 'pynit.Preprocessing.warp_anat_to_template'
        tempobj     : pynit.Template
            The template object which contains set of atlas
        dtype       : str
            Surfix for the step path

        Returns
        -------
        step_paths  : dict
        """
        dataclass, anat = methods.check_dataclass(anat)
        print("Warp-Atlas to {} and Check it's registration".format(anat))
        step01 = self.init_step('Warp-atlas2{}'.format(dtype))
        num_step = os.path.basename(step01).split('_')[0]
        step02 = self.final_step('{}_CheckAtlasRegistration-{}'.format(num_step, dtype))
        # Loop the subjects
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                # Grab the warping map and transform matrix
                mats, warps, warped = methods.get_warp_matrix(self, warped_anat, subj, inverse=True)
                temp_path = os.path.join(warped_anat, subj, "base")
                tempobj.save_as(temp_path, quiet=True)
                anats = self._prjobj(dataclass, anat, subj)
                output_path = os.path.join(step01, subj, "{}_atlas.nii".format(subj))
                methods.mkdir(os.path.join(step01, subj), os.path.join(step02, 'AllSubjects'))
                print(" +Filename: {}".format(warped.Filename))
                self._prjobj.run('ants_WarpImageMultiTransform', output_path,
                                 '{}_atlas.nii'.format(temp_path), warped.Abspath,
                                 True, '-i', mats, warps)
                tempobj.atlas.save_as(os.path.join(step01, subj, "{}_atlas".format(subj)), label_only=True)
                for i, finfo in anats:
                    subjatlas = methods.load_temp(finfo.Abspath, output_path)
                    fig = subjatlas.show(**kwargs)
                    if type(fig) is tuple:
                        fig = fig[0]
                    fig.suptitle('Check atlas registration of {}'.format(subj), fontsize=12, color='yellow')
                    fig.savefig(os.path.join(step02, '{}.png'.format('-'.join([subj, 'checkatlas']))),
                                facecolor=fig.get_facecolor())
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    # Grab the warping map and transform matrix
                    mats, warps, warped = methods.get_warp_matrix(self, warped_anat, subj, sess, inverse=True)
                    temp_path = os.path.join(step01, subj, sess, "base")
                    tempobj.save_as(temp_path, quiet=True)
                    anats = self._prjobj(dataclass, anat, subj, sess)
                    output_path = os.path.join(step01, subj, sess, "{}_atlas.nii".format(sess))
                    methods.mkdir(os.path.join(step01, subj, sess), os.path.join(step02, subj, 'AllSessoions'))
                    print(" +Filename: {}".format(warped.Filename))
                    self._prjobj.run('ants_WarpImageMultiTransform', output_path,
                                     '{}_atlas.nii'.format(temp_path), warped.Abspath, True, '-i', mats, warps)
                    tempobj.atlas.save_as(os.path.join(step01, subj, sess, "{}_atlas".format(sess)), label_only=True)
                    for i, finfo in anats:
                        subjatlas = methods.load_temp(finfo.Abspath, output_path)
                        fig = subjatlas.show(**kwargs)
                        if type(fig) is tuple:
                            fig = fig[0]
                        fig.suptitle('Check atlas registration of {}'.format(sess), fontsize=12, color='yellow')
                        fig.savefig(os.path.join(step02, subj, 'AllSessions',
                                                 '{}.png'.format('-'.join([sess, 'checkatlas']))),
                                    facecolor=fig.get_facecolor())
        self._prjobj.reset(True)
        self._prjobj.apply()
        return {'atlas': step01, 'checkreg': step02}

    def get_timetrace(self, func, atlas, dtype='func', file_tag=None, ignore=None, subjects=None, **kwargs):
        """ Method for extracting timecourse from mask

        Parameters
        ----------
        func       : str
            Datatype or absolute path of the input mean functional image
        atlas      : str
            template object which has atlas, or folder name which includes all mask
        dtype      : str
            Surfix for the step path
        kwargs     :

        Returns
        -------
        step_paths : dict

        """
        if not subjects: #TODO: Subject selection testcode, need to apply for all steps
            subjects = self.subjects[:]
        dataclass, func = methods.check_dataclass(func)
        atlas, tempobj = methods.check_atals_datatype(atlas)
        print('ExtractTimeCourseData-{}'.format(func))
        step01 = self.init_step('ExtractTimeCourse-{}'.format(dtype))
        for subj in subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                if not tempobj:
                    # atlas = self._prjobj(1, self._pipeline, atlas, subj).df.Abspath.loc[0]
                    warped = self._prjobj(1, self._processing, subj, file_tag='_InverseWarped').df.Abspath.loc[0]
                    tempobj = methods.load_temp(warped, atlas)
                if not file_tag:
                    if not ignore:
                        funcs = self._prjobj(dataclass, func, subj)
                    else:
                        funcs = self._prjobj(dataclass, func, subj, ignore=ignore)
                else:
                    if not ignore:
                        funcs = self._prjobj(dataclass, func, subj, file_tag=file_tag)
                    else:
                        funcs = self._prjobj(dataclass, func, subj, file_tag=file_tag, ignore=ignore)
                for i, finfo in funcs:
                    print(" +Filename: {}".format(finfo.Filename))
                    df = Analysis.get_timetrace(methods.load(finfo.Abspath), tempobj, afni=True, **kwargs)
                    df.to_excel(os.path.join(step01, subj, "{}.xlsx".format(os.path.splitext(finfo.Filename)[0])))
            else:
                methods.mkdir(os.path.join(step01, subj))
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    if not tempobj:
                        # atlas = self._prjobj(1, self._pipeline, atlas, subj, sess).df.Abspath.loc[0]
                        warped = self._prjobj(1, self._processing, subj, sess,
                                              file_tag='_InverseWarped').df.Abspath.loc[0]
                        tempobj = methods.load_temp(warped, atlas)
                    if not file_tag:
                        if not ignore:
                            funcs = self._prjobj(dataclass, func, subj, sess)
                        else:
                            funcs = self._prjobj(dataclass, func, subj, sess, ignore=ignore)
                    else:
                        if not ignore:
                            funcs = self._prjobj(dataclass, func, subj, sess, file_tag=file_tag)
                        else:
                            funcs = self._prjobj(dataclass, func, subj, sess, file_tag=file_tag, ignore=ignore)
                    methods.mkdir(os.path.join(step01, subj, sess))
                    for i, finfo in funcs:
                        print("  +Filename: {}".format(finfo.Filename))
                        df = Analysis.get_timetrace(methods.load(finfo.Abspath), tempobj, afni=True, **kwargs)
                        df.to_excel(os.path.join(step01, subj, sess, "{}.xlsx".format(
                            os.path.splitext(finfo.Filename)[0])))
        self._prjobj.reset(True)
        self._prjobj.apply()
        return {'timecourse': step01}

    def get_correlation_matrix(self, func, atlas, dtype='func', file_tag=None, ignore=None, subjects=None, **kwargs):
        """ Method for extracting timecourse, correlation matrix and calculating z-score matrix

        Parameters
        ----------
        func       : str
            Datatype or absolute path of the input mean functional image
        atlas      : str
        dtype      : str
            Surfix for the step path
        kwargs     :

        Returns
        -------
        step_paths : dict

        """
        if not subjects: #TODO: Subject selection testcode, need to apply for all steps
            subjects = self.subjects[:]
        dataclass, func = methods.check_dataclass(func)
        atlas, tempobj = methods.check_atals_datatype(atlas)
        print('ExtractTimeCourseData-{}'.format(func))
        step01 = self.init_step('ExtractTimeCourse-{}'.format(dtype))
        step02 = self.init_step('CC_Matrix-{}'.format(dtype))
        num_step = os.path.basename(step02).split('_')[0]
        step03 = self.final_step('{}_Zscore_Matrix-{}'.format(num_step, dtype))
        for subj in subjects:
            print("-Subject: {}".format(subj))
            methods.mkdir(os.path.join(step01, subj), os.path.join(step02, subj), os.path.join(step03, subj))
            if self._prjobj.single_session:
                if not tempobj:
                    # atlas = self._prjobj(1, self._pipeline, atlas, subj).df.Abspath.loc[0]
                    warped = self._prjobj(1, self._processing, subj, file_tag='_InverseWarped').df.Abspath.loc[0]
                    tempobj = methods.load_temp(warped, atlas)
                if not file_tag:
                    if not ignore:
                        funcs = self._prjobj(dataclass, func, subj)
                    else:
                        funcs = self._prjobj(dataclass, func, subj, ignore=ignore)
                else:
                    if not ignore:
                        funcs = self._prjobj(dataclass, func, subj, file_tag=file_tag)
                    else:
                        funcs = self._prjobj(dataclass, func, subj, file_tag=file_tag, ignore=ignore)
                for i, finfo in funcs:
                    print(" +Filename: {}".format(finfo.Filename))
                    df = Analysis.get_timetrace(methods.load(finfo.Abspath), tempobj, afni=True, **kwargs)
                    df.to_excel(os.path.join(step01, subj, "{}.xlsx".format(os.path.splitext(finfo.Filename)[0])))
                    df.corr().to_excel(os.path.join(step02, subj, "{}.xlsx".format(
                        os.path.splitext(finfo.Filename)[0])))
                    np.arctanh(df.corr()).to_excel(
                        os.path.join(step03, subj, "{}.xlsx").format(os.path.splitext(finfo.Filename)[0]))
            else:
                methods.mkdir(os.path.join(step01, subj), os.path.join(step02, subj),
                                os.path.join(step03, subj))
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    if not tempobj:
                        # atlas = self._prjobj(1, self._pipeline, atlas, subj, sess).df.Abspath.loc[0]
                        warped = self._prjobj(1, self._processing, subj, sess,
                                              file_tag='_InverseWarped').df.Abspath.loc[0]
                        tempobj = methods.load_temp(warped, atlas)
                    if not file_tag:
                        if not ignore:
                            funcs = self._prjobj(dataclass, func, subj, sess)
                        else:
                            funcs = self._prjobj(dataclass, func, subj, sess, ignore=ignore)
                    else:
                        if not ignore:
                            funcs = self._prjobj(dataclass, func, subj, sess, file_tag=file_tag)
                        else:
                            funcs = self._prjobj(dataclass, func, subj, sess, file_tag=file_tag, ignore=ignore)
                    methods.mkdir(os.path.join(step01, subj, sess), os.path.join(step02, subj, sess),
                                    os.path.join(step03, subj, sess))
                    for i, finfo in funcs:
                        print("  +Filename: {}".format(finfo.Filename))
                        df = Analysis.get_timetrace(methods.load(finfo.Abspath), tempobj, afni=True, **kwargs)
                        df.to_excel(os.path.join(step01, subj, sess, "{}.xlsx".format(
                            os.path.splitext(finfo.Filename)[0])))
                        df.corr().to_excel(
                            os.path.join(step02, subj, sess, "{}.xlsx".format(os.path.splitext(finfo.Filename)[0])))
                        np.arctanh(df.corr()).to_excel(
                            os.path.join(step03, subj, sess, "{}.xlsx".format(os.path.splitext(finfo.Filename)[0])))
        self._prjobj.reset(True)
        self._prjobj.apply()
        return {'timecourse': step01, 'cc_matrix': step02}

    def final_step(self, title):
        path = os.path.join(self._prjobj.path, self._prjobj.ds_type[2],
                            self.processing, title)
        methods.mkdir(os.path.join(self._prjobj.path, self._prjobj.ds_type[2],
                                   self.processing), path)
        self._prjobj.scan_prj()
        return path
