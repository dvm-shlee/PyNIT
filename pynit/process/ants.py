from base import *
from pynit.handler.step import Step
import multiprocessing

class ANTs_Process(BaseProcess):
    # def __init__(self, *args, **kwargs):
    #     super(ANTs_Process, self).__init__(*args, **kwargs)

    def ants_Coreg(self, anat, meanfunc, surfix='func', debug=False):
        """This step align the anatomical data to given template brain space using ANTs non-linear SyN algorithm

        :param anat:        input path for anatomical image, three type of path can be used
                            1. datatype path from the raw data (e.g. 'anat' or 'dti')
                            2. absolute path
                            3. index of path which is shown on 'executed' method
        :param meanfunc:    input path for mean functional image, same as above input path
        :param surfix:      surfix for output path
        :param debug:       set True if you want to print out the executing function of this process (default: False)
        :type anat:         str or int
        :type meanfunc:         str or int
        :type surfix:       str
        :return:            output path
        :rtype:             dict
        """
        display(title(value='** Processing non-linear coregistration.....'))
        anat = self.check_input(anat)
        meanfunc = self.check_input(meanfunc)
        step = Step(self, n_thread=1)
        step.set_input(name='meanfunc', path=meanfunc, idx=0)
        step.set_input(name='anat', path=anat, type=1, idx=0)
        step.set_var(name='thread', value=multiprocessing.cpu_count())
        step.set_output(name='prefix', ext='remove')
        cmd = 'antsSyN -f {anat} -m {meanfunc} -o {prefix} -n {thread}'
        step.set_cmd(cmd)
        output_path = step.run('NonLinearCoreg', surfix, debug=debug)
        return dict(normanat=output_path)

    def ants_ApplyCoreg(self):
        pass

    def ants_MotionCorrection(self, func, surfix='func', debug=False):
        display(title(value='** Extracting time-course data from ROIs'))
        func = self.check_input(func)
        step = Step(self)
        step.set_input(name='func', path=func, type=False)
        step.set_output(name='prefix', ext='remove')
        cmd01 = "antsMotionCorr -d 3 -a {func} -o {prefix}-avg.nii.gz"
        cmd02 = "antsMotionCorr -d 3 -o [{prefix},{prefix}.nii.gz,{prefix}-avg.nii.gz] " \
                "-m gc[ {prefix}-avg.nii.gz ,{func}, 1, 1, Random, 0.05  ] -t Affine[ 0.005 ] " \
                "-i 20 -u 1 -e 1 -s 0 -f 1 -n 10"
        step.set_cmd(cmd01)
        step.set_cmd(cmd02)
        output_path = step.run('MotionCorrection', surfix=surfix, debug=debug)
        return dict(func=output_path)

    def ants_BiasFieldCorrection(self, anat, func, debug=False):
        """N4BiasFieldCorrection

        :param anat:
        :param func:
        :param debug:
        :return:
        """
        anat = self.check_input(anat)
        step = Step(self)
        step.set_input(name='anat', path=anat)
        step.set_output(name='output')
        cmd1 = 'N4BiasFieldCorrection -i {anat} -o {output}'
        step.set_cmd(cmd1)
        anat_path = step.run('BiasFiled', 'anat', debug=debug)

        func = self.check_input(func)
        step.reset()
        step.set_input(name='func', path=func)
        step.set_output(name='output')
        cmd2 = 'N4BiasFieldCorrection -i {func} -o {output}'
        step.set_cmd(cmd2)
        func_path = step.run('BiasField', 'func', debug=debug)
        return dict(anat=anat_path, func=func_path)

    def ants_SpatialNorm(self, anat, tmpobj, surfix='anat', debug=False):
        """This step align the anatomical data to given template brain space using ANTs non-linear SyN algorithm

        :param anat     :   str or int
            Folder name of anatomical data in Data class or absolute path
            If you put integer, the path will inputted by indexing the executed path with given integer
        :param tmpobj:
        :param surfix:
        :return:
        """
        anat = self.check_input(anat)
        step = Step(self, n_thread=1)
        step.set_message('** Processing spatial normalization.....')
        step.set_input(name='anat', path=anat, idx=0)
        step.set_var(name='tmpobj', value=tmpobj.template_path)
        step.set_var(name='thread', value=multiprocessing.cpu_count())
        step.set_output(name='prefix', ext='remove')
        cmd = 'antsSyN -f {tmpobj} -m {anat} -o {prefix} -n {thread}'
        step.set_cmd(cmd)
        output_path = step.run('SpatialNorm', surfix, debug=debug)
        return dict(normanat=output_path)

    def ants_ApplySpatialNorm(self, func, warped, surfix='func', debug=False, **kwargs):
        """This step applying the non-linear transform matrix from the anatomical image to functional images

        :param func     :   str or int
            Folder name of functional data in Data class or absolute path of one of executed step
            If you put integer, the path will inputted by indexing the executed path with given integer

        :param warped   :   str or int
            Absolute path that ANTs SyN based spatial normalization registration is applied
            If you put integer, the path will inputted by indexing the executed path with given integer

        :param surfix   :   str
            The given string will be set as surfix of output folder

        :param kwargs   :   dict
            This arguments will be used for filtering the input data, available keywords are as below
            'subs', 'sess', 'file_tag', 'ignore', 'ext'

        :return:
            Output path as dictionary format
        """
        # Check and correct inputs
        func = self.check_input(func)
        warped = self.check_input(warped)
        step = Step(self)
        step.set_message('** Processing spatial normalization.....')
        # Set filters for input transform data
        baseimg_filter = dict(ignore=['_1InverseWarp', '_1Warp', '_inversed'])
        dmorph_filter = dict(file_tag='_1Warp')
        tmatrix_filter = dict(ext='.mat')

        # Set inputs
        step.set_input(name='func', path=func, filters=kwargs)
        step.set_input(name='base', path=warped, filters=baseimg_filter, type=1, idx=0)
        step.set_input(name='morph', path=warped, filters=dmorph_filter, type=1, idx=0)
        step.set_input(name='mat', path=warped, filters=tmatrix_filter, type=1, idx=0)
        step.set_output(name='output')

        # Set commend that need to executes for all subjects
        cmd = 'WarpTimeSeriesImageMultiTransform 4 {func} {output} -R {base} {morph} {mat}'
        step.set_cmd(cmd)
        output_path = step.run('ApplySpatialNorm', surfix, debug=debug)
        return dict(normfunc=output_path)