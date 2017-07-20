from base import *
from pynit.handler.step import Step
import multiprocessing

class ANTs_Process(BaseProcess):
    def __init__(self, *args, **kwargs):
        super(ANTs_Process, self).__init__(*args, **kwargs)

    def ants_Coreg(self):
        pass

    def ants_ApplyCoreg(self):
        pass

    def ants_SpatialNorm(self, anat, tmpobj, surfix='anat'):
        """This step align the anatomical data to given template brain space using ANTs non-linear SyN algorithm

        :param anat     :   str or int
            Folder name of anatomical data in Data class or absolute path
            If you put integer, the path will inputted by indexing the executed path with given integer
        :param tmpobj:
        :param surfix:
        :return:
        """

        parallel = False
        if self._parallel:
            parallel = True
        self._parallel = False          # turn of parallel processing mode

        display(title(value='** Processing spatial normalization.....'))
        anat = self.check_input(anat)
        step = Step(self)
        step.set_input(name='anat', input_path=anat, static=True)
        step.set_staticinput(name='tmpobj', value=tmpobj.template_path)
        step.set_staticinput(name='thread', value=multiprocessing.cpu_count())
        cmd = 'antsSyN -f {tmpobj} -m {anat} -o {prefix} -n {thread}'
        step.set_command(cmd)
        output_path = step.run('SpatialNorm', surfix, debug=False)

        if parallel:
            self._parallel = True
        return dict(normanat=output_path)

    def ants_ApplySpatialNorm(self, func, warped, surfix='func', **kwargs):
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
        display(title(value='** Processing spatial normalization.....'))

        # Check and correct inputs
        func = self.check_input(func)
        warped = self.check_input(warped)
        filters, subj, sess = self.check_filters(**kwargs)
        step = Step(self, subjects=subj, sessions=sess)

        # Set filters for input transform data
        baseimg_filter = dict(ignore=['_1InverseWarp', '_1Warp', '_inversed'])
        dmorph_filter = dict(file_tag='_1Warp')
        tmatrix_filter = dict(ext='.mat')

        # Set inputs
        step.set_input(name='func', input_path=func, filters=filters)
        step.set_input(name='base', input_path=warped, static=True, filters=baseimg_filter, side=True)
        step.set_input(name='morph', input_path=warped, static=True, filters=dmorph_filter, side=True)
        step.set_input(name='mat', input_path=warped, static=True, filters=tmatrix_filter, side=True)

        # Set commend that need to executes for all subjects
        cmd = 'WarpTimeSeriesImageMultiTransform 4 {func} {output} -R {base} {morph} {mat}'
        step.set_command(cmd)
        output_path = step.run('ApplySpatialNorm', surfix, debug=False)
        return dict(normfunc=output_path)