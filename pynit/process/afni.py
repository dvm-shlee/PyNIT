from base import *
from pynit.handler.images import TempFile, Template
from pynit.handler.step import Step
import multiprocessing

class AFNI_Process(BaseProcess):
    # def __init__(self, *args, **kwargs):
    #     super(AFNI_Process, self).__init__(*args, **kwargs)

    def afni_MeanImgCalc(self, func, cbv=False, surfix='func', debug=False):
        """ Calculate mean image through time axis using '3dcalc' to get better SNR image
        this process do motion correction before calculate mean image

        :param func:        input path, three type of path can be used
                            1. datatype path from the raw data (e.g. 'func' or 'cbv')
                            2. absolute path
                            3. index of path which is shown on 'executed' method
        :param cbv:         set True if the input path MION infusion image for CBV (default: False)
        :param surfix:      surfix for output path
        :param debug:       set True if you want to print out the executing function of this process (default: False)
        :type func:         str
        :type cbv:          bool
        :type surfix:       str
        :type debug:        bool
        :return:            output path
        :rtype:             dict
        """
        func = self.check_input(func)
        step = Step(self)
        step.set_message('** Processing mean image calculation.....')
        step.set_input(name='func', path=func, idx=0)
        step.set_output(name='output', type=0)
        step.set_output(name='mparam', ext='1D', type=0)
        step.set_output(name='temp_01', type=3)
        cmd01 = "3dvolreg -prefix {temp_01} -1Dfile {mparam} -Fourier -verbose -base 0 {func}"
        step.set_cmd(cmd01)
        if cbv:
            cmd02 = "3dinfo -nv {func}"
            step.set_var(name='bold', value='int(int(ttime)/3)', type=1)
            step.set_var(name='bold_output', value='methods.splitnifti(output)+"_BOLD.nii.gz"', type=1)
            step.set_var(name='cbv', value='int(int(ttime)*2/3)', type=1)
            step.set_var(name='cbv_output', value='methods.splitnifti(output)+"_CBV.nii.gz"', type=1)
            step.set_cmd(cmd02, name='ttime')
            options = ['"[0..{bold}]"',
                       '"[{cbv}..$]"']
            cmd03 = "3dTstat -prefix {bold_output} -mean {temp_01}" + options[0]
            step.set_cmd(cmd03)
            cmd04 = "3dTstat -prefix {cbv_output} -mean {temp_01}" + options[1]
            step.set_cmd(cmd04)
            output_path = step.run('MeanImgCalc-CBV', surfix, debug=debug)
        else:
            cmd02 = "3dTstat -prefix {output} -mean {temp_01}"
            step.set_cmd(cmd02)
            output_path = step.run('MeanImgCalc-BOLD', surfix, debug=debug)
        return dict(meanfunc=output_path)

    def afni_SliceTimingCorrection(self, func, tr=None, tpattern='altplus', surfix='func', debug=False):
        """ Correct temporal mismatch of image acquisition timing due to the slice timing
        this process use AFNI's '3dTshift'.

        :param func:        input path, three type of path can be used
                            1. datatype path from the raw data (e.g. 'func' or 'cbv')
                            2. absolute path
                            3. index of path which is shown on 'executed' method
        :param tr:          repetition time
        :param tpattern:    slice timing pattern which is defined on 3dTshift
                            (e.g. altplus, altminus, seqplut, seqminus)
        :param surfix:      surfix for output path
        :param debug:       set True if you want to print out the executing function of this process (default: False)
        :type func:         str or int
        :type tr:           int
        :type tpattern:     str
        :type surfix:       str
        :type debug:        bool
        :return:            output path
        :rtype:             dict
        """
        func = self.check_input(func)
        options = str()
        step = Step(self)
        step.set_message('** Processing slice timing correction.....')
        step.set_input(name='func', path=func)
        step.set_output(name='output', type=0)
        cmd = "3dTshift -prefix {output}"
        if tr:
            options += " -TR {0}".format(tr)
        if tpattern:
            options += " -tpattern {0}".format(tpattern)
        else:
            options += " -tpattern altplus"
        input_str = " {func}"
        cmd = cmd + options + input_str
        step.set_cmd(cmd)
        output_path = step.run('SliceTmCorrect', surfix, debug=debug)
        return dict(func=output_path)

    def afni_MotionCorrection(self, func, base, surfix='func', debug=False):
        """ Applying rigid transformation through time axis to correct head motion
        all images in the input path will be aligned on first image of base path you provides
        this process use AFNI's '3dvolreg'

        :param func:        input path, three type of path can be used
                            1. datatype path from the raw data (e.g. 'func' or 'cbv')
                            2. absolute path
                            3. index of path which is shown on 'executed' method
        :param base:        base image path, same as above input path
        :param surfix:      surfix for output path
        :param debug:       set True if you want to print out the executing function of this process (default: False)
        :type func:         str or int
        :type base:         str or int
        :type surfix:       str
        :type debug:        bool
        :return:            output path
        :rtype:             dict
        """
        func = self.check_input(func)
        base = self.check_input(base)
        step = Step(self)
        step.set_message('** Processing motion correction.....')
        step.set_input(name='func', path=func, type=False)

        try:
            if '-CBV-' in base:
                mimg_filters = {'file_tag': '_CBV', 'ignore': 'BOLD'}
                step.set_input(name='base', path=base, filters=mimg_filters, idx=0, type=1)
            else:
                step.set_input(name='base', path=base, idx=0, type=1)
        except:
            methods.raiseerror(messages.Errors.MissingPipeline,
                               'Initial Mean image calculation step has not been executed!')
        step.set_output(name='output')
        step.set_output(name='mparam', ext='1D')
        step.set_output(name='transmat', ext='aff12.1D')
        step.set_output(name='temp_01', type=3)
        step.set_output(name='temp_02', type=3)
        step.set_output(name='temp_03', type=3)
        cmd01 = "3dvolreg -prefix {temp_01} -1Dfile {mparam} -Fourier -verbose -base 0 {func}"
        step.set_cmd(cmd01)
        cmd02 = "3dTstat -mean -prefix {temp_02} {temp_01}"
        step.set_cmd(cmd02)
        cmd03 = "3dAllineate -prefix {temp_03} -warp sho -base {base} -1Dmatrix_save {transmat} {temp_02}"
        step.set_cmd(cmd03)
        cmd04 = '3dAllineate -prefix {output} -1Dmatrix_apply {transmat} -warp sho {temp_01}'
        step.set_cmd(cmd04)
        output_path = step.run('MotionCorrection', surfix, debug=debug)
        return dict(func=output_path)

    def afni_MaskPrep(self, anat, meanfunc, tmpobj, surfix='func', ui=False, debug=False):
        """ Prepare mask images by reorient template mask image to individual image
        and provide launcher to open image viewer to help manual mask drawing
        this process use AFNI's '3dAllineate'

        :param anat:        input path for anatomical image, three type of path can be used
                            1. datatype path from the raw data (e.g. 'anat' or 'dti')
                            2. absolute path
                            3. index of path which is shown on 'executed' method
        :param tmpobj:      brain template image object
        :param meanfunc:    input path for functional image, same as above input path
        :param surfix:      surfix for output path
        :param ui:          Enable UI feature in jupyter notebook
        :param debug:       set True if you want to print out the executing function of this process (default: False)
        :type anat:         str or int
        :type tmpobj:       pn.Template
        :type meanfunc:         str or int
        :type surfix:       str
        :type ui:           bool
        :return:            output path
        :rtype:             dict
        """
        anat = self.check_input(anat)
        step = Step(self)
        step.set_message('** Processing mask image preparation.....')
        mimg_path = None
        # try:
        step.set_input(name='anat', path=anat, idx=0)
        # except:
        #     methods.raiseerror(messages.Errors.MissingPipeline,
        #                        'No anatomy file!')
        try:
            step.set_var(name='mask', value=str(tmpobj.mask), type=1)
        except:
            methods.raiseerror(messages.InputPathError,
                               'No mask template file!')
        cmd01 = 'N4BiasFieldCorrection -d 3 -i {anat} -o {temp_01}'
        cmd02 = '3dAllineate -prefix {temp_02} -NN -onepass -EPI -base {temp_01} -cmass+xy {mask}'
        cmd03 = '3dcalc -prefix {output} -expr "astep(a, 0.5)" -a {temp_02}'
        step.set_output(name='output', type=0)
        step.set_output(name='temp_01', type=3)
        step.set_output(name='temp_02', type=3)
        step.set_cmd(cmd01)
        step.set_cmd(cmd02)
        step.set_cmd(cmd03)
        anat_mask = step.run('MaskPrep', 'anat', debug=debug)
        step.reset()
        step.set_message('** Processing mask image preparation.....')
        try:
            mimg_path = self.check_input(meanfunc)
            if '-CBV-' in mimg_path:
                mimg_filters = {'file_tag': '_BOLD'}
                step.set_input(name='func', path=mimg_path, filters=mimg_filters, idx=0)
            else:
                step.set_input(name='func', path=mimg_path, idx=0)
        except:
            methods.raiseerror(messages.Errors.MissingPipeline,
                               'Initial Mean image calculation step has not been executed!')
        try:
            step.set_var(name='mask', value=str(tmpobj.mask), type=1)
        except:
            methods.raiseerror(messages.InputPathError,
                               'No mask template file!')
        cmd01 = 'N4BiasFieldCorrection -d 3 -i {func} -o {temp_01}'
        cmd02 = '3dAllineate -prefix {temp_02} -NN -onepass -EPI -base {temp_01} -cmass+xy {mask}'
        cmd03 = '3dcalc -prefix {output} -expr "astep(a, 0.5)" -a {temp_02}'
        step.set_output(name='output', type=0)
        step.set_output(name='temp_01', type=3)
        step.set_output(name='temp_02', type=3)
        step.set_cmd(cmd01)
        step.set_cmd(cmd02)
        step.set_cmd(cmd03)
        func_mask = step.run('MaskPrep', surfix, debug=debug)
        if ui:
            if self._viewer == 'itksnap':
                display(widgets.VBox([title(value='-'*43 + ' Anatomical images ' + '-'*43),
                                      gui.itksnap(self, anat_mask, anat),
                                      title(value='<br>' + '-'*43 + ' Functional images ' + '-'*43),
                                      gui.itksnap(self, func_mask, mimg_path)]))
            else:
                methods.raiseerror(messages.Errors.InputValueError,
                                   '"{}" is not available'.format(self._viewer))
        else:
            step.reset()
            step.set_message('** Move files to [{}] folder.....'.format(self.prj.ds_type[2]))
            step.set_input(name='anat', path=anat)
            step.set_input(name='anat_mask', path=anat_mask, type=1)
            step.set_output(name='output', dc=1, ext='remove')
            step.set_var(name='mask_output', value="'{}_mask.nii.gz'.format(output)", type=1)
            cmd01 = '3dcopy {anat} {output}.nii.gz'
            cmd02 = '3dcopy {anat_mask} {mask_output}'
            step.set_cmd(cmd01)
            step.set_cmd(cmd02)
            step.run('MaskPrep', 'anat', debug=debug)
            step.reset()
            step.set_message('** Move files to [{}] folder.....'.format(self.prj.ds_type[2]))
            if '-CBV-' in mimg_path:
                mimg_filters = {'file_tag': '_BOLD'}
                step.set_input(name='meanfunc', path=mimg_path, filters=mimg_filters)
            else:
                step.set_input(name='meanfunc', path=mimg_path)
            step.set_input(name='func_mask', path=func_mask, type=1)
            step.set_output(name='output', dc=1, ext='remove')
            step.set_var(name='mask_output', value="'{}_mask.nii.gz'.format(output)", type=1)
            cmd01 = '3dcopy {meanfunc} {output}.nii.gz'
            cmd02 = '3dcopy {func_mask} {mask_output}'
            step.set_cmd(cmd01)
            step.set_cmd(cmd02)
            step.run('MaskPrep', surfix, debug=debug)
            return dict(anat_mask=anat_mask,
                        func_mask=func_mask,)

    def afni_PasteMask(self, mask, destination, debug=False):
        """ Paste the updated mask into the Processing folder

        :param mask:
        :param destination:
        :return:
        """
        mask = self.check_input(mask, dc=1)
        destination = self.check_input(destination)
        step = Step(self)
        step.set_message('** Upload updated mask files.....')
        step.set_input(name='mask', path=mask, filters=dict(file_tag='_mask'))
        step.set_input(name='target', path=destination, type=1)
        step.set_output(name='output', type=4)
        step.set_module('shutil', sub='copy', rename='scopy')
        cmd = 'scopy({mask},{target})'
        step.set_cmd(cmd, type=1)
        step.run('PasteMask', debug=debug)

    def afni_SkullStrip(self, anat, meanfunc, surfix='func', debug=False):
        """ Subtract out the image outside of the brain mask,
        'afni_MaskPrep' must be executed before run this process.

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
        anat = self.check_input(anat)
        meanfunc = self.check_input(meanfunc)
        anat_mask = [self.steps[idx] for idx, step in self.executed.items() if 'MaskPrep-anat' in step][0]
        anat_mask = self.check_input(anat_mask)
        func_mask = [self.steps[idx] for idx, step in self.executed.items() if 'MaskPrep-{}'.format(surfix) in step][0]
        func_mask = self.check_input(func_mask)
        step = Step(self)
        step.set_message('** Processing skull stripping.....')
        step.set_input(name='anat', path=anat, idx=0)
        step.set_input(name='anat_mask', path=anat_mask, type=1, idx=0)
        step.set_output(name='output')
        cmd01 = '3dcalc -prefix {output} -expr "a*step(b)" -a {anat} -b {anat_mask}'
        step.set_cmd(cmd01)
        anat_path = step.run('SkullStrip', 'anat')
        step = Step(self)
        if '-CBV-' in meanfunc:
            func_filter = {'file_tag': '_BOLD'}
            step.set_input(name='meanfunc', path=meanfunc, filters=func_filter, idx=0)
        else:
            step.set_input(name='meanfunc', path=meanfunc)
        step.set_input(name='func_mask', path=func_mask, type=1, idx=0)
        step.set_output(name='output')
        cmd02 = '3dcalc -prefic {output} -expr "a*step(b)" -a {meanfunc} -b {func_mask}'
        step.set_cmd(cmd02)
        func_path = step.run('SkullStrip', surfix, debug=debug)
        return dict(anat=anat_path, func=func_path)

    def afni_Coreg(self, anat, meanfunc, aniso=False, inverse=False, surfix='func', debug=False):
        """ Applying bias field correction with ANTs N4 algorithm and then align functional image to
        anatomical space using Afni's 3dAllineate command

        :param anat:        input path for anatomical image, three type of path can be used
                            1. datatype path from the raw data (e.g. 'anat' or 'dti')
                            2. absolute path
                            3. index of path which is shown on 'executed' method
        :param meanfunc:    input path for mean functional image, same as above input path
        :param aniso:       anisotropic 2D slices
        :param inverse:     If this parameter is True, register anatomical image to functional image instead
        :param surfix:      surfix for output path
        :param debug:       set True if you want to print out the executing function of this process (default: False)
        :type anat:         str or int
        :type meanfunc:     str or int
        :type aniso:        bool
        :type inverse:      bool
        :type surfix:       str
        :return:            output path
        :rtype:             dict
        """
        anat = self.check_input(anat)
        meanfunc = self.check_input(meanfunc)
        step = Step(self)
        step.set_message('** Processing coregistration.....')
        step.set_input(name='func', path=meanfunc, idx=0)
        step.set_input(name='anat', path=anat, type=1, idx=0)
        step.set_output(name='output')
        step.set_output(name='transmat', ext='aff12.1D')
        step.set_output(name='temp_01', type=3)
        step.set_output(name='temp_02', type=3)
        if inverse:
            cmd01 = "N4BiasFieldCorrection -d 3 -i {func} -o {temp_01}"
            cmd02 = "N4BiasFieldCorrection -d 3 -i {anat} -o {temp_02}"
        else:
            cmd01 = "N4BiasFieldCorrection -d 3 -i {anat} -o {temp_01}"
            cmd02 = "N4BiasFieldCorrection -d 3 -i {func} -o {temp_02}"
        step.set_cmd(cmd01)
        step.set_cmd(cmd02)
        if aniso == 1:
            cmd03 = "3dAllineate -prefix {output} -onepass -EPI -base {temp_01} -cmass+xy " \
                    "-1Dmatrix_save {transmat} {temp_02}"
        else:
            cmd03 = "3dAllineate -prefix {output} -twopass -EPI -base {temp_01} " \
                    "-1Dmatrix_save {transmat} {temp_02}"
        step.set_cmd(cmd03)
        output_path = step.run('Coregistration', surfix, debug=debug)
        step.reset()
        step.set_message('** Plotting images to check registration.....')
        if inverse:
            step.set_input(name='anat', path=output_path, idx=0)
            step.set_input(name='func', path=meanfunc, type=1, idx=0)
        else:
            step.set_input(name='func', path=output_path, idx=0)
            step.set_input(name='anat', path=anat, type=1, idx=0)
        step.set_output(name='chkreg1', ext='png', dc=1, prefix='Func2Anat')
        step.set_output(name='chkreg2', ext='png', dc=1, prefix='Anat2Func')
        step.set_var(name='test1', value=3)
        cmd04 = 'check_reg {anat} {func} {chkreg1}'
        cmd05 = 'check_reg {func} {anat} {chkreg2}'
        step.set_cmd(cmd04)
        step.set_cmd(cmd05)
        result_path = step.run('Check_Registration', surfix, debug=debug)
        return dict(func=output_path, checkreg = result_path)

    def afni_SkullStripAll(self, func, funcmask, surfix='func', debug=False):
        """ Applying arithmetic skull stripping

        :param func:        input path for all functional image, three type of path can be used
                            1. datatype path from the raw data (e.g. 'anat' or 'dti')
                            2. absolute path
                            3. index of path which is shown on 'executed' method
        :param funcmask:    input path for mean functional image, same as above input path
        :param surfix:      surfix for output path
        :param debug:       set True if you want to print out the executing function of this process (default: False)
        :type func:         str or int
        :type funcmask:     str or int
        :type surfix:       str
        :return:            output path
        :rtype:             dict
        """
        funcmask = self.check_input(funcmask)
        func = self.check_input(func)
        step = Step(self)
        step.set_message('** Processing skull stripping to all {} data.....'.format(surfix))
        step.set_input(name='func', path=func)
        step.set_input(name='mask', path=funcmask, type=1, idx=0)
        step.set_output(name='output')
        cmd = '3dcalc -prefix {output} -expr "a*step(b)" -a {func} -b {mask}'
        step.set_cmd(cmd)
        output_path = step.run('Apply_SkullStrip', surfix, debug=debug)
        return dict(func=output_path)

    def afni_ApplyCoregAll(self, func, coregfunc, surfix='func', debug=False):
        """ Applying transform matrix to all functional data using AFNI's '3dAllineate'

        :param func:        input path for all functional image, three type of path can be used
                            1. datatype path from the raw data (e.g. 'anat' or 'dti')
                            2. absolute path
                            3. index of path which is shown on 'executed' method
        :param coregfunc:   input path for realigned functional image, same as above input path
        :param surfix:      surfix for output path
        :param debug:       set True if you want to print out the executing function of this process (default: False)
        :type func:         str or int
        :type coregfunc:    str or int
        :type surfix:       str
        :return:            output path
        :rtype:             dict
        """
        coregfunc = self.check_input(coregfunc)
        func = self.check_input(func)
        step = Step(self)
        step.set_message('** Applying coregistration to all {} data.....'.format(surfix))
        tform_filters = {'ext': '.aff12.1D'}
        step.set_input(name='func', path=func)
        step.set_input(name='tform', path=coregfunc, filters=tform_filters, type=1, idx=0)
        step.set_input(name='coreg', path=coregfunc, type=1, idx=0)
        step.set_output(name='output')
        cmd = '3dAllineate -prefix {output} -master {coreg} -1Dmatrix_apply {tform} {func}'
        step.set_cmd(cmd)
        output_path = step.run('Apply_Coreg', surfix, debug=debug)
        return dict(func=output_path)

    def afni_SpatialNorm(self, anat, tmpobj, surfix='anat', debug=False):
        """ Align anatomical image to template brain space using AFNI's '3dAllineate'

        :param anat:        input path for anatomical image, three type of path can be used
                            1. datatype path from the raw data (e.g. 'anat' or 'dti')
                            2. absolute path
                            3. index of path which is shown on 'executed' method
        :param tmpobj:      brain template image object
        :param surfix:      surfix for output path
        :param debug:       set True if you want to print out the executing function of this process (default: False)
        :type anat:         str or int
        :type tmpobj:       pn.Template
        :type surfix:       str
        :return:            output path
        :rtype:             dict
        """
        anat = self.check_input(anat)
        step = Step(self)
        step.set_message('** Processing spatial normalization.....')
        step.set_input(name='anat', path=anat, idx=0)
        step.set_var(name='tmpobj', value=tmpobj.template_path, type=1)
        step.set_output(name='output')
        step.set_output(name='transmat', ext='.aff12.1D')
        cmd = '3dAllineate -prefix {output} -twopass -cmass+xy -zclip -conv 0.01 -base {tmpobj} ' \
              '-cost crM -check nmi -warp shr -1Dmatrix_save {transmat} {anat}'
        step.set_cmd(cmd)
        output_path = step.run('SpatialNorm', surfix, debug=debug)
        return dict(normanat=output_path)

    def afni_ApplySpatialNorm(self, func, normanat, surfix='func', debug=False):
        """ Applying transform matrix to all functional data for spatial normalization
        this processor use AFNI's '3dAllineate'

        :param func:        input path for functional image, three type of path can be used
                            1. datatype path from the raw data (e.g. 'anat' or 'dti')
                            2. absolute path
                            3. index of path which is shown on 'executed' method
        :param normanat:    input path for normalized anatomical image, same as above input path
        :param surfix:      surfix for output path
        :param debug:       set True if you want to print out the executing function of this process (default: False)
        :type func:         str or int
        :type normanat:     str or int
        :type surfix:       str
        :return:            output path
        :rtype:             dict
        """
        func = self.check_input(func)
        normanat = self.check_input(normanat)
        step = Step(self)
        step.set_message('** Applying spatial normalization to all {} data.....'.format(surfix))
        step.set_input(name='func', path=func)
        step.set_input(name='normanat', path=normanat, type=1, idx=0)
        transmat_filter = {'ext': '.aff12.1D'}
        step.set_input(name='transmat', path=normanat, filters=transmat_filter, type=1, idx=0)
        step.set_output(name='output')
        cmd = '3dAllineate -prefix {output} -master {normanat} -warp shr -1Dmatrix_apply {transmat} {func}'
        step.set_cmd(cmd)
        output_path = step.run('ApplySpatialNorm', surfix, debug=debug)
        return dict(normfunc=output_path)

    def afni_SpatialSmoothing(self, func, fwhm=0.5, tmpobj=None, surfix='func', debug=False, **kwargs):
        """ Apply gaussian smoothing kernel with given FWHM,
        this process use AFNI's '3dBlurInMask'

        :param func:        input path for functional image, three type of path can be used
                            1. datatype path from the raw data (e.g. 'anat' or 'dti')
                            2. absolute path
                            3. index of path which is shown on 'executed' method
        :param fwhm:        Full with half maximum of Gaussian kernel
        :param tmpobj:      brain template image object
        :param surfix:      surfix for output path
        :param debug:       set True if you want to print out the executing function of this process (default: False)
        :type func:         str or int
        :type fwhm:         float
        :type tmpobj:       pn.Template
        :type surfix:       str
        :return:            output path
        :rtype:             dict
        """
        func = self.check_input(func)
        step = Step(self)
        step.set_message('** Processing spatial smoothing.....')
        step.set_input(name='func', path=func, filters=kwargs)
        if not fwhm:
            methods.raiseerror(messages.Errors.InputValueError, 'the FWHM value have to specified')
        else:
            step.set_var(name='fwhm', value=fwhm, type=1)
        step.set_output(name='output')
        cmd = '3dBlurInMask -prefix {output} -FWHM {fwhm}'
        if tmpobj:
            step.set_var(name='mask', value=str(tmpobj.mask), type=1)
            cmd += ' -mask {mask}'
        cmd += ' -quiet {func}'
        step.set_cmd(cmd)
        output_path = step.run('SpatialSmoothing', surfix, debug=debug)
        return dict(func=output_path)

    def afni_GLManalysis(self, func, paradigm, clip_range=None, surfix='func', debug=False, **kwargs):
        """ run task-based fMRI analysis including estimating the temporal auto-correlation,
        this process use AFNI's '3dDeconvolve' and '3dREMLfit'

        :param func:        input path for functional image, three type of path can be used
                            1. datatype path from the raw data (e.g. 'anat' or 'dti')
                            2. absolute path
                            3. index of path which is shown on 'executed' method
        :param paradigm:    task paradigm for each subject
        :param clip_range:  time point ranges for temporal cropping
        :param surfix:      surfix for output path
        :param debug:       set True if you want to print out the executing function of this process (default: False)
        :type func:         str or int
        :type paradigm:     list
        :type clip_range:   list
        :type surfix:       str
        :type debug:        bool
        :return:            output path
        :rtype:             dict
        """
        func = self.check_input(func)
        step = Step(self)
        step.set_message('** Processing General Linear Analysis')
        step.set_input(name='func', path=func, filters=kwargs)
        step.set_var(name='paradigm', value=paradigm)
        step.set_var(name='param', value='" ".join(map(str, paradigm[idx][0]))', type=1)
        step.set_var(name='model', value='paradigm[idx][1][0]', type=1)
        step.set_var(name='mparam', value='" ".join(map(str, paradigm[idx][1][1]))', type=1)
        step.set_output(name='output')
        step.set_output(name='prefix', ext='remove')
        if clip_range:
            cmd01 = '3dDeconvolve -input {func}'
            cmd01 += '"[{}..{}]" '.format(clip_range[0], clip_range[1])
            cmd01 += '-num_stimts 1 -polort 2 -stim_times 1 "1D: {param}" ' \
                    '"{model}({mparam})" -stim_label 1 STIM -tout -bucket {output}'
        else:
            cmd01 = '3dDeconvolve -input {func} -num_stimts 1 -polort 2 -stim_times 1 "1D: {param}" ' \
                    '"{model}({mparam})" -stim_label 1 STIM -tout -bucket {output} -x1D {prefix}'
        step.set_cmd(cmd01)
        glm = step.run('GLMAnalysis', surfix, debug=debug)
        step.reset()
        step.set_message('** Estimating the temporal auto-correlation structure')
        step.set_input(name='func', path=func, filters=kwargs)
        filters = dict(ext='.xmat.1D')
        step.set_input(name='glm', path=glm, filters=filters, type=1)
        step.set_output(name='output')
        if clip_range:
            cmd02 = '3dREMLfit -matrix {glm} -input {func}'
            cmd02 += '"[{}..{}]" '.format(clip_range[0], clip_range[1])
            cmd02 += '-tout -Rbuck {output} -verb'
        else:
            cmd02 = '3dREMLfit -matrix {glm} -input {func} -tout -Rbuck {output} -verb'
        step.set_cmd(cmd02)
        output_path = step.run('REMLfit', surfix, debug=debug)
        return dict(GLM=output_path)

    def afni_ClusterMap(self, stats, func, pval=0.01, clst_size=40, surfix='func', debug=False):
        """ function to generate mask images from cluster using given threshold parameter.
        this process use AFNI's '3dAttribute', 'cdf', and '3dclust'

        :param stats:       input path for individual level activity map, three type of path can be used
                            1. datatype path from the raw data (e.g. 'funt')
                            2. absolute path
                            3. index of path which is shown on 'executed' method
        :param func:        input path for functional data, three type of path can be used
                            1. datatype path from the raw data (e.g. 'funt')
                            2. absolute path
                            3. index of path which is shown on 'executed' method
        :param pval:        threshold p value
        :param clst_size:   threshold voxel size
        :param surfix:      surfix for output path
        :return:            output path
        """
        stats = self.check_input(stats)
        func = self.check_input(func)
        step = Step(self)
        step.set_message('** Generating clustered masks')
        step.set_input(name='glm', path=stats)
        step.set_input(name='func', path=func, type=1)
        step.set_output(name='output')
        step.set_var(name='pval', value=pval)
        step.set_var(name='csize', value=clst_size)
        cmd01 = '3dAttribute BRICK_STATAUX {glm}'
        step.set_cmd(cmd01, name='dof')
        step.set_var(name='dof', value='dof.split()[-1]', type=1)
        cmd02 = 'cdf -p2t fitt {pval} {dof}'
        step.set_cmd(cmd02, name='tval')
        step.set_var(name='tval', value='tval.split("=")[1].strip()', type=1)
        cmd03 = '3dclust -1Dformat -nosum -1dindex 2 -1tindex 2 -2thresh -{tval} {tval} ' \
                '-dxyz=1 -savemask {output} 1.01 {csize} {glm}'
        step.set_cmd(cmd03)
        step.set_cmd('with open(methods.splitnifti(output) + ".json", "wb") as f:', type=1)
        step.set_cmd('\tjson.dump(dict(source=func[i].Abspath), f)', type=1)
        output_path = step.run('ClusteredMask', surfix=surfix, debug=debug)
        # if jupyter_env:
        #     if ui:
        #         if self._viewer == 'itksnap':
        #             display(gui.itksnap(self, output_path, tmpobj.image.get_filename()))
        #         else:
        #             methods.raiseerror(messages.Errors.InputValueError,
        #                                '"{}" is not available'.format(self._viewer))
        #     else:
        #         pass
        # else:
        #     return dict(mask=output_path)

        return dict(mask=output_path)

    def afni_EstimateSubjectROIs(self, cluster, mask, surfix='func', debug=False):
        cluster = self.check_input(cluster)
        step = Step(self)
        step.set_message('** Estimate ROI masks for each subject')
        step.set_input(name='cluster', path=cluster)
        step.set_input(name='func', path=cluster, filters=dict(ext='.json'), type=1)
        if isinstance(mask, Template):
            mask = str(mask.atlas_path)
        step.set_var(name='mask', value=mask)
        step.set_output(name='output', type=0)
        step.set_output(name='json', type=0, ext='json')
        cmd01 = '3dcalc -prefix {output} -expr "a*step(b)" -a {mask} -b {cluster}'
        step.set_cmd(cmd01)
        cmd02 = 'cp {func} {json}'
        step.set_cmd(cmd02)
        output_path = step.run('SubjectROIs', surfix=surfix, debug=debug)
        return dict(mask=output_path)

    def afni_SignalProcessing(self, func, norm=True, ort=None, clip_range=None, mask=None, bpass=None,
                              fwhm=None, dt=None, surfix='func', debug=False, **kwargs):
        """Wrapper method of afni's 3dTproject for signal processing of resting state data

        :param func:
        :param norm:
        :param ort:
        :param mask:
        :param bpass:
        :param fwhm:
        :param dt:
        :param surfix:
        :return:
        """
        display(title(value='** Run signal processing for resting state data'))
        step = Step(self)
        func = self.check_input(func)
        ort = self.check_input(ort)
        step.set_input(name='func', path=func, filters=kwargs)
        step.set_output(name='output')
        cmd = ['3dTproject -prefix {output}']
        orange, irange = None, None         # orange= range of ort file, irange= range of image file

        # Parameters
        if bpass:                           # set bandpass filter
            if isinstance(bpass, list) and len(bpass) == 2:
                cmd.append('-passband {} {}'.format(*bpass))
            else:
                pass
        if norm:                            # set signal normalization
            cmd.append('-norm')
        if ort:                             # set ort (nuisance signal regression)
            if clip_range:
                if isinstance(clip_range, list):
                    if len(clip_range) == 2:
                        orange = "'{" + "{}..{}".format(*clip_range) + "}'"
                        irange = "'[" + "{}..{}".format(*clip_range) + "]'"
                        step.set_var(name='orange', value=orange, type=1)
                        step.set_var(name='irange', value=irange, type=1)

            ort_filter = {'ext': '.1D', 'ignore': ['.aff12']}
            if kwargs:
                for key in kwargs.keys():
                    if 'ignore' in key:
                        if isinstance(kwargs['ignore'], list):
                            ort_filter['ignore'].extend(kwargs.pop('ignore'))
                        else:
                            ort_filter['ignore'].append(kwargs.pop('ignore'))
                    if 'ext' in key:
                        kwargs.pop('ext')
                ort_filter.update(kwargs)
            if isinstance(ort, dict):
                for key, value in ort.items():
                    ortpath = self.check_input(value)
                    if clip_range:
                        cmd.append('-ort {{}}'.format(key)+'{orange}')
                    else:
                        cmd.append('-ort {{}}'.format(key))
                    step.set_input(name=key, path=ortpath, filters=ort_filter, type=1)
            elif isinstance(ort, list):
                for i, o in enumerate(ort):
                    exec('ort_{} = self.check_input({})'.format(str(i), o))
                    ort_name = 'ort_{}'.format(str(i))
                    if clip_range:
                        cmd.append('-ort {}'.format(ort_name)+'{orange}')
                    else:
                        cmd.append('-ort {}'.format(ort_name))
                    step.set_input(name=ort_name, path=o, filters=ort_filter, type=1)
            elif isinstance(ort, str):
                ort = self.check_input(ort)
                if clip_range:
                    cmd.append('-ort {ort}"{orange}"')
                else:
                    cmd.append('-ort {ort}')
                step.set_input(name='ort', path=ort, filters=ort_filter, type=1)
            else:
                self.logger.debug('TypeError on input ort.')
        if mask:                            # set mask
            if os.path.isfile(mask):
                step.set_var(name='mask', value=mask, type=1)
            elif os.path.isdir(mask):
                step.set_input(name='mask', path=mask, type=1)
            else:
                pass
            cmd.append('-mask {mask}')
        if fwhm:                            # set smoothness
            step.set_var(name='fwhm', value=fwhm)
            cmd.append('-blur {fwhm}')
        if dt:                              # set sampling rate (TR)
            step.set_var(name='dt', value=dt)
            cmd.append('-dt {dt}')
        if clip_range:                           # set range
            cmd.append('-input {func}"{irange}"')
        else:
            cmd.append('-input {func}')
        step.set_cmd(" ".join(cmd))
        output_path = step.run('SignalProcess', surfix=surfix, debug=debug)
        return dict(signalprocess=output_path)

    def afni_ROIStats(self, func, rois, cbv=False, cbv_param=None, clip_range=None, option=None,
                      label=None, surfix='func', debug=False, **kwargs):
        """Extracting time-course data from ROIs

        :param func:    Input path for functional data
        :param roi:     Template instance or mask path
        :param cbv:     Input path for MION infusion data
        :param cbv_param:     [echotime, number of volumes (TR) to average]
        :param clip_range:
        :param option:   if roi is Template instance
        :param label:
        :param surfix:
        :type func:     str
        :type roi:      Template or str
        :type cbv:      str
        :type cbv_param:list
        :type option:   {'bilateral', 'merge', 'contra'}
        :type surfix:   str
        :return:        Current step path
        :rtype:         dict
        """
        func = self.check_input(func)
        # Check if given rois is Template instance
        tmp = None
        list_of_roi = None
        if not isinstance(rois, str):
            try:
                if option:
                    if option == 'bilateral':
                        tmp = TempFile(rois.atlas, atlas=True, bilateral=True)
                    elif option == 'merge':
                        tmp = TempFile(rois.atlas, atlas=True, merge=True)
                    elif option == 'contra':
                        tmp = TempFile(rois.atlas, atlas=True, flip=True)
                    else:
                        tmp = TempFile(rois.atlas, atlas=True)
                else:
                    tmp = TempFile(rois.atlas, atlas=True)
                rois = str(tmp.path)
                list_of_roi = list(tmp.label)
            except:
                pass
        else:
            pass
        if label:
            list_of_roi = [roi for roi, cmap in label.label.values()][1:]
        # Check if given rois path is existed in the list of executed steps
        rois = self.check_input(rois)

        # Initiate step instance
        step = Step(self)
        step.set_message('** Extracting time-course data from ROIs')

        # If given roi path is single file
        if os.path.isfile(rois):
            step.set_var(name='rois', value=rois)
            step.set_input(name='func', path=func)
            cmd = '3dROIstats -mask {rois} {func}'
        # Else, given roi path is directory
        else:
            step.set_input(name='rois', path=rois)
            step.set_input(name='func', path=rois, filters=dict(ext='.json'), type=1)
            step.set_cmd('json.load(open(func[i].Abspath))["source"]', type=1, name='func_path')
            step.set_var(name='func_path', value='func_path', type=1)
            cmd = '3dROIstats -mask {rois} {func_path}'
        step.set_output(name='output', dc=1, ext='xlsx')
        step.set_module('pandas', rename='pd')
        step.set_module('numpy', rename='np')
        step.set_module('StringIO', sub='StringIO')
        # If CBV parameters are given, parsing the CBV infusion file path from json file
        if any(isinstance(cbv, t) for t in [int, str]):
            cbv = self.check_input(cbv)
            step.set_input(name='cbv', path=cbv, type=1, filters=dict(ext='.json'))
        if clip_range:
            cmd += '"[{}..{}]"'.format(clip_range[0], clip_range[1])
        step.set_cmd(cmd, name='out')
        step.set_cmd('df = pd.read_table(StringIO(out))', type=1)
        step.set_cmd('df = df[df.columns[2:]]', type=1)
        # If given roi is Template instance
        if list_of_roi:
            step.set_var(name='list_rois', value=list_of_roi)
            step.set_cmd('avail_rois = [int(roi.strip().split("_")[1])-1 for roi in list(df.columns)]', type=1)
            step.set_cmd('final_list_rois = list(np.array(list_rois)[avail_rois])', type=1)
            step.set_cmd('df.columns = final_list_rois', type=1)
        # again, if CBV parameter are given, put commends and methods into custom build function
        if cbv_param:
            if isinstance(cbv_param, list) and (len(cbv_param) == 2):
                step.set_var(name='te', value=cbv_param[0])
                step.set_var(name='n_tr', value=cbv_param[1])
                step.set_cmd('cbv_path = json.load(open(cbv[i].Abspath))["cbv"]', type=1)
                step.set_var(name='cbv_path', value='cbv_path', type=1)
                cbv_cmd = '3dROIstats -mask {rois} {cbv_path}'
                step.set_cmd(cbv_cmd, name='cbv_out')
                # step.set_cmd('temp_outputs.append([out, err])', type=1)
                step.set_cmd('cbv_df = pd.read_table(StringIO(cbv_out))', type=1)
                step.set_cmd('cbv_df = cbv_df[cbv_df.columns[2:]]', type=1)
                if list_of_roi:
                    step.set_cmd('cbv_df.columns = final_list_rois', type=1)
            else:
                methods.raiseerror(messages.Errors.InputValueError, 'Please check input CBV parameters')
        step.set_cmd('if len(df.columns):', type=1)
        # again, if CBV parameter are given, correct the CBV changes.
        if cbv_param:
            step.set_cmd('\tdR2_mion = (-1 / te) * np.log(df.loc[:n_tr, :].mean(axis=0) / '
                         'cbv_df.loc[:n_tr, :].mean(axis=0))', type=1)
            step.set_cmd('\tdR2_stim = (-1 / te) * np.log(df / df.loc[:n_tr, :].mean(axis=0))', type=1)
            step.set_cmd('\tdf = dR2_stim/dR2_mion', type=1)
        # Generating excel files
        step.set_cmd('fname = os.path.splitext(str(func[i].Filename))[0]', type=1, level=1)
        step.set_cmd('df.to_excel({output}, index=False)', type=1, level=1)
        step.set_cmd('else:', type=1)
        step.set_cmd('pass', type=1, level=1)

        # Run the steps
        output_path = step.run('ExtractROIs', surfix=surfix, debug=debug)
        if tmp:
            tmp.close()
        return dict(timecourse=output_path)

    def afni_TemporalClipping(self, func, clip_range, surfix='func', debug=False, **kwargs):
        """

        :param func:
        :param clip_range:
        :param surfix:
        :param kwargs:
        :return:
        """
        step = Step(self)
        step.set_message('** Temporal clipping of functional image')
        func = self.check_input(func)
        step.set_input(name='func', path=func, filters=kwargs)
        step.set_output(name='output')
        if clip_range:
            if isinstance(clip_range, list):
                if len(clip_range) == 2:
                    irange = "'[" + "{}..{}".format(*clip_range) + "]'"
                    step.set_var(name='irange', value=irange, type=1)
        cmd = '3dcalc -prefix {output} -expr "a" -a {func}"{irange}"'
        step.set_cmd(cmd)
        output_path = step.run('TemporalClipping', surfix, debug=debug)
        return dict(clippedfunc=output_path)

    def afni_GroupAverage(self, func, idx_coef=1, idx_tval=2, surfix='func',
                          outliers=None, debug=False):
        """ This processor performing the Mixed Effects Meta Analysis to estimate group mean
        It's required to install R, plus 'snow' package.

        If you want to cite the analysis approach, use the following at this moment:

        Chen et al., 2012. FMRI Group Analysis Combining Effect Estimates
        and Their Variances. NeuroImage. NeuroImage 60: 747-765.

        :param func:
        :param idx_coef:
        :param idx_tval:
        :param outliers:
        :param surfix:
        :param kwargs:
        :return:
        """
        step = Step(self, n_thread=1)
        step.set_message('** Estimate group mean using Mixed effect meta analysis')
        func = self.check_input(func)
        if outliers:
            filters = dict(ignore=outliers)
        else:
            filters = None
        step.set_input(name='func', path=func, type=2, filters=filters)
        step.set_output(name='output', dc=1, type=1, prefix='MEMA_1sampTtest')
        step.set_var(name='idx_coef', value=idx_coef)
        step.set_var(name='idx_tval', value=idx_tval)
        step.set_var(name='jobs', value=multiprocessing.cpu_count())
        step.set_var(name='group', value='subj')
        cmd = 'onesample_ttest -i {func} -o {output} -b {idx_coef} -t {idx_tval} -j {jobs} -g {group}'
        step.set_cmd(cmd)
        output_path = step.run('GroupAverage', surfix, debug=debug)
        return dict(groupavr=output_path)