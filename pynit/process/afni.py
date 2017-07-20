from base import *
from pynit.handler.images import TempFile
from pynit.handler.step import Step

class AFNI_Process(BaseProcess):
    def __init__(self, *args, **kwargs):
        super(AFNI_Process, self).__init__(*args, **kwargs)

    def afni_MeanImgCalc(self, func, cbv=False, surfix='func'):
        """Mean image calculation for functional image : Initial preparation

        :param func: str, Name of functional data folder at source location (eg. 'func')
        :param cbv: boolean, True if MION contrast agent is infused
        :param surfix: str, Output folder surfix
        :return: output_path: dict, Absolute path of outputs
        """
        display(title(value='** Processing mean image calculation.....'))
        func = self.check_input(func)
        step = Step(self)
        step.set_input(name='func', input_path=func, static=True)
        step.set_outparam(name='mparam', ext='.1D')
        cmd01 = "3dvolreg -prefix {temp_01} -1Dfile {mparam} -Fourier -verbose -base 0 {func}"
        step.set_command(cmd01)
        if cbv:
            cmd02 = "3dinfo -nv {func}"
            step.set_staticinput('bold', 'int(int(ttime)/3)')
            step.set_staticinput('bold_output', 'methods.splitnifti(output)+"_BOLD.nii.gz"')
            step.set_staticinput('cbv', 'int(int(ttime)*2/3)')
            step.set_staticinput('cbv_output', 'methods.splitnifti(output)+"_CBV.nii.gz"')
            step.set_command(cmd02, stdout='ttime')
            options = ['"[0..{bold}]"',
                       '"[{cbv}..$]"']
            cmd03 = "3dTstat -prefix {bold_output} -mean {temp_01}" + options[0]
            step.set_command(cmd03)
            cmd04 = "3dTstat -prefix {cbv_output} -mean {temp_01}" + options[1]
            step.set_command(cmd04)
            # step.get_executefunc('test', verbose=True)
            output_path = step.run('MeanImgCalc-CBV', surfix)
        else:
            cmd02 = "3dTstat -prefix {output} -mean {temp_01}"
            step.set_command(cmd02)
            output_path = step.run('MeanImgCalc-BOLD', surfix)
        return dict(meanfunc=output_path)

    def afni_SliceTimingCorrection(self, func, tr=None, tpattern=None, surfix='func'):
        """Corrects for slice time differences when individual 2D slices are recorded over a 3D image

        :param func: str,
        :param tr: int,
        :param tpattern: str,
        :param surfix: str,
        :return: output_path: dict, Absolute path of outputs
        """
        display(title(value='** Processing slice timing correction.....'))
        func = self.check_input(func)
        options = str()
        step = Step(self)
        step.set_input(name='func', input_path=func, static=False)
        cmd = "3dTshift -prefix {output}"
        if tr:
            options += " -TR {0}".format(tr)
        if tpattern:
            options += " -tpattern {0}".format(tpattern)
        else:
            options += " -tpattern altplus"
        input_str = " {func}"
        cmd = cmd+options+input_str
        step.set_command(cmd)
        output_path = step.run('SliceTmCorrect', surfix)
        return dict(func=output_path)

    def afni_MotionCorrection(self, func, base, surfix='func'):
        """

        :param func:
        :param base:
        :param surfix:
        :return:
        """
        display(title(value='** Processing motion correction.....'))
        func = self.check_input(func)
        base = self.check_input(base)
        step = Step(self)
        step.set_input(name='func', input_path=func, static=False)
        try:
            if '-CBV-' in base:
                mimg_filters = {'file_tag': '_CBV', 'ignore': 'BOLD'}
                step.set_input(name='base', input_path=base, filters=mimg_filters, static=True, side=True)
            else:
                step.set_input(name='base', input_path=base, static=True, side=True)
            # mimg_path = self.steps[0]
            # if '-CBV-' in mimg_path:
            #     mimg_filters = {'file_tag': '_CBV'}
            #     step.set_input(name='base', input_path=mimg_path, filters=mimg_filters, static=True, side=True)
            # else:
            #     step.set_input(name='base', input_path=mimg_path, static=True, side=True)
        except:
            methods.raiseerror(messages.Errors.MissingPipeline,
                               'Initial Mean image calculation step has not been executed!')
        step.set_outparam(name='mparam', ext='.1D')
        step.set_outparam('transmat', ext='.aff12.1D')
        cmd01 = "3dvolreg -prefix {temp_01} -1Dfile {mparam} -Fourier -verbose -base 0 {func}"
        step.set_command(cmd01)
        cmd02 = "3dTstat -mean -prefix {temp_02} {temp_01}"
        step.set_command(cmd02)
        cmd03 = "3dAllineate -prefix {temp_03} -warp sho -base {base} -1Dmatrix_save {transmat} {temp_02}"
        step.set_command(cmd03)
        cmd04 = '3dAllineate -prefix {output} -1Dmatrix_apply {transmat} -warp sho {temp_01}'
        step.set_command(cmd04)
        output_path = step.run('MotionCorrection', surfix)
        return dict(func=output_path)

    def afni_MaskPrep(self, anat, tmpobj, func=None, surfix='func'):
        """

        :param anat:
        :return:
        """
        display(title(value='** Processing mask image preparation.....'))
        anat = self.check_input(anat)
        step = Step(self)
        mimg_path = None
        try:
            step.set_input(name='anat', input_path=anat, static=True)
        except:
            methods.raiseerror(messages.Errors.MissingPipeline,
                               'No anatomy file!')
        try:
            step.set_staticinput(name='mask', value=str(tmpobj.mask))
        except:
            methods.raiseerror(messages.InputPathError,
                               'No mask template file!')
        cmd01 = "3dAllineate -prefix {temp1} -NN -onepass -EPI -base {anat} -cmass+xy {mask}"
        cmd02 = '3dcalc -prefix {output} -expr "astep(a, 0.5)" -a {temp1}'
        step.set_command(cmd01)
        step.set_command(cmd02)
        anat_mask = step.run('MaskPrep', 'anat')
        step = Step(self)
        try:
            if func:
                mimg_path = self.check_input(func)
            else:
                mimg_path = self.steps[0]
            if '-CBV-' in mimg_path:
                mimg_filters = {'file_tag': '_BOLD'}
                step.set_input(name='func', input_path=mimg_path, filters=mimg_filters, static=True)
            else:
                step.set_input(name='func', input_path=mimg_path, static=True)
        except:
            methods.raiseerror(messages.Errors.MissingPipeline,
                               'Initial Mean image calculation step has not been executed!')
        try:
            step.set_staticinput(name='mask', value=str(tmpobj.mask))
        except:
            methods.raiseerror(messages.InputPathError,
                               'No mask template file!')
        cmd01 = "3dAllineate -prefix {temp1} -NN -onepass -EPI -base {func} -cmass+xy {mask}"
        cmd02 = '3dcalc -prefix {output} -expr "astep(a, 0.5)" -a {temp1}'
        step.set_command(cmd01, idx=0)
        step.set_command(cmd02)
        func_mask = step.run('MaskPrep', surfix)
        if jupyter_env:
            if self._viewer == 'itksnap':
                display(widgets.VBox([title(value='-'*43 + ' Anatomical images ' + '-'*43),
                                      viewer.itksnap(self, anat_mask, anat),
                                      title(value='<br>' + '-'*43 + ' Functional images ' + '-'*43),
                                      viewer.itksnap(self, func_mask, mimg_path)]))
            elif self._viewer == 'fslview':
                display(widgets.VBox([title(value='-'*43 + ' Anatomical images ' + '-'*43),
                                      viewer.fslview(self, anat_mask, anat),
                                      title(value='<br>' + '-'*43 + ' Functional images ' + '-'*43),
                                      viewer.fslview(self, func_mask, mimg_path)]))
            else:
                methods.raiseerror(messages.Errors.InputValueError,
                                   '"{}" is not available'.format(self._viewer))
        else:
            return dict(anat_mask=anat_mask, func_mask=func_mask)

    def afni_SkullStrip(self, anat, func, surfix='func'):
        """ The pre-defined step for skull stripping with AFNI

        :param anat:
        :param func:
        :return:
        """
        display(title(value='** Processing skull stripping.....'))
        anat = self.check_input(anat)
        func = self.check_input(func)
        anat_mask = [self.steps[idx] for idx, step in self.executed.items() if 'MaskPrep-anat' in step][0]
        anat_mask = self.check_input(anat_mask)
        func_mask = [self.steps[idx] for idx, step in self.executed.items() if 'MaskPrep-{}'.format(surfix) in step][0]
        func_mask = self.check_input(func_mask)
        step = Step(self)
        step.set_input(name='anat', input_path=anat, static=True)
        step.set_input(name='anat_mask', input_path=anat_mask, static=True, side=True)
        cmd01 = '3dcalc -prefix {output} -expr "a*step(b)" -a {anat} -b {anat_mask}'
        step.set_command(cmd01)
        anat_path = step.run('SkullStrip', 'anat')
        step = Step(self)
        if '-CBV-' in func:
            func_filter = {'file_tag':'_BOLD'}
            step.set_input(name='func', input_path=func, filters=func_filter, static=True)
        else:
            step.set_input(name='func', input_path=func, static=True)
        step.set_input(name='func_mask', input_path=func_mask, static=True, side=True)
        cmd02 = '3dcalc -prefic {output} -expr "a*step(b)" -a {func} -b {func_mask}'
        step.set_command(cmd02, idx=0)
        func_path = step.run('SkullStrip', surfix)
        return dict(anat=anat_path, func=func_path)

    def afni_Coreg(self, anat, meanfunc, surfix='func'):
        """Applying bias field correction with ANTs N4 algorithm and then align funtional image to
        anatomical space using Afni's 3dAllineate command

        :param anat:
        :param meanfunc:
        :param surfix:
        :return:
        """
        display(title(value='** Processing coregistration.....'))
        anat = self.check_input(anat)
        meanfunc = self.check_input(meanfunc)
        step = Step(self)
        step.set_input(name='anat', input_path=anat, static=True, side=True)
        step.set_input(name='func', input_path=meanfunc, static=True)
        step.set_outparam(name='transmat', ext='.aff12.1D')
        cmd01 = "N4BiasFieldCorrection -i {anat} -o {temp_01}"
        step.set_command(cmd01)
        cmd02 = "N4BiasFieldCorrection -i {func} -o {temp_02}"
        step.set_command(cmd02)
        cmd03 = "3dAllineate -prefix {output} -onepass -EPI -base {temp_01} -cmass+xy " \
                "-1Dmatrix_save {transmat} {temp_02}"
        step.set_command(cmd03)
        output_path = step.run('Coregistration', surfix)
        return dict(func=output_path)

    def afni_SkullStripAll(self, func, meanfunc, surfix='func'):
        """Applying arithmetic skull stripping

        :param func:
        :param meanfunc:
        :param surfix:
        :return:
        """
        display(title(value='** Processing skull stripping to all {} data.....'.format(surfix)))
        meanfunc = self.check_input(meanfunc)
        func = self.check_input(func)
        step = Step(self)
        step.set_input(name='mask', input_path=meanfunc, static=True, side=True)
        step.set_input(name='func', input_path=func)
        cmd = '3dcalc -prefix {output} -expr "a*step(b)" -a {func} -b {mask}'
        step.set_command(cmd)
        output_path = step.run('Apply_SkullStrip', surfix)
        return dict(func=output_path)

    def afni_ApplyCoregAll(self, func, coregfunc, surfix='func'):
        """Applying transform matrix to all functional data using Afni's 3dAllineate command

        :param func:
        :param coregfunc:
        :param surfix:
        :return:
        """
        display(title(value='** Applying coregistration to all {} data.....'.format(surfix)))
        coregfunc = self.check_input(coregfunc)
        func = self.check_input(func)
        step = Step(self)
        tform_filters = {'ext':'.aff12.1D'}
        step.set_input(name='tform', input_path=coregfunc, filters=tform_filters, static=True, side=True)
        step.set_input(name='coreg', input_path=coregfunc, static=True, side=True)
        step.set_input(name='func', input_path=func)
        cmd = '3dAllineate -prefix {output} -master {coreg} -1Dmatrix_apply {tform} {func}'
        step.set_command(cmd)
        output_path = step.run('Apply_Coreg', surfix)
        return dict(func=output_path)

    def afni_SpatialNorm(self, anat, tmpobj, surfix='anat'):
        """Align anatomical image to template brain space using Afni's 3dAllineate command

        :param anat:
        :param tmpobj:
        :param surfix:
        :return:
        """
        display(title(value='** Processing spatial normalization.....'))
        anat = self.check_input(anat)
        step = Step(self)
        step.set_input(name='anat', input_path=anat, static=True)
        step.set_staticinput(name='tmpobj', value=tmpobj.template_path)
        step.set_outparam(name='transmat', ext='.aff12.1D')
        cmd = '3dAllineate -prefix {output} -twopass -cmass+xy -zclip -conv 0.01 -base {tmpobj} ' \
              '-cost crM -check nmi -warp shr -1Dmatrix_save {transmat} {anat}'
        step.set_command(cmd)
        output_path = step.run('SpatialNorm', surfix)
        return dict(normanat=output_path)

    def afni_ApplySpatialNorm(self, func, normanat, surfix='func'):
        """Applying transform matrix to all functional data for spatial normalization

        :param func:
        :param normanat:
        :param surfix:
        :return:
        """
        display(title(value='** Applying spatial normalization to all {} data.....'.format(surfix)))
        func = self.check_input(func)
        normanat = self.check_input(normanat)
        step = Step(self)
        step.set_input(name='func', input_path=func)
        step.set_input(name='normanat', input_path=normanat, static=True, side=True)
        transmat_filter = {'ext':'.aff12.1D'}
        step.set_input(name='transmat', input_path=normanat, filters=transmat_filter, static=True, side=True)
        cmd = '3dAllineate -prefix {output} -master {normanat} -warp shr -1Dmatrix_apply {transmat} {func}'
        step.set_command(cmd)
        output_path = step.run('ApplySpatialNorm', surfix)
        return dict(normfunc=output_path)

    def afni_SpatialSmoothing(self, func, fwhm=0.5, tmpobj=None, surfix='func', **kwargs):
        """

        :param func:
        :param fwhm:
        :param tmpobj:
        :param surfix:
        :return:
        """
        display(title(value='** Processing spatial smoothing.....'))
        func = self.check_input(func)
        filters, subj, sess = self.check_filters(**kwargs)
        step = Step(self, subjects=subj, sessions=sess)
        step.set_input(name='func', input_path=func, filters=filters)
        if not fwhm:
            methods.raiseerror(messages.Errors.InputValueError, 'the FWHM value have to specified')
        else:
            step.set_staticinput('fwhm', fwhm)
        cmd = '3dBlurInMask -prefix {output} -FWHM {fwhm}'
        if tmpobj:
            step.set_staticinput('mask', value=str(tmpobj.mask))
            cmd += ' -mask {mask}'
        cmd += ' -quiet {func}'
        step.set_command(cmd)
        output_path = step.run('SpatialSmoothing', surfix)
        return dict(func=output_path)

    def afni_GLManalysis(self, func, paradigm, clip_range=None, surfix='func', **kwargs):
        """

        :param func:
        :param paradigm:
        :param clip_range:
        :param surfix:
        :return:
        """
        display(title(value='** Processing General Linear Analysis'))
        func = self.check_input(func)
        filters, subj, sess = self.check_filters(**kwargs)
        step = Step(self, subjects=subj, sessions=sess)
        step.set_input(name='func', input_path=func, filters=filters)
        step.set_variable(name='paradigm', value=paradigm)
        step.set_staticinput(name='param', value='" ".join(map(str, paradigm[idx][0]))')
        step.set_staticinput(name='model', value='paradigm[idx][1][0]')
        step.set_staticinput(name='mparam', value='" ".join(map(str, paradigm[idx][1][1]))')
        if clip_range:
            cmd01 = '3dDeconvolve -input {func}'
            cmd01 += '"[{}..{}]" '.format(clip_range[0], clip_range[1])
            cmd01 += '-num_stimts 1 -polort 2 -stim_times 1 "1D: {param}" ' \
                    '"{model}({mparam})" -stim_label 1 STIM -tout -bucket {output}'
        else:
            cmd01 = '3dDeconvolve -input {func} -num_stimts 1 -polort 2 -stim_times 1 "1D: {param}" ' \
                    '"{model}({mparam})" -stim_label 1 STIM -tout -bucket {output} -x1D {prefix}'
        step.set_command(cmd01)
        glm = step.run('GLMAnalysis', surfix, debug=False)
        display(title(value='** Estimating the temporal auto-correlation structure'))
        step = Step(self, subjects=subj, sessions=sess)
        step.set_input(name='func', input_path=func, filters=filters)
        filter = dict(ext='.xmat.1D')
        step.set_input(name='glm', input_path=glm, filters=filter, side=True)
        if clip_range:
            cmd02 = '3dREMLfit -matrix {glm} -input {func}'
            cmd02 += '"[{}..{}]" '.format(clip_range[0], clip_range[1])
            cmd02 += '-tout -Rbuck {output} -verb'
        else:
            cmd02 = '3dREMLfit -matrix {glm} -input {func} -tout -Rbuck {output} -verb'
        step.set_command(cmd02)
        output_path = step.run('REMLfit', surfix, debug=False)
        return dict(GLM=output_path)

    def afni_ClusterMap(self, glm, func, tmpobj, pval=0.01, cluster_size=40, surfix='func'):
        """Wrapper method of afni's 3dclust for generating clustered mask

        :param glm:
        :param func:
        :param tmpobj:
        :param pval:
        :param cluster_size:
        :param surfix:
        :return:
        """
        display(title(value='** Generating clustered masks'))
        glm = self.check_input(glm)
        func = self.check_input(func)
        step = Step(self)
        step.set_input(name='glm', input_path=glm)
        step.set_input(name='func', input_path=func, side=True)
        step.set_staticinput(name='pval', value=pval)
        step.set_staticinput(name='csize', value=cluster_size)
        cmd01 = '3dAttribute BRICK_STATAUX {glm}'
        step.set_command(cmd01, stdout='dof')
        step.set_staticinput(name='dof', value='dof.split()[-1]')
        cmd02 = 'cdf -p2t fitt {pval} {dof}'
        step.set_command(cmd02, stdout='tval')
        step.set_execmethod('tval.split("=")[1].strip()', var='tval')
        cmd03 = '3dclust -1Dformat -nosum -1dindex 2 -1tindex 2 -2thresh -{tval} {tval} ' \
                '-dxyz=1 -savemask {output} 1.01 {csize} {glm}'
        step.set_command(cmd03)
        step.set_execmethod('with open(methods.splitnifti(output) + ".json", "wb") as f:')
        step.set_execmethod('\tjson.dump(dict(source=func[i].Abspath), f)')
        output_path = step.run('ClusteredMask', surfix=surfix)
        if jupyter_env:
            if self._viewer == 'itksnap':
                display(viewer.itksnap(self, output_path, tmpobj.image.get_filename()))
            elif self._viewer == 'fslview':
                display(viewer.fslview(self, output_path, tmpobj.image.get_filename()))
            else:
                methods.raiseerror(messages.Errors.InputValueError,
                                   '"{}" is not available'.format(self._viewer))
        else:
            return dict(mask=output_path)

    def afni_SignalProcessing(self, func, norm=True, ort=None, clip_range=None, mask=None, bpass=None,
                              fwhm=None, dt=None, surfix='func', **kwargs):
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
        filters, subj, sess = self.check_filters(**kwargs)
        step = Step(self, subjects=subj, sessions=sess)
        func = self.check_input(func)
        ort = self.check_input(ort)
        step.set_input(name='func', input_path=func, filters=filters)
        cmd = ['3dTproject -prefix {output}']
        orange, irange = None, None         # orange= range of ort file, irange= range of image file
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
                        step.set_staticinput(name='orange', value=orange)
                        step.set_staticinput(name='irange', value=irange)

            ort_filter = {'ext': '.1D', 'ignore': ['.aff12']}
            if filters:
                for key in filters.keys():
                    if 'ignore' in key:
                        if isinstance(filters['ignore'], list):
                            ort_filter['ignore'].extend(filters.pop('ignore'))
                        else:
                            ort_filter['ignore'].append(filters.pop('ignore'))
                    if 'ext' in key:
                        filters.pop('ext')
                ort_filter.update(filters)
            if isinstance(ort, dict):
                for key, value in ort.items():
                    ortpath = self.check_input(value)
                    if clip_range:
                        cmd.append('-ort {{}}'.format(key)+'{orange}')
                    else:
                        cmd.append('-ort {{}}'.format(key))
                    step.set_input(name=key, input_path=ortpath, filters=ort_filter, side=True)
            elif isinstance(ort, list):
                for i, o in enumerate(ort):
                    exec('ort_{} = self.check_input({})'.format(str(i), o))
                    ort_name = 'ort_{}'.format(str(i))
                    if clip_range:
                        cmd.append('-ort {}'.format(ort_name)+'{orange}')
                    else:
                        cmd.append('-ort {}'.format(ort_name))
                    step.set_input(name=ort_name, input_path=o, filters=ort_filter, side=True)
            elif isinstance(ort, str):
                ort = self.check_input(ort)
                if clip_range:
                    cmd.append('-ort {ort}"{orange}"')
                else:
                    cmd.append('-ort {ort}')
                step.set_input(name='ort', input_path=ort, filters=ort_filter, side=True)
            else:
                self.logger.debug('TypeError on input ort.')
        if mask:                            # set mask
            if os.path.isfile(mask):
                step.set_staticinput(name='mask', value=mask)
            elif os.path.isdir(mask):
                step.set_input(name='mask', input_path=mask, static=True, side=True)
            else:
                pass
            cmd.append('-mask {mask}')
        if fwhm:                            # set smoothness
            step.set_staticinput(name='fwhm', value=fwhm)
            cmd.append('-blur {fwhm}')
        if dt:                              # set sampling rate (TR)
            step.set_staticinput(name='dt', value=dt)
            cmd.append('-dt {dt}')
        if clip_range:                           # set range
            cmd.append('-input {func}"{irange}"')
        else:
            cmd.append('-input {func}')
        step.set_command(" ".join(cmd))
        output_path = step.run('SignalProcess', surfix=surfix, debug=False)
        return dict(signalprocess=output_path)

    def afni_ROIStats(self, func, rois, cbv=None, clip_range=None, option=None, surfix='func', **kwargs):
        """Extracting time-course data from ROIs

        :param func:    Input path for functional data
        :param roi:     Template instance or mask path
        :param cbv:     [echotime, number of volumes (TR) to average]
        :param clip_range:
        :param option:  bilateral or merge if roi is Template instance
        :param surfix:
        :type func:     str
        :type roi:      Template or str
        :type cbv:      list
        :type surfix:   str
        :return:        Current step path
        :rtype:         dict
        """
        display(title(value='** Extracting time-course data from ROIs'))
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
        # Check if given rois path is existed in the list of executed steps
        rois = self.check_input(rois)

        # Initiate step instance
        filters, subj, sess = self.check_filters(**kwargs)
        step = Step(self, subjects=subj, sessions=sess)

        # If given roi path is single file
        if os.path.isfile(rois):
            step.set_staticinput(name='rois', value=rois)
            step.set_input(name='func', input_path=func)
            cmd = '3dROIstats -mask {rois} {func}'
        # Else, given roi path is directory
        else:
            step.set_input(name='rois', input_path=rois)
            step.set_input(name='func', input_path=rois, filters=dict(ext='json'), side=True)
            step.set_execmethod('func_path = json.load(open(func[i].Abspath))["source"]')
            step.set_staticinput('func_path', value='func_path')
            cmd = '3dROIstats -mask {rois} {func_path}'
        # If CBV parameters are given, parsing the CBV infusion file path from json file
        if cbv:
            step.set_input(name='cbv', input_path=func, side=True, filters=dict(ext='.json'))
        if clip_range:
            cmd += '"[{}..{}]"'.format(clip_range[0], clip_range[1])
        step.set_command(cmd, stdout='out')
        step.set_execmethod('temp_outputs.append([out, err])')
        step.set_execmethod('pd.read_table(StringIO(out))', var='df')
        step.set_execmethod('df[df.columns[2:]]', var='df')
        # If given roi is Template instance
        # if list_of_roi:
        #     step.set_variable(name='list_roi', value=list_of_roi)
        #     step.set_execmethod('list_roi', var='df.columns')
        # again, if CBV parameter are given, put commends and methods into custom build function
        if cbv:
            if isinstance(cbv, list) and len(cbv) == 2:
                step.set_variable(name='te', value=cbv[0])
                step.set_variable(name='n_tr', value=cbv[1])
                step.set_execmethod('cbv_path = json.load(open(cbv[i].Abspath))["cbv"]')
                step.set_staticinput(name='cbv_path', value='cbv_path')
                cbv_cmd = '3dROIstats -mask {rois} {cbv_path}'
                step.set_command(cbv_cmd, stdout='cbv_out')
                step.set_execmethod('temp_outputs.append([out, err])')
                step.set_execmethod('pd.read_table(StringIO(cbv_out))', var='cbv_df')
                step.set_execmethod('cbv_df[cbv_df.columns[2:]]', var='cbv_df')
                if list_of_roi:
                    step.set_execmethod('list_roi', var='cbv_df.columns')
            else:
                methods.raiseerror(messages.Errors.InputValueError, 'Please check input CBV parameters')
        step.set_execmethod('if len(df.columns):')
        # again, if CBV parameter are given, correct the CBV changes.
        if cbv:
            step.set_execmethod('\tdR2_mion = (-1 / te) * np.log(df.loc[:n_tr, :].mean(axis=0) / '
                                'cbv_df.loc[:n_tr, :].mean(axis=0))')
            step.set_execmethod('\tdR2_stim = (-1 / te) * np.log(df / df.loc[:n_tr, :].mean(axis=0))')
            step.set_execmethod('\tdf = dR2_stim/dR2_mion')
        # Generating excel files
        step.set_execmethod('\tfname = os.path.splitext(str(func[i].Filename))[0]')
        step.set_execmethod('\tdf.to_excel(os.path.join(sub_path, methods.splitnifti(fname)+".xlsx"), '
                            'index=False)')
        step.set_execmethod('\tpass')
        step.set_execmethod('else:')
        step.set_execmethod('\tpass')

        # Run the steps
        output_path = step.run('ExtractROIs', surfix=surfix)#, debug=True)
        if tmp:
            tmp.close()
        return dict(timecourse=output_path)

    def afni_TemporalClipping(self, func, clip_range, surfix='func', **kwargs):
        """

        :param func:
        :param clip_range:
        :param surfix:
        :param kwargs:
        :return:
        """
        display(title(value='** Temporal clipping of functional image'))
        filters, subj, sess = self.check_filters(**kwargs)
        step = Step(self, subjects=subj, sessions=sess)
        func = self.check_input(func)
        step.set_input(name='func', input_path=func, filters=filters)
        if clip_range:
            if isinstance(clip_range, list):
                if len(clip_range) == 2:
                    irange = "'[" + "{}..{}".format(*clip_range) + "]'"
                    step.set_staticinput(name='irange', value=irange)
        cmd = '3dcalc -prefix {output} -expr "a" -a {func}"{irange}"'
        step.set_command(cmd)
        output_path = step.run('TemporalClipping', surfix, debug=False)
        return dict(clippedfunc=output_path)