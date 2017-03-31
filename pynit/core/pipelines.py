class PipeTemplate(object):
    """ Pipeline template class
    """
    @property
    def avail(self):
        pipes = [pipe[5:] for pipe in dir(self) if 'pipe_' in pipe]
        output = dict(zip(range(len(pipes)), pipes))
        # for i, values in output.items():
        #     output[i] = values[3:]
        return output


class A_fMRI_preprocess(PipeTemplate):
    def __init__(self, proc, tmpobj, anat='anat', func='func', tr=None, tpattern=None,
                 bpass=None, fwhm=None, cbv=False, surfix='func'):
        """Collection of preprocessing pipelines for Shihlab at UNC
        Author  : SungHo Lee(shlee@unc.edu)
        Revised : Feb.27th.2017

        Parameters:
            tmpobj  : TemplateObject
            anat    : str
                Path of anatomical images (default: 'anat')
            func    : str
                Path of functional images (default: 'func')
            tr      : int
                Temporal sampling time for volume (default: None)
            tpattern: str
                Slice order code based on afni command '3dTshift' (default: None)
            bpass   : list [low, high]
                Filter range (default: None)
            fwhm    : float
                Desired spatial smoothness
            cbv     : str
                Path of MION infusion image (default: False)
            surfix  : str

        """
        # Define attributes
        self.tmpobj = tmpobj
        self.proc = proc
        self.func = func
        self.anat = anat
        self.tr = tr
        self.tpattern = tpattern
        self.bpass = bpass
        self.fwhm = fwhm
        self.cbv = cbv
        self.surfix = surfix
        # Reset subjects
        self.proc.reload()

    def pipe_01_Brain_Mask_Preparation(self):
        # Mean image calculation
        if self.cbv:
            self.proc.afni_MeanImgCalc(self.cbv, cbv=True, surfix=self.surfix)
        else:
            self.proc.afni_MeanImgCalc(self.func, surfix=self.surfix)
        # Mask preparation
        self.proc.afni_MaskPrep(self.anat, self.tmpobj)

    def pipe_02_Preprocessing_2d_evoked_data(self):
        # Skull stripping
        self.proc.afni_SkullStrip(self.anat, self.proc.steps[0])
        # Coregistration
        self.proc.afni_Coreg(self.proc.steps[3], self.proc.steps[4], surfix=self.surfix)
        # Slice timing correction
        if self.tr or self.tpattern:
            self.proc.afni_SliceTimingCorrection(self.func, tr=self.tr, tpattern=self.tpattern, surfix=self.surfix)
        else:
            self.proc.afni_SliceTimingCorrection(self.func, surfix=self.surfix)
        # Motion correction
        self.proc.afni_MotionCorrection(self.proc.steps[6], surfix=self.surfix)
        # Skull stripping all functional data
        self.proc.afni_SkullStripAll(self.proc.steps[7], self.proc.steps[2], surfix=self.surfix)
        # Apply coregistration transform matrix to all functional data
        self.proc.afni_ApplyCoregAll(self.proc.steps[8], self.proc.steps[5], surfix=self.surfix)
        # Spatial normalization
        self.proc.afni_SpatialNorm(self.proc.steps[3], self.tmpobj)
        self.proc.afni_ApplySpatialNorm(self.proc.steps[9], self.proc.steps[10], surfix=self.surfix)
        if self.cbv:
            self.proc.afni_ApplyCoregAll(self.cbv, self.proc.steps[5], surfix='cbv')
            self.proc.afni_ApplySpatialNorm(self.proc.steps[12], self.proc.steps[10], surfix='cbv')
        if self.fwhm:
            self.proc.afni_SpatialSmoothing(self.proc.steps[11], fwhm=self.fwhm, tmpobj=self.tmpobj, surfix=self.surfix)

    def pipe_03_Preprocessing_2d_restingstate_data(self):
        # Skull stripping
        self.proc.afni_SkullStrip(self.anat, self.proc.steps[0])
        # Coregistration
        self.proc.afni_Coreg(self.proc.steps[3], self.proc.steps[4], surfix=self.surfix)
        # Slice timing correction
        if self.tr or self.tpattern:
            self.proc.afni_SliceTimingCorrection(self.func, tr=self.tr, tpattern=self.tpattern, surfix=self.surfix)
        else:
            self.proc.afni_SliceTimingCorrection(self.func, surfix=self.surfix)
        # Motion correction
        self.proc.afni_MotionCorrection(self.proc.steps[6], surfix=self.surfix)
        # Skull stripping all functional data
        self.proc.afni_SkullStripAll(self.proc.steps[7], self.proc.steps[2], surfix=self.surfix)
        # Apply coregistration transform matrix to all functional data
        self.proc.afni_ApplyCoregAll(self.proc.steps[8], self.proc.steps[5], surfix=self.surfix)
        # Spatial normalization
        self.proc.afni_SpatialNorm(self.proc.steps[3], self.tmpobj)
        self.proc.afni_ApplySpatialNorm(self.proc.steps[9], self.proc.steps[10], surfix=self.surfix)
        if self.cbv:
            self.proc.afni_ApplyCoregAll(self.cbv, self.proc.steps[5], surfix='cbv')
            self.proc.afni_ApplySpatialNorm(self.proc.steps[12], self.proc.steps[10], surfix='cbv')
        self.proc.afni_SignalProcessing(self.proc.steps[11], ort=self.proc.step[7], fwhm=self.fwhm,  dt=self.tr,
                                        mask=self.tmpobj.mask.get_filename(), bpass = self.bpass, surfix=self.surfix)

class B_evoked_fMRI_analysis(PipeTemplate):
    def __init__(self, proc, tmpobj, paradigm, thresholds=None, mask=None, cbv=None, crop=None, surfix='func'):
        """Collection of GLM analysis pipelines for Shihlab at UNC
        Author  : SungHo Lee(shlee@unc.edu)
        Revised : Mar.2nd.2017

        Parameters:
            paradigm: list
                Mandatary input for evoked paradigm
            thresholds : list (default: None)
                Threshold for generating clusters [pval, num_of_voxels]
            mask    : path (default: None)
                ROIs mask for extracting timecourses
                if not provided, then generating cluster map using evoked responses
            cbv     : [echotime, number_of_volume_to_calc_average], list (default: None)
                parameters to calculate CBV, if this parameters are given, CBV correction will be calculated
            crop    : list [start, end]
                range that you want to crop the time-course data
            surfix  : str
                """
        # Define attributes
        self.tmpobj = tmpobj
        self.proc = proc
        self.thr = thresholds
        self.paradigm = paradigm
        self.cbv = cbv
        self.crop = crop
        self.mask = mask
        self.surfix = surfix

    def pipe_01_GLM_analysis(self):
        # Perform GLM analysis
        self.proc.afni_GLManalysis(self.proc.steps[0], self.paradigm, crop=self.crop, surfix=self.surfix)
        if not self.mask:
            # Extract clusters using evoked results
            step = [step for step in self.proc.steps if self.surfix in step and 'REMLfit' in step][0]
            self.proc.afni_ClusterMap(step, self.proc.steps[0], self.tmpobj, surfix=self.surfix)

    def pipe_02_Extract_Timecourse(self):
        if self.mask:
            # If mask given, extract timecourse using the given mask
            self.proc.afni_ROIStats(self.proc.steps[0], self.mask, cbv=self.cbv, crop=self.crop, surfix=self.surfix)
        # Extract timecourse using the mask you generated at step1
        else:
            step = [step for step in self.proc.steps if self.surfix in step and 'ClusteredMask' in step][0]
            self.proc.afni_ROIStats(self.proc.steps[0], self.proc.steps[3], crop=self.crop,
                                    cbv=self.cbv, surfix=self.surfix)