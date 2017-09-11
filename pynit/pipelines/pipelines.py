class PipeTemplate(object):
    """ Pipeline template class
    """
    @property
    def avail(self):
        pipes = [pipe[5:] for pipe in dir(self) if 'pipe_' in pipe]
        output = dict(zip(range(len(pipes)), pipes))
        return output


# class Q_Quality_assesments(PipeTemplate):
#     def __init__(self):
#         pass


class A_fMRI_preprocess(PipeTemplate):
    def __init__(self, proc, tmpobj, anat='anat', func='func', tr=None, tpattern=None, aniso=False,
                 cbv=False, ui=False, surfix='func'):
        """Collection of preprocessing pipelines for Shihlab at UNC
        Author  : SungHo Lee(shlee@unc.edu)
        Revised : Sep.9th.2017

        Parameters:
            anat    : str
                Path of anatomical images (default: 'anat')
            func    : str
                Path of functional images (default: 'func')
            tr      : int
                Temporal sampling time for volume (default: None)
            tpattern: str
                Slice order code based on afni command '3dTshift' (default: None)
            aniso   : bool
                Set True if you use 2D anitotropic sliced image (default: False)
            cbv     : str
                Path of MION infusion image (default: False)
            ui      : bool
                UI supports (default: False)
            surfix  : str
                Surfix for output folder (default: 'func')
        """
        # Define attributes
        self.proc = proc
        self.func = func
        self.anat = anat
        self.tmpobj = tmpobj
        self.tr = tr
        self.tpattern = tpattern
        self.aniso = aniso
        self.cbv = cbv
        self.ui = ui
        self.surfix = surfix

    def pipe_01_Brain_Mask_Preparation(self):
        """ Mask preparation step
        """
        # Mean image calculation (0)
        if self.cbv:
            self.proc.afni_MeanImgCalc(self.cbv, cbv=True, surfix=self.surfix)
        else:
            self.proc.afni_MeanImgCalc(self.func, surfix=self.surfix)
        # Mask preparation (1-anat, 2-func)
        if self.ui:
            self.proc.afni_MaskPrep(self.anat, 0, self.tmpobj, ui=True)
        else:
            self.proc.afni_MaskPrep(self.anat, 0, self.tmpobj)

    def pipe_02_Standard_Preprocessing(self):
        """
        """
        # Update mask files (1-anat, 2-func)
        if not self.ui:
            self.proc.afni_PasteMask(0, 1)
            self.proc.afni_PasteMask(1, 2)
        # Skull stripping (3-anat, 4-func)
        self.proc.afni_SkullStrip(self.anat, 0)
        # Coregistration (5)
        self.proc.afni_Coreg(3, 4, aniso=self.aniso, surfix=self.surfix)
        # Slice timing correction (6)
        if self.tr or self.tpattern:
            self.proc.afni_SliceTimingCorrection(self.func, tr=self.tr, tpattern=self.tpattern, surfix=self.surfix)
        else:
            self.proc.afni_SliceTimingCorrection(self.func, surfix=self.surfix)
        # Motion correction (7)
        self.proc.afni_MotionCorrection(6, 0, surfix=self.surfix)
        # Skull stripping all functional data (8)
        self.proc.afni_SkullStripAll(7, 2, surfix=self.surfix)
        # Apply coregistration transform matrix to all functional data (9)
        self.proc.afni_ApplyCoregAll(8, 5, surfix=self.surfix)
        if not self.aniso:
            # Align anatomy image to template space (10)
            self.proc.ants_SpatialNorm(3, self.tmpobj, surfix=self.surfix)
            # Align functional images to template space (11)
            self.proc.ants_ApplySpatialNorm(9, 10, surfix=self.surfix)
        else:
            self.proc.afni_SpatialNorm(3, self.tmpobj, surfix='anat')
            self.proc.afni_ApplySpatialNorm(9, 10, surfix=self.surfix)
        if self.cbv:
            # Coregistration MION infusion image to anatomy image (12)
            self.proc.afni_ApplyCoregAll(self.cbv, 5, surfix='cbv')
            # Align MION infusion image to template space (13)
            self.proc.afni_ApplySpatialNorm(12, 10, surfix='cbv')

    # def pipe_03_Advanced_Preprocessing(self):
    #     # Update mask files (1-anat, 2-func)
    #     if not self.ui:
    #         self.proc.afni_PasteMask(0, 1)
    #         self.proc.afni_PasteMask(1, 2)
    #     # Skull stripping (3-anat, 4-func)
    #     self.proc.afni_SkullStrip(self.anat, 0)
    #     # Coregistration (5)
    #     self.proc.afni_Coreg(3, 4, aniso=self.aniso, inverse=True, surfix='anat')
    #     # Slice timing correction (6)
    #     if self.tr or self.tpattern:
    #         self.proc.afni_SliceTimingCorrection(self.func, tr=self.tr, tpattern=self.tpattern, surfix=self.surfix)
    #     else:
    #         self.proc.afni_SliceTimingCorrection(self.func, surfix=self.surfix)
    #     # Motion correction (7)
    #     self.proc.afni_MotionCorrection(6, 0, surfix=self.surfix)
    #     # Skull stripping all functional data (8)
    #     self.proc.afni_SkullStripAll(7, 2, surfix=self.surfix)


class B_evoked_fMRI_analysis(PipeTemplate):
    def __init__(self, proc, tmpobj, paradigm=None, fwhm=0.5, thresholds=None, mask=None, cbv_param=None, crop=None,
                 option=None, ui=False, surfix='func'):
        """Collection of GLM analysis pipelines for Shihlab at UNC
        Author  : SungHo Lee(shlee@unc.edu)
        Revised : Sep.9st.2017

        Parameters:
            paradigm    : list
                Mandatary input for evoked paradigm
            fwhm        : float
                Voxel Smoothness
            thresholds : list (default: None)
                Threshold for generating clusters [pval, num_of_voxels]
            mask        : path (default: None)
                ROIs mask for extracting timecourses
                if not provided, then generating cluster map using evoked responses
            cbv_param   : [echotime, number_of_volume_to_calc_average], list (default: None)
                parameters to calculate CBV, if this parameters are given, CBV correction will be calculated
            crop        : list [start, end]
                range that you want to crop the time-course data
            option      : str
                option for ROIs extraction ('bilateral', 'merge', or 'contra')
            ui          : bool
                UI supports (default: False)
            surfix      : str
        """
        # Define attributes
        self.tmpobj = tmpobj
        self.proc = proc
        self.thr = thresholds
        self.paradigm = paradigm
        self.cbv = cbv_param
        self.crop = crop
        self.fwhm = str(fwhm)
        self.option = option
        self.mask = mask
        self.ui = ui
        self.surfix = surfix

    def pipe_01_GLM_analysis(self):
        # Spatial smoothing (1)
        self.proc.afni_SpatialSmoothing(0, fwhm=self.fwhm, tmpobj=self.tmpobj, surfix=self.surfix)
        # Perform GLM analysis (2)
        self.proc.afni_GLManalysis(1, self.paradigm, clip_range=self.crop, surfix=self.surfix)
        if not self.mask:
            # Extract clusters using evoked results
            try:
                step = [step for step in self.proc.steps if self.surfix in step and 'REMLfit' in step][0]
                if self.thr:
                    self.proc.afni_ClusterMap(step, 1, self.tmpobj, ui=self.ui, pval=self.thr[0],
                                              clst_size=self.thr[1], surfix=self.surfix)
                else:
                    self.proc.afni_ClusterMap(step, 1, self.tmpobj, ui=self.ui, surfix=self.surfix)
            except:
                pass

    def pipe_02_Extract_Timecourse(self):
        if self.cbv:
            cbv_id = 0
        else:
            cbv_id = False
        if self.crop:
            total = [step for step in self.proc.steps if 'fullts' in step and 'ExtractROIs' in step]
            if len(total):
                pass
            else:
                if self.mask:
                    self.proc.afni_ROIStats(1, self.mask, cbv=cbv_id, cbv_param=self.cbv, surfix='fullts')
                else:
                    step = [step for step in self.proc.steps if self.surfix in step and 'ClusteredMask' in step][0]
                    self.proc.afni_ROIStats(1, step, cbv=cbv_id, cbv_param=self.cbv, surfix='fullts')
        if self.mask:
            # If mask given, extract timecourse using the given mask
            self.proc.afni_ROIStats(1, self.mask, cbv=cbv_id, cbv_param=self.cbv, clip_range=self.crop, surfix=self.surfix)
        # Extract timecourse using the mask you generated at step1
        else:
            step = [step for step in self.proc.steps if self.surfix in step and 'ClusteredMask' in step][0]
            self.proc.afni_ROIStats(1, step, clip_range=self.crop, option=self.option,
                                    cbv_param=self.cbv, surfix=self.surfix, cbv=cbv_id)

# class C_resting_state_fMRI_analysis(PipeTemplate):
#     def __init__(self, proc, tmpobj, surfix='func'):
#         """Collection of GLM analysis pipelines for Shihlab at UNC
#         Author  : SungHo Lee(shlee@unc.edu)
#         Revised : Sep.6nd.2017
#
#         Parameters:
#         """
#         # Define attributes
#         self.tmpobj = tmpobj
#         self.proc = proc
#         self.surfix = surfix