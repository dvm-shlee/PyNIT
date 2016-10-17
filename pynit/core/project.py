import os
from .objects import Reference, ImageObj
from .process import Analysis, Interface, TempFile
from .utility import InternalMethods, pd, np
from .visual import Viewer
import error


class Preprocess(object):
    """ Preprocessing pipeline
    """
    def __init__(self, prjobj, pipeline):
        prjobj.reset()
        self._subjects = None
        self._sessions = None
        if prjobj.subjects:
            self._subjects = sorted(prjobj.subjects[:])
            if not prjobj.single_session:
                self._sessions = sorted(prjobj.sessions[:])
        self._prjobj = prjobj
        self._prjobj.initiate_pipeline(pipeline)
        self._pipeline = pipeline

    @property
    def subjects(self):
        return self._subjects

    @property
    def sessions(self):
        return self._sessions

    def cbv_meancalculation(self, func):
        """ CBV image preparation
        """
        dataclass, func = InternalMethods.check_input_dataclass(func)
        print("MotionCorrection")
        step01 = self.init_step('MotionCorrection-CBVinduction')
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            InternalMethods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                cbv_img = self._prjobj(dataclass, func, subj)
                for i, finfo in cbv_img.iterrows():
                    print(" +Filename: {}".format(finfo.Filename))
                    self._prjobj.run('afni_3dvolreg', os.path.join(step01, subj, finfo.Filename), finfo.Abspath)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    InternalMethods.mkdir(os.path.join(step01, subj, sess))
                    cbv_img = self._prjobj(dataclass, func, subj, sess)
                    for i, finfo in cbv_img.iterrows():
                        print("  +Filename: {}".format(finfo.Filename))
                        self._prjobj.run('afni_3dvolreg', os.path.join(step01, subj, sess, finfo.Filename),
                                         finfo.Abspath)
        step02 = self.init_step('MeanImageCalculation-BOLD')
        step03 = self.init_step('MeanImageCalculation-CBV')
        print("MeanImageCalculation-BOLD&CBV")
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            InternalMethods.mkdir(os.path.join(step02, subj), os.path.join(step03, subj))
            if self._prjobj.single_session:
                cbv_img = self._prjobj(1, self._pipeline, os.path.basename(step01), subj)
                for i, finfo in cbv_img.iterrows():
                    print(" +Filename: {}".format(finfo.Filename))
                    shape = ImageObj.load(finfo.Abspath).shape
                    self._prjobj.run('afni_3dTstat', os.path.join(step02, subj, finfo.Filename),
                                     "{path}'[{start}..{end}]'".format(path=finfo.Abspath,
                                                                       start=0,
                                                                       end=(int(shape[-1] / 3))))
                    self._prjobj.run('afni_3dTstat', os.path.join(step03, subj, finfo.Filename),
                                     "{path}'[{start}..{end}]'".format(path=finfo.Abspath,
                                                                       start=int(shape[-1] * 2 / 3),
                                                                       end=shape[-1] - 1))
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    InternalMethods.mkdir(os.path.join(step02, subj, sess), os.path.join(step03, subj, sess))
                    cbv_img = self._prjobj(0, os.path.basename(step01), subj, sess)
                    for i, finfo in cbv_img.iterrows():
                        print("  +Filename: {}".format(finfo.Filename))
                        shape = InternalMethods.load(finfo.Abspath).shape
                        self._prjobj.run('afni_3dTstat', os.path.join(step02, subj, sess, finfo.Filename),
                                         "{path}'[{start}..{end}]'".format(path=finfo.Abspath,
                                                                           start=0,
                                                                           end=(int(shape[-1] / 3))))
                        self._prjobj.run('afni_3dTstat', os.path.join(step03, subj, sess, finfo.Filename),
                                         "{path}'[{start}..{end}]'".format(path=finfo.Abspath,
                                                                           start=int(shape[-1] * 2 / 3),
                                                                           end=shape[-1] - 1))
        return {'CBVinduction': step01, 'meanBOLD': step02, 'meanCBV': step03}

    def mean_calculation(self, func, dtype='func'):
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
        dataclass, func = InternalMethods.check_input_dataclass(func)
        # if os.path.exists(func):
        #     dataclass = 1
        #     func = InternalMethods.path_splitter(func)[-1]
        # else:
        #     dataclass = 0
        step01 = self.init_step('InitialPreparation-{}'.format(dtype))
        print("MotionCorrection")
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            InternalMethods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                firstfunc = self._prjobj(dataclass, func, subj)
                for i, finfo in firstfunc.iterrows():
                    if not i:
                        print(" +Filename: {}".format(finfo.Filename))
                        self._prjobj.run('afni_3dvolreg', os.path.join(step01, subj, finfo.Filename), finfo.Abspath)
                    else:
                        pass
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    InternalMethods.mkdir(os.path.join(step01, subj, sess))
                    firstfunc = self._prjobj(dataclass, func, subj, sess)
                    for i, finfo in firstfunc.iterrows():
                        if not i:
                            print("  +Filename: {}".format(finfo.Filename))
                            self._prjobj.run('afni_3dvolreg', os.path.join(step01, subj, sess, finfo.Filename),
                                             finfo.Abspath)
                        else:
                            pass
        step02 = self.init_step('MeanImageCalculation-{}'.format(dtype))
        print("MeanImageCalculation-{}".format(func))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            InternalMethods.mkdir(os.path.join(step02, subj))
            if self._prjobj.single_session:
                funcs = self._prjobj(1, self._pipeline, os.path.basename(step01), subj).loc[0]
                print(" +Filename: {}".format(funcs.Filename))
                self._prjobj.run('afni_3dTstat', os.path.join(step02, subj, funcs.Filename),
                                 funcs.Abspath)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    InternalMethods.mkdir(os.path.join(step02, subj, sess))
                    funcs = self._prjobj(1, self._pipeline, os.path.basename(step01), subj, sess).loc[0]
                    print(" +Filename: {}".format(funcs.Filename))
                    self._prjobj.run('afni_3dTstat', os.path.join(step02, subj, sess, funcs.Filename),
                                     funcs.Abspath)
        return {'firstfunc': step01, 'meanfunc': step02}

    def slicetiming_correction(self, func, tr=1, tpattern='altplus', dtype='func'):
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
        dataclass, func = InternalMethods.check_input_dataclass(func)
        # if os.path.exists(func):
        #     dataclass = 1
        #     func = InternalMethods.path_splitter(func)[-1]
        # else:
        #     dataclass = 0
        print('SliceTimingCorrection-{}'.format(func))
        step01 = self.init_step('SliceTimingCorrection-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            InternalMethods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                epi = self._prjobj(dataclass, func, subj)
                for i, finfo in epi.iterrows():
                    print(" +Filename: {}".format(finfo.Filename))
                    self._prjobj.run('afni_3dTshift', os.path.join(step01, subj, finfo.Filename),
                                     finfo.Abspath, tr=tr, tpattern=tpattern)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    InternalMethods.mkdir(os.path.join(step01, subj, sess))
                    epi = self._prjobj(dataclass, func, subj, sess)
                    for i, finfo in epi.iterrows():
                        print("  +Filename: {}".format(finfo.Filename))
                        self._prjobj.run('afni_3dTshift', os.path.join(step01, subj, sess, finfo.Filename),
                                         finfo.Abspath, tr=tr, tpattern=tpattern)
        return {'func': step01}

    def motion_correction(self, func, meanfunc=None, dtype='func'):
        """ Corrects for motion artifacts in the  input functional image

        Parameters
        ----------
        func       : str
            Datatype or absolute step path for the input functional image
        meanfunc   : str
            Datatype or absolute step path for the mean functional image
        dtype      : str
            Surfix for the step path


        Returns
        -------
        step_paths : dict
        """
        dataclass, func = InternalMethods.check_input_dataclass(func)
        # if os.path.exists(func):
        #     dataclass = 1
        #     func = InternalMethods.path_splitter(func)[-1]
        # else:
        #     dataclass = 0
        print('MotionCorrection-{}'.format(func))
        step01 = self.init_step('MotionCorrection-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            InternalMethods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                epi = self._prjobj(dataclass, func, subj)
                if meanfunc:
                    meanimg = self._prjobj(1, self._pipeline, os.path.basename(meanfunc), subj)
                    meanimg = meanimg.Abspath[0]
                else:
                    meanimg = 0
                for i, finfo in epi.iterrows():
                    print(" +Filename: {}".format(finfo.Filename))
                    self._prjobj.run('afni_3dvolreg', os.path.join(step01, subj, finfo.Filename), finfo.Abspath,
                                     base_slice=meanimg)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    InternalMethods.mkdir(os.path.join(step01, subj, sess))
                    epi = self._prjobj(dataclass, func, subj, sess)
                    meanimg = self._prjobj(1, self._pipeline, os.path.basename(meanfunc), subj, sess)
                    for i, finfo in epi.iterrows():
                        print("  +Filename: {}".format(finfo.Filename))
                        self._prjobj.run('afni_3dvolreg', os.path.join(step01, subj, sess, finfo.Filename),
                                         finfo.Abspath, base_slice=meanimg.Abspath[0])
        return {'func': step01}

    def maskdrawing_preparation(self, meanfunc, anat, padding=True, zaxis=2):
        f_dataclass, meanfunc = InternalMethods.check_input_dataclass(meanfunc)
        a_dataclass, anat = InternalMethods.check_input_dataclass(anat)
        # if os.path.exists(meanfunc):
        #     f_dataclass = 1
        #     meanfunc = InternalMethods.path_splitter(meanfunc)[-1]
        # else:
        #     f_dataclass = 0
        # if os.path.exists(anat):
        #     a_dataclass = 1
        #     anat = InternalMethods.path_splitter(anat)[-1]
        # else:
        #     a_dataclass = 0
        print('MaskDrawing-{} & {}'.format(meanfunc, anat))

        step01 = self.init_step('MaskDrwaing-func')
        step02 = self.init_step('MaskDrawing-anat')
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            InternalMethods.mkdir(os.path.join(step01, subj), os.path.join(step02, subj))
            if self._prjobj.single_session:
                epi = self._prjobj(f_dataclass, meanfunc, subj)
                t2 = self._prjobj(a_dataclass, anat, subj)
                for i, finfo in epi.iterrows():
                    print(" +Filename: {}".format(finfo.Filename))
                    epiimg = InternalMethods.load(finfo.Abspath)
                    if padding:
                        epiimg.padding(low=1, high=1, axis=zaxis)
                    epiimg.save_as(os.path.join(step01, subj, finfo.Filename), quiet=True)
                for i, finfo in t2.iterrows():
                    print(" +Filename: {}".format(finfo.Filename))
                    t2img = InternalMethods.load(finfo.Abspath)
                    if padding:
                        t2img.padding(low=1, high=1, axis=zaxis)
                    t2img.save_as(os.path.join(step02, subj, finfo.Filename), quiet=True)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    InternalMethods.mkdir(os.path.join(step01, subj, sess), os.path.join(step02, subj, sess))
                    epi = self._prjobj(f_dataclass, meanfunc, subj, sess)
                    t2 = self._prjobj(a_dataclass, anat, subj, sess)
                    for i, finfo in epi.iterrows():
                        print("  +Filename: {}".format(finfo.Filename))
                        epiimg = InternalMethods.load(finfo.Abspath)
                        if padding:
                            epiimg.padding(low=1, high=1, axis=zaxis)
                        epiimg.save_as(os.path.join(step01, subj, sess, finfo.Filename))
                    for i, finfo in t2.iterrows():
                        print("  +Filename: {}".format(finfo.Filename))
                        t2img = InternalMethods.load(finfo.Abspath)
                        if padding:
                            t2img.padding(low=1, high=1, axis=zaxis)
                        t2img.save_as(os.path.join(step02, subj, sess, finfo.Filename))
        return {'meanfunc': step01, 'anat': step02}

    def compute_skullstripping(self, meanfunc, anat, padded=True, zaxis=2):
        axis = {0:'x', 1:'y', 2:'z'}
        f_dataclass, meanfunc = InternalMethods.check_input_dataclass(meanfunc)
        a_dataclass, anat = InternalMethods.check_input_dataclass(anat)
        # meanfunc = InternalMethods.path_splitter(meanfunc)[-1]
        # anat = InternalMethods.path_splitter(anat)[-1]
        print('SkullStripping-{} & {}'.format(meanfunc, anat))
        step01 = self.init_step('SkullStripped-meanfunc')
        step02 = self.init_step('SkullStripped-anat')
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            InternalMethods.mkdir(os.path.join(step01, subj), os.path.join(step02, subj))
            if self._prjobj.single_session:
                # Load image paths
                epi = self._prjobj(1, self._pipeline, meanfunc, subj, ignore='_mask')
                t2 = self._prjobj(1, self._pipeline, anat, subj, ignore='_mask')
                # Load mask image obj
                epimask = self._prjobj(1, self._pipeline, meanfunc, subj, file_tag='_mask').Abspath[0]
                t2mask = self._prjobj(1, self._pipeline, anat, subj, file_tag='_mask').Abspath[0]
                # Execute process
                for i, finfo in epi.iterrows():
                    print(" +Filename of meanfunc: {}".format(finfo.Filename))
                    filename = finfo.Filename
                    fpath = os.path.join(step01, subj, '_{}'.format(filename))
                    self._prjobj.run('afni_3dcalc', fpath, 'a*step(b)',
                                     finfo.Abspath, epimask)
                    ss_epi = InternalMethods.load(fpath)
                    if padded:
                        exec('ss_epi.crop({}=[1, {}])'.format(axis[zaxis], ss_epi.shape[zaxis]-1))
                    ss_epi.save_as(os.path.join(step01, subj, filename), quiet=True)
                    os.remove(fpath)
                for i, finfo in t2.iterrows():
                    print(" +Filename of anat: {}".format(finfo.Filename))
                    filename = finfo.Filename
                    fpath = os.path.join(step02, subj, '_{}'.format(filename))
                    self._prjobj.run('afni_3dcalc', fpath, 'a*step(b)',
                                     finfo.Abspath, t2mask)
                    ss_t2 = InternalMethods.load(fpath)
                    if padded:
                        exec('ss_t2.crop({}=[1, {}])'.format(axis[zaxis], ss_t2.shape[zaxis] - 1))
                    ss_t2.save_as(os.path.join(step02, subj, filename), quiet=True)
                    os.remove(fpath)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    InternalMethods.mkdir(os.path.join(step01, subj, sess), os.path.join(step02, subj, sess))
                    # Load image paths
                    epi = self._prjobj(1, self._pipeline, meanfunc, subj, sess, ignore='_mask')
                    t2 = self._prjobj(1, self._pipeline, anat, subj, sess, ignore='_mask')
                    # Load mask image obj
                    epimask = self._prjobj(1, self._pipeline, meanfunc, subj, sess, file_tag='_mask').Abspath[0]
                    t2mask = self._prjobj(1, self._pipeline, anat, subj, sess, file_tag='_mask').Abspath[0]
                    # Execute process
                    for i, finfo in epi.iterrows():
                        print("  +Filename of meanfunc: {}".format(finfo.Filename))
                        filename = finfo.Filename
                        fpath = os.path.join(step01, subj, sess, '_{}'.format(filename))
                        self._prjobj.run('afni_3dcalc', fpath, 'a*step(b)',
                                         finfo.Abspath, epimask)
                        ss_epi = InternalMethods.load(fpath)
                        if padded:
                            exec('ss_epi.crop({}=[1, {}])'.format(axis[zaxis], ss_epi.shape[zaxis] - 1))
                        ss_epi.save_as(os.path.join(step01, subj, sess, filename), quiet=True)
                        os.remove(fpath)
                    for i, finfo in t2.iterrows():
                        print("  +Filename of anat: {}".format(finfo.Filename))
                        filename = finfo.Filename
                        fpath = os.path.join(step02, subj, sess, '_{}'.format(filename))
                        self._prjobj.run('afni_3dcalc', fpath, 'a*step(b)',
                                         finfo.Abspath, t2mask)
                        ss_t2 = InternalMethods.load(fpath)
                        if padded:
                            exec('ss_t2.crop({}=[1, {}])'.format(axis[zaxis], ss_t2.shape[zaxis] - 1))
                        ss_t2.save_as(os.path.join(step02, subj, sess, filename), quiet=True)
                        os.remove(fpath)
        return {'meanfunc': step01, 'anat': step02}

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
        f_dataclass, meanfunc = InternalMethods.check_input_dataclass(meanfunc)
        a_dataclass, anat = InternalMethods.check_input_dataclass(anat)
        print('BiasFieldCorrection-{} & {}'.format(meanfunc, anat))
        step01 = self.init_step('BiasFieldCorrection-{}'.format(dtype))
        step02 = self.init_step('BiasFieldCorrection-{}'.format(anat.split('-')[-1]))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            InternalMethods.mkdir(os.path.join(step01, subj), os.path.join(step02, subj))
            if self._prjobj.single_session:
                epi = self._prjobj(f_dataclass, meanfunc, subj)
                t2 = self._prjobj(a_dataclass, anat, subj)
                for i, finfo in epi.iterrows():
                    print(" +Filename of func: {}".format(finfo.Filename))
                    self._prjobj.run('ants_BiasFieldCorrection', os.path.join(step01, subj, finfo.Filename),
                                     finfo.Abspath, algorithm='n4')
                for i, finfo in t2.iterrows():
                    print(" +Filename of anat: {}".format(finfo.Filename))
                    self._prjobj.run('ants_BiasFieldCorrection', os.path.join(step02, subj, finfo.Filename),
                                     finfo.Abspath, algorithm='n4')
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    InternalMethods.mkdir(os.path.join(step01, subj, sess), os.path.join(step02, subj, sess))
                    epi = self._prjobj(f_dataclass, meanfunc, subj, sess)
                    t2 = self._prjobj(f_dataclass, anat, subj, sess)
                    for i, finfo in epi.iterrows():
                        print("  +Filename of func: {}".format(finfo.Filename))
                        self._prjobj.run('ants_BiasFieldCorrection', os.path.join(step01, subj, sess, finfo.Filename),
                                         finfo.Abspath, algorithm='n4')
                    for i, finfo in t2.iterrows():
                        print("  +Filename of anat: {}".format(finfo.Filename))
                        self._prjobj.run('ants_BiasFieldCorrection', os.path.join(step02, subj, sess, finfo.Filename),
                                         finfo.Abspath, algorithm='n4')
        print('Coregistration-{} to {}'.format(meanfunc, anat))
        step03 = self.init_step('Coregistration-{}2{}'.format(dtype, anat.split('-')[-1]))
        num_step = os.path.basename(step03).split('_')[0]
        step04 = self.final_step('{}_CheckRegistraton-{}'.format(num_step, dtype))
        for subj in self.subjects:
            InternalMethods.mkdir(os.path.join(step04, 'AllSubjects'))
            print("-Subject: {}".format(subj))
            InternalMethods.mkdir(os.path.join(step03, subj))
            if self._prjobj.single_session:
                epi = self._prjobj(1, self._pipeline, os.path.basename(step01), subj)
                t2 = self._prjobj(1, self._pipeline, os.path.basename(step02), subj)
                for i, finfo in epi.iterrows():
                    print(" +Filename: {}".format(finfo.Filename))
                    fixed_img = t2.Abspath[0]
                    moved_img = os.path.join(step03, subj, finfo.Filename)
                    self._prjobj.run('afni_3dAllineate', moved_img, finfo.Abspath, onepass=True, EPI=True,
                                     base=fixed_img, cmass='+xy', matrix_save=os.path.join(step03, subj, subj))
                    fig1 = Viewer.check_reg(InternalMethods.load(fixed_img),
                                            InternalMethods.load(moved_img), sigma=2, **kwargs)
                    fig1.suptitle('EPI to T2 for {}'.format(subj), fontsize=12, color='yellow')
                    fig1.savefig(os.path.join(step04, 'AllSubjects', '{}.png'.format('-'.join([subj, 'func2anat']))),
                                 facecolor=fig1.get_facecolor())
                    fig2 = Viewer.check_reg(InternalMethods.load(moved_img),
                                            InternalMethods.load(fixed_img), sigma=2, **kwargs)
                    fig2.suptitle('T2 to EPI for {}'.format(subj), fontsize=12, color='yellow')
                    fig2.savefig(os.path.join(step04, 'AllSubjects', '{}.png'.format('-'.join([subj, 'anat2func']))),
                                 facecolor=fig2.get_facecolor())
            else:
                InternalMethods.mkdir(os.path.join(step04, subj), os.path.join(step04, subj, 'AllSessions'))
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    InternalMethods.mkdir(os.path.join(step03, subj, sess))
                    epi = self._prjobj(1, self._pipeline, os.path.basename(step01), subj, sess)
                    t2 = self._prjobj(1, self._pipeline, os.path.basename(step02), subj, sess)
                    for i, finfo in epi.iterrows():
                        print("  +Filename of anat: {}".format(finfo.Filename))
                        fixed_img = t2.Abspath[0]
                        moved_img = os.path.join(step03, subj, sess, finfo.Filename)
                        self._prjobj.run('afni_3dAllineate', moved_img, finfo.Abspath, onepass=True, EPI=True,
                                         base=fixed_img, cmass='+xy',
                                         matrix_save=os.path.join(step03, subj, sess, sess))
                        fig1 = Viewer.check_reg(InternalMethods.load(fixed_img),
                                                InternalMethods.load(moved_img), sigma=2, **kwargs)
                        fig1.suptitle('EPI to T2 for {}'.format(subj), fontsize=12, color='yellow')
                        fig1.savefig(os.path.join(step04, subj, 'AllSessions',
                                                  '{}.png'.format('-'.join([sess, 'func2anat']))),
                                     facecolor=fig1.get_facecolor())
                        fig2 = Viewer.check_reg(InternalMethods.load(moved_img),
                                                InternalMethods.load(fixed_img), sigma=2, **kwargs)
                        fig2.suptitle('T2 to EPI for {}'.format(subj), fontsize=12, color='yellow')
                        fig2.savefig(os.path.join(step04, subj, 'AllSessions',
                                                  '{}.png'.format('-'.join([sess, 'anat2func']))),
                                     facecolor=fig2.get_facecolor())
        return {'meanfunc': step01, 'anat':step02, 'realigned_func': step03, 'checkreg': step04}

    def apply_brainmask(self, func, mask, padded=True, zaxis=2, dtype='func'):
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
        dataclass, func = InternalMethods.check_input_dataclass(func)
        print('ApplyingBrainMask-{}'.format(func))
        step01 = self.init_step('ApplyingBrainMask-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            InternalMethods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                epi = self._prjobj(dataclass, func, subj)
                epimask = self._prjobj(1, self._pipeline, os.path.basename(mask), subj, file_tag='_mask')
                maskobj = InternalMethods.load(epimask.Abspath[0])
                if padded:
                    exec ('maskobj.crop({}=[1, {}])'.format(axis[zaxis], maskobj.shape[zaxis] - 1))
                temp_epimask = TempFile(maskobj, 'epimask_{}'.format(subj))
                for i, finfo in epi.iterrows():
                    print(" +Filename: {}".format(finfo.Filename))
                    self._prjobj.run('afni_3dcalc', os.path.join(step01, subj, finfo.Filename), 'a*step(b)',
                                     finfo.Abspath, str(temp_epimask))
                temp_epimask.close()
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    InternalMethods.mkdir(os.path.join(step01, subj, sess))
                    epi = self._prjobj(dataclass, func, subj, sess)
                    epimask = self._prjobj(1, self._pipeline, os.path.basename(mask), subj, sess, file_tag='_mask')
                    maskobj = InternalMethods.load(epimask.Abspath[0])
                    if padded:
                        exec ('maskobj.crop({}=[1, {}])'.format(axis[zaxis], maskobj.shape[zaxis] - 1))
                    temp_epimask = TempFile(maskobj, 'epimask_{}_{}'.format(subj, sess))
                    for i, finfo in epi.iterrows():
                        print("  +Filename: {}".format(finfo.Filename))
                        self._prjobj.run('afni_3dcalc', os.path.join(step01, subj, sess, finfo.Filename), 'a*step(b)',
                                         finfo.Abspath, temp_epimask)
                    temp_epimask.close()
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
        dataclass, func = InternalMethods.check_input_dataclass(func)
        print('ApplyingTransformation-{}'.format(func))
        step01 = self.init_step('ApplyingTransformation-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            InternalMethods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                ref = self._prjobj(1, self._pipeline, os.path.basename(realigned_func), subj)
                param = self._prjobj(1, self._pipeline, os.path.basename(realigned_func), subj, ext='.1D')
                funcs = self._prjobj(dataclass, os.path.basename(func), subj)
                for i, finfo in funcs.iterrows():
                    print(" +Filename: {}".format(finfo.Filename))
                    moved_img = os.path.join(step01, subj, finfo.Filename)
                    self._prjobj.run('afni_3dAllineate', moved_img, finfo.Abspath, master=ref.Abspath.loc[0],
                                     matrix_apply=param.Abspath.loc[0])
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    InternalMethods.mkdir(os.path.join(step01, subj, sess))
                    ref = self._prjobj(1, self._pipeline, os.path.basename(realigned_func), subj, sess)
                    param = self._prjobj(1, self._pipeline, os.path.basename(realigned_func), subj, sess, ext='.1D')
                    funcs = self._prjobj(dataclass, os.path.basename(func), subj, sess)
                    for i, finfo in funcs.iterrows():
                        print("  +Filename: {}".format(finfo.Filename))
                        moved_img = os.path.join(step01, subj, sess, finfo.Filename)
                        self._prjobj.run('afni_3dAllineate', moved_img, finfo.Abspath, master=ref.Abspath.loc[0],
                                         matrix_apply=param.Abspath.loc[0])
        return {'func': step01}

    def global_regression(self, func, dtype='func', detrend=-1):
        """ Method for global signal regression of individual functional image

        Parameters
        ----------
        func       : str
            Datatype or absolute step path for the input functional image
        dtype      : str
            Surfix for the step path
        detrend    : int

        Returns
        -------
        step_paths : dict
        """
        dataclass, func = InternalMethods.check_input_dataclass(func)
        print('GlobalRegression-{}'.format(func))
        step01 = self.init_step('GlobalRegression-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            InternalMethods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                funcs = self._prjobj(dataclass, func, subj)
                for i, finfo in funcs.iterrows():
                    print(" +Filename: {}".format(finfo.Filename))
                    regressor = os.path.join(step01, subj, "{}.1D".format(os.path.splitext(finfo.Filename)[0]))
                    self._prjobj.run('afni_3dmaskave', regressor, finfo.Abspath, finfo.Abspath)
                    self._prjobj.run('afni_3dDetrend', os.path.join(step01, subj, finfo.Filename), finfo.Abspath,
                                     vector=regressor, polort='-1')
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    InternalMethods.mkdir(os.path.join(step01, subj, sess))
                    funcs = self._prjobj(dataclass, func, subj)
                    for i, finfo in funcs.iterrows():
                        print("  +Filename: {}".format(finfo.Filename))
                        regressor = os.path.join(func, subj, sess,
                                                 "{}.1D".format(os.path.splitext(finfo.Filename)[0]))
                        self._prjobj.run('afni_3dmaskave', regressor, finfo.Abspath, finfo.Abspath)
                        self._prjobj.run('afni_3dDetrend', os.path.join(step01, subj, sess, finfo.Filename),
                                         finfo.Abspath, vector=regressor, polort=str(detrend))
        return {'func': step01}

    def motion_parameter_regression(self, func, motioncorrected_func, dtype='func', detrend=-1):
        """ Method for motion parameter regression of individual functional image

        Parameters
        ----------
        func                 : str
            Datatype or absolute path of the input functional image
        motioncorrected_func : str
            Absolute step path which contains the motion corrected functional image
        dtype                : str
            Surfix for the step path
        detrend              : int

        Returns
        -------
        step_paths          : dict
        """
        dataclass, func = InternalMethods.check_input_dataclass(func)
        print('GlobalRegression-{}'.format(func))
        step01 = self.init_step('GlobalRegression-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            InternalMethods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                funcs = self._prjobj(dataclass, func, subj)
                for i, finfo in funcs.iterrows():
                    print(" +Filename: {}".format(finfo.Filename))
                    regressor = self._prjobj(dataclass, motioncorrected_func, subj, ext='.1D', ignore='.aff12',
                                             file_tag=os.path.splitext(finfo.Filename)[0]).Abspath[0]
                    self._prjobj.run('afni_3dDetrend', os.path.join(step01, subj, finfo.Filename), finfo.Abspath,
                                     vector=regressor, polort=str(detrend))
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    InternalMethods.mkdir(os.path.join(step01, subj, sess))
                    funcs = self._prjobj(dataclass, func, subj, sess)
                    for i, finfo in funcs.iterrows():
                        print("  +Filename: {}".format(finfo.Filename))
                        regressor = self._prjobj(dataclass, motioncorrected_func, subj, ext='.1D', ignore='.aff12',
                                                 file_tag=os.path.splitext(finfo.Filename)[0]).Abspath[0]
                        self._prjobj.run('afni_3dDetrend', os.path.join(step01, subj, sess, finfo.Filename),
                                         finfo.Abspath, vector=regressor, polort=str(detrend))
        return {'func': step01}

    def signal_processing(self, func, dt=1, norm=False, despike=False, detrend=False,
                          blur=False, band=False, dtype='func'):
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
        dataclass, func = InternalMethods.check_input_dataclass(func)
        print('SignalProcessing-{}'.format(func))
        step01 = self.init_step('SignalProcessing-{}'.format(dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            InternalMethods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                funcs = self._prjobj(dataclass, func, subj)
                for i, finfo in funcs.iterrows():
                    print(" +Filename: {}".format(finfo.Filename))
                    self._prjobj.run('afni_3dBandpass', os.path.join(step01, subj, finfo.Filename), finfo.Abspath,
                                     norm=norm, despike=despike, detrend=detrend, blur=blur, band=band, dt=dt)
            else:
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    InternalMethods.mkdir(os.path.join(step01, subj, sess))
                    funcs = self._prjobj(dataclass, func, subj, sess)
                    for i, finfo in funcs.iterrows():
                        print("  +Filename: {}".format(finfo.Filename))
                        self._prjobj.run('afni_3dBandpass', os.path.join(step01, subj, finfo.Filename), finfo.Abspath,
                                         norm=norm, despike=despike, detrend=detrend, blur=blur, band=band, dt=dt)
        return {'func': step01}

    def warp_func(self, func, warped_anat, tempobj, dtype='func', **kwargs):
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
        dataclass, func = InternalMethods.check_input_dataclass(func)
        print("Warp-{} to Atlas and Check it's registration".format(func))
        step01 = self.init_step('Warp-{}2atlas'.format(dtype))
        num_step = os.path.basename(step01).split('_')[0]
        step02 = self.final_step('{}_CheckAtlasRegistration-{}'.format(num_step, dtype))
        # Loop the subjects
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            InternalMethods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                InternalMethods.mkdir(os.path.join(step02, 'AllSubjects'))
                # Grab the warping map and transform matrix
                mats, warps, warped = InternalMethods.get_warp_matrix(self, warped_anat, subj, inverse=True)
                temp_path = os.path.join(step01, subj, "base")
                tempobj.save_as(temp_path, quiet=True)
                funcs = self._prjobj(dataclass, func, subj)
                print(" +Filename of fixed image: {}".format(warped.Filename))
                for i, finfo in funcs.iterrows():
                    print(" +Filename of moving image: {}".format(finfo.Filename))
                    output_path = os.path.join(step01, subj, finfo.Filename)
                    self._prjobj.run('ants_WarpTimeSeriesImageMultiTransform', output_path,
                                     finfo.Abspath, warped.Abspath, warps, mats)
                # subjatlas = InternalMethods.load_temp(warped.Abspath, '{}_atlas.nii'.format(temp_path))
                subjatlas = InternalMethods.load_temp(output_path, '{}_atlas.nii'.format(temp_path))
                fig = subjatlas.show(**kwargs)
                if type(fig) is tuple:
                    fig = fig[0]
                fig.suptitle('Check atlas registration of {}'.format(subj), fontsize=12, color='yellow')
                fig.savefig(os.path.join(step02, 'AllSubjects', '{}.png'.format('-'.join([subj, 'checkatlas']))),
                            facecolor=fig.get_facecolor())
                os.remove('{}_atlas.nii'.format(temp_path))
                os.remove('{}_atlas.label'.format(temp_path))
                os.remove('{}_template.nii'.format(temp_path))
            else:
                InternalMethods.mkdir(os.path.join(step02, subj))
                for sess in self.sessions:
                    InternalMethods.mkdir(os.path.join(step02, subj, 'AllSessions'), os.path.join(step01, subj, sess))
                    print(" :Session: {}".format(sess))
                    # Grab the warping map and transform matrix
                    mats, warps, warped = InternalMethods.get_warp_matrix(self, warped_anat, subj, sess, inverse=True)
                    temp_path = os.path.join(step01, subj, sess, "base")
                    tempobj.save_as(temp_path, quiet=True)
                    funcs = self._prjobj(dataclass, func, subj, sess)
                    print(" +Filename of fixed image: {}".format(warped.Filename))
                    for i, finfo in funcs.iterrows():
                        print(" +Filename of moving image: {}".format(finfo.Filename))
                        output_path = os.path.join(step01, subj, sess, finfo.Filename)
                        self._prjobj.run('ants_WarpTimeSeriesImageMultiTransform', output_path,
                                         finfo.Abspath, warped.Abspath, warps, mats)
                    # subjatlas = InternalMethods.load_temp(warped.Abspath, '{}_atlas.nii'.format(temp_path))
                    subjatlas = InternalMethods.load_temp(output_path, '{}_atlas.nii'.format(temp_path))
                    fig = subjatlas.show(**kwargs)
                    if type(fig) is tuple:
                        fig = fig[0]
                    fig.suptitle('Check atlas registration of {}'.format(subj), fontsize=12, color='yellow')
                    fig.savefig(os.path.join(step02, subj, 'AllSessions',
                                             '{}.png'.format('-'.join([subj, sess, 'checkatlas']))),
                                facecolor=fig.get_facecolor())
                    os.remove('{}_atlas.nii'.format(temp_path))
                    os.remove('{}_atlas.label'.format(temp_path))
                    os.remove('{}_template.nii'.format(temp_path))
        return {'func': step01, 'checkreg': step02}

    def warp_anat_to_template(self, anat, tempobj, dtype='anat', **kwargs): # TODO: This code not work if the template image resolution is different with T2 image
        """ Method for warping the individual anatomical image to template

        Parameters
        ----------
        anat        : str
            Datatype or Absolute step path for the input anatomical image
        tempobj     : pynit.Template
            The template object which contains set of atlas
        dtype       : str
            Surfix for the step path

        Returns
        -------
        step_paths  : dict
        """
        # Check the source of input data
        if os.path.exists(anat):
            dataclass = 1
            anat = InternalMethods.path_splitter(anat)[-1]
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
            InternalMethods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                InternalMethods.mkdir(os.path.join(step02, 'AllSubjects'))
                anats = self._prjobj(dataclass, anat, subj)
                InternalMethods.mkdir(os.path.join(step01, subj))
                for i, finfo in anats.iterrows():
                    print(" +Filename: {}".format(finfo.Filename))
                    output_path = os.path.join(step01, subj, "{}".format(subj))
                    self._prjobj.run('ants_RegistrationSyn', output_path,
                                     finfo.Abspath, base_path=tempobj.template_path, quick=False)
                    fig1 = Viewer.check_reg(InternalMethods.load(tempobj.template_path),
                                            InternalMethods.load("{}_Warped.nii.gz".format(output_path)), sigma=2, **kwargs)
                    fig1.suptitle('T2 to Atlas for {}'.format(subj), fontsize=12, color='yellow')
                    fig1.savefig(os.path.join(step02, 'AllSubjects', '{}.png'.format('-'.join([subj, 'anat2temp']))),
                                 facecolor=fig1.get_facecolor())
                    fig2 = Viewer.check_reg(InternalMethods.load("{}_Warped.nii.gz".format(output_path)),
                                            InternalMethods.load(tempobj.template_path), sigma=2, **kwargs)
                    fig2.suptitle('Atlas to T2 for {}'.format(subj), fontsize=12, color='yellow')
                    fig2.savefig(os.path.join(step02, 'AllSubjects', '{}.png'.format('-'.join([subj, 'temp2anat']))),
                                 facecolor=fig2.get_facecolor())
            else:
                InternalMethods.mkdir(os.path.join(step02, subj), os.path.join(step02, subj, 'AllSessions'))
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    anats = self._prjobj(dataclass, anat, subj, sess)
                    InternalMethods.mkdir(os.path.join(step01, subj, sess))
                    for i, finfo in anats.iterrows():
                        print("  +Filename: {}".format(finfo.Filename))
                        output_path = os.path.join(step01, subj, sess, "{}".format(subj))
                        self._prjobj.run('ants_RegistrationSyn',
                                         output_path,
                                         finfo.Abspath, base_path=tempobj.template_path, quick=False)
                        fig1 = Viewer.check_reg(InternalMethods.load(tempobj.template_path),
                                                InternalMethods.load("{}_Warped.nii.gz".format(output_path)),
                                                sigma=2, **kwargs)
                        fig1.suptitle('T2 to Atlas for {}'.format(subj), fontsize=12, color='yellow')
                        fig1.savefig(
                            os.path.join(step02, subj, 'AllSessions', '{}.png'.format('-'.join([subj, 'anat2temp']))),
                            facecolor=fig1.get_facecolor())
                        fig2 = Viewer.check_reg(InternalMethods.load("{}_Warped.nii.gz".format(output_path)),
                                                InternalMethods.load(tempobj.template_path), sigma=2, **kwargs)
                        fig2.suptitle('Atlas to T2 for {}'.format(subj), fontsize=12, color='yellow')
                        fig2.savefig(
                            os.path.join(step02, subj, 'AllSessions', '{}.png'.format('-'.join([subj, 'temp2anat']))),
                            facecolor=fig2.get_facecolor())
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
        dataclass, anat = InternalMethods.check_input_dataclass(anat)
        print("Warp-Atlas to {} and Check it's registration".format(anat))
        step01 = self.init_step('Warp-atlas2{}'.format(dtype))
        num_step = os.path.basename(step01).split('_')[0]
        step02 = self.final_step('{}_CheckAtlasRegistration-{}'.format(num_step, dtype))
        # Loop the subjects
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            InternalMethods.mkdir(os.path.join(step01, subj))
            if self._prjobj.single_session:
                # Grab the warping map and transform matrix
                mats, warps, warped = InternalMethods.get_warp_matrix(self, warped_anat, subj, inverse=True)
                temp_path = os.path.join(warped_anat, subj, "base")
                tempobj.save_as(temp_path, quiet=True)
                anats = self._prjobj(dataclass, anat, subj)
                output_path = os.path.join(step01, subj, "{}_atlas.nii".format(subj))
                InternalMethods.mkdir(os.path.join(step01, subj), os.path.join(step02, 'AllSubjects'))
                print(" +Filename: {}".format(warped.Filename))
                self._prjobj.run('ants_WarpImageMultiTransform', output_path,
                                 '{}_atlas.nii'.format(temp_path), warped.Abspath,
                                 True, '-i', mats, warps)
                tempobj.atlasobj.save_as(os.path.join(step01, subj, "{}_atlas".format(subj)), label_only=True)
                for i, finfo in anats.iterrows():
                    subjatlas = InternalMethods.load_temp(finfo.Abspath, output_path)
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
                    mats, warps, warped = InternalMethods.get_warp_matrix(self, warped_anat, subj, sess, inverse=True)
                    temp_path = os.path.join(step01, subj, sess, "base")
                    tempobj.save_as(temp_path, quiet=True)
                    anats = self._prjobj(dataclass, anat, subj, sess)
                    output_path = os.path.join(step01, subj, sess, "{}_atlas.nii".format(sess))
                    InternalMethods.mkdir(os.path.join(step01, subj, sess), os.path.join(step02, subj, 'AllSessoions'))
                    print(" +Filename: {}".format(warped.Filename))
                    self._prjobj.run('ants_WarpImageMultiTransform', output_path,
                                     '{}_atlas.nii'.format(temp_path), warped.Abspath, True, '-i', mats, warps)
                    tempobj.atlasobj.save_as(os.path.join(step01, subj, sess, "{}_atlas".format(sess)), label_only=True)
                    for i, finfo in anats.iterrows():
                        subjatlas = InternalMethods.load_temp(finfo.Abspath, output_path)
                        fig = subjatlas.show(**kwargs)
                        if type(fig) is tuple:
                            fig = fig[0]
                        fig.suptitle('Check atlas registration of {}'.format(sess), fontsize=12, color='yellow')
                        fig.savefig(os.path.join(step02, subj, 'AllSessions',
                                                 '{}.png'.format('-'.join([sess, 'checkatlas']))),
                                    facecolor=fig.get_facecolor())
        return {'atlas': step01, 'checkreg': step02}

    # def warp_atlas(self, anat, tempobj, dtype='anat', **kwargs):
    #     """ Warp anatomical image to template and inverse transform the atlas image
    #     """
    #     # Check the source of input data
    #     dataclass, anat = InternalMethods.check_input_dataclass(anat)
    #     print('Warp-{} to Tempalte'.format(anat))
    #     step01 = self.init_step('Warp-{}2temp'.format(dtype))
    #     # Loop the subjects
    #     for subj in self.subjects:
    #         print("-Subject: {}".format(subj))
    #         InternalMethods.mkdir(os.path.join(step01, subj))
    #         if self._prjobj.single_session:
    #             anats = self._prjobj(dataclass, anat, subj)
    #             InternalMethods.mkdir(os.path.join(step01, subj))
    #             for i, finfo in anats.iterrows():
    #                 print(" +Filename: {}".format(finfo.Filename))
    #                 self._prjobj.run('ants_RegistrationSyn', os.path.join(step01, subj, "{}".format(subj)),
    #                                  finfo.Abspath, base_path=tempobj.template_path, quick=False)
    #         else:
    #             for sess in self.sessions:
    #                 print(" :Session: {}".format(sess))
    #                 anats = self._prjobj(dataclass, anat, subj, sess)
    #                 InternalMethods.mkdir(os.path.join(step01, subj, sess))
    #                 for i, finfo in anats.iterrows():
    #                     print("  +Filename: {}".format(finfo.Filename))
    #                     self._prjobj.run('ants_RegistrationSyn',
    #                                      os.path.join(step01, subj, sess, "{}".format(sess)),
    #                                      finfo.Abspath, base_path=tempobj.template_path, quick=False)
    #     # Print step
    #     print("Warp-Atlas to {} and Check it's registration".format(anat))
    #     step02 = self.init_step('Warp-atlas2{}'.format(dtype))
    #     step03 = self.init_step('CheckAtlasRegistration-{}'.format(dtype))
    #     # Loop the subjects
    #     for subj in self.subjects:
    #         print("-Subject: {}".format(subj))
    #         InternalMethods.mkdir(os.path.join(step02, subj))
    #         if self._prjobj.single_session:
    #             # Grab the warping map and transform matrix
    #             mats, warps, warped = InternalMethods.get_warp_matrix(self, step01, subj, inverse=True)
    #             #
    #             temp_path = os.path.join(step01, subj, "base")
    #             tempobj.save_as(temp_path, quiet=True)
    #             anats = self._prjobj(dataclass, self._pipeline, anat, subj)
    #             output_path = os.path.join(step02, subj, "{}_atlas.nii".format(subj))
    #             InternalMethods.mkdir(os.path.join(step02, subj), os.path.join(step03, subj))
    #             print(" +Filename: {}".format(warped.Filename))
    #             self._prjobj.run('ants_WarpImageMultiTransform', output_path,
    #                              '{}_atlas.nii'.format(temp_path), warped.Abspath,
    #                              True, '-i', mats, warps)
    #             tempobj.atlasobj.save_as(os.path.join(step02, subj, "{}_atlas".format(subj)), label_only=True)
    #             for i, finfo in anats.iterrows():
    #                 subjatlas = InternalMethods.load_temp(finfo.Abspath, output_path)
    #                 fig = subjatlas.show(**kwargs)
    #                 if type(fig) is tuple:
    #                     fig = fig[0]
    #                 fig.suptitle('Check atlas registration of {}'.format(subj), fontsize=12, color='yellow')
    #                 fig.savefig(os.path.join(step03, '{}.png'.format('-'.join([subj, 'checkatlas']))),
    #                              facecolor=fig.get_facecolor())
    #         else:
    #             for sess in self.sessions:
    #                 print(" :Session: {}".format(sess))
    #                 # Grab the warping map and transform matrix
    #                 mats, warps, warped = InternalMethods.get_warp_matrix(self, step01, subj, sess, inverse=True)
    #                 temp_path = os.path.join(step01, subj, sess, "base")
    #                 tempobj.save_as(temp_path, quiet=True)
    #                 anats = self._prjobj(dataclass, self._pipeline, anat, subj, sess)
    #                 output_path = os.path.join(step02, subj, sess, "{}_atlas.nii".format(sess))
    #                 InternalMethods.mkdir(os.path.join(step02, subj, sess), os.path.join(step03, subj, sess))
    #                 print(" +Filename: {}".format(warped.Filename))
    #                 self._prjobj.run('ants_WarpImageMultiTransform', output_path,
    #                                  '{}_atlas.nii'.format(temp_path), warped.Abspath, True, '-i', mats, warps)
    #                 tempobj.atlasobj.save_as(os.path.join(step02, subj, sess, "{}_atlas".format(sess)), label_only=True)
    #                 for i, finfo in anats.iterrows():
    #                     subjatlas = InternalMethods.load_temp(finfo.Abspath, output_path)
    #                     fig = subjatlas.show(**kwargs)
    #                     if type(fig) is tuple:
    #                         fig = fig[0]
    #                     fig.suptitle('Check atlas registration of {}'.format(sess), fontsize=12, color='yellow')
    #                     fig.savefig(os.path.join(step03, subj, '{}.png'.format('-'.join([sess, 'checkatlas']))),
    #                                 facecolor=fig.get_facecolor())
    #     return {'anat': step01, 'atlas': step02, 'checkreg': step03}

    def get_correlation_matrix(self, func, atlas, dtype='func', **kwargs):
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
        dataclass, func = InternalMethods.check_input_dataclass(func)
        atlas, tempobj = InternalMethods.check_atals_datatype(atlas)
        print('ExtractTimeCourseData-{}'.format(func))
        step01 = self.init_step('ExtractTimeCourse-{}'.format(dtype))
        step02 = self.init_step('CC_Matrix-{}'.format(dtype))
        num_step = os.path.basename(step02).split('_')[0]
        step03 = self.final_step('{}_Zscore_Matrix-{}'.format(num_step, dtype))
        for subj in self.subjects:
            print("-Subject: {}".format(subj))
            InternalMethods.mkdir(os.path.join(step01, subj), os.path.join(step02, subj), os.path.join(step03, subj))
            if self._prjobj.single_session:
                if not tempobj:
                    atlas = self._prjobj(1, self._pipeline, atlas, subj).Abspath.loc[0]
                    warped = self._prjobj(1, self._pipeline, atlas, subj, file_tag='_InverseWarped').Abspath.loc[0]
                    tempobj = InternalMethods.load_temp(warped, atlas)
                funcs = self._prjobj(dataclass, func, subj)
                for i, finfo in funcs.iterrows():
                    print(" +Filename: {}".format(finfo.Filename))
                    df = Analysis.get_timetrace(InternalMethods.load(finfo.Abspath), tempobj, afni=True, **kwargs)
                    df.to_excel(os.path.join(step01, subj, "{}.xlsx".format(os.path.splitext(finfo.Filename)[0])))
                    df.corr().to_excel(os.path.join(step02, subj, "{}.xlsx".format(
                        os.path.splitext(finfo.Filename)[0])))
                    np.arctanh(df.corr()).to_excel(
                        os.path.join(step03, subj, "{}.xlsx").format(os.path.splitext(finfo.Filename)[0]))
            else:
                InternalMethods.mkdir(os.path.join(step01, subj), os.path.join(step02, subj),
                                      os.path.join(step03, subj))
                for sess in self.sessions:
                    print(" :Session: {}".format(sess))
                    if not tempobj:
                        atlas = self._prjobj(1, self._pipeline, atlas, subj, sess).Abspath.loc[0]
                        warped = self._prjobj(1, self._pipeline, atlas, subj, sess, file_tag='_InverseWarped').Abspath.loc[0]
                        tempobj = InternalMethods.load_temp(warped, atlas)
                    funcs = self._prjobj(dataclass, func, subj, sess)
                    InternalMethods.mkdir(os.path.join(step01, subj, sess), os.path.join(step02, subj, sess),
                                          os.path.join(step03, subj, sess))
                    for i, finfo in funcs.iterrows():
                        print("  +Filename: {}".format(finfo.Filename))
                        df = Analysis.get_timetrace(InternalMethods.load(finfo.Abspath), tempobj, afni=True, **kwargs)
                        df.to_excel(os.path.join(step01, subj, sess, "{}.xlsx".format(
                            os.path.splitext(finfo.Filename)[0])))
                        df.corr().to_excel(
                            os.path.join(step02, subj, sess, "{}.xlsx".format(os.path.splitext(finfo.Filename)[0])))
                        np.arctanh(df.corr()).to_excel(
                            os.path.join(step03, subj, sess, "{}.xlsx".format(os.path.splitext(finfo.Filename)[0])))
        return {'timecourse': step01, 'cc_matrix': step02}

    def set_stim_paradigm(self, num_of_time, tr, filename='stim_paradigm', **kwargs):
        onset = []
        num_stimts = 1
        duration = None
        peak = 1
        stim_type = None
        if kwargs:
            for kwarg in kwargs.keys():
                if kwarg is 'onset':
                    if type(kwargs[kwarg]) is not list:
                        raise error.CommandExecutionFailure
                    else:
                        onset = kwargs[kwarg]
                if kwarg is 'duration':
                    if type(kwargs[kwarg]) is not int:
                        raise error.CommandExecutionFailure
                    else:
                        duration = str(kwargs[kwarg])
                if kwarg is 'peak':
                    if type(kwargs[kwarg]) is not int:
                        raise error.CommandExecutionFailure
                    else:
                        peak = str(kwargs[kwarg])
                if kwarg is 'hrf_function':
                    if type(kwargs[kwarg]) is not str:
                        raise error.CommandExecutionFailure
                    else:
                        if kwargs[kwarg] is 'MION':
                            stim_type = "MIONN({})".format(duration)
                        elif kwargs[kwarg] is 'BLOCK':
                            stim_type = "BLOCK({},{})".format(duration, peak)
                        else:
                            raise error.CommandExecutionFailure
        output_path = os.path.join('.tmp', '{}.xmat.1D'.format(filename))
        Interface.afni_3dDeconvolve(output_path, None, nodata=[str(num_of_time), str(tr)],
                                    num_stimts=num_stimts, polort=-1,
                                    stim_times=['1', '1D: {}'.format(" ".join(onset)),
                                                "'{}'".format(stim_type)])
        return {'paradigm': output_path}

    # def general_linear_model(self, func, paradigm, dtype='func'):
    #     if os.path.exists(func):
    #         dataclass = 1
    #         func = InternalMethods.path_splitter(func)[-1]
    #     else:
    #         dataclass = 0
    #     print('GLM Analysis-{}'.format(func))
    #     step01 = self.init_step('ExtractTimeCourse-{}'.format(dtype))
    #     # num_step = os.path.basename(step02).split('_')[0]
    #     # step02 = self.final_step('{}_ActivityMap-{}'.format(num_step, dtype))
    #     for subj in self.subjects:
    #         print("-Subject: {}".format(subj))
    #         InternalMethods.mkdir(os.path.join(step01, subj))
    #         if self._prjobj.single_session:
    #             funcs = self._prjobj(dataclass, func, subj)
    #             for i, finfo in funcs.iterrows():
    #                 print(" +Filename: {}".format(finfo.Filename))
    #                 output_path = os.path.join(step01, subj, finfo.Filename)
    #                 self._prjobj.run('afni_3dDeconvolve', str(output_path), str(finfo.Abspath),
    #                                  num_stimts='1', nfirst='0', polort='-1', stim_file=['1', "'{}'".format(paradigm)],
    #                                  stim_label=['1', "'STIM'"], num_glt='1', glt_label=['1', "'STIM'"],
    #                                  gltsym='SYM: +STIM')
    #     return {'func': step01}

    def init_step(self, title):
        path = self._prjobj.initiate_step(title)
        self._prjobj.reload()
        return path

    def final_step(self, title):
        path = os.path.join(self._prjobj.path, self._prjobj.ds_type[2],
                            self._prjobj.pipeline, title)
        InternalMethods.mkdir(os.path.join(self._prjobj.path, self._prjobj.ds_type[2],
                                           self._prjobj.pipeline), path)
        self._prjobj.reload()
        return path


class Project(object):
    """Project Handler for Neuroimage data
    """

    def __init__(self, project_path, ds_ref='NIRAL', img_format='NifTi-1', **kwargs):
        """Load and initiate the project

        Parameters
        ----------
        project_path:   str
            Path of particular project
        ds_ref:         str
            Reference of data structure (default: 'NIRAL')
        img_format:     str
            Reference img format
        """
        # Variables for attributes
        max_rows = 100
        max_colwidth = 100
        if kwargs:
            if 'max_rows' in kwargs.keys():
                max_rows = kwargs['max_rows']
            if 'max_colwidth' in kwargs.keys():
                max_colwidth = kwargs['max_colwidth']
        pd.options.display.max_rows = max_rows
        pd.options.display.max_colwidth = max_colwidth
        self.single_session = False
        self.__path = project_path
        self.__filters = [None] * 6
        # Each values are represented subject, session, dtype(or pipeline), step(or results) file_tags, ignores
        self.__df = pd.DataFrame()
        # Parsing the information from the reference
        self.__ref = [ds_ref, img_format]
        self.__empty_project = False
        ref = Reference(*self.__ref)
        self.img_ext = ref.imgext
        self.ds_type = ref.ref_ds
        # Define basic variables for initiating instance
        self.__dc_idx = 0           # Data class index
        self.__ext_filter = self.img_ext
        InternalMethods.mk_main_folder(self)
        self.__pipeline = None
        self.interface = Interface()
        try:
            self.reload()
        except:
            raise error.ReloadFailure

    @property
    def df(self):
        columns = self.__df.columns
        return self.__df.reset_index()[columns]

    @property
    def path(self):
        return self.__path

    @property
    def dataclass(self):
        return self.ds_type[self.__dc_idx]

    @dataclass.setter
    def dataclass(self, idx):
        if idx in range(3):
            self.__dc_idx = idx
            self.reset()
            self.__update()
        else:
            raise error.NotExistingDataclass

    @property
    def subjects(self):
        return self.__subjects

    @property
    def sessions(self):
        return self.__sessions

    @property
    def dtypes(self):
        return self.__dtypes

    @property
    def pipelines(self):
        return self.__pipelines

    @property
    def pipeline(self):
        return self.__pipeline

    @property
    def steps(self):
        return self.__steps

    @property
    def results(self):
        return self.__results

    @property
    def filters(self):
        return self.__filters

    @property
    def summary(self):
        return self.__summary()

    @property
    def ext(self):
        return self.__ext_filter

    @ext.setter
    def ext(self, value):
        if type(value) == str:
            self.__ext_filter = [value]
        elif type(value) == list:
            self.__ext_filter = value
        elif not value:
            self.__ext_filter = None
        else:
            raise error.FilterInputTypeError

    def initiate_pipeline(self, pipeline):
        InternalMethods.mkdir(os.path.join(self.path, self.ds_type[1], pipeline))
        self.__pipeline = pipeline

    def initiate_step(self, stepname):
        if self.__pipeline:
            steppath = InternalMethods.get_step_name(self, stepname)
            steppath = os.path.join(self.path, self.ds_type[1], self.__pipeline, steppath)
            InternalMethods.mkdir(steppath)
            return steppath
        else:
            raise error.PipelineNotSet

    def reset(self, ext=None):
        """Reset filter - Clear all filter information and extension
        """
        self.__filters = [None] * 6
        self.__pipeline = None
        if not ext:
            self.ext = self.img_ext
        else:
            self.ext = ext
        self.reload()
        self.__update()

    def reload(self):
        """Reload the dataframe based on current set data class and extension

        :return:
        """
        # Parsing command works
        self.__df, self.single_session, empty_prj = InternalMethods.parsing(self.path, self.ds_type, self.__dc_idx)
        if not empty_prj:
            self.__df = InternalMethods.initial_filter(self.__df, self.ds_type, self.__ext_filter)
            if len(self.__df):
                self.__df = self.__df[InternalMethods.reorder_columns(self.__dc_idx, self.single_session)]
            self.__update()
            self.__empty_project = False
        else:
            self.__empty_project = True

    def copy(self):
        """Make copy of current project

        :return: niph.Project instance
        """
        return Project(self.__path, *self.__ref)

    def set_filters(self, *args, **kwargs):
        """Set filters

        :param args:    str[, ]
            String arguments regarding hierarchical data structures
        :param kwargs:  key=value pair[, ]
            Key and value pairs regarding the filename
            :key file_tag:  str or list of str
                Keywords of interest for filename
            :key ignore:    str of list of str
                Keywords of neglect for filename
            :key keep:    boolean
                If this argument is exist and True, keep previous filter information
        :return:
        """
        if kwargs:
            if 'ext' in kwargs.keys():
                self.ext = kwargs['ext']
        if 'keep' in kwargs.keys():
            # This option allows to keep previous filter
            if kwargs['keep']:
                self.__update()
            else:
                self.reset(self.ext)
        else:
            self.reset(self.ext)
        if args or kwargs:
            if args:
                if self.subjects:
                    if self.__filters[0]:
                        self.__filters[0].extend([arg for arg in args if arg in self.subjects])
                    else:
                        self.__filters[0] = [arg for arg in args if arg in self.subjects]
                    if not self.single_session:
                        if self.__filters[1]:
                            self.__filters[1].extend([arg for arg in args if arg in self.sessions])
                        else:
                            self.__filters[1] = [arg for arg in args if arg in self.sessions]
                    else:
                        self.__filters[1] = None
                else:
                    self.__filters[0] = None
                    self.__filters[1] = None
                if self.__dc_idx == 0:
                    if self.dtypes:
                        if self.__filters[2]:
                            self.__filters[2].extend([arg for arg in args if arg in self.dtypes])
                        else:
                            self.__filters[2] = [arg for arg in args if arg in self.dtypes]
                    else:
                        self.__filters[2] = None
                    self.__filters[3] = None
                elif self.__dc_idx == 1:
                    if self.pipelines:
                        if self.__filters[2]:
                            self.__filters[2].extend([arg for arg in args if arg in self.pipelines])
                        else:
                            self.__filters[2] = [arg for arg in args if arg in self.pipelines]
                    else:
                        self.__filters[2] = None
                    if self.steps:
                        if self.__filters[3]:
                            self.__filters[3].extend([arg for arg in args if arg in self.steps])
                        else:
                            self.__filters[3] = [arg for arg in args if arg in self.steps]
                    else:
                        self.__filters[3] = None
                else:
                    if self.pipelines:
                        if self.__filters[2]:
                            self.__filters[2].extend([arg for arg in args if arg in self.pipelines])
                        else:
                            self.__filters[2] = [arg for arg in args if arg in self.pipelines]
                    else:
                        self.__filters[2] = None
                    if self.results:
                        if self.__filters[3]:
                            self.__filters[3].extend([arg for arg in args if arg in self.results])
                        else:
                            self.__filters[3] = [arg for arg in args if arg in self.results]
                    else:
                        self.__filters[3] = None
            if kwargs:
                if 'file_tag' in kwargs.keys():
                    if type(kwargs['file_tag']) == str:
                        self.__filters[4] = [kwargs['file_tag']]
                    elif type(kwargs['file_tag']) == list:
                        self.__filters[4] = kwargs['file_tag']
                    else:
                        raise error.FilterInputTypeError
                if 'ignore' in kwargs.keys():
                    if type(kwargs['ignore']) == str:
                        self.__filters[5] = [kwargs['ignore']]
                    elif type(kwargs['ignore']) == list:
                        self.__filters[5] = kwargs['ignore']
                    else:
                        raise error.FilterInputTypeError
        self.__df = self.applying_filters(self.__df)
        self.__update()

    def applying_filters(self, df):
        """Applying current filters to the input dataframe

        :param df: pandas.DataFrame
        :return: pandas.DataFrame
        """
        if len(df):
            if self.__filters[0]:
                df = df[df.Subject.isin(self.__filters[0])]
            if self.__filters[1]:
                df = df[df.Session.isin(self.__filters[1])]
            if self.__filters[2]:
                if self.__dc_idx == 0:
                    df = df[df.DataType.isin(self.__filters[2])]
                else:
                    df = df[df.Pipeline.isin(self.__filters[2])]
            if self.__filters[3]:
                if self.__dc_idx == 1:
                    df = df[df.Step.isin(self.__filters[3])]
                elif self.__dc_idx == 2:
                    df = df[df.Result.isin(self.__filters[3])]
                else:
                    pass
            if self.__filters[4]:
                df = df[df.Filename.str.contains('|'.join(self.__filters[4]))]
            if self.__filters[5]:
                df = df[~df.Filename.str.contains('|'.join(self.__filters[5]))]
            return df
        else:
            return df

    def help(self, command=None):
        """Print doc string for command or pipeline

        :param command:
        :return:
        """
        if command:
            if command in dir(Interface):
                exec 'help(Interface.{})'.format(command)
            elif command in dir(Analysis):
                exec 'help(Analysis.{})'.format(command)
            else:
                raise error.UnableInterfaceCommand

    def run(self, command, *args, **kwargs):
        """Execute processing tools
        """
        if command in dir(Interface):
            try:
                if os.path.exists(args[0]):
                    pass
                else:
                    getattr(Interface, command)(*args, **kwargs)
            except:
                exec('help(Interface.{})'.format(command))
                raise error.CommandExecutionFailure
        else:
            raise error.NotExistingCommand

    def __summary(self):
        """Print summary of current project
        """
        summary = 'Project summary'
        summary = '{}\nProject: {}'.format(summary, os.path.dirname(self.path).split(os.sep)[-1])
        if self.__empty_project:
            summary = '{}\n[Empty project]'.format(summary)
        else:
            summary = '{}\nSelected DataClass: {}\n'.format(summary, self.dataclass)
            if self.pipelines:
                summary = '{}\nApplied Pipeline(s): {}'.format(summary, self.pipelines)
            if self.steps:
                summary = '{}\nApplied Step(s): {}'.format(summary, self.steps)
            if self.results:
                summary = '{}\nProcessed Result(s): {}'.format(summary, self.results)
            if self.subjects:
                summary = '{}\nSubject(s): {}'.format(summary, self.subjects)
            if self.sessions:
                summary = '{}\nSession(s): {}'.format(summary, self.sessions)
            if self.dtypes:
                summary = '{}\nDataType(s): {}'.format(summary, self.dtypes)
            if self.single_session:
                summary = '{}\nSingle session dataset'.format(summary)
            summary = '{}\n\nApplied filters'.format(summary)
            if self.__filters[0]:
                summary = '{}\nSet subject(s): {}'.format(summary, self.__filters[0])
            if self.__filters[1]:
                summary = '{}\nSet session(s): {}'.format(summary, self.__filters[1])
            if self.__dc_idx == 0:
                if self.__filters[2]:
                    summary = '{}\nSet datatype(s): {}'.format(summary, self.__filters[2])
            else:
                if self.__filters[2]:
                    summary = '{}\nSet Pipeline(s): {}'.format(summary, self.__filters[2])
                if self.__filters[3]:
                    if self.__dc_idx == 1:
                        summary = '{}\nSet Step(s): {}'.format(summary, self.__filters[3])
                    else:
                        summary = '{}\nSet Result(s): {}'.format(summary, self.__filters[3])
            if self.__ext_filter:
                summary = '{}\nSet file extension(s): {}'.format(summary, self.__ext_filter)
            if self.__filters[4]:
                summary = '{}\nSet file tag(s): {}'.format(summary, self.__filters[4])
            if self.__filters[5]:
                summary = '{}\nSet ignore(s): {}'.format(summary, self.__filters[5])
            if self.__pipeline:
                summary = '{}\nInitiated pipeline: {}'.format(summary, self.__pipeline)
        print(summary)

    def __update(self):
        """Update sub variables based on current set filter information
        """
        if len(self.df):
            try:
                self.__subjects = sorted(list(set(self.df.Subject.tolist())))
                if self.single_session:
                    self.__sessions = None
                else:
                    self.__sessions = sorted(list(set(self.df.Session.tolist())))
                if self.__dc_idx == 0:
                    self.__dtypes = sorted(list(set(self.df.DataType.tolist())))
                    self.__pipelines = None
                    self.__steps = None
                    self.__results = None
                elif self.__dc_idx == 1:
                    self.__dtypes = None
                    self.__pipelines = sorted(list(set(self.df.Pipeline.tolist())))
                    self.__steps = sorted(list(set(self.df.Step.tolist())))
                    self.__results = None
                elif self.__dc_idx == 2:
                    self.__dtypes = None
                    self.__pipelines = sorted(list(set(self.df.Pipeline.tolist())))
                    self.__results = sorted(list(set(self.df.Result.tolist())))
                    self.__steps = None
            except:
                raise error.UpdateFailed
        else:
            self.__subjects = None
            self.__sessions = None
            self.__dtypes = None
            self.__pipelines = None
            self.__steps = None
            self.__results = None

    def __call__(self, dc_id, *args, **kwargs):
        """Return DataFrame followed applying filters
        """
        if self.__empty_project:
            return None
        else:
            copy = self.copy()
            copy.dataclass = dc_id
            copy.set_filters(*args, **kwargs)
            return copy.df

    def __repr__(self):
        """Return absolute path for current filtered dataframe
        """
        if self.__empty_project:
            return str(self.summary)
        else:
            return str(self.df.Abspath)

    def __getitem__(self, index):
        """Return particular data based on input index
        """
        if self.__empty_project:
            return None
        else:
            return self.df.loc[index]

    def __iter__(self):
        """Iterator for dataframe
        """
        if self.__empty_project:
            raise error.EmptyProject
        else:
            for row in self.df.iterrows():
                yield row

    def __len__(self):
        """Return number of data
        """
        if self.__empty_project:
            return 0
        else:
            return len(self.df)