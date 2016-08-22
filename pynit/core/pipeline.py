import os
import time
from os.path import join

import process
import error
from .utility import InternalMethods


class Pipeline(process.Pipeline):
    def __init__(self, obj):
        super(Pipeline, self).__init__(obj)

    def _pipe_MakeStudySpecificTemplate(self, tempobj, anat, func, cbv=None):
        """Pipeline to build study specific template based on provided template
        * This pipeline has been build to use Manual skull stripping. The automateone should be included later

        template:   str
            Path of the template
        anat:       str
            Folder name of the anatomical image
        cbv:        boolean
            If cbv is True, the pipeline run for contrast injected images,
        """
        self.prj.reset()
        # Check template path
        if type(tempobj) is not str:
            try:
                template = tempobj.image.get_filename()
            except:
                raise error.InputPathError
        else:
            template = tempobj

        # Step 01. Motion Correction
        command = 'afni_3dvolreg'
        if cbv:
            kwargs = {'dc_id': 0,
                      'filters': [func],
                      'file_index': 0}
            step01 = self('MotionCorrection-{}'.format(func), command, **kwargs)

            # Step 02. Calculate mean image for BOLD and CBV
            command = 'cal_mean_cbv'
            kwargs = {'dc_id': 1,
                      'filters': [step01],
                      'postfix_bold': 'bold',
                      'postfix_cbv': func}
            step02 = self('MeanImage-{}_and_{}'.format(func, 'BOLD'), command, **kwargs)
        else:
            kwargs = {'dc_id': 0,
                      'filters': [func],
                      'file_index': 1}
            step01 = self('MotionCorrection-{}'.format(func), command, **kwargs)

            # Step 02. Calculate mean image for BOLD and CBV
            command = 'cal_mean'
            kwargs = {'dc_id': 1,
                      'filters': [step01]}
            step02 = self('MeanImage-{}'.format(func), command, **kwargs)

        # Step 03. Copy functional image to step03 folder for masking
        command = 'copyfile'
        kwargs = {'dc_id': 1, 'filters': [step02, {'file_tag': 'bold'}]}
        step03 = self('MaskDrawing-func', command, **kwargs)

        # Step 04. Copy anatomical image to step04 folder for masking
        command = 'copyfile'
        kwargs = {'dc_id': 0, 'filters': [anat]}
        step04 = self('MaskDrawing-anat', command, **kwargs)

        # Request Mask Generation TODO: Put it Message class (Pipeline sometime need to use message anyway)
        start_time = time.time()
        mask_files = self.prj.copy()
        mask_files.set_filters(1, step03, file_tag='_mask')
        self.set_filters(1, step03, ignore='_mask')
        step03_mask = len(mask_files.df)
        step03_func = len(self.prj.df)
        mask_files.set_filters(1, step04, file_tag='_mask')
        self.set_filters(1, step04, ignore='_mask')
        step04_mask = len(mask_files.df)
        step04_anat = len(self.prj.df)
        if (step03_mask + step04_mask) == 0:
            print("\nPlease put the brain mask with suffix '_mask' folder.\n"
                  "e.g.) if the the filename is 'sub1.nii', mask file should be 'sub1_mask.nii'")
            return
        else:
            if step03_mask != step03_func and step04_mask != step04_anat:
                if step03_mask != step03_func:
                    print('\nWarning: The mask files for the functional images are not fully provided')
                    return
                elif step04_mask != step04_anat:
                    print('\nWarning: The mask files for the anatomical images are not fully provided')
                    return
                else:
                    print('\nWarning: The mask files for the some images are not provided')
                    return
            else:
                pass
        del step03_func, step04_anat, step03_mask, step04_mask
        end_time = time.time() - start_time
        print('\n{} takes {} sec'.format("Checking mask files", round(end_time, 2)))

        # Step 05. Skull stripping-func
        def skull_stripping(pipeline_obj, step_name, output_path, *args):
            pipeline_obj.set_filters(step_name, *args, file_tag='_mask')
            mask = pipeline_obj.prj.df.Abspath.tolist()[0]
            pipeline_obj.set_filters(step_name, *args, ignore='_mask')
            img = pipeline_obj.prj.df.Abspath.tolist()[0]
            output_path = join(output_path, pipeline_obj.prj.df.Filename.tolist()[0])
            inputs = [img, mask]
            pipeline_obj.run_cmd('afni_3dcalc', output_path, 'a*step(b)', *inputs)

        start_time = time.time()
        step05, step_path = self.init_step('SkullStripping-func')
        for subject in self.prj.subjects:
            output_path = join(step_path, subject)
            try:
                os.mkdir(output_path)
            except:
                pass
            if self.prj.single_session:
                skull_stripping(self, step03, output_path, subject)
            else:
                for session in self.prj.sessions:
                    output_path = join(output_path, session)
                    try:
                        os.mkdir(output_path)
                    except:
                        pass
                    skull_stripping(self, step03, output_path, subject, session)
        end_time = time.time() - start_time
        print('\nStep{} takes {} sec'.format(step05, round(end_time, 2)))

        # Step 06. Skull stripping-anat
        start_time = time.time()
        step06, step_path = self.init_step('SkullStripping-anat')
        for subject in self.prj.subjects:
            output_path = join(step_path, subject)
            try:
                os.mkdir(output_path)
            except:
                pass
            if self.prj.single_session:
                skull_stripping(self, step04, output_path, subject)
            else:
                for session in self.prj.sessions:
                    output_path = join(output_path, session)
                    try:
                        os.mkdir(output_path)
                    except:
                        pass
                    skull_stripping(self, step04, output_path, subject, session)
        end_time = time.time() - start_time
        print('\nStep{} takes {} sec'.format(step06, round(end_time, 2)))

        # Step 07. Bias Field Correction-func
        command = 'ants_BiasFieldCorrection'
        kwargs = {'filters': [step05]}
        step07 = self('BiasFieldCorrection-func', command, **kwargs)

        # Step 08. Bias Field Correction-anat
        command = 'ants_BiasFieldCorrection'
        kwargs = {'filters': [step06]}
        step08 = self('BiasFieldCorrection-anat', command, **kwargs)

        # Step 09. Coregistration-func to anat
        def coregistration(pipeline_obj, move_step_name, fix_step_name, output_path, *args):
            pipeline_obj.set_filters(move_step_name, *args)
            input_path = pipeline_obj.prj.df.Abspath.tolist()[0]
            output_path = join(output_path, pipeline_obj.prj.df['Filename'].tolist()[0])
            pipeline_obj.set_filters(fix_step_name, *args)
            base_path = pipeline_obj.prj.df['Abspath'].tolist()[0]
            pipeline_obj.run_cmd('ants_RegistrationSyn', output_path, input_path, base_path)
        start_time = time.time()
        step09, step_path = self.init_step('Coregistration-func')
        for subject in self.prj.subjects:
            output_path = join(step_path, subject)
            try:
                os.mkdir(output_path)
            except:
                pass
            if self.prj.subjects == self.prj.sessions:
                coregistration(self, step07, step08, output_path, subject)
            else:
                for session in self.prj.sessions:
                    output_path = join(output_path, session)
                    try:
                        os.mkdir(output_path)
                    except:
                        pass
                    coregistration(self, step07, step08, output_path, subject, session)
        end_time = time.time() - start_time
        print('\nStep{} takes {} sec'.format(step09, round(end_time, 2)))

        # Step 10. Normalization anat to Template
        command = 'ants_RegistrationSyn'
        kwargs = {'filters': [step08],
                  'base_path': template}
        step10 = self('Normalization-anat', command, **kwargs)

        def warp_cbv_image(pipeline_obj, input_step, func_step, coreg_step, norm_step, output_path,
                           template, *args):
            pipeline_obj.set_filters(input_step, *args, file_tag=func)
            input_path = pipeline_obj.prj.df['Abspath'].tolist()[0]
            pipeline_obj.set_filters(func_step, *args, file_tag='_mask')
            mask = pipeline_obj.prj.df['Abspath'].tolist()[0]
            pipeline_obj.set_filters(coreg_step, *args, exts=['.mat'])
            coreg_mat = pipeline_obj.prj.df['Abspath'].tolist()[0]
            pipeline_obj.set_filters(coreg_step, *args, file_tag='_1Warp')
            coreg_warp_map = pipeline_obj.prj.df['Abspath'].tolist()[0]
            pipeline_obj.set_filters(norm_step, *args, exts=['.mat'])
            norm_mat = pipeline_obj.prj.df['Abspath'].tolist()[0]
            pipeline_obj.set_filters(norm_step, *args, file_tag='_1Warp')
            norm_warp_map = pipeline_obj.prj.df['Abspath'].tolist()[0]
            output_file = join(output_path, '{}_mean_{}.nii'.format('_'.join(args), func))
            output_mask = join(output_path, '{}_mean_{}_mask.nii'.format('_'.join(args), func))
            pipeline_obj.run_cmd('ants_WarpImageMultiTransform', output_file, input_path, template, coreg_mat,
                                 coreg_warp_map, norm_mat, norm_warp_map)
            pipeline_obj.run_cmd('ants_WarpImageMultiTransform', output_mask, mask, template, coreg_mat,
                                 coreg_warp_map, norm_mat, norm_warp_map)

        def warp_func_image(pipeline_obj, input_step, coreg_step, norm_step, output_path, template, *args):
            pipeline_obj.set_filters(input_step, *args, file_tag=func)
            input_path = pipeline_obj.prj.df['Abspath'].tolist()[0]
            pipeline_obj.set_filters(coreg_step, *args, exts=['.mat'])
            coreg_mat = pipeline_obj.prj.df['Abspath'].tolist()[0]
            pipeline_obj.set_filters(coreg_step, *args, file_tag='_1Warp')
            coreg_warp_map = pipeline_obj.prj.df['Abspath'].tolist()[0]
            pipeline_obj.set_filters(norm_step, *args, exts=['.mat'])
            norm_mat = pipeline_obj.prj.df['Abspath'].tolist()[0]
            pipeline_obj.set_filters(norm_step, *args, file_tag='_1Warp')
            norm_warp_map = pipeline_obj.prj.df['Abspath'].tolist()[0]
            output_file = join(output_path, '{}_mean_{}.nii'.format('_'.join(args), func))
            pipeline_obj.run_cmd('ants_WarpImageMultiTransform', output_file, input_path, template, coreg_mat,
                                 coreg_warp_map, norm_mat, norm_warp_map)

        # Step 11. Applying transform matrix for CBV
        step11, step_path = self.init_step('ApplyingTransformMatrix-func')
        for subject in self.prj.subjects:
            output_path = join(step_path, subject)
            try:
                os.mkdir(output_path)
            except:
                pass
            if self.prj.subjects == self.prj.sessions:
                if cbv:
                    warp_cbv_image(self, step02, step03, step09, step10, output_path, template, subject)
                else:
                    warp_func_image(self,step02, step09, step10, output_path, template, subject)

            else:
                for session in self.prj.sessions:
                    output_path = join(output_path, session)
                    try:
                        os.mkdir(output_path)
                    except:
                        pass
                    if cbv:
                        warp_cbv_image(self, step02, step03, step09, step10, output_path, template, subject, session)
                    else:
                        warp_func_image(self, step02, step09, step10, output_path, template, subject, session)

        # Final step. StudySpecificTemplate with mask
        result_path = join(self.prj._path, self.prj.ds_type[2], self.pipeline)
        try:
            os.mkdir(result_path)
        except:
            pass
        result_path_template = join(result_path, 'Template')
        try:
            os.mkdir(result_path_template)
        except:
            pass
        result_path_norm_subs = join(result_path, 'Normalized_subjects')
        try:
            os.mkdir(result_path_norm_subs)
        except:
            pass
        output_path = join(result_path_template, '{}_func_template.nii'.format(InternalMethods.path_splitter(self.prj._path)[-1]))
        output_mask_path = join(result_path_template, '{}_func_template_mask.nii'.format(InternalMethods.path_splitter(self.prj._path)[-1]))
        self.set_filters(step11, ignore='_mask')
        warped_imgs = self.prj.df['Abspath'].tolist()
        self.run_cmd('afni_3dMean', output_path, *warped_imgs)
        self.set_filters(step11)
        for path in self:
            self.run_cmd('shihlab_copyfile', join(result_path_norm_subs, path['Filename']), path['Abspath'])
        if cbv:
            self.set_filters(step11, file_tag='_mask')
            warped_masks = self.prj.df['Abspath'].tolist()
            self.run_cmd('afni_3dMean', output_mask_path, *warped_masks)
        # elif func:
        #     self.set_filters(step11, file_tag='_mask')
        #     for path in self:
        #         self.run_cmd('shihlab_copyfile', join(output_path, path['Filename']), path['Abspath'])

        print('All processes for pipeline {} are finished'.format(self.pipeline))

    def _pipe_rsfMRI_Preprocessing(self, func, **kwargs): # add template,
        # Preprocessing pipeline for resting state fMRI
        # Prior processing of MakeStudySpecificTemplate is required.
        # Step00. Check requirements
        try:
            self.prj.set_filters(self.prj.ds_type[2], 'MakeStudySpecificTemplate')
        except:
            raise NotImplemented('Prior requirement is not implemented,-MakeStuydSpecificTemplate')
        # Step01. SliceTiming Correction
        command = 'afni_3dTshift'
        kwargs = {'filters': [func],
                  'tr':1,
                  'tpattern': 'altplus'}
        step01 = self('SliceTimingCorrection-{}'.format(func), command, 1, **kwargs)

        # Step02. Motion Correction
        command = 'afni_3dvolreg'
        kwargs = {'filters': [step01]}
        step02 = self('MotionCorrection-{}'.format(func), command, **kwargs)

        # Step03. Calculate mean for functional images
        command = 'shihlab_cal_mean'
        kwargs = {'filters': [step02]}
        step03 = self('MeanImageCalculation-{}'.format(func), command, **kwargs)

        # Step04. Registration to template images
        self.prj.set_filters(self.prj.ds_type[2], 'MakeStudySpecificTemplate', 'Template', ignore='_mask')
        template = self.prj.df['Abspath'][0]
        command = 'afni_3dvolreg'
        kwargs = {'filters': [step03], 'base_slice': template}
        step04 = self('Registration-{}'.format(func), command, **kwargs)

        # Step05. Apply transformation file TODO: multisession
        start_time = time.time()
        step05, step_path = self.init_step('ApplyingTransform-{}'.format(func))
        for subject in self.prj.subjects:
            if self.prj.subjects == self.prj.sessions:
                output_path = join(step_path, subject)
                try:
                    os.mkdir(output_path)
                except:
                    pass
                self.set_filters(step04, subject, exts=['.aff12.1D'])
                tfm_paths = self.prj.df['Abspath']
                self.set_filters(step02, subject)
                for i, path in enumerate(self):
                    self.run_cmd('afni_3dAllineate', join(output_path, path['Filename']), path['Abspath'],
                                 matrix_apply=tfm_paths[i], master=template, warp='shr')
        end_time = time.time() - start_time
        print('\nStep{} takes {} sec'.format(step05, round(end_time, 2)))

        # Step06. SkullStripping TODO: multisession
        start_time = time.time()
        step06, step_path = self.init_step('SkullStripping-{}'.format(func))
        self.prj.set_filters(self.prj.ds_type[2], 'MakeStudySpecificTemplate', 'Template',
                               file_tag='_mask')
        mask = self.prj.df['Abspath'][0]
        for subject in self.prj.subjects:
            if self.prj.subjects == self.prj.sessions:
                output_path = join(step_path, subject)
                try:
                    os.mkdir(output_path)
                except:
                    pass
                self.set_filters(step05, subject)
                for i, path in enumerate(self):
                    self.run_cmd('afni_3dcalc', join(output_path, path['Filename']), 'a*b',
                                 path['Abspath'], mask)
        end_time = time.time() - start_time
        print('\nStep{} takes {} sec'.format(step06, round(end_time, 2)))

        # Step07. Temporal preprocessing TODO: This is the final step. Dont generate step07 folder, output to results
        command = 'afni_3dBandpass'
        kwargs = {'norm':True, 'despike':True,
                  'dt': '1', 'blur': '0.3', 'band': ['0.008', '0.3'],
                  'mask':mask}
        step07 = self('TemporalProcessing-{}'.format(func), command, **kwargs)
