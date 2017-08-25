from base import *
from pynit.handler.step import Step

class PN_Process(BaseProcess):
    def __init__(self, *args, **kwargs):
        super(PN_Process, self).__init__(*args, **kwargs)

    def pn_MaskPrep(self, anat, tmpobj, func=None, surfix='func'): #TODO: Need to develop own skullstrip alrorithm
        """

        :param anat:
        :return:
        """
        display(title(value='** Processing mask image preparation.....'))
        anat = self.check_input(anat)
        step = Step(self)
        mimg_path = None
        try:
            step.set_input(name='anat', path=anat, idx=0)
        except:
            methods.raiseerror(messages.Errors.MissingPipeline,
                               'No anatomy file!')
        try:
            step.set_var(name='mask', value=str(tmpobj.mask))
        except:
            methods.raiseerror(messages.InputPathError,
                               'No mask template file!')
        cmd01 = "mask_prep 10 {anat} {temp1}"
        cmd02 = 'mask_reg -o {output} -f {temp1} -m {mask} -n 1'
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
                step.set_input(name='func', path=mimg_path, filters=mimg_filters, type=True)
            else:
                step.set_input(name='func', path=mimg_path, type=True)
        except:
            methods.raiseerror(messages.Errors.MissingPipeline,
                               'Initial Mean image calculation step has not been executed!')
        try:
            step.set_staticinput(name='mask', value=str(tmpobj.mask))
        except:
            methods.raiseerror(messages.InputPathError,
                               'No mask template file!')
        cmd01 = "mask_prep -f 10 {func} {temp1}"
        cmd02 = 'mask_reg -o {output} -f {temp1} -m {mask} -n 1'
        step.set_command(cmd01)
        step.set_command(cmd02)
        func_mask = step.run('MaskPrep', surfix)
        if jupyter_env:
            if self._viewer == 'itksnap':
                display(widgets.VBox([title(value='-' * 43 + ' Anatomical images ' + '-' * 43),
                                      gui.itksnap(self, anat_mask, anat),
                                      title(value='<br>' + '-' * 43 + ' Functional images ' + '-' * 43),
                                      gui.itksnap(self, func_mask, mimg_path)]))
            elif self._viewer == 'fslview':
                display(widgets.VBox([title(value='-' * 43 + ' Anatomical images ' + '-' * 43),
                                      gui.fslview(self, anat_mask, anat),
                                      title(value='<br>' + '-' * 43 + ' Functional images ' + '-' * 43),
                                      gui.fslview(self, func_mask, mimg_path)]))
            else:
                methods.raiseerror(messages.Errors.InputValueError,
                                   '"{}" is not available'.format(self._viewer))
        else:
            return dict(anat_mask=anat_mask, func_mask=func_mask)

