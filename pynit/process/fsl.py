from base import *

class FSL_Process(Process):
    def __init__(self, *args, **kwargs):
        super(FSL_Process, self).__init__(*args, **kwargs)

    def ants_MotionCorrection(self, func, surfix='func', debug=False):
        display(title(value='** Extracting time-course data from ROIs'))
        func = self.check_input(func)
        step = Step(self)
        step.set_input(name='func', input_path=func, static=False)
        cmd01 = "antsMotionCorr -d 3 -a {func} -o {prefix}-avg.nii.gz"
        cmd02 = "antsMotionCorr -d 3 -o [{prefix},{prefix}.nii.gz,{prefix}-avg.nii.gz] " \
                "-m gc[ {prefix}-avg.nii.gz ,{func}, 1, 1, Random, 0.05  ] -t Affine[ 0.005 ] " \
                "-i 20 -u 1 -e 1 -s 0 -f 1 -n 10"
        step.set_command(cmd01)
        step.set_command(cmd02)
        output_path = step.run('MotionCorrection', surfix=surfix, debug=debug)
        return dict(func=output_path)

    def ants_BiasFieldCorrection(self, anat, func):
        anat = self.check_input(anat)
        func = self.check_input(func)
        step1 = Step(self)
        step2 = Step(self)
        filters = dict(file_tag='_mask')
        step1.set_input(name='anat', input_path=anat, static=True)
        step2.set_input(name='func', input_path=func, static=True)
        cmd1 = 'N4BiasFieldCorrection -i {anat} -o {output}'
        cmd2 = 'N4BiasFieldCorrection -i {func} -o {output}'
        step1.set_command(cmd1)
        step2.set_command(cmd2)
        anat_path = step1.run('BiasFiled', 'anat')
        func_path = step2.run('BiasField', 'func')
        return dict(anat=anat_path, func=func_path)

    def fsl_IndividualICA(self, func, tr=2.0, alpha=0.5, bgthreshold=10, surfix='func'):
        func = self.check_input(func)
        step = Step(self)
        step.set_input(name='func', input_path=func)
        cmd = ['melodic -i {func} -o {sub_path}', '--tr={}'.format(tr), '--mmthresh={}'.format(alpha),
               '--bgthreshold={} --nobet --nomask'.format(bgthreshold)]
        step.set_command(' '.join(cmd))
        func_path = step.run('IndividualICA', 'func')
        return dict(func=func_path)

    def fsl_BiasFieldCalculation(self, anat, func, n_class=3, smoothing=2, image_type=2, debug=False):
        anat = self.check_input(anat)
        func = self.check_input(func)
        step1 = Step(self)
        step2 = Step(self)
        step1.set_input(name='anat', input_path=anat, static=True)
        step2.set_input(name='func', input_path=func)
        cmd1 = ['fast --class={} --lowpass={} --type={} -b'.format(n_class, smoothing, image_type),
                '--out={prefix} {anat}']
        cmd2 = ['fast --class={} --lowpass={} --type={} -b'.format(n_class, smoothing, image_type),
                '--out={prefix} {func}']
        step1.set_command(' '.join(cmd1))
        step2.set_command(' '.join(cmd2))
        anat_path = step1.run('BiasFieldCalculation', 'anat', debug=debug)
        func_path = step2.run('BiasFieldCalculation', 'func', debug=debug)
        return dict(anat=anat_path, func=func_path)

    def fsl_BiasFieldCorrection(self, anat, anat_bias, func, func_bias, debug=False):
        anat = self.check_input(anat)
        anat_bias = self.check_input(anat_bias)
        func = self.check_input(func)
        func_bias = self.check_input(func_bias)
        step1 = Step(self)
        step2 = Step(self)
        step1.set_input(name='anat', input_path=anat, static=True)
        step2.set_input(name='func', input_path=func)
        step1.set_input(name='anat_bias', input_path=anat_bias, static=True, filters=dict(file_tag='_bias'), side=True)
        step2.set_input(name='func_bias', input_path=func_bias, filters=dict(file_tag='_bias'), side=True)
        cmd1 = '3dcalc -prefix {output} -expr "a/b" -a {anat} -b {anat_bias}'
        cmd2 = '3dcalc -prefix {output} -expr "a/b" -a {func} -b {func_bias}'
        step1.set_command(cmd1)
        step2.set_command(cmd2)
        anat_path = step1.run('BiasFieldCorrection', 'anat', debug=debug)
        func_path = step2.run('BiasFieldCorrection', 'func', debug=debug)
        return dict(anat=anat_path, func=func_path)

    def fsl_DualRegression(self, func, surfix='func'): #TODO: Implant DualRegression
        pass