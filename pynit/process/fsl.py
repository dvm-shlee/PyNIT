from base import *
from pynit.handler.step import Step


class FSL_Process(BaseProcess):
    # def __init__(self, *args, **kwargs):
    #     super(FSL_Process, self).__init__(*args, **kwargs)

    def fsl_IndividualICA(self, func, tr=2.0, alpha=0.5, bgthreshold=10, surfix='func'):
        func = self.check_input(func)
        step = Step(self, n_thread=1)
        step.set_input(name='func', path=func)
        step.set_output(name='sub_path', type=1, dc=1)
        cmd = ['melodic -i {func} -o {sub_path}', '--tr={}'.format(tr), '--mmthresh={}'.format(alpha),
               '--bgthreshold={} --nobet --nomask'.format(bgthreshold)]
        step.set_cmd(' '.join(cmd))
        func_path = step.run('IndividualICA', 'func')
        return dict(func=func_path)

    def fsl_BiasFieldCalculation(self, anat, func, n_class=3, smoothing=2, image_type=2, debug=False):
        anat = self.check_input(anat)
        func = self.check_input(func)
        step1 = Step(self)
        step2 = Step(self)
        step1.set_input(name='anat', path=anat, type=True)
        step1.set_output(name='prefix', ext='remove')
        step2.set_input(name='func', path=func)
        step2.set_output(name='prefix', ext='remove')
        cmd1 = ['fast --class={} --lowpass={} --type={} -b'.format(n_class, smoothing, image_type),
                '--out={prefix} {anat}']
        cmd2 = ['fast --class={} --lowpass={} --type={} -b'.format(n_class, smoothing, image_type),
                '--out={prefix} {func}']
        step1.set_cmd(' '.join(cmd1))
        step2.set_cmd(' '.join(cmd2))
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
        step1.set_input(name='anat', path=anat, type=True)
        step2.set_input(name='func', path=func)
        step1.set_input(name='anat_bias', path=anat_bias, filters=dict(file_tag='_bias'), type=1, idx=0)
        step2.set_input(name='func_bias', path=func_bias, filters=dict(file_tag='_bias'), type=1)
        step1.set_output(name='output')
        step2.set_output(name='output')
        cmd1 = '3dcalc -prefix {output} -expr "a/b" -a {anat} -b {anat_bias}'
        cmd2 = '3dcalc -prefix {output} -expr "a/b" -a {func} -b {func_bias}'
        step1.set_cmd(cmd1)
        step2.set_cmd(cmd2)
        anat_path = step1.run('BiasFieldCorrection', 'anat', debug=debug)
        func_path = step2.run('BiasFieldCorrection', 'func', debug=debug)
        return dict(anat=anat_path, func=func_path)

    # def fsl_DualRegression(self, func, surfix='func'): #TODO: Implant DualRegression
    #     pass