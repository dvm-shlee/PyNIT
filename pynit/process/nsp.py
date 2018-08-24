from base import *
from pynit.handler.step import Step

class NSP_Process(BaseProcess):

    def nsp_SignalProcessing(self, func, mask, dt, param=None, param_filter=None, polort=3,
                             band=None, surfix='func', n_thread='max', debug=False):
        """

        :param func:
        :param tmpobj:
        :param dt:
        :param param:
        :param polort:
        :param band:
        :param n_thread:
        :param debug:
        :return:
        """
        func = self.check_input(func)
        step = Step(self, n_thread=n_thread)
        step.set_message('** Run signal processing for resting state data')
        if param_filter is None:
            param_filter = {'ext': '.1D', 'ignore': ['.aff12']}
        if param is not None:
            param = self.check_input(param)
            step.set_input(name='param', path=param, filters=param_filter, type=1)
            cmd = 'pynsp nuisance -i {func} -m {mask} --dt {dt} -p {param} -b {band} --polort {polort} -o {output}'
        else:
            cmd = 'pynsp nuisance -i {func} -m {mask} --dt {dt} -b {band} --polort {polort} -o {output}'
        step.set_input(name='func', path=func)
        step.set_var(name='mask', value=mask)
        step.set_var(name='dt', value=dt)
        step.set_var(name='polort', value=polort)
        step.set_var(name='band', value="'{} {}'".format(*band))
        step.set_output(name='output', ext='remove')
        step.set_cmd(cmd)
        output_path = step.run('SignalProcessing', surfix, debug=debug)
        return dict(func=output_path)

    def nsp_ROIbasedConnectivity(self, func, tmpobj, mask=None, use_PCA=True, FDR=False,
                                 surfix='func', n_thread='max', debug=False):
        """

        :param func:
        :param tmpobj:
        :param mask:
        :param use_PCA:
        :param FDR:
        :param surfix:
        :param n_thread:
        :param debug:
        :return:
        """
        func = self.check_input(func)
        step = Step(self, n_thread=n_thread)
        step.set_message('** Estimate ROI-base Correlation Coefficient')
        step.set_input(name='func', path=func)
        step.set_var(name='atlas', value=tmpobj.atlas_path)
        step.set_var(name='label', value='{}.label'.format(methods.splitnifti(str(tmpobj.atlas_path))))
        if mask is not None:
            step.set_var(name='mask', value=mask)
        else:
            step.set_var(name='mask', value=tmpobj.mask.path)
        step.set_output(name='output', ext='remove')
        cmd = 'pynsp roi-conn -i {func} -a {atlas} -l {label} -m {mask} -o {output}'
        if use_PCA is True:
            cmd = '{} --PCA'.format(cmd)
        if FDR is True:
            cmd = '{} --FDR'.format(cmd)
        step.set_cmd(cmd)
        output_path = step.run('ROIbasedConnectivity', surfix, debug=debug)
        return dict(qt=output_path)

    def nsp_ReHo(self, func, mask, NN=3, surfix='func', n_thread='max', debug=False):
        """

        :param func:
        :param mask:
        :param NN:
        :param surfix:
        :param n_thread:
        :param debug:
        :return:
        """
        func = self.check_input(func)
        step = Step(self, n_thread=n_thread)
        step.set_message('** Calculate ReHo')
        step.set_input(name='func', path=func)
        step.set_var(name='mask', value=mask)
        step.set_var(name='NN', value=NN)
        step.set_output(name='output', ext='remove')
        cmd = 'pynsp reho -i {func} -n {NN} -m {mask} -o {output}'
        step.set_cmd(cmd)
        output_path = step.run('ReHo', surfix, debug=debug)
        return dict(reho=output_path)

    def nsp_ALFF(self, func, mask, dt, band, surfix='func', n_thread='max', debug=False):
        """

        :param func:
        :param mask:
        :param dt:
        :param band:
        :param surfix:
        :param n_thread:
        :param debug:
        :return:
        """
        func = self.check_input(func)
        step = Step(self, n_thread=n_thread)
        step.set_message('** Calculate ALFF')
        step.set_input(name='func', path=func)
        step.set_var(name='mask', value=mask)
        step.set_var(name='dt', value=dt)
        step.set_var(name='band', value="'{} {}'".format(*band))
        step.set_output(name='output', ext='remove')
        cmd = 'pynsp alff -i {func} -b {band} -t {dt} -m {mask} -o {output}'
        step.set_cmd(cmd)
        output_path = step.run('ALFF', surfix, debug=debug)
        return dict(alff=output_path)

    def nsp_QualityControl(self, func, mparam, mask, surfix='func', n_thread='max', debug=False):
        """

        :param func:
        :param mparam:
        :param mask:
        :param surfix:
        :param n_thread:
        :param debug:
        :return:
        """
        func = self.check_input(func)
        mparam = self.check_input(mparam)
        param_filter = {'ext': '.1D', 'ignore': ['.aff12']}
        step = Step(self, n_thread=n_thread)
        step.set_message('** Calculate QC parameters')
        step.set_input(name='func', path=func)
        step.set_var(name='mask', value=mask)
        step.set_input(name='mparam', path=mparam, filters=param_filter, type=1)
        step.set_output(name='output', ext='remove')
        cmd = 'pynsp qc -i {func} -p {mparam} -m {mask} -o {output}'
        step.set_cmd(cmd)
        output_path = step.run('QualityControl', surfix, debug=debug)
        return dict(qt=output_path)
