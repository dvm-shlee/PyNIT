from base import *
from pynit.handler.step import Step

class NSP_Process(BaseProcess):

    def nsp_SeedPCCorr(self, func, tmpobj, index,
                       surfix='func', n_thread='max', debug=False):
        """
        """
        func = self.check_input(func)
        if isinstance(index, list) is not True:
            raise Exception
        step = Step(self, n_thread=n_thread)
        step.set_message('** Processing seed PC based correlation analysis.....')
        step.set_input(name='func', path=func)
        step.set_var(name='template', value=tmpobj.template_path, type=1)
        step.set_var(name='index', value="'{}'".format(" ".join(map(str, index))), type=1)
        step.set_var(name='atlas', value=tmpobj.atlas_obj.get_filename(), type=1)
        step.set_output(name='output', type=0)
        cmd = "pynsp seedpc -i {func} -o {output} -t {template} -a {atlas} -d {index}"
        step.set_cmd(cmd)
        output_path = step.run('SeedPCCorr', surfix, debug=debug)
        return dict(corr=output_path)

    def nsp_SeedCorr(self, func, tmpobj, index,
                     surfix='func', n_thread='max', debug=False):
        """
        """
        func = self.check_input(func)
        if isinstance(index, list) is not True:
            raise Exception
        step = Step(self, n_thread=n_thread)
        step.set_message('** Processing seed-based correlation analysis.....')
        step.set_input(name='func', path=func)
        step.set_var(name='template', value=tmpobj.template_path, type=1)
        step.set_var(name='index', value="'{}'".format(" ".join(map(str, index))), type=1)
        step.set_var(name='atlas', value=tmpobj.atlas_obj.get_filename(), type=1)
        step.set_output(name='output', type=0)
        cmd = "pynsp seed -i {func} -o {output} -t {template} -a {atlas} -d {index}"
        step.set_cmd(cmd)
        output_path = step.run('SeedCorr', surfix, debug=debug)
        return dict(corr=output_path)

    def nsp_SignalProcessing(self, func, mask, dt, ort=None, ort_filter=None, band=None,
                             surfix='func', n_thread='max', debug=False):
        """

        :param func:
        :param tmpobj:
        :param dt:
        :param ort:
        :param band:
        :param n_thread:
        :param debug:
        :return:
        """
        func = self.check_input(func)
        step = Step(self, n_thread=n_thread)
        step.set_message('** Run signal processing for resting state data')
        if ort_filter is None:
            ort_filter = {'ext': '.1D', 'ignore': ['.aff12']}
        if ort is not None:
            ort = self.check_input(ort)
            step.set_input(name='ort', path=ort, filters=ort_filter, type=1)
            cmd = 'pynsp nuisance -i {func} -m {mask} --dt {dt} --ort {ort} --alff {band} -o {output}'
        else:
            cmd = 'pynsp nuisance -i {func} -m {mask} --dt {dt} --alff {band} -o {output}'
        step.set_input(name='func', path=func)
        step.set_var(name='mask', value=mask)
        step.set_var(name='dt', value=dt)
        step.set_var(name='band', value="'{} {}'".format(*band))
        step.set_output(name='output', ext='remove')
        step.set_cmd(cmd)
        output_path = step.run('SignalProcessing', surfix, debug=debug)
        return dict(func=output_path)