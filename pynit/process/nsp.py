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
