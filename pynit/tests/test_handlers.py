from ..core.handlers import Project, Process, Step
from ..core import methods
from unittest import TestCase
from shutil import rmtree
import os

# @ut.skip("Skip this class")
class ProjectTest(TestCase):
    def setUp(self):
        self.prj = Project('test_prj')

    def tearDown(self):
        # This method will be run ehether runTest() succeeded or not
        # clean up the other stuff
        # close the connection to the database
        # remove the temporary directory and all of its contents
        return

    # @ut.expectedFailure
    def test_init(self):
        print('Testing project class...')
        self.assertEqual(len(self.prj), 4, 'Error on parsing Data folder')
        self.assertEqual(len(self.prj(1)), 0, 'Error on parsing Processing folder')
        self.assertEqual(len(self.prj(1, ext='.1D')), 0, 'Error on parsing different file extension')

# @ut.skip("Skip this class")
class ProcessTest(TestCase):
    def setUp(self):
        self.prj = Project('./test_prj')
        self.proc = Process(self.prj, 'TestProcess')

    def test_init_step(self):
        path = os.path.join(self.prj.path, self.prj.ds_type[1], 'TestProcess')
        print('Testing initiation of step...')
        step_path = self.proc.init_step('TestStep').split(os.sep)[-1]
        listdir = os.listdir(path)
        self.assertEqual( step_path in listdir, True, 'Error on Step initiation')

    def tearDown(self):
        rmtree('test_prj/Processing/TestProcess')
        rmtree('test_prj/Results/TestProcess', ignore_errors=True)


class StepTest(TestCase):
    def setUp(self):
        self.prj = Project('./test_prj')
        self.proc = Process(self.prj, 'TestProcess')
        self.step = Step(self.proc)

    def test_GenerateExecuteFunction(self):
        main_filters = {'file_tag':'file2', 'ext':'nii.gz'}
        self.step.set_input(name='func1', input_path='func', filters=main_filters)
        side_filters = {'file_tag':'file4', 'ext':'nii.gz'}
        self.step.set_input(name='func2', input_path='func', filters=side_filters, side=True, static=True)
        self.assertEqual(isinstance(self.step.get_inputcode(), list), True, 'Error on Dataset filtering')
        self.step.set_outparam(name='mparam', ext='.1D')
        self.step.set_staticinput('times', 'times')
        command_0 = '3dinfo -nv {func1}'
        self.step.set_command(command_0, stdout='times')
        command_1 = '3dcalc -prefix {temp_01} -expr "a+b+c" -a {func1} -b {func2} -c {times}'
        self.step.set_command(command_1)
        command_2 = '3dcalc -prefix {temp_02} -expr "-a" -a {temp_01}'
        self.step.set_command(command_2)
        command_3 = '3dcalc -prefix {output} -expr "a*2" -a {temp_02} > {mparam}'
        self.step.set_command(command_3)

        self.assertEqual(self.step.get_executefunc('test', verbose=True), None, 'Error on step function generation')

    def tearDown(self):
        rmtree('test_prj/Processing/TestProcess', ignore_errors=True)
        rmtree('test_prj/Results/TestProcess', ignore_errors=True)
