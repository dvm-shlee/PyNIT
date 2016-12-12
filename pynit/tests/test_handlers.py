from ..core.handlers import Project, Process, Step
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

    def test_filtering(self):
        main_filters = {'file_tag':'file1', 'ext':'nii.gz'}
        self.step.set_input(name='t2data', input_path='anat', filters=main_filters)
        side_filters = {'file_tag':'file3', 'ext':'nii.gz'}
        self.step.set_input(name='sidet2', input_path='anat', filters=side_filters, side=True, static=True)
        command = "3dcalc -prefix {output} -expr 'a*step(b)' -a {t2data} -b {sidet2}"
        self.step.set_command(command)
        self.assertEqual(isinstance(self.step.get_inputcode(), list), True, 'Error on Dataset filtering')

    def test_more(self):
        main_filters = {'file_tag':'file2', 'ext':'nii.gz'}
        self.step.set_input(name='func1', input_path='func', filters=main_filters)
        side_filters = {'file_tag':'file4', 'ext':'nii.gz'}
        self.step.set_input(name='func2', input_path='func', filters=side_filters, side=True, static=True)
        command_1 = "3dcalc -prefix {temp_01} -expr 'a+b' -a {func1} -b {func2}"
        self.step.set_command(command_1)
        command_2 = "3dcalc -prefix {temp_02} -expr '-a' -a {temp_01}"
        self.step.set_command(command_2)
        command_3 = "3dcalc -prefix {output} -expr 'a*2' -a {temp_02}"
        self.step.set_command(command_3)
        print(self.step.get_executefunc('test'))
        self.assertEqual(isinstance(self.step.get_inputcode(), list), True, 'Error on Dataset filtering')

    def tearDown(self):
        rmtree('test_prj/Processing/TestProcess', ignore_errors=True)
        rmtree('test_prj/Results/TestProcess', ignore_errors=True)
