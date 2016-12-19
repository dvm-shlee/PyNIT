from unittest import TestCase
from ..core.handlers import Project, Process
from shutil import rmtree


class TestMethods(TestCase):
    def setUp(self):
        self.prj = Project('test_prj')
        self.proc = Process(self.prj, 'TestProcess')

    def tearDown(self):
        rmtree('test_prj/Processing/TestProcess')
        rmtree('test_prj/Results/TestProcess', ignore_errors=True)
