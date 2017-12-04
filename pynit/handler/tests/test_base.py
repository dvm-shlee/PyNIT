from __future__ import absolute_import
import unittest
from pynit.handler.base import BaseProcessor
from pynit import Project, Process

testpath = '/Users/shlee419/DataFiles/2_Testing/171101_TestProject'

class TestBaseProcessor(unittest.TestCase):

    def setUp(self):
        self.prc = Process(Project(testpath), 'TestProcess')

    def test_1_DMIFO(self):  # Dynamic Major Inputs, File Outputs
        print('{0} Testing case of Dynamic Inputs, and File Outputs {0}'.format('*-'*10))
        step = BaseProcessor(self.prc)
        step.set_input(name='test_input', path='func', type=0)
        step.set_input(name='test_side', path='anat', type=1)
        step.set_output(name='test_output', dc=0)
        step.set_cmd('3dcopy {test_input} {test_side} {test_output}')
        print(step.build_func(name='test')+'\n\n')
        # self.assertEqual()

    def test_2_SMIFO(self):  # Static Major Inputs, File Outputs
        print('{0} Testing case of Static Inputs, File Outputs {0}'.format('*-' * 10))
        step = BaseProcessor(self.prc)
        step.set_input(name='test_input', path='func', idx=0, type=0)
        step.set_output(name='test_output', dc=0)
        step.set_cmd('3dcopy {test_input} {test_output}')
        print(step.build_func(name='test')+'\n\n')
        # self.assertEqual()

    def test_3_DMIPO(self):  # Dynamic Major Inputs, Prefix-only Outputs
        print('{0} Testing case of Dynamic Inputs, Prefix-only Outputs {0}'.format('*-' * 10))
        step = BaseProcessor(self.prc)
        step.set_input(name='test_input', path='func', type=0)
        step.set_output(name='test_output', dc=0, ext='remove')
        step.set_cmd('3dcopy {test_input} {test_output}')
        print(step.build_func(name='test')+'\n\n')
        # self.assertEqual()

    def test_4_DMIBFOL0(self):    # Dynamic Major Inputs, Base Folder Outputs at level 0
        print('{0} Testing case of Dynamic Major Inputs, Base Folder Outputs {0}'.format('*-' * 10))
        step = BaseProcessor(self.prc)
        step.set_input(name='test_input', path='func', type=0)
        step.set_output(name='test_output', dc=1, type=1)
        step.set_cmd('3dcopy {test_input} {test_output}.')
        print(step.build_func(name='test') + '\n\n')

    def test_5_DMIBFOL1(self):    # Dynamic Major Inputs, Base Folder Outputs at level 1
        print('{0} Testing case of Dynamic Major Inputs, Base Folder Outputs {0} at level 1'.format('*-' * 10))
        step = BaseProcessor(self.prc)
        step.set_input(name='test_input', path='func', type=0)
        step.set_output(name='test_output', dc=1, type=1, level=1)
        step.set_cmd('3dcopy {test_input} {test_output}')
        print(step.build_func(name='test') + '\n\n')

    def test_6_DMIBFOL1(self):    # Dynamic Major Inputs, Base Folder Outputs at level 2
        print('{0} Testing case of Dynamic Major Inputs, Base Folder Outputs {0} at level 2'.format('*-' * 10))
        step = BaseProcessor(self.prc)
        step.set_input(name='test_input', path='func', type=0)
        step.set_output(name='test_output', dc=1, type=1, level=2)
        step.set_cmd('3dcopy {test_input} {test_output}')
        print(step.build_func(name='test') + '\n\n')

    def test_7_DMINFO(self):    # Dynamic Major Inputs, New Folder Outputs at level 0
        print('{0} Testing case of Dynamic Major Inputs, New Folder Outputs at level 0 {0}'.format('*-' * 10))
        step = BaseProcessor(self.prc)
        step.set_input(name='test_input', path='func', type=0)
        step.set_output(name='test_output', dc=1, type=2, prefix='test')
        step.set_cmd('3dcopy {test_input} {test_output}')
        print(step.build_func(name='test') + '\n\n')

    def test_8_DMINFO(self):    # Dynamic Major Inputs, New Folder Outputs at level 1
        print('{0} Testing case of Dynamic Major Inputs, New Folder Outputs at level 1 {0}'.format('*-' * 10))
        step = BaseProcessor(self.prc)
        step.set_input(name='test_input', path='func', type=0)
        step.set_output(name='test_output', dc=1, type=2, prefix='test', level=1)
        step.set_cmd('3dcopy {test_input} {test_output}')
        print(step.build_func(name='test') + '\n\n')

    def test_9_DMINFO(self):    # Dynamic Major Inputs, New Filename Outputs
        print('{0} Testing case of Dynamic Major Inputs, New Filename Outputs {0}'.format('*-' * 10))
        step = BaseProcessor(self.prc)
        step.set_input(name='test_input', path='func', type=0)
        step.set_output(name='test_output', dc=1, type=2, prefix='test', ext='remove')
        step.set_cmd('3dcopy {test_input} {test_output}')
        print(step.build_func(name='test') + '\n\n')

    def test_0_1_GIBFO(self):    # Group Inputs, Base Folder Outputs
        print('{0} Testing case of Group Inputs, Base Folder Outputs {0}'.format('*-' * 10))
        step = BaseProcessor(self.prc)
        step.set_input(name='test_group1', path='func', filters=dict(subj=['a', 'b', 'c']), type=3, kwargs=dict(test='123'))
        step.set_input(name='test_group2', path='func', filters=dict(subj=['d', 'e', 'f']), type=3)
        step.set_output(name='test_output', dc=1, type=1, prefix='test')
        step.set_cmd('3dcopy {test_group1} {test_group1_test} {test_group2} {test_output}')
        print(step.build_func(name='test') + '\n\n')

    def test_0_2_GIBFO(self):    # Group Inputs, New Folder Outputs
        print('{0} Testing case of Group Inputs, New Folder Outputs {0}'.format('*-' * 10))
        step = BaseProcessor(self.prc)
        step.set_input(name='test_group1', path='func', filters=dict(subj=['a', 'b', 'c']), type=3, kwargs=dict(test='123'))
        step.set_input(name='test_group2', path='func', filters=dict(subj=['d', 'e', 'f']), type=3)
        step.set_output(name='test_output', dc=1, type=2, prefix='test')
        step.set_cmd('3dcopy {test_group1} {test_group1_test} {test_group2} {test_output}')
        print(step.build_func(name='test') + '\n\n')

    def test_0_3_GIBFO(self):    # Group Inputs, New Folder Outputs
        print('{0} Testing case of Group Inputs, New Folder Outputs {0}'.format('*-' * 10))
        step = BaseProcessor(self.prc)
        step.set_input(name='test_group1', path='func', filters=dict(subj=['a', 'b', 'c']), type=3, kwargs=dict(test='123'))
        step.set_input(name='test_group2', path='func', filters=dict(subj=['d', 'e', 'f']), type=3)
        step.set_output(name='test_output', dc=1, type=2, prefix='test', ext='remove')
        step.set_cmd('3dcopy {test_group1} {test_group1_test} {test_group2} {test_output}')
        print(step.build_func(name='test') + '\n\n')

    def test_0_4_GIBFO(self):    # Group Inputs, New File Outputs
        print('{0} Testing case of Group Inputs, New File Outputs with extension {0}'.format('*-' * 10))
        step = BaseProcessor(self.prc)
        step.set_input(name='test_group1', path='func', filters=dict(subj=['a', 'b', 'c']), type=3, kwargs=dict(test='123'))
        step.set_input(name='test_group2', path='func', filters=dict(subj=['d', 'e', 'f']), type=3)
        step.set_output(name='test_output', dc=1, type=2, prefix='test', ext='nii.gz')
        step.set_cmd('3dcopy {test_group1} {test_group1_test} {test_group2} {test_output}')
        print(step.build_func(name='test') + '\n\n')

    def test_10_MIBFO(self):    # Group Inputs, New File Outputs
        print('{0} Testing case of Multi Inputs, New File Outputs with extension {0}'.format('*-' * 10))
        step = BaseProcessor(self.prc)
        step.set_input(name='func', path='func', type=2)
        step.set_output(name='test_output', dc=1, type=1, level=1,
                        prefix='test', ext='nii.gz')
        step.set_cmd('3dcopy {func} {test_output}')
        print(step.build_func(name='test') + '\n\n')

    def tearDown(self):
        # This method will be run ehether runTest() succeeded or not
        # clean up the other stuff
        # close the connection to the database
        # remove the temporary directory and all of its contents
        return

if __name__ == '__main__':
    unittest.main()