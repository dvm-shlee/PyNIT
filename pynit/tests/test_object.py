from unittest import TestCase
from ..core.objects import Template
from shutil import rmtree


class TestObject(TestCase):
    def setUp(self):
        prj_path = './test_template/Bilateral_8slices_template.nii'
        atl_path = './test_template/Bilateral_8slices_atlas.nii'
        self.tmpobj = Template(prj_path, atl_path)

    def test_ROIlist(self):
        pass

    def tearDown(self):
        pass
