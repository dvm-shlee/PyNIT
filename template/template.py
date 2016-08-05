import nibabel as nib
from ..core.visualization import Image


class Template(object):
    """Template tools
    """
    def __init__(self, path, roi=None):
        self.__path = path
        self.__img = nib.load(self.__path)

    @property
    def data(self):
        return self.__img.get_data()

    @property
    def affine(self):
        return self.__img.affine()

    @property
    def roi(self):
        return self.__roi

    def avail(self):
        pass

    def set_template(self):
        pass

    def set_roi(self, roi):
        # If the given parameter roi is pre-defined, use it,
        # else if it is the filepath, import it
        pass

    def show(self, scale=10, **kwargs):
        return Image.mosaic(self.__img, scale, **kwargs)

    def atlas(self, scale=10, **kwargs):
        return Image.atlas(self.__img, self.roi, scale, **kwargs)

    def reslice(self, ac_slice, total_slice):
        pass

