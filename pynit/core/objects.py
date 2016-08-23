# import os
import numpy as np
import nibabel as nib
import pandas as pd
from .visual import Viewer
from .utility import InternalMethods
from .process import Analysis
import error


class Reference(object):
    """Class of reference informations for image processing and data analysis
    """
    img = {'NifTi-1':           ['.nii', '.nii.gz'],
           'ANALYZE7.5':        ['.img', '.hdr'],
           'AFNI':              ['.BRIK', '.HEAD'],
           'Shihlab':           ['.sdt', '.spr']
           }
    txt = {'Common':            ['.txt', '.cvs', '.tvs'],
           'AFNI':              ['.1D'],
           'MATLAB':            ['.mat'],
           'Slicer_Transform':  ['.tfm'],
           'JSON':              ['.json']
           }
    data_structure = {'NIRAL': ['Data', 'Processing', 'Results'],
                      'BIDS': ['sourcedata', 'derivatives']
                      }

    def __init__(self, *args):
        try:
            self._img = [arg for arg in args if arg in self.img.keys()]
            self._txt = [arg for arg in args if arg in self.txt.keys()]
            self._ds = [arg for arg in args if arg in self.data_structure.keys()]
            if (len(self._img) or len(self._txt) or len(self._ds)) > 1:
                raise AttributeError
        except:
            raise AttributeError

    def __repr__(self):
        title = 'Predefined values'
        img = 'Image format:\t{}'.format(self.img.keys())
        txt = 'Text format:\t{}'.format(self.txt.keys())
        ds = 'Data structure:\t{}'.format(self.data_structure.keys())
        output = '{}\n{}\n{}\n{}\n{}'.format(title,'-'*len(title), img, txt, ds)
        return output

    @property
    def imgext(self):
        return self.img[self._img[0]]

    @property
    def txtext(self):
        return self.txt[self._txt[0]]

    @property
    def ref_ds(self):
        return self.data_structure[self._ds[0]]

    def set_img_format(self, img_format):
        if img_format in self.img.keys():
            raise AttributeError
        else:
            self._img = img_format

    def set_txt_format(self, txt_format):
        if txt_format in self.txt.keys():
            raise AttributeError
        else:
            self._txt = txt_format

    def set_ref_data_structure(self, ds_ref):
        if ds_ref in self.data_structure.keys():
            raise AttributeError
        else:
            self._ds = ds_ref


class ImageObj(nib.nifti1.Nifti1Image):
    """ ImageObject for PyNIT
    """
    # def __init__(self):
    #     super(ImageObj, self).__init__()

    def show(self, **kwargs):
        """ Plotting slice of the object
        """
        Viewer.slice(self, **kwargs)

    def mosaic(self, *args, **kwargs):
        """ Mosaic view for the object
        """
        fig = Viewer.mosaic(self, *args, **kwargs)

    def swap_axis(self, axis1, axis2):
        """ Swap input axis with given axis of the object
        """
        InternalMethods.swap_axis(self, axis1, axis2)

    def flip(self, **kwargs):
        invert = InternalMethods.check_invert(kwargs)
        self._dataobj = InternalMethods.apply_invert(self._dataobj, *invert)

    def crop(self, **kwargs):
        InternalMethods.crop(self, **kwargs)

    def reslice(self, ac_slice, ac_loc, slice_thickness, total_slice, axis=2):
        """ Reslice the image with given number of slice and slice thinkness

        :param ac_slice: int
            Slice location of anterior commissure in original image
        :param ac_loc: int
            The slice number of anterior commissure want to be in resliced image
        :param slice_thickness:
            Desired slice thickness for re-sliced image
        :param total_slice:
            Desired total number of slice for  re-sliced image
        :param axis:
            Axis want to be re-sliced
        :return:
        """
        InternalMethods.down_reslice(self, ac_slice, ac_loc, slice_thickness, total_slice, axis)

    def save_as(self, filename):
        """ Save as a new file with current affine information
        """
        nii = nib.Nifti1Image(self.dataobj, self.affine)
        nii.header['sform_code'] = 0
        nii.header['qform_code'] = 1
        nii.to_filename(filename)
        print("NifTi1 format image is saved to '{}'".format(filename))

    def padding(self, low, high, axis):
        dataobj = self._dataobj[...]
        dataobj = np.swapaxes(dataobj, axis, 2)
        shape = list(dataobj.shape[:])
        shape[2] = low
        lower_pad = np.zeros(shape)
        shape[2] = high
        higher_pad = np.zeros(shape)
        dataobj = np.concatenate((lower_pad, dataobj, higher_pad), axis=2)
        self._dataobj = np.swapaxes(dataobj, axis, 2)

    def check_reg(self, imageobj, scale=10, **kwargs):
        fig = Viewer.check_reg(imageobj, self, scale=scale, norm=True, **kwargs)

    def check_mask(self, maskobj, scale=15, **kwargs):
        fig = Viewer.check_mask(self, maskobj, scale=scale, **kwargs)

    @property
    def affine(self):
        return self._affine


class Template(object):
    """ TemplateObject for PyNIT
    """
    def __init__(self, path=None):
        self._atlas = None
        if type(path) is ImageObj:
            self._image = path
        elif type(path) is str:
            try:
                self.load(path)
            except:
                raise error.InputPathError
        else:
            raise error.InputFileError

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, imageobj):
        if type(imageobj) is ImageObj:
            self._image = imageobj
        else:
            raise error.InputObjectError

    @property
    def atlas(self):
        return self._atlas.image

    @property
    def label(self):
        return self._atlas.label

    def load(self, path):
        """ Import template

        :param path:
        :return:
        """
        self._image = ImageObj.load(path)

    def set_atlas(self, path):
        self._atlas = Atlas(path)

    def swap_axis(self, axis1, axis2):
        if self._atlas:
            self.atlas.swap_axis(axis1, axis2)
        self.image.swap_axis(axis1, axis2)

    def flip(self, **kwargs):
        if self._atlas:
            self.atlas.flip(**kwargs)
        self.image.flip(**kwargs)

    def crop(self, **kwargs):
        if self._atlas:
            self.atlas.crop(**kwargs)
        self.image.crop(**kwargs)

    def reslice(self, ac_slice, ac_loc, slice_thickness, total_slice, axis=2):
        if self._atlas:
            self.atlas.reslice(self, ac_slice, ac_loc, slice_thickness, total_slice, axis=axis)
        self.image.reslice(self, ac_slice, ac_loc, slice_thickness, total_slice, axis=axis)

    def show(self, scale=15, **kwargs):
        if self._atlas:
            Viewer.atlas(self.image, self._atlas, scale=scale, **kwargs)
        else:
            Viewer.mosaic(self.image, scale=scale, **kwargs)

    def save_as(self, filename):
        self.image.save_as('{}_template.nii'.format(filename))
        self._atlas.save_as('{}_atlas'.format(filename))

    def __getitem__(self, idx):
        return self._atlas.__getitem__(idx)

    def __repr__(self):
        return self._atlas.__repr__()


class Atlas(object):
    def __init__(self, path=None):
        self._atlas = None
        self._label = None
        if type(path) is ImageObj:
            self._image = path
        elif type(path) is str:
            try:
                self.load(path)
            except:
                raise error.InputPathError
        else:
            raise error.InputFileError

    @property
    def image(self):
        return self._image

    @property
    def label(self):
        return self._label

    @image.setter
    def image(self, imageobj):
        if type(imageobj) is ImageObj:
            self._image = imageobj
        else:
            raise error.InputObjectError

    def load(self, path):
        self._image, self._label = InternalMethods.parsing_atlas(path)

    def save_as(self, filename):
        self._image.save_as("{}.nii".format(filename))
        InternalMethods.save_label(self._label, "{}.label".format(filename))

    def __getitem__(self, idx):
        if not self._image:
            return None
        else:
            if idx != 0:
                mask = np.zeros(self.image.shape)
                mask[self.image.get_data() == idx] = 1
                maskobj = ImageObj(mask, self.image.affine)
                return self.label[idx][0], maskobj
            else:
                return None

    def __repr__(self):
        labels = None
        for idx in self.label.keys():
            if not idx:
                pass
                labels = '[{:>3}] {:>40}\n'.format(idx, self.label[idx][0])
            else:
                labels = '{}[{:>3}] {:>40}\n'.format(labels, idx, self.label[idx][0])
        return labels


class GroupImages(object):
    """ Group handler for multiple but same sized of Image objects
    """
    def __init__(self):
        self._container = dict()
        self._panel = pd.Panel()

    def __setitem__(self, key, value):
        if self._container:
            natives = [obj for obj in self._container.values()]
            if not isinstance(natives[0], value):
                raise error.ObjectMismatch
            else:
                if isinstance(value, ImageObj):
                    if value.shape != natives[0].shape:
                        raise error.ObjectMismatch
                else:
                    pass
        self._container[key] = value

    def __getitem__(self, key):
        return self._container[key]

    @property
    def timetraces(self):
        return self._panel

    def collect_timetrace(self, tempobj, **kwargs):
        for sub, imageobj in self._container.iteritems():
            self._panel[sub] = Analysis.get_timetrace(imageobj, tempobj, **kwargs)


