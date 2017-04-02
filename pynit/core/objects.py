import os, sys
import numpy as np
from nibabel import Nifti1Image

from .visualizers import Viewer
from .processors import TempFile
import messages
import methods
import pipelines
try:
    if len([key for key in sys.modules.keys() if key == 'ipykernel']):
        from ipywidgets import widgets
        from .visualizers import display
        jupyter_env = True
    else:
        jupyter_env = False
except:
    pass


class Reference(object):
    """Class of reference informations for image processing and data analysis
    """
    img = {'NifTi-1':           ['.nii', '.nii.gz'],
           'ANALYZE7.5':        ['.img', '.hdr'],
           'AFNI':              ['.BRIK', '.HEAD'],
           'Shihlab':           ['.sdt', '.spr'],
           'Nrrd':              ['.nrrd', '.nrdh']
           }
    txt = {'Common':            ['.txt', '.cvs', '.tvs'],
           'Mictosoft':         ['.xlsx', '.xls'],
           'AFNI':              ['.1D'],
           'MATLAB':            ['.mat'],
           'Slicer_Transform':  ['.tfm'],
           'JSON':              ['.json']
           }
    data_structure = {'NIRAL': ['Data', 'Processing', 'Results'],
                      }
    pipelines = [pipe for pipe in dir(pipelines) if 'PipeTemplate' not in pipe if '__' not in pipe]

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
        pipe = 'Available pipelines:\t{}'.format(self.pipelines)
        output = '{}\n{}\n{}\n{}\n{}\n{}'.format(title,'-'*len(title), img, txt, ds, pipe)
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

    @property
    def avail(self):
        n_pipe = len(self.pipelines)
        output = dict(zip(range(n_pipe), self.pipelines))
        # for i, values in output.items():
        #     output[i] = values[2:]
        return output

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


class ImageObj(Nifti1Image):
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
        methods.swap_axis(self, axis1, axis2)

    def flip(self, **kwargs):
        invert = methods.check_invert(kwargs)
        self._dataobj = methods.apply_invert(self._dataobj, *invert)

    def crop(self, **kwargs):
        methods.crop(self, **kwargs)

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
        methods.down_reslice(self, ac_slice, ac_loc, slice_thickness, total_slice, axis)

    def save_as(self, filename, quiet=False):
        """ Save as a new file with current affine information
        """
        self.header['sform_code'] = 0
        self.header['qform_code'] = 1
        self.to_filename(filename)
        if not quiet:
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
    def __init__(self, path=None, atlas=None):
        # Initiate default attributes
        self._atlas = None
        self._atlas_path = None
        self._mask = None
        self._object = False
        # If input is instance of the ImageObj
        if type(path) is ImageObj:
            self._image = path
            self._object = True
        # If input is path string
        elif type(path) is str:
            try:
                self.load(path)
            except:
                raise messages.InputPathError
            # if atlas path also given
            if atlas:
                self.set_atlas(atlas)
        else:
            raise messages.InputFileError
        # Generating mask image
        self._mask = self.get_mask()
        if self._object:
            self._path = TempFile(self._image, 'temp_template')
        else:
            self._path = path

    # Attributes
    @property
    def mask(self):
        return self._mask

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, imageobj):
        if type(imageobj) is ImageObj:
            self._image = imageobj
        else:
            raise messages.InputObjectError

    @property
    def atlas_obj(self):
        """Return ImageObj of Atlas
        """
        return self._atlas.image

    @property
    def label(self):
        return self._atlas.label

    @property
    def template_path(self):
        return self._path

    @property
    def atlas_path(self):
        return self._atlas_path

    @property
    def atlas(self):
        """Return atlas instance
        """
        return self._atlas

    def load(self, path):
        """ Import template

        :param path:
        :return:
        """
        self._image = ImageObj.load(path)
        self._path = path

    def set_atlas(self, path):
        self._atlas = Atlas(path)
        self._atlas_path = TempFile(self.atlas.image, 'temp_atlas')

    def get_mask(self):
        """Calculate mask from image data and generating template file
        """
        mask = np.asarray(self._image.dataobj)
        mask[abs(mask) > 0] = 1
        maskobj = ImageObj(mask, self.image.affine)
        return TempFile(maskobj, 'template_mask')

    def get_bg_cordinate(self):
        maskdata = self.mask._image._dataobj
        coronal = maskdata.sum(axis=2)
        coronal[coronal>0] = 1
        axial = maskdata.sum(axis=1)
        axial[axial>0] = 1
        x = np.argmax(coronal.sum(axis=1))
        y = np.argmax(coronal.sum(axis=0))
        z = np.argmax(axial.sum(axis=0))
        return x, y, z

    def swap_axis(self, axis1, axis2):
        if self._atlas:
            self.atlas_obj.swap_axis(axis1, axis2)
        self.image.swap_axis(axis1, axis2)

    def flip(self, **kwargs):
        if self._atlas:
            self.atlas_obj.flip(**kwargs)
        self.image.flip(**kwargs)

    def crop(self, **kwargs):
        if self._atlas:
            self.atlas_obj.crop(**kwargs)
        self.image.crop(**kwargs)

    def reslice(self, ac_slice, ac_loc, slice_thickness, total_slice, axis=2):
        if self._atlas:
            self.atlas_obj.reslice(self, ac_slice, ac_loc, slice_thickness, total_slice, axis=axis)
        self.image.reslice(self, ac_slice, ac_loc, slice_thickness, total_slice, axis=axis)

    def show(self, scale=15, **kwargs):
        if self._atlas:
            fig = Viewer.atlas(self.image, self._atlas, scale=scale, **kwargs)
        else:
            fig = Viewer.mosaic(self.image, scale=scale, **kwargs)
        try:
            fig, legend = fig
            display(fig)
            display(legend)
        except:
            display(fig)

    def save_as(self, filename, quiet=False):
        self.image.save_as('{}_template.nii.gz'.format(filename), quiet=quiet)
        if self._atlas:
            self._atlas.save_as('{}_atlas'.format(filename), quiet=quiet)

    def extract(self, path, **kwargs):
        if self._atlas:
            self._atlas.extract(path, **kwargs)
        else:
            methods.raiseerror(messages.Notice.MethodNotActivated, 'Atlas is not defined')

    def close(self):
        if self._object:
            os.remove(self._path)
        if self._atlas:
            os.remove(str(self._atlas_path))
        if self._mask:
            os.remove(str(self._mask))

    def __getitem__(self, idx):
        return self._atlas.__getitem__(idx)

    def __repr__(self):
        if self._atlas:
            return self._atlas.__repr__()
        else:
            return self._path


class Atlas(object):
    """ This class templating the segmentation image object to handle atlas related attributes

    """
    def __init__(self, path=None):
        self.path = path
        self._label = None
        self._coordinates = None
        if type(path) is ImageObj:
            self._image = path
        elif type(path) is str:
            self.load(path)
        else:
            raise messages.InputFileError

    @property
    def coordinates(self):
        return self._coordinates

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
            raise messages.InputObjectError

    def load(self, path):
        self._image, self._label, self._coordinates = methods.parsing_atlas(path)

    def save_as(self, filename, label_only=False, quiet=False):
        if not label_only:
            self._image.save_as("{}.nii".format(filename), quiet=quiet)
        methods.save_label(self._label, "{}.label".format(filename))

    def extract(self, path, contra=False, merge=False, surfix=None):
        if not os.path.exists(path):
            try:
                methods.mkdir(path)
            except:
                raise messages.InputPathError
        else:
            if merge:
                self._image._dataobj += self._image._dataobj[::-1,...]
        num_of_rois = int(np.max(self._image._dataobj))
        for i in range(num_of_rois+1):
            if not i:
                pass
            else:
                try:
                    label, roi = self[i]
                    if contra:
                        label = 'contra_'+label
                        roi._dataobj = roi._dataobj[::-1,...]
                    else:
                        if merge:
                            label = 'bilateral_'+label
                    if surfix:
                        label = "{}_{}".format(surfix, label)
                    roi.to_filename(os.path.join(path, "{}.nii".format(label)))
                except:
                    pass
                
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
                labels = '[{:>3}] {:>40}\n'.format(idx, self.label[idx][0])
            else:
                labels = '{}[{:>3}] {:>40}\n'.format(labels, idx, self.label[idx][0])
        return labels
