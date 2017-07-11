import numpy as np
from nibabel import Nifti1Image, affines
from ..core.visualizers import check_invert, apply_invert
from ..core.visualizers import Viewer


def reset_orient(imageobj, affine):
    """ Reset to the original scanner space

    :param imageobj:
    :param affine:
    :return:
    """
    imageobj.set_qform(affine)
    imageobj.set_sform(affine)
    imageobj.header['sform_code'] = 0
    imageobj.header['qform_code'] = 1
    imageobj._affine = affine


def swap_axis(imageobj, axis1, axis2):
    """ Swap axis of image object

    :param imageobj:
    :param axis1:
    :param axis2:
    :return:
    """
    resol, origin = affines.to_matvec(imageobj.get_affine())
    resol = np.diag(resol).copy()
    origin = origin
    imageobj._dataobj = np.swapaxes(imageobj._dataobj, axis1, axis2)
    resol[axis1], resol[axis2] = resol[axis2], resol[axis1]
    origin[axis1], origin[axis2] = origin[axis2], origin[axis1]
    affine = affines.from_matvec(np.diag(resol), origin)
    reset_orient(imageobj, affine)


def down_reslice(imageobj, ac_slice, ac_loc, slice_thickness, total_slice, axis=2):
    """ Reslicing

    :param imageobj:
    :param ac_slice:
    :param ac_loc:
    :param slice_thickness:
    :param total_slice:
    :param axis:
    :return:
    """
    data = np.asarray(imageobj.dataobj)
    resol, origin = affines.to_matvec(imageobj.affine)
    resol = np.diag(resol).copy()
    scale = float(slice_thickness) / resol[axis]
    resol[axis] = slice_thickness
    idx = []
    for i in range(ac_loc):
        idx.append(ac_slice - int((ac_loc - i) * scale))
    for i in range(total_slice - ac_loc):
        idx.append(ac_slice + int(i * scale))
    imageobj._dataobj = data[:, :, idx]
    affine, origin = affines.to_matvec(imageobj.affine[:, :])
    affine = np.array(np.diag(affine))
    affine[axis] = slice_thickness
    affine_mat = affines.from_matvec(np.diag(affine), origin)
    imageobj._affine = affine_mat
    imageobj.set_qform(affine_mat)
    imageobj.set_sform(affine_mat)
    imageobj.header['sform_code'] = 0
    imageobj.header['qform_code'] = 1


def crop(imageobj, **kwargs):
    """ Crop

    :param imageobj:
    :param kwargs:
    :return:
    """
    x = None
    y = None
    z = None
    t = None
    for arg in kwargs.keys():
        if arg == 'x':
            x = kwargs[arg]
        if arg == 'y':
            y = kwargs[arg]
        if arg == 'z':
            z = kwargs[arg]
        if arg == 't':
            t = kwargs[arg]
        else:
            pass
    if x:
        if (type(x) != list) and (len(x) != 2):
            raise TypeError
    else:
        x = [None, None]
    if y:
        if (type(y) != list) and (len(y) != 2):
            raise TypeError
    else:
        y = [None, None]
    if z:
        if (type(z) != list) and (len(z) != 2):
            raise TypeError
    else:
        z = [None, None]
    if t:
        if (type(t) != list) and (len(t) != 2):
            raise TypeError
    else:
        t = [None, None]
    if len(imageobj.shape) == 3:
        imageobj._dataobj = imageobj._dataobj[x[0]:x[1], y[0]:y[1], z[0]:z[1]]
    if len(imageobj.shape) == 4:
        imageobj._dataobj = imageobj._dataobj[x[0]:x[1], y[0]:y[1], z[0]:z[1], t[0]:t[1]]


def set_center(imageobj, corr):
    """ Applying center corrdinate to the object
    """
    resol, origin = affines.to_matvec(imageobj.affine[:, :])
    affine = affines.from_matvec(resol, corr)
    reset_orient(imageobj, affine)


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
        output = '{}\n{}\n{}\n{}\n{}'.format(title, '-'*len(title), img, txt, ds)
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


class ImageObj(Nifti1Image):
    """ ImageObject for PyNIT
    """
    # def __init__(self):
    #     super(ImageObj, self).__init__()

    def show(self, **kwargs):
        """ Plotting slice of the object
        """
        Viewer.slice(self, **kwargs)

    def mosaic(self, *args, **kwargs):                          #TODO: update needed
        """ Mosaic view for the object
        """
        fig = Viewer.mosaic(self, *args, **kwargs)

    def swap_axis(self, axis1, axis2):
        """ Swap input axis with given axis of the object
        """
        swap_axis(self, axis1, axis2)

    def flip(self, **kwargs):
        invert = check_invert(kwargs)
        self._dataobj = apply_invert(self._dataobj, *invert)

    def crop(self, **kwargs):
        crop(self, **kwargs)

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
        down_reslice(self, ac_slice, ac_loc, slice_thickness, total_slice, axis)

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

    def check_reg(self, imageobj, scale=10, **kwargs):          #TODO: update needed
        fig = Viewer.check_reg(imageobj, self, scale=scale, norm=True, **kwargs)

    def check_mask(self, maskobj, scale=15, **kwargs):          #TODO: update needed
        fig = Viewer.check_mask(self, maskobj, scale=scale, **kwargs)

    @property
    def affine(self):
        return self._affine
