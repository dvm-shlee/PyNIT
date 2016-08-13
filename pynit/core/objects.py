import numpy as np
import nibabel as nib
import pandas as pd
from .visual import Viewer
from .utility import Interface, Internal


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


class Image(nib.nifti1.Nifti1Image):
    def show(self, *args, **kwargs):
        Viewer.slice(self, *args, **kwargs)

    def mosaic(self, *args, **kwargs):
        fig = Viewer.mosaic(self, *args, **kwargs)

    def swapaxis(self, axis1, axis2):
        resol, origin = nib.affines.to_matvec(self.get_affine())
        resol = np.diag(resol).copy()
        origin = origin
        self._dataobj = np.swapaxes(self._dataobj, axis1, axis2)
        resol[axis1], resol[axis2] = resol[axis2], resol[axis1]
        origin[axis1], origin[axis2] = origin[axis2], origin[axis1]
        affine = nib.affines.from_matvec(np.diag(resol), origin)
        self.header['sform_code'] = 0
        self.header['qform_code'] = 1
        self.set_qform(affine)

    def flip(self, **kwargs):
        invert = Internal.check_invert(kwargs)
        self._dataobj = Internal.apply_invert(self._dataobj, *invert)

    def crop(self, **kwargs):
        Internal.crop(self, **kwargs)

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
        dataobj = Internal.down_reslice(self, ac_slice, ac_loc, slice_thickness, total_slice, axis)
        affine, origin = nib.affines.to_matvec(self.affine)
        affine = np.diag(affine)
        affine[axis] = slice_thickness
        affine_mat = nib.affines.from_matvec(np.diag(affine), origin)
        self._dataobj = dataobj
        self._affine = affine_mat
        self.set_qform(affine_mat)
        self.set_sform(affine_mat)
        self.header['sform_code'] = 0
        self.header['qform_code'] = 1

    def save_as(self, filename):
        """ Save as a new file with current affine information
        """
        nii = nib.Nifti1Image(self._dataobj, self._affine)
        nii.header['sform_code'] = 1
        nii.header['qform_code'] = 1
        nii.to_filename(filename)
        print("NifTi1 format image is saved as '{}'".format(filename))

    def pedding(self, axis, ):

        pass

    def saveas(self, filename):
        self.to_filename(filename)
        self.to_file_map()

    def timetrace(self, roiobj):
        number_of_rois = np.max(roiobj.dataobj)
        df = pd.DataFrame()
        for i in range(number_of_rois-1):
            data = Interface.afni_3dmaskave(None, self.get_filename(), "{}<{}>".format(roiobj.get_filename(), i+1))
            data = pd.Series(data)
            df[i] = data
        return df

    def atlas(self, roiobj, *args, **kwargs):
        Viewer.atlas(self, roiobj, *args, **kwargs)


class Template(object):
    """
    """
    #TODO: using this to parsing ROI images and name and generate single atlas with ITK stype label
    #TODO: also import template
    def __init__(self):
        pass
    def set_templalte(self):
        pass
    def set_path(self):
        # If path, grap all
        # If obj, grap as
        pass
    # All reslice, reorientation function need to be integrated
    # And will be applied for both mask and template together