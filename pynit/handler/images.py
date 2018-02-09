import os
import re
import numpy as np
import copy as ccopy
import nibabel as nib
from .base import ImageObj
from .base import BrainPlot
from ..tools import messages, methods, display
from scipy import ndimage
from shutil import rmtree
from nibabel import affines


def load(filename):
    """ load available file
    available exts: .nii(.gz), .mga, .xls(x), .csv, .tsv, .json

    :param filename: file want to load
    :type filename: str
    :return: object
    """
    if '.nii' in filename:
        img = ImageObj.load(filename)
    elif '.mha' in filename:
        try:
            import SimpleITK as sitk
            mha = sitk.ReadImage(filename)
        except:
            raise messages.ImportItkFailure
        data = sitk.GetArrayFromImage(mha)
        resol = mha.GetSpacing()
        origin = mha.GetOrigin()
        affine = affines.from_matvec(np.diag(resol), origin)
        img = ImageObj(data, affine)
    else:
        import pandas as pd
        if '.xls' in filename:
            img = pd.read_excel(filename)
        elif '.csv' in filename:
            img = pd.read_csv(filename)
        elif '.tsv' in filename:
            img = pd.read_table(filename)
        elif '.json' in filename:
            import json
            img = json.load(open(filename))
        else:
            raise messages.InputPathError
    return img


def load_temp(path=None, atlas=None):
    """ Load imagefile
    """
    tempobj = Template(path, atlas)
    return tempobj


def parsing_atlas(path):
    """Parsing atlas imageobj and label

    :param path:
    :return:
    """
    label = dict()
    affine = list()
    if os.path.isdir(path):
        atlasdata = None
        list_of_rois = [img for img in os.listdir(path) if '.nii' in img]
        rgbs = np.random.rand(len(list_of_rois), 3)
        label[0] = 'Clear Label', [.0, .0, .0]

        for idx, img in enumerate(list_of_rois):
            imageobj = ImageObj.load(os.path.join(path, img))
            affine.append(imageobj.affine)
            if not idx:
                atlasdata = np.asarray(imageobj.dataobj)
            else:
                atlasdata += np.asarray(imageobj.dataobj) * (idx + 1)
            label[idx+1] = methods.splitnifti(img), rgbs[idx]
        atlas = ImageObj(atlasdata, affine[0])
    elif os.path.isfile(path):
        atlas = ImageObj.load(path)
        if '.nii' in path:
            filepath = os.path.basename(methods.splitnifti(path))
            dirname = os.path.dirname(path)
            if dirname == '':
                dirname = '.'
            for f in os.listdir(dirname):
                if filepath in f:
                    if '.lbl' in f:
                        filepath = os.path.join(dirname, "{}.lbl".format(filepath))
                    elif '.label' in f:
                        filepath = os.path.join(dirname, "{}.label".format(filepath))
                    else:
                        filepath = filepath
            if filepath == os.path.basename(methods.splitnifti(path)):
                raise messages.NoLabelFile
        else:
            raise messages.NoLabelFile
        pattern = r'^\s+(?P<idx>\d+)\s+(?P<R>\d+)\s+(?P<G>\d+)\s+(?P<B>\d+)\s+' \
                  r'(\d+|\d+\.\d+)\s+\d+\s+\d+\s+"(?P<roi>.*)$'
        with open(filepath, 'r') as labelfile:
            for line in labelfile:
                if re.match(pattern, line):
                    idx = int(re.sub(pattern, r'\g<idx>', line))
                    roi = re.sub(pattern, r'\g<roi>', line)
                    roi = roi.split('"')[0]
                    rgb = re.sub(pattern, r'\g<R>\s\g<G>\s\g<B>', line)
                    rgb = rgb.split(r'\s')
                    rgb = np.array(map(float, rgb))/255
                    label[idx] = roi, rgb
    else:
        raise messages.InputPathError
    data = np.asarray(atlas.dataobj)
    # Calculate centor of mass (coordinate of the rois)
    com = dict()
    for i, roi in enumerate(zip(*label.values())[0]):
        if not i:
            pass
        else:
            roi_mask = (data == i)*1.0
            com[roi] = np.array(map(round, ndimage.center_of_mass(roi_mask)))
    return atlas, label, com


def save_label(label, filename):
    """ Save label instance to file

    :param label:
    :param filename:
    :return:
    """
    with open(filename, 'w') as f:
        line = list()
        for idx in label.keys():
            roi, rgb = label[idx]
            rgb = np.array(rgb) * 255
            rgb = rgb.astype(int)
            if idx == 0:
                line = '{:>5}   {:>3}  {:>3}  {:>3}        0  0  0    "{}"\n'.format(idx, rgb[0], rgb[1], rgb[2],
                                                                                     roi)
            else:
                line = '{}{:>5}   {:>3}  {:>3}  {:>3}        1  1  0    "{}"\n'.format(line, idx, rgb[0], rgb[1],
                                                                                       rgb[2], roi)
        f.write(line)


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
        coronal[coronal > 0] = 1
        axial = maskdata.sum(axis=1)
        axial[axial > 0] = 1
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
            fig = BrainPlot.atlas(self.image, self._atlas, scale=scale, **kwargs)
        else:
            fig = BrainPlot.mosaic(self.image, scale=scale, **kwargs)
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
        self._image, self._label, self._coordinates = parsing_atlas(path)

    def save_as(self, filename, label_only=False, quiet=False):
        if not label_only:
            self._image.save_as("{}.nii".format(filename), quiet=quiet)
        save_label(self._label, "{}.label".format(filename))

    def extract(self, path, contra=False, merge=False, surfix=None):
        if not os.path.exists(path):
            # try:
            methods.mkdir(path)
            # except:
            #     raise messages.InputPathError
        num_of_rois = int(np.max(self._image._dataobj))
        for i in range(num_of_rois + 1):
            if not i:
                pass
            else:
                try:
                    label, roi = self[i]
                    if contra:
                        label = 'contra_' + label
                        roi._dataobj = roi._dataobj[::-1, ...]
                    else:
                        if merge:
                            label = 'bilateral_' + label
                            roi._dataobj += roi._dataobj[::-1, ...]
                            roi._dataobj[roi._dataobj > 0] = 1
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


class TempFile(object):
    """This class is designed to make Template Object can be utilized on Processing steps
    Due to the major processing steps using command line tools(AFNI, ANTs so on..), the numpy
    object cannot be used on these tools.

    Using this class, loaded ImageObj now has temporary files on the location at './.tmp' or './atlas_tmp'
    """
    def __init__(self, obj, filename='image_cache', atlas=False, flip=False, merge=False, bilateral=False):
        """Initiate instance

        :param obj:         ImageObj
        :param filename:    Temporary filename
        :param atlas:       True if the input is atlas data
        :param flip:        True if you want to flip
        :param merge:       True if you want to merge flipped ImageObj
        """
        # If given input object is atlas
        if atlas:
            self._image = None
            # Copy object to protect the intervention between object
            self._atlas = ccopy.copy(obj)
            if flip:
                self._atlas.extract('./.atlas_tmp', contra=True)
            if merge:
                self._atlas.extract('./.atlas_tmp', merge=True)
            if bilateral:
                self._atlas.extract('./.atlas_tmp')
                obj.extract('./.atlas_tmp', contra=True)
            else:
                self._atlas.extract('./.atlas_tmp')
            self._listdir = [ f for f in os.listdir('./.atlas_tmp') if '.nii' in f ]
            atlas = Atlas('./.atlas_tmp')
            methods.mkdir('./.tmp')
            self._path = os.path.join('./.tmp', "{}.nii".format(filename))
            self._fname = filename
            atlas.save_as(os.path.join('./.tmp', filename), quiet=True)
            self._label = [roi for roi, color in atlas.label.values()][1:]
        else:
            # Copy object to protect the intervention between object
            self._image = ccopy.copy(obj)
            if flip:
                self._image.flip(invertx=True)
            if merge:
                self._image._dataobj += self._image._dataobj[::-1,]
            self._fname = filename
            methods.mkdir('./.tmp')
            self._image.save_as(os.path.join('./.tmp', filename), quiet=True)
            self._atlas = None
            self._label = None
            self._path = os.path.join('./.tmp', "{}.nii".format(filename))

    @property
    def path(self):
        return self._path

    @property
    def label(self):
        return self._label

    def __getitem__(self, idx):
        if self._image:
            raise IndexError
        else:
            if self._atlas:
                return os.path.abspath(os.path.join('.atlas_tmp', self._listdir[idx]))
            else:
                return None

    def __repr__(self):
        if self._image:
            return os.path.abspath(os.path.join('.tmp', "{}.nii".format(self._fname)))
        else:
            if self._atlas:
                output = []
                for i, roi in enumerate(self._listdir):
                    output.append('{:>3}  {:>100}'.format(i, os.path.abspath(os.path.join('.atlas_tmp', roi))))
                return str('\n'.join(output))

    def close(self):
        if self._image:
            os.remove(os.path.join('.tmp', "{}.nii".format(self._fname)))
        if self._atlas:
            rmtree('.atlas_tmp', ignore_errors=True)
            os.remove(os.path.join('.tmp', "{}.nii".format(self._fname)))
            os.remove(os.path.join('.tmp', "{}.label".format(self._fname)))
        self._atlas = None
        self._image = None
        self._path = None


class BrukerRawData(object):
    def __init__(self, path, num=None, pid=None, ori=False, convert=False):
        # load parameter files and parsing header information
        self._acqp = None
        self._method = None
        self._reco = None
        if num:
            num = str(num)
            with open(os.path.join(path, num, 'acqp')) as f:
                acqp = f.readlines()[:]
            with open(os.path.join(path, num, 'method')) as f:
                method = f.readlines()[:]
            self._acqp = self.parsing(acqp, 'acqp')
            self._method = self.parsing(method, 'method')

            if isinstance(pid, int):
                try:
                    with open(os.path.join(path, num, 'pdata', str(pid), 'reco')) as f:
                        reco = f.readlines()[:]
                        self._reco = self.parsing(reco, 'reco')
                    # Check datatype
                    dtype = self._reco['RECO_wordtype']
                    if dtype == '_16BIT_SGN_INT':
                        dt = np.dtype('int16')
                    elif dtype == '_32BIT_SGN_INT':
                        dt = np.dtype('int32')
                    else:
                        dt = np.dtype('float64')
                    # Get image resolution
                    self._resol = self.get_resol()
                    self._affine = np.diag(self.resol + [1])
                    self._2dseq = None
                    self._nii = None
                    if convert:
                        # Load binary image file
                        img = os.path.join(path, num, 'pdata', str(pid), '2dseq')
                        self._2dseq = np.fromfile(img, dtype=dt)

                        # Convert to Nifti
                        self._nii = nib.Nifti1Image(self.img, self.affine)

                        # Correct orientation
                        if ori == True:
                            self.correct_orient()

                        # Convert to Nifti
                        self.set_default_header()
                except:
                    pass

        with open(os.path.join(path, 'subject')) as f:
            subject = f.readlines()[:]
        self._subject = self.parsing(subject, 'subject')

    @property
    def acqp(self):
        return self._acqp

    @property
    def method(self):
        return self._method

    @property
    def reco(self):
        return self._reco

    @property
    def subject(self):
        return self._subject

    @property
    def resol(self):
        return self._resol

    @property
    def img(self):
        return self._2dseq.reshape(self.get_shape()).T

    @property
    def affine(self):
        return self._affine

    @property
    def nii(self):
        return self._nii

    def parsing(self, profiles, key):
        p_sprofile = r'^\#\#\$(.*)\=([^(].*[^)])'
        p_vprofile = r'^\#\#\$(.*)\=\((.*)\)'
        p_vis = r'^\$\$.*'
        p_string = r'^\<(.*)\>$'
        output_obj = dict()
        for i, line in enumerate(profiles):
            if re.search(p_sprofile, line):
                key = re.sub(p_sprofile, r'\1', line).strip()
                value = re.sub(p_sprofile, r'\2', line).strip()
                value = self.check_dt(value)
                output_obj[key] = value
            elif re.search(p_vprofile, line):
                key = re.sub(p_vprofile, r'\1', line).strip()
                n_value = re.sub(p_vprofile, r'\2', line).strip()
                try:
                    n_value = map(int, map(str.strip, n_value.split(',')))
                except:
                    n_value = map(str, map(str.strip, n_value.split(',')))
                if len(n_value) == 1:
                    n_value = n_value[0]
                values = list()
                for next_line in profiles[i + 1:]:
                    if re.search(p_sprofile, next_line):
                        break
                    elif re.search(p_vprofile, next_line):
                        break
                    elif re.search(p_vis, next_line):
                        break
                    else:
                        values.append(next_line.strip())
                values = ' '.join(values)
                if isinstance(n_value, list):
                    try:
                        values = np.array(self.check_array(n_value, values)).reshape(n_value)
                    except:
                        values = self.check_dt(values)
                    output_obj[key] = values
                else:
                    if re.match(p_string, values):
                        output_obj[key] = re.sub(p_string, r'\1', values)
                    else:
                        if n_value == 1:
                            values = self.check_dt(values)
                        else:
                            try:
                                values = self.check_array(n_value, values)
                            except:
                                print('{}({})={}'.format(key, n_value, values))
                        output_obj[key] = values
            else:
                pass
        return output_obj

    def check_dt(self, value):
        p_int = r'^-?[0-9]+$'
        p_float = r'^-?(\d+\.?)?\d+([eE][-+]?\d+)?$'
        p_string = r'^\<(.*)\>$'
        value = value.strip(' ')
        if re.match(p_float, value):
            if re.match(p_int, value):
                value = int(value)
            else:
                value = float(value)
        else:
            try:
                value = int(value)
            except:
                if re.match(p_string, value):
                    value = re.sub(p_string, r'\1', value).strip(" ")
        return value

    def check_array(self, n_value, values):
        p_groups = r'\(([^)]*)\)'
        if re.match(p_groups, values):
            values = re.findall(p_groups, values)
            values = [map(self.check_dt, value.split(', ')) for value in values]
        else:
            values = map(self.check_dt, values.split())
        return values

    def get_shape(self):
        try:
            matrix = self.reco['RECO_size'][::-1]
        except:
            matrix = self.method['PVM_EncMatrix'][::-1]
        try:
            slices = self.acqp['NSLICES']
        except:
            slices = None
        try:
            echo_img = self.method['PVM_NEchoImages']
        except:
            echo_img = None
        if isinstance(matrix, int):
            matrix = [matrix]
        if echo_img > 1:
            matrix = [echo_img] + matrix
        try:
            dim = self.method['PVM_SpatDimEnum']
            if dim == '2D':
                matrix = [slices] + matrix
            elif dim == '3D':
                sliceori, readori, pos1, pow2 = self.get_orient()
                if readori == 'H_F':
                    matrix = [matrix[0], matrix[2], matrix[1]]
        except:
            pass
        try:
            num_rep = self.method['PVM_NRepetitions']
        except:
            num_rep = None
        if self.method['Method'] == 'DtiEpi':
            num_b0 = self.method['PVM_DwAoImages']
            num_dir = self.method['PVM_DwNDiffDir']
            matrix = [(num_b0 + num_dir)] + matrix
        if num_rep:
            if self.method['Method'] == 'DtiEpi':
                matrix[0] *= num_rep
            else:
                matrix = [num_rep] + matrix
        return matrix

    def get_tempresol(self):
        tr = self.method['PVM_RepetitionTime']
        num_avr = self.method['PVM_NAverages']
        try:
            num_seg = self.method['NSegments']
            return tr * num_seg * num_avr
        except:
            return tr * num_avr

    def get_resol(self):
        if self.method['PVM_SpatDimEnum'] == '2D':
            # dx, dy = self.method['PVM_Spat    Resol']
            dy, dx = list(np.array(self.reco['RECO_fov']).astype(float)/np.array(self.reco['RECO_size']).astype(float)*10)
            dz = self.acqp['ACQ_slice_thick']
        elif self.method['PVM_SpatDimEnum'] == '3D':
            # dx, dy, dz = self.method['PVM_SpatResol']
            dy, dx, dz = list(np.array(self.reco['RECO_fov']).astype(float)/np.array(self.reco['RECO_size']).astype(float)*10)
        else:
            dx, dy, dz = (1, 1, 1)
        return [dx, dy, dz]

    def get_center(self):
        if self.method['PVM_SpatDimEnum'] == '2D':
            center_x, center_y = np.array(self.method['PVM_Fov'])/2
            center_z = (self.acqp['ACQ_slice_thick'] + self.method['PVM_SPackArrSliceGap']) * self.acqp['NSLICES'] / 2
        elif self.method['PVM_SpatDimEnum'] == '3D':
            center_x, center_y, center_z = np.array(self.method['PVM_Fov'])/2
        else:
            center_x, center_y, center_z = (0, 0, 0)
        return center_x, center_y, center_z

    def get_orient(self):
        slice_orient = self.method['PVM_SPackArrSliceOrient']
        read_orient = self.method['PVM_SPackArrReadOrient']
        pos = self.acqp['ACQ_patient_pos'].split('_')
        return [slice_orient, read_orient] + pos

    def get_geometry(self):
        phase1_offset = self.method['PVM_SPackArrPhase1Offset']
        phase2_offset = self.method['PVM_SPackArrPhase2Offset']
        slice_offset = self.method['PVM_SPackArrSliceOffset']
        read_offset = self.method['PVM_SPackArrReadOffset']
        return [slice_offset, phase1_offset, phase2_offset, read_offset]

    def set_default_header(self):
        self.nii.header.default_x_flip = False
        tr = self.get_tempresol()
        if self.method['Method'] == 'EPI':
            self.nii.header.set_xyzt_units('mm', 'sec')
            self.nii.header['pixdim'][4] = float(tr) / 1000
            self.nii.header.set_dim_info(slice=2)
            self.nii.header['slice_duration'] = float(tr) / (1000 * self.acqp['NSLICES'])
            if self.method['PVM_ObjOrderScheme'] == 'User_defined_slice_scheme':
                self.nii.header['slice_code'] = 0
            elif self.method['PVM_ObjOrderScheme'] == 'Sequential':
                self.nii.header['slice_code'] = 1
            elif self.method['PVM_ObjOrderScheme'] == 'Reverse_sequential':
                self.nii.header['slice_code'] = 2
            elif self.method['PVM_ObjOrderScheme'] == 'Interlaced':
                self.nii.header['slice_code'] = 3
            elif self.method['PVM_ObjOrderScheme'] == 'Reverse_interlacesd':
                self.nii.header['slice_code'] = 4
            elif self.method['PVM_ObjOrderScheme'] == 'Angiopraphy':
                self.nii.header['slice_code'] = 0
            self.nii.header['slice_start'] = min(self.acqp['ACQ_obj_order'])
            self.nii.header['slice_end'] = max(self.acqp['ACQ_obj_order'])
        else:
            self.nii.header.set_xyzt_units('mm', 'unknown')
            self.nii.header['qform_code'] = 1
            self.nii.header['sform_code'] = 0

    def correct_orient(self, human=0):
        """
        Readout direction: - this setup only affected axis of readout direction not other axis
            L_R: left to right (x)
            A_P: anterior to posterior (y)
            H_F: head to foot (z)
        position:
            Supine, Prone
            Head, Foot
        """
        ori = self.get_orient()
        if human:
            pass
        else:
            self._resol[0] = -1 * float(self._resol[0])
            self._resol[1] = -1 * float(self._resol[1])
            affine = np.diag(self._resol + [1])
            centers = self.get_center()
            print(centers)
            offset = np.round(self.get_geometry(), decimals=4)
            affine[0, 3] = (centers[0] - offset[3])
            affine[1, 3] = (centers[1] + offset[1])
            affine[2, 3] = -1 * (centers[2] - offset[0])
            if ori[0] == 'axial':
                self._affine = affine[[0, 2, 1, 3], :]
            elif ori[0] == 'coronal':
                self._affine = affine
            else:
                self._affine = affine[[2, 1, 0, 3], :]
            self.nii.set_qform(self.affine, code=1)
            self.nii.set_sform(self.affine, code=0)

    def __repr__(self):
        return 'BrukerRawData'