from __future__ import print_function
import os
import re
import shutil
import sys
import logging
import logging.handlers
from pandas import Panel, DataFrame, Series, read_table, read_excel
import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage import exposure
from nibabel import affines as affns
import objects
import messages
import shlex
from subprocess import PIPE, Popen
from matplotlib import colors

# Network analysis
import networkx as nx
import bct
import community

try:
    import SimpleITK as sitk
except ImportError:
    pass


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
    resol, origin = affns.to_matvec(imageobj.get_affine())
    resol = np.diag(resol).copy()
    origin = origin
    imageobj._dataobj = np.swapaxes(imageobj._dataobj, axis1, axis2)
    resol[axis1], resol[axis2] = resol[axis2], resol[axis1]
    origin[axis1], origin[axis2] = origin[axis2], origin[axis1]
    affine = affns.from_matvec(np.diag(resol), origin)
    reset_orient(imageobj, affine)


def load(filename):
    """ Load imagefile

    :param filename:
    :return:
    """
    if '.nii' in filename:
        img = objects.ImageObj.load(filename)
    elif '.mha' in filename:
        try:
            mha = sitk.ReadImage(filename)
        except:
            raise messages.ImportItkFailure
        data = sitk.GetArrayFromImage(mha)
        resol = mha.GetSpacing()
        origin = mha.GetOrigin()
        affine = affns.from_matvec(np.diag(resol), origin)
        img = objects.ImageObj(data, affine)
    else:
        raise messages.InputPathError
    return img


def load_temp(path=None, atlas=None):
    """ Load imagefile

    :param filename:
    :return:
    """
    tempobj = objects.Template(path, atlas)
    return tempobj


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
    resol, origin = affns.to_matvec(imageobj.affine)
    resol = np.diag(resol).copy()
    scale = float(slice_thickness) / resol[axis]
    resol[axis] = slice_thickness
    idx = []
    for i in range(ac_loc):
        idx.append(ac_slice - int((ac_loc - i) * scale))
    for i in range(total_slice - ac_loc):
        idx.append(ac_slice + int(i * scale))
    imageobj._dataobj = data[:, :, idx]
    affine, origin = affns.to_matvec(imageobj.affine[:, :])
    affine = np.array(np.diag(affine))
    affine[axis] = slice_thickness
    affine_mat = affns.from_matvec(np.diag(affine), origin)
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
    resol, origin = affns.to_matvec(imageobj.affine[:, :])
    affine = affns.from_matvec(resol, corr)
    reset_orient(imageobj, affine)


# Method collection for templateObject handler

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
            imageobj = objects.ImageObj.load(os.path.join   (path, img))
            affine.append(imageobj.affine)
            if not idx:
                atlasdata = np.asarray(imageobj.dataobj)
            else:
                atlasdata += np.asarray(imageobj.dataobj) * (idx + 1)
            label[idx+1] = splitnifti(img), rgbs[idx]
        atlas = objects.ImageObj(atlasdata, affine[0])
    elif os.path.isfile(path):
        atlas = objects.ImageObj.load(path)
        if '.nii' in path:
            filepath = os.path.basename(splitnifti(path))
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
            if filepath == os.path.basename(splitnifti(path)):
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

# Method collection for viewer handler
def set_viewaxes(axes):
    """ Set View Axes

    :param axes:
    :return:
    """
    ylim = axes.get_ylim()
    xlim = axes.get_xlim()
    axes.set_ylabel('L', rotation=0, fontsize=20)
    axes.set_xlabel('I', fontsize=20)
    axes.set_facecolor('white')
    axes.tick_params(labeltop=True, labelright=True, labelsize=8)
    axes.grid(False)
    axes.text(xlim[1]/2, ylim[1] * 1.1, 'P', fontsize=20)
    axes.text(xlim[1]*1.1, sum(ylim)/2*1.05, 'R', fontsize=20)
    return axes


def check_invert(kwargs):
    """ Check image invertion
    """
    invertx = False
    inverty = False
    invertz = False
    if kwargs:
        for arg in kwargs.keys():
            if arg == 'invertx':
                invertx = kwargs[arg]
            if arg == 'inverty':
                inverty = kwargs[arg]
            if arg == 'invertz':
                invertz = kwargs[arg]
    return invertx, inverty, invertz


def apply_invert(data, *invert):
    """ Apply image invertion
    """
    if invert[0]:
        data = nib.orientations.flip_axis(data, axis=0)
    if invert[1]:
        data = nib.orientations.flip_axis(data, axis=1)
    if invert[2]:
        data = nib.orientations.flip_axis(data, axis=2)
    return data


def convert_to_3d(imageobj):
    """ Reduce demension to 3D
    """
    dim = len(imageobj.shape)
    if dim == 4:
        data = np.asarray(imageobj.dataobj)[..., 0]
    elif dim == 3:
        data = np.asarray(imageobj.dataobj)
    elif dim == 5:
        data = np.asarray(imageobj.dataobj)[..., 0, 0]
    else:
        raise messages.ImageDimentionMismatched
    return data


def apply_p2_98(data):
    """ Image normalization
    """
    p2 = np.percentile(data, 2)
    p98 = np.percentile(data, 98)
    data = exposure.rescale_intensity(data, in_range=(p2, p98))
    return data


def set_mosaic_fig(data, dim, resol, slice_axis, scale):
    """ Set environment for mosaic figure
    """
    num_of_slice = dim[slice_axis]
    num_height = int(np.sqrt(num_of_slice))
    num_width = int(round(num_of_slice / num_height))
    # Swap axis
    data = np.swapaxes(data, slice_axis, 2)
    resol[2], resol[slice_axis] = resol[slice_axis], resol[2]
    dim[2], dim[slice_axis] = dim[slice_axis], dim[2]
    # Check the size of each slice
    size_height = num_height * dim[1] * resol[1] * scale / max(dim)
    size_width = num_width * dim[0] * resol[0] * scale / max(dim)
    # Figure generation
    slice_grid = [num_of_slice, num_height, num_width]
    size = [size_width, size_height]
    return data, slice_grid, size


def check_sliceaxis_cmap(imageobj, kwargs):
    """ Check sliceaxis (minimal number os slice) and cmap
    """
    slice_axis = int(np.argmin(imageobj.shape[:3]))
    cmap = 'gray'
    for arg in kwargs.keys():
        if arg == 'slice_axis':
            slice_axis = kwargs[arg]
        if arg == 'cmap':
            cmap = kwargs[arg]
    return slice_axis, cmap


def check_slice(dataobj, axis, slice_num):
    """ Check initial slice number to show
    """
    if slice_num:
        slice_num = slice_num
    else:
        slice_num = dataobj.shape[axis]/2
    return slice_num


# Method collection for preprocessing
def check_atals_datatype(atlas):
    if type(atlas) is str:
        # atlas = os.path.basename(atlas)
        tempobj = None
    else:
        try:
            tempobj = atlas
            atlas = tempobj.atlas_path
        except:
            raise messages.InputObjectError
    return atlas, tempobj


def get_warp_matrix(preproc, *args, **kwargs): #TODO: will be deprecated
    if 'inverse' in kwargs.keys():
        inverse = kwargs['inverse']
    else:
        inverse = None
    if inverse:
        mats = preproc._prjobj(1, preproc._processing, os.path.basename(args[0]),   # path name
                               *args[1:], ext='.mat').df.Abspath.loc[0]             # subject or session
        warps = preproc._prjobj(1, preproc._processing, os.path.basename(args[0]),
                                *args[1:], file_tag='_1InverseWarp').df.Abspath.loc[0]
        warped = preproc._prjobj(1, preproc._processing, os.path.basename(args[0]),
                                 *args[1:], file_tag='_InverseWarped').df.loc[0]
    else:
        mats = preproc._prjobj(1, preproc._processing, os.path.basename(args[0]),
                               *args[1:], ext='.mat').df.Abspath.loc[0]
        warps = preproc._prjobj(1, preproc._processing, os.path.basename(args[0]),
                                *args[1:], file_tag='_1Warp').df.Abspath.loc[0]
        warped = preproc._prjobj(1, preproc._processing, os.path.basename(args[0]),
                                 *args[1:], file_tag='_Warped').df.loc[0]
    return mats, warps, warped


# Method collection for dynamic analysis
def seed_coords(tractobj, start_point, end_point):
    data = tractobj.dataobj
    coords = np.argwhere(data == 1)
    coords = map(list, coords)
    coords.remove(start_point)
    seed_coords = []

    x, y, z = start_point
    cubic = data[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]
    cubic_corrds = np.argwhere(cubic == 1)
    cubic_results = []

    for dx, dy, dz in cubic_corrds:
        cubic_results.append([x + dx - 1, y + dy - 1, z + dz - 1])
    for cubic_corrd in cubic_results:
        if list(cubic_corrd) in (start_point, end_point):
            pass
        else:
            seed_coords.append(cubic_corrd)

    for i in range(len(coords)):
        x, y, z = seed_coords[-1]
        cubic = data[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]
        cubic_corrds = np.argwhere(cubic == 1)
        cubic_results = []

        for dx, dy, dz in cubic_corrds:
            cubic_results.append([x + dx - 1, y + dy - 1, z + dz - 1])
        for cubic_corrd in cubic_results:
            if list(cubic_corrd) in seed_coords:
                pass
            else:
                seed_coords.append(cubic_corrd)
    return seed_coords


def splitnifti(path):
    while '.nii' in path:
        path = os.path.splitext(path)[0]
    return str(path)


def gen_travel_seed(tractobj, start_point, end_point, filename=None):
    seed_crds = seed_coords(tractobj, start_point, end_point)
    shape = list(tractobj.shape[:])
    shape.append(len(seed_crds))
    data = np.zeros(shape, np.int16)
    for i, coord in enumerate(seed_crds):
        x, y, z = seed_crds
        data[x, y, z, i] = 1
        data[x, y, z + 1, i] = 1
        data[x + 1, y, z, i] = 1
        data[x + 1, y, z + 1, i] = 1
    travelseed_obj = objects.ImageObj(data, tractobj.affine)
    if filename:
        travelseed_obj.to_filename(filename)
    return travelseed_obj


def check_dataclass(datatype):
    if os.path.exists(datatype):
        dataclass = 1
        datatype = path_splitter(datatype)
        datatype = datatype[-1]
    else:
        dataclass = 0
    return dataclass, datatype

def shell(cmd):
    """Execute shell command

    :param cmd: str, command to execute
    :return: stdout, error
    """
    try:
        processor = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
        out, err = processor.communicate()
        return out, err
    except OSError as e:
        raiseerror(messages.Errors.InputValueError, 'Command can not be executed.')

def get_logger(path, name):
    # create logger
    logger = logging.getLogger('{0}Logger'.format(name))
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(path, '{0}.log'.format(name)))
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handler to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def raiseerror(exception, message):
    """ Raise User friendly error message

    Parameters
    ----------
    exception : Exception
        Excaption want to be raised
    message : str
        Message
    Returns
    -------
    None
    """
    try:
        raise exception(message)
    except Exception as e:
        sys.stderr.write("ERROR({0}): {1}\n".format(e.__doc__, e.message))
        messages.warnings.simplefilter("ignore")
        sys.exit()


def mkdir(*paths):
    """ Make directory

    Parameters
    ----------
    paths : str or list
        list of path

    Returns
    -------
    None
    """
    for path in paths:
        try:
            os.mkdir(path)
        except:
            pass


def path_splitter(path):
    """ Split path structure into list

    Parameters
    ----------
    path : str

    Returns
    -------
    path : list
    """
    return path.strip(os.sep).split(os.sep)


def parsing(path, ds_type, idx):
    """ Parsing the information of dataset from the pointed data class

    Parameters
    ----------
    path    : str
        Main path of the project
    ds_type : list
        Project.ds_type object
    idx     : int
        Dataclass index

    Returns
    -------
    pandas.DataFrame
        Dataset
    single_session  : boolean
        True if single session dataset
    empty_project   : boolean
        True if empty project folder

    """
    single_session = False
    empty_prj = False
    df = DataFrame()
    for f in os.walk(os.path.join(path, ds_type[idx])):
        if f[2]:
            for filename in f[2]:
                row = Series(path_splitter(os.path.relpath(f[0], path)))
                row['Filename'] = filename
                row['Abspath'] = os.path.join(f[0], filename)
                df = df.append(DataFrame([row]), ignore_index=True)
    if idx == 0:
        if len(df.columns) is 5:
            single_session = True
    else:
        if len(df.columns) is 6:
            single_session = True
    columns = update_columns(idx, single_session)
    df = df.rename(columns=columns)
    if 'Subject' not in df.columns:
        empty_prj = True
    elif not len(df):
        empty_prj = True
    if empty_prj:
        return DataFrame(), single_session, empty_prj
    else:
        return df.sort_values('Abspath'), single_session, empty_prj

def initial_filter(df, data_class, exts):
    """ Filtering out only selected file type in the project folder

    Parameters
    ----------
    df          : pandas.DataFrame
        Dataframe of project boject
    data_class  : list
        Dataclass want to be filtered
        e.g.) One of value in ['Data', 'Processing', 'Results'] for NIRAL method
    ext         : list
        Extension want to be filtered

    Returns
    -------
    df          : pandas.DataFrame
        Filtered dataframe

    """
    if data_class:
        if not type(data_class) is list:
            data_class = [data_class]
        try:
            df = df[df['DataClass'].isin(data_class)]
        except TypeError as e:
            print("Type error({0}): {1}".format(e.errno, e.strerror))
    if exts:
        df = df[df['Filename'].str.contains('|'.join([r"{ext}$".format(ext=ext) for ext in exts]))]
    columns = df.columns
    return df.reset_index()[columns]


def update_columns(idx, single_session):
    """ Update name of columns according to the set Dataclass

    Parameters
    ----------
    idx             : int
        Dataclass index
    single_session  : boolean
        True if the project is single session

    Returns
    -------
    column          : dict
        New list of columns
    """
    if idx == 0:
        if single_session:
            subject, session, dtype = (1, 3, 2)
        else:
            subject, session, dtype = (1, 2, 3)
        columns = {0: 'DataClass', subject: 'Subject', session: 'Session', dtype: 'DataType'}
    elif idx == 1:
        columns = {0: 'DataClass', 1: 'Pipeline', 2: 'Step', 3: 'Subject', 4: 'Session'}
    elif idx == 2:
        columns = {0: 'DataClass', 1: 'Pipeline', 2: 'Result', 3: 'Subject', 4: 'Session'}
    else:
        columns = {0: 'DataClass'}
    return columns


def reorder_columns(idx, single_session):
    """ Reorder the name of columns

    Parameters
    ----------
    idx             : int
        Dataclass index
    single_session  : boolean
        True if the project is single session

    Returns
    -------
    column          : list or None
        Reordered column
    """
    if idx == 0:
        if single_session:
            return ['Subject', 'DataType', 'Filename', 'Abspath']
        else:
            return ['Subject', 'Session', 'DataType', 'Filename', 'Abspath']
    elif idx == 1:
        if single_session:
            return ['Pipeline', 'Step', 'Subject', 'Filename', 'Abspath']
        else:
            return ['Pipeline', 'Step', 'Subject', 'Session', 'Filename', 'Abspath']
    elif idx == 2:
        if single_session:
            return ['Pipeline', 'Result', 'Subject', 'Filename', 'Abspath']
        else:
            return ['Pipeline', 'Result', 'Subject', 'Session', 'Filename', 'Abspath']
    else:
        return None


def mk_main_folder(prj):
    """ Make processing and results folders

    Parameters
    ----------
    prj         : pynit.Project

    Returns
    -------
    None
    """
    mkdir(os.path.join(prj.path, prj.ds_type[0]),
                    os.path.join(prj.path, prj.ds_type[1]),
                    os.path.join(prj.path, prj.ds_type[2]))


def check_arguments(args, residuals, lists):
    """ Parse the values in the list to be used as filter

    Parameters
    ----------
    args        : tuple
        Input arguments for filtering
    residuals   : list
        Residual values
    lists       : list
        Attributes of project object

    Returns
    -------
    filter      : list
        Values need to be filtered
    residuals   : list
        Residual values
    """
    filter = [arg for arg in args if arg in lists]
    residuals = list(residuals)
    if len(filter):
        for comp in filter:
            if comp in residuals:
                residuals.remove(comp)
    return list(set(filter)), list(set(residuals))

def get_step_name(procobj, step, results=False, verbose = None):
    if results:
        idx = 2
    else:
        idx = 1
    processing_path = os.path.join(procobj._prjobj.path, procobj._prjobj.ds_type[idx], procobj.processing)
    executed_steps = [f for f in os.listdir(processing_path) if os.path.isdir(os.path.join(processing_path, f))]
    if len(executed_steps):
        overlapped = [old_step for old_step in executed_steps if step in old_step]
        if len(overlapped):
            if verbose:
                print('Notice: existing path')
            checked_files = []
            for f in os.walk(os.path.join(processing_path, overlapped[0])):
                checked_files.extend(f[2])
            if len(checked_files):
                if verbose:
                    print('Notice: Last step path is not empty')
            return overlapped[0]
        else:
            return "_".join([str(len(executed_steps) + 1).zfill(3), step])
    else:
        if verbose:
            print('The pipeline [{pipeline}] is initiated'.format(pipeline=procobj.processing))
        return "_".join([str(1).zfill(3), step])


def copyfile(output_path, input_path, *args):
    """ Copy File
    """
    shutil.copyfile(input_path, output_path)


# Networks
def get_graph(dset, tmpobj, louvain=False):
    G = nx.Graph()
    G.add_nodes_from(zip(*tmpobj.label.values())[0][1:])
    for roi in dset.columns:
        for i, value in enumerate(dset[roi]):
            if not np.isnan(value):
                if value:
                    G.add_edge(roi, dset.index[i], weight=value)
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

    if louvain:
        modules = dict(zip(dset.columns, bct.modularity_louvain_und(bct.binarize(dset.values, copy=True))[0]))
    else:
        modules = dict(zip(dset.columns, bct.modularity_und(dset.values)[0]))
    comm_idx = dict()
    for node, comm in modules.items():
        if comm not in comm_idx.keys():
            comm_idx[comm] = [node]
        else:
            comm_idx[comm].append(node)
    colors_set = colors.XKCD_COLORS.keys()[::-1]
    node_color_map = []
    for node in G.nodes():
        node_color_map.append(colors_set[modules[node]])
    return G, edges, weights, modules, node_color_map, comm_idx, colors_set