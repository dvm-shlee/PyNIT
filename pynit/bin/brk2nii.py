#!/usr/bin/env python

__author__ = 'SungHo Lee (shlee@unc.edu)'
__version_info__ = ('2016', '08', '07')
__version__ = '-'.join(__version_info__) + 'REV04'

import re, os, argparse, sys
try:
    import nibabel as nib
    import numpy as np
    import json
except:
    print("Required libraries (numpy, nibabel, json) are not installed")
    sys.exit(0)

parser = argparse.ArgumentParser(prog='Brk2Nii', description="Convert Bruker raw '2dseq' to Nifti formated image")
parser.add_argument("pid", help="Proccessed ID (in case image is reconstructed)")
parser.add_argument("path", help="Folder location for the Bruker raw data", type=str)
parser.add_argument("filename", help="Filename w/o extension to export NifTi image", type=str)
parser.add_argument("-p", "--param", help="Export parameters to 'txt' file", action="store_true", default=0)
parser.add_argument("-V", "--version", action="version", version="%(prog)s ("+__version__+")")
args = parser.parse_args()

if args.path:
    mtd_path = os.path.join(args.path,'method')
    img_path = os.path.join(args.path,'pdata',str(args.pid),'2dseq')
    reco_path = os.path.join(args.path,'pdata',str(args.pid),'reco')

# Parsing method parameters from Bruker raw data
param = [r'.*(Method)=(.*)',
         r'.*RECO_wor(dtype)=(.*)',
         r'.*(SpatResol)=(.*)',
         r'.*(SliceThick)=(.*)',
         r'.*SPackArr(SliceGap)=(.*)',
         r'.*SPackArr(SliceDistance)=(.*)',
         r'.*PVM_(Matrix)=(.*)',
         r'.*SPackArr(NSlices)=(.*)',
         r'.*N(Repetitions)=(.*)',
         r'.*SPackArr(SliceOrient)=(.*)',
         r'.*SPackArr(ReadOrient)=(.*)',
         r'.*(NSegments)=(.*)',
         r'.*(RepetitionTime)=(.*)',
         r'.*(NAverages)=(.*)',
         r'.*Obj(OrderScheme)=(.*)',
         r'.*PVM_Spat(Dim)Enum=(.*)']
param_dict = dict()
for p in param:
    with open(mtd_path, 'r') as mtdfile:
        for line in mtdfile:
            if re.search(p, line):
                key = re.sub(p, r'\1', line).strip()
                next_line = next(mtdfile)
                if re.search(r'.*=.*', next_line):
                    value = re.sub(p, r'\2', line).strip()
                else:
                    nvalue = re.sub('[\(\)\{\}\<>]', '', re.sub(p, r'\2', line)).strip()
                    if int(nvalue) > 1:
                        value = next_line.strip().split()
                        try: 
                            value = map(int, value)
                        except:
                            try:
                                value = map(float, value)
                            except:
                                pass
                    else:
                        value = next_line.strip()
                try:
                    value = int(value)
                except:
                    try:
                        value = float(value)
                    except:
                        pass
                param_dict[key] = value
            else:
                pass

# Check reconstruction has been processed or not
try:
    with open(reco_path, 'r') as rcofile:
        for line in rcofile:
            reco_param = r'.*RECO_size=(.*)'
            if re.search(reco_param, line):
                param_dict['Matrix'] = map(int, re.sub(reco_param, r'\1', line).strip().split())
                param_dict['RECO'] = True
except:
    pass

# Check Datatype - Need to be update if Bruker provide more set of dtype
dt = np.dtype('uint16')
if 'dtype' in param_dict.keys():
    if param_dict['dtype'] == '_32BIT_SGN_INT':
        dt = np.dtype('uint32')

# Check Temporal Dimension and resolution
frames = param_dict['Repetitions']
if 'NSegments' in param_dict.keys():
    tr = param_dict['RepetitionTime'] * param_dict['NSegments'] * param_dict['NAverages']
else:
    tr = param_dict['RepetitionTIme'] * param_dict['NAverages']
if frames == 1:
    dim = []
else:
    dim = [frames]

# Check Spatial Dimention, resolution and orientation
# 'L_R'     left to right            (x axis of gradient coil)
# 'A_P'     anterior to posterior    (y axis of gradient coil)
# 'H_F'     head to foot             (z axis of gradient coil)
corrcoef = {'L_R': [-1, 0], 'A_P': [-1, 2], 'H_F': [1, 1]}
ori_dict = {'axial': ['L_R', 'A_P'],
            'coronal': ['L_R', 'H_F'],
            'sagittal': ['A_P', 'H_F']}
rdout_ori = param_dict['ReadOrient']
slice_ori = param_dict['SliceOrient']
matrix = param_dict['Matrix']
slices = param_dict['NSlices']
resol = param_dict['SpatResol']
axis_order = ori_dict[slice_ori][:]
axis_order.extend([ori for ori in corrcoef.keys() if ori not in axis_order])

redirect = [corrcoef[axis][0] for axis in axis_order]

# coronal view has oposite direction, need to check sagittal view later
if slice_ori == 'coronal':
    redirect = list(-np.array(redirect))
redirect.append(1)
reori = [corrcoef[axis][1] for axis in axis_order]
reori.append(3)

if param_dict['Dim'] == '2D':
    matrix.append(slices)
    resol.append(param_dict['SliceDistance'])
    matrix = matrix[::-1]
elif param_dict['Dim'] == '3D':
    matrix = np.roll(matrix, 1)
    resol = np.roll(resol, 1)[::-1]
else:
    dim = [len(slice_ori)]
dim.extend(matrix)

# Make json file - BIDS standard need to be integrated
if args.param:
    print('Exporting parameters to %s.json' % args.filename)
    with open('%s.json' % args.filename, 'w') as fp:
        json.dump(param_dict, fp, ensure_ascii=False)

# Print out the result image dimension and resolution
shape = param_dict['Matrix']
if len(shape) == 3:
    shape.append(frames)
else:
    shape.append(slices)
    shape.append(frames)
print("Image dimension: %d, Image shape: %s" % (len(shape), str(shape)))

# Import image binary from Bruker raw
try:
    img = np.transpose(np.fromfile(img_path, dtype=dt).reshape(dim))
except BaseException("Unmatch matrix size!") as e:
    print(e)
    sys.exit(0)

# Integrate header information from parameters
img_affn = nib.affines.from_matvec(np.diag(resol), np.zeros(3))
img_nii = nib.Nifti1Image(img, img_affn)
affine = img_nii.header.get_base_affine()
# SliceOrder - need to be modified if different case is exist
# img_nii.header['intent_code'] need to be added if DTI image is performed
if param_dict['OrderScheme'] == 'Interlaced':
    slice_code = 3
else:
    slice_code = 0
# img_nii.header['freq_dim'], img_nii.header['slice_dim'], img_nii.header['phase_dim'] need to be added
#   if spiral scan is performed
# img_nii.header['slice_start'], img_nii.header['slice_end'] need to be added if slice padding is performed
if len(dim) == 4:
    img_nii.header.set_xyzt_units('mm', 'sec')
    img_nii.header['pixdim'][4] = float(tr)/1000
    img_nii.header['slice_duration'] = float(tr)/(1000 * slices)
    img_nii.header['slice_code'] = slice_code
else:
    img_nii.header.set_xyzt_units('mm')

img_nii.header['sform_code'] = 0
img_nii.header['qform_code'] = 1
# Apply qform affine matrix
qform = np.diag(img_affn) * redirect
i = np.argsort(reori)
img_nii.set_qform(np.diag(qform)[i, :])

# Save to NifTi file
img_nii.to_filename(args.filename)
img_nii.to_file_map()
