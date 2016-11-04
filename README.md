# PyNIT (Python NeuroImaging Toolkit)

PyNIT is a Python project that provides a useful tools for *in vivo*  animal neuroimaging analysis.

The major function of this package is **project handler** (Project() class) which provides collect and print out information of  structured neuroimage project folder (mainly based on [BIDS](http://bids.neuroimaging.io) with filtering function and iterator to utilized the information. This class is mainly utilize [pandas](http://pandas.pydata.org) DataFrame to make easy to handle the hierarchical data structure and multiple database-like informations including subjects, sessions, filename, absolute path, so on. This feature can be utilize for collecting and iterating the absolute path of the image files to load, apply preprocessing steps, and analyze multiple image data easily. 

The purpose of developing this package is to provide the handy environments for processing, manipulating, and visualizing neuroimage data to translational and preclinical neuroscientists who are familiar with python language or jupyter environment.

While we are focusing on preclinical neuroimaging, we provide the Bruker raw data converter **brk2nii** in this package at 'pynit/bin' folder. The converter is optimized for Bruker 9.4T scanner, Paravision version 5.1 environments. If you have any problems to use this converter for the raw data from other version of Paravision, please provide us the sample rawfiles to test a code. Currently, DTI image cannot be converted whith this converter. We will update soon.

We also provide command-line **checkscans** tools that can print out the session information into terminal windows includes scan number, name of protocol, and brief information for acquisition parameters. We will soon integrate this commands to Django based project organizer to help researcher organize their scan data to BIDS standard.

# Other features

- Preprocessing() class is developed to apply AFNI and ANTs commands for resting state fMRI research. All steps will run though all subjects, session, and file you have. It need to be updated further. Current steps are optimized for [Shihlab](http://shihlab.org) functional MRI preprocessing pipeline

- Template() class provides some handy opiton for merging multiple binary ROI into ITK-snap segmentation file with labels, image mannipulating such as crop, flip, swap-axis, and plotting tempalate image with ROIs. This class mainly utilize [Nibabel](http://nipy.org/nibabel/) package for handling Nifti file format.

- Other hidden function need to be optimized more. Will update soon.

## Code
Install PyNIT with:
```
git clone https://github.com/dvm-shlee/pynit.git
```

Update code
```
git pull
```

## Examples
Initiate project
```
[IN]
import pynit as pn
prj_path = '[Your project path]'
prj = pn.Project(prj_path)

prj.summary

[OUT]
Project summary
Project: [Your project path]
Selected DataClass: Data

Subject(s): ['Subj1', 'Subj2', .... ]
Session(s): ['Sess1', 'Sess2']
DataType(s): ['func', 'anat']

Applied filters
Set file extension(s): ['.nii', '.nii.gz']
```
