# PyNIT (Python NeuroImaging Toolkit)
### Version: 0.1.1

PyNIT is a Python project that provides a useful tools for *in vivo*  animal neuroimaging analysis.

The major function of this package is **project handler** (Project() class) which provides methods to parse and print out the information of structured neuroimage project folder based on [BIDS](http://bids.neuroimaging.io) with filtering function and iterator to utilized the information. This class mainly utilizes [pandas](http://pandas.pydata.org) DataFrame to make easy to handle the hierarchical structured neuroimaging data and database-like meta information includes subjects, sessions, filename, absolute path. This feature can be used for iterating the absolute path of the image files to load, apply preprocessing steps, and analyze multiple image data. 

The purpose of developing this package is to provide the handy environments for processing, manipulating, and visualizing neuroimage data for translational and preclinical neuroscientists who are familiar with python language and jupyter notebook environment. Also aimed to make all preprocessing reproducible, and also easy to archive the finished project with well-shaped data structure.

While we are focusing on preclinical neuroimaging, we provide the Bruker raw data converter **brk2nii** in this package located at 'pynit/bin' folder. The converter is optimized for Bruker 9.4T scanner, Paravision version 5.1 environments. If you have any problems to use this converter for the raw data from other version of Paravision, please provide us the sample rawfiles to test a code. Currently, DTI image cannot be converted with this converter. We will update this function soon.

We also provide command-line **checkscans** tools that can print out the session information into terminal windows includes scan number, name of protocol, and brief information for acquisition parameters. Above commandline tools will be integrate into Django based project organizer to help researcher organize their scan data to BIDS standard using web-based user interface.

## Guideline of Project folder structure

- Project folder has to have three data class as subfolders : Data, Processing, Results
- Non-processed dataset need to be organized based on BIDS guideline as linked above, and have to be placed at subfolder 'Data'.
- After initiate project, 'Processing' and 'Results' folders will be generated.
- All preprocessed files derived from 'Data' folder will be generated under 'Processing' folder while results images from the preprocessing steps and excel files for the z-scored matrixes are generated under 'Results' folder.

## Additional features

- Preprocess() class is developed to apply [AFNI](https://afni.nimh.nih.gov) and [ANTs](http://stnava.github.io/ANTs/) commandline tools for resting state fMRI. All steps will run though all subjects, session, and file you have. It need to be updated further. Current steps are optimized for functional MRI preprocessing pipeline of [Shihlab](http://shihlab.org)-located at University of North Carolina at Chapel Hill to extract time course from ROIs, generating Z-scored correlation matrix. Example jupyter notebook with sample data will be provided soon.

- Template() class provides some handy opiton for merging multiple binary ROI into [ITK-snap](http://www.itksnap.org/pmwiki/pmwiki.php) segmentation file with labels, providing some useful methods to manipulate template image such as crop, flip, reslice, swap-axis, and plotting tempalate image with ROIs to generating figure for publication. This class mainly utilize [Nibabel](http://nipy.org/nibabel/) package to load and write Nifti files and [Matplotlib](http://matplotlib.org) to plot image into Jupyter notebook environment.

## License
PyNIT is licensed under the term of the GNU GENERAL PUBLIC LICENSE Version 3

## Code
Install PyNIT with:
```
git clone https://github.com/dvm-shlee/pynit.git
```

Update with:
```
git pull
```

## Documentation
Comming soon...


## Usage
Initiate project
```
import pynit as pn

prj_path = '~/2016_Rat_rsfMRI' # Project root folder, which has 'Data' as subfolder
prj = pn.Project(prj_path)
```

Summarizing project information
```
prj.summary
```

Applying filter
```
prj.set_filter('subj1', 'sess1', file_tag='run01', ignore='evoked')
prj.reload()
```

Printing project data structure
```
prj.df
```

Print out absolute path for each file from filtered project instance
```
for i, finfo in prj:
    print(finfo.Abspath)
```

Load Nifti image and plot interactive brain slice
```
img = pn.load('image_path')
img.show()
```

## Author
The main author of PyNIT is currently SungHo Lee, please join us if you want to involve this project.
