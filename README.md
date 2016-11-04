# PyNIT (Python NeuroImaging Toolkit)
### Version: 0.1.1

PyNIT is a Python project that provides a useful tools for *in vivo*  animal neuroimaging analysis.

The major function of this package is **project handler** (Project() class) which provides methods to collect and print out information of structured neuroimage project folder (based on [BIDS](http://bids.neuroimaging.io) with filtering function and iterator to utilized the information. This class is mainly utilize [pandas](http://pandas.pydata.org) DataFrame to make easy to handle the hierarchical data structure and multiple database-like informations including subjects, sessions, filename, absolute path, so on. This feature can be utilize for collecting and iterating the absolute path of the image files to load, apply preprocessing steps, and analyze multiple image data easily. 

The purpose of developing this package is to provide the handy environments for processing, manipulating, and visualizing neuroimage data for translational and preclinical neuroscientists who are familiar with python language and jupyter notebook environment. Also aimed to make all preprocessing reproducible, and also easy to archive the finished project with well-shaped data structure.

While we are focusing on preclinical neuroimaging, we provide the Bruker raw data converter **brk2nii** in this package located at 'pynit/bin' folder. The converter is optimized for Bruker 9.4T scanner, Paravision version 5.1 environments. If you have any problems to use this converter for the raw data from other version of Paravision, please provide us the sample rawfiles to test a code. Currently, DTI image cannot be converted whith this converter. We will update soon.

We also provide command-line **checkscans** tools that can print out the session information into terminal windows includes scan number, name of protocol, and brief information for acquisition parameters. We will soon integrate this commands to Django based project organizer to help researcher organize their scan data to BIDS standard.

## Additional features

- Preprocessing() class is developed to apply [AFNI](https://afni.nimh.nih.gov) and [ANTs](http://stnava.github.io/ANTs/) commands for resting state fMRI research. All steps will run though all subjects, session, and file you have. It need to be updated further. Current steps are optimized for functional MRI preprocessing pipeline in [Shihlab](http://shihlab.org)-located at University of North Carolina at Chapel Hill to extract time course from ROIs, generating Z-scored correlation matrix.

- Template() class provides some handy opiton for merging multiple binary ROI into [ITK-snap](http://www.itksnap.org/pmwiki/pmwiki.php) segmentation file with labels, providing some useful methods to manipulate template image such as crop, flip, reslice, swap-axis, and plotting tempalate image with ROIs to generating figure for publication. This class mainly utilize [Nibabel](http://nipy.org/nibabel/) package for handling Nifti file format and [Matplotlib](http://matplotlib.org) for plotting image.

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
Under construction...

## Author
The main author of PyNIT is currently SungHo Lee, please join us to improve this project.
