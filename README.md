# PyNIT (Python NeuroImaging Toolkit)
### Version: 0.2.1

PyNIT is a pipeline tool developed for using in preclinical neuroimaging analysis, focusing on fMRI data, that runs on the [Jupyter Notebook](http://jupyter-notebook.readthedocs.io/en/stable/), an interactive research computing environment. This package provides various object classes to help easily create and apply the pipelines using commandline tools such as [FSL](https://fsl.fmrib.ox.ac.uk), [AFNI](https://afni.nimh.nih.gov), and [ANTs](http://stnava.github.io/ANTs/), as well as the custom command line tools [PyNSP](https://github.com/dvm-shlee/PyNSP) (python neuro-signal processing tool). For consistent behavior, currently, this pipeline tool only compatible with BIDS ([Brain Imaging Data Structure](http://bids.neuroimaging.io)) guidelines.

### Python version support
Officially only Python 2.7; Since the Python core team announce that they plans to stop supporting Python 2.7 on January 1st, 2020, PyNIT for Python 3 is currently under-developing from the scratch.

## Installation
### installing from PyPi 
Install jupyter and enable widget extension

```
pip install jupyter
jupyter nbextension enable --py widgetsnbextension
```

To install PyNIT module:
```angular2html
$ pip install pynit
```

Update with:
```
pip install --upgrade pynit
```

### Dependencies
#### Python modules
- numpy
- jupyter 
- tqdm
- pandas
- nibabel
- pynsp
#### Command line tools
- AFNI
- ANTs
- FSL

## Guideline of Project folder structure

- Project folder has to have three data class as subfolders : Data, Processing, Results
- Non-processed dataset need to be organized based on BIDS guideline as linked above, and have to be placed at subfolder 'Data'.
- After initiate project, 'Processing' and 'Results' folders will be automatically generated.

### Example of project data structure
#### Single-session project
- Project_main_folder
    - Data
        - sub-01
            - anat
                - sub-01_T2w.nii.gz
            - func<
                - sub-01_task-optoP1_run-01_bold.nii.gz
                - sub-01_task-optoP1_run-02_bold.nii.gz
                - sub-01_task-optoP2_run-01_bold.nii.gz
                - sub-01_task-optoP2_run-02_bold.nii.gz
                - sub-01_task-rest_bold.nii.gz
            - cbv
                - sub-01_task-infusion_run-01_cbv.nii.gz
        - sub-02
            - anat
                ...

#### Multi-session project
- Project_main_folder
    - Data</il>
        - sub-01
            - ses-01
                - anat
                    - sub-01_T2w.nii.gz
                - func<
                    - sub-01_task-optoP1_run-01_bold.nii.gz
                    - sub-01_task-optoP1_run-02_bold.nii.gz
                    - sub-01_task-optoP2_run-01_bold.nii.gz
                    - sub-01_task-optoP2_run-02_bold.nii.gz
                    - sub-01_task-rest_bold.nii.gz
                - cbv
                    - sub-01_task-infusion_run-01_cbv.nii.gz
            - ses-02
                - anat
                    - ...
        - sub-02
            - ses-01
                - anat
                    ...
            - ses-02
                - anat
                    ...

## Usage
### Dataset handling
```angular2html

```
### Process handling
```angular2html

```
### Process debuging
```angular2html

```
### Pipeline processing
```angular2html

```
## License
PyNIT is licensed under the term of the GNU GENERAL PUBLIC LICENSE Version 3

## Author
The main author of PyNIT is currently SungHo Lee, please join us if you want to involve this project.
