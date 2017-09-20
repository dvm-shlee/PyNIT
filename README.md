# PyNIT (Python NeuroImaging Toolkit)
### Version: 0.1.3

PyNIT is a pipeline tool for in vivo neuroimaging, especially optimized for rodent, that runs on the [Jupyter Notebook](http://jupyter-notebook.readthedocs.io/en/stable/), an interactive research computing environment. This package provides hierarchically designed classes to help easily create and apply the pipelines using [FSL](https://fsl.fmrib.ox.ac.uk), [AFNI](https://afni.nimh.nih.gov), and [ANTs](http://stnava.github.io/ANTs/), which are the most used packaged in fMRI analysis. For consistent behavior, it is designed to work only with datasets that follow the BIDS ([Brain Imaging Data Structure](http://bids.neuroimaging.io)) guidelines.

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

## Author
The main author of PyNIT is currently SungHo Lee, please join us if you want to involve this project.
