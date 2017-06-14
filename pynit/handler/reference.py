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
    pipelines = [pipe for pipe in dir(pipelines) if 'PipeTemplate' not in pipe if '__' not in pipe]

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
        pipe = 'Available pipelines:\t{}'.format(self.pipelines)
        output = '{}\n{}\n{}\n{}\n{}\n{}'.format(title,'-'*len(title), img, txt, ds, pipe)
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

    @property
    def avail(self):
        n_pipe = len(self.pipelines)
        output = dict(zip(range(n_pipe), self.pipelines))
        # for i, values in output.items():
        #     output[i] = values[2:]
        return output

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