import sys
from base import BaseProcessor

#########################################
# The imported modules belows           #
# check jupyter notebook environment    #
#########################################
jupyter_env = False
try:
    if len([key for key in sys.modules.keys() if 'ipykernel' in key]):
        from tqdm import tqdm_notebook as progressbar
        from ipywidgets import widgets
        from ipywidgets.widgets import HTML as title
        from IPython.display import display, display_html
        jupyter_env = True
    else:
        from tqdm import tqdm as progressbar
except:
    pass


class Step(BaseProcessor):
    """ Template for a processing step

    This class simply allows you to design processing steps, that needs to combine multiple command line tools in
    several fMRI imaging package such as AFNI, ANTs, and FSL.
    The fundamental mechanism is that by applying given inputs, outputs, and command, this class generating
    customized function and executed it.
    """
    def __init__(self, procobj, n_thread='max'):
        super(Step, self).__init__(procobj, n_thread=n_thread)


