from .base import BaseProcess
from .ants import ANTs_Process
from .afni import AFNI_Process
from .fsl import FSL_Process
from .nsp import NSP_Process

class Process(ANTs_Process, AFNI_Process, FSL_Process, NSP_Process):
    def __init__(self, prjobj, name, tag=None, logging=True, viewer='itksnap'):
        super(Process, self).__init__(prjobj, name, tag=tag, logging=logging, viewer=viewer)