from .base import BaseProcess
from .ants import ANTs_Process
from .afni import AFNI_Process
from .fsl import FSL_Process


class Process(ANTs_Process, AFNI_Process, FSL_Process):
    def __init__(self, prjobj, name, logging=True, viewer='itksnap'):
        super(Process, self).__init__(prjobj, name, logging, viewer=viewer)