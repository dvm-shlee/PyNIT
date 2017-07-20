from .base import BaseProcess
from .ants import ANTs_Process
from .afni import AFNI_Process
from .fsl import FSL_Process
from .pn import PN_Process


class Process(ANTs_Process, AFNI_Process, FSL_Process, PN_Process):
    def __init__(self, prjobj, name, parallel=True, logging=True, viewer='itksnap'):
        super(Process, self).__init__(prjobj, name, parallel, logging, viewer)
        # self.__super = super(Process, self)
        # self.__super.__init__(prjobj, name, parallel, logging, viewer)