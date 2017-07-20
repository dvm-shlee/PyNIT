from pynit.pipelines.base import Pipelines
from pynit.process import Process

from .tools import methods, viewer, visualizers
from .handler import Project, Step, TempFile, Template, Atlas, ImageObj, Reference
from .handler.images import load, load_temp
from ipywidgets.widgets import HTML as title

def display(message):
    return visualizers.display(title(message))

__all__ = ['Project', 'Process', 'Step', 'Pipelines', 'TempFile', 'Template', 'Atlas', 'ImageObj', 'Reference']
shell = methods.shell