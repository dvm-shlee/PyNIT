from pynit.pipelines.base import Pipelines
from pynit.pipelines.pipelines import PipeTemplate
from pynit.process import Process
from .tools import methods, gui, HTML as _HTML, display as _display
from .handler import Project, Step, TempFile, \
    Template, Atlas, ImageObj
from .handler.images import load, load_temp


def display(message): return _display(_HTML(message))


__all__ = ['Project', 'Process', 'Step', 'Pipelines', 'PipeTemplate',
           'TempFile', 'Template', 'Atlas',
           'ImageObj']

__version__ = '0.2.1'

# Shortcuts of system methods
shell = methods.shell
update = methods.update
install = methods.install