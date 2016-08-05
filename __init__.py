from core.niph import Project as ph
from core.niph import Pipeline as pipe
from core.visualization import Image as viewer
# from core.statics import InternalMethods
from core.commands import Commands as tools
# from pipelines.pipeline import Pipeline as pipe
from template.template import Template

__version__ = '0.1.dev1'
__Author__ = 'SungHo Lee (shlee@unc.edu)'
__all__ = ['ph', 'pipe', 'viewer', 'tools', 'Template']
print('PyNIT(Python based NeuroImaging Toolkit)')