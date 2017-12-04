import methods
import messages
from IPython import get_ipython
try:
    from ipywidgets import widgets
except:
    pass

notebook_env = False

if get_ipython() and len(get_ipython().config.keys()):
    from tqdm import tqdm_notebook as progressbar
    from ipywidgets.widgets import HTML as HTML
    from IPython.display import display, clear_output
    notebook_env = True

else:
    from pprint import pprint as display
    from tqdm import tqdm as progressbar

    def HTML(message): return message

    def clear_output(): pass

def display_html(message):
    return display(HTML(message))