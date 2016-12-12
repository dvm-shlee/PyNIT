from .core import *

# try:
#     out, err = methods.shell('afni --ver')
# except:
#     methods.raiseerror(core.messages.Errors.DependenceError, 'AFNI is not installed')

load = methods.load

__path = methods.os.path.join(methods.os.sep, *__file__.split(methods.os.sep)[:-2])
def update():
    out, err = methods.shell(methods.os.path.join(__path, "git pull"))
    if err:
        methods.raiseerror(messages.Errors.PackageUpdateFailure, 'Update failed!')
    else:
        print(out)