from base import BaseProcessor


class Step(BaseProcessor):
    """ Template for a processing step

    This class simply allows you to design processing steps, that needs to combine multiple command line tools in
    several fMRI imaging package such as AFNI, ANTs, and FSL.
    The fundamental mechanism is that by applying given inputs, outputs, and command, this class generating
    customized function and executed it.
    """
    def __init__(self, procobj, n_thread='max'):
        super(Step, self).__init__(procobj, n_thread=n_thread)


