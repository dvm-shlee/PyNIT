import numpy as np


class Calc(object):
    @staticmethod
    def linear_norm(img, newMin, newMax):
        "Linear normalization of the grayscale digital image"
        return (img - np.min(img)) * (newMax - newMin) / (np.max(img) - np.min(img)) - newMin