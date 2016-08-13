from __future__ import division

# Import external packages
import os
import numpy as np
import nibabel as nib
from skimage import exposure

# Import interactive plot in jupyter notebook
try:
    if __IPYTHON__:
        from ipywidgets import interact, fixed
except:
    pass

# Import matplotlib for visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors

# The commented codes below are used for save figure later (mayby?)
# import matplotlib.patches as mpatches
# from matplotlib.backends.backend_agg import FigureCanvasAgg

# Set figure style here
mpl.rcParams['figure.dpi'] = 120
plt.style.use('ggplot')

# Import internal packages
from .utility import Internal, Interface

# R-Python interface for advanced plotting
import rpy2.robjects as robj
from IPython.display import Image
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()

# TODO: Check registration, Check atlas, Show Image
# TODO: Simple Video generation or showing over the slice or time
# TODO: Plotting graph (Time)


class Viewer(object):
    @staticmethod
    def slice(img, slice_num=None, norm=True, **kwargs):
        """ Image single slice viewer

        :param img: obj
            ImageObject to see the slice
        :param slice_num: int
            slice
        :param norm: boolean
            norm
        :param kwargs: key, value args
            options
        :return:
        """
        # Parsing the axis information from kwargs
        axis = 2
        if kwargs:
            for arg in kwargs.keys():
                if arg == 'axis':
                    axis = kwargs[arg]
        # Load image data array and swap to given axis
        data = img.dataobj
        data = np.swapaxes(data, axis, 2)
        # Parsing the resolution info
        resol, origin = nib.affines.to_matvec(img.header.get_base_affine())
        resol = np.diag(resol).copy()
        # Swap the affine matrix
        resol[axis], resol[2] = resol[2], resol[axis]
        # Parsing arguments
        if slice_num:
            slice_num = slice_num
        else:
            slice_num = img.shape[axis]/2
        # Image normalization if norm is True
        if norm:
            p2 = np.percentile(data, 2)
            p98 = np.percentile(data, 98)
            data = exposure.rescale_intensity(data, in_range=(p2, p98))
        else:
            pass
        # Check invert states using given kwargs and apply
        invert = Internal.check_invert(kwargs)
        data = Internal.apply_invert(data, *invert)

        # Internal show slice function for interact python
        def imshow(slice_num, frame=0):
            fig, axes = plt.subplots()
            if len(data.shape) == 3:
                axes.imshow(data[..., int(slice_num)].T, origin='lower', interpolation='nearest', cmap='gray')
            elif len(data.shape) == 4:
                axes.imshow(data[:, :, int(slice_num), frame].T, origin='lower', interpolation='nearest', cmap='gray')
            else:
                raise ImportError
            axes = Internal.set_viewaxes(axes)
            if resol[1] != resol[0]:
                axes.set_aspect(abs(resol[1] / resol[0]))

        # Check image dimension, only 3D and 4D is available
        try:
            if len(data.shape) == 3:
                interact(imshow, slice_num=(0, img.shape[axis]-1), frame=fixed(0))
            elif len(data.shape) == 4:
                interact(imshow, slice_num=(0, img.shape[axis]-1), frame=(0, img.shape[axis+1]-1))
            else:
                raise ImportError
        except:
            fig, axes = plt.subplots()
            axes.imshow(data[..., int(slice_num)].T, origin='lower', cmap='gray')
            axes.set_axis_off()

    # @staticmethod
    # def orthogonal(path, coordinate):
    #     pass
    #
    # @staticmethod
    # def atlas(path):
    #     pass
    #
    # @staticmethod
    # def check_reg(moved_img, fixed_img):
    #     pass

    @staticmethod
    def mosaic(img, scale=1, norm=True, **kwargs):
        """function for generating mosaic figure

        :param img: nibabel object
        :param scale
        :param kwargs
        :return:
        """
        # if type(img) is nib.nifti1.Nifti1Image:
        #     pass
        # elif type(img) is str:
        #     try:
        #         img = nib.load(img)
        #     except:
        #         raise ImportError
        dim = list(img.shape)
        resol = list(img.header['pixdim'][1:4])
        if len(dim) > 3:
            data = np.asarray(img.dataobj)[..., 0]
        else:
            data = np.asarray(img.dataobj)
        # Check normalization
        if norm:
            p2 = np.percentile(data, 2)
            p98 = np.percentile(data, 98)
            data = exposure.rescale_intensity(data, in_range=(p2, p98))
        else:
            pass
        slice_axis = int(np.argmin(data.shape))
        if kwargs:
            for arg in kwargs.keys():
                if arg == 'slice_axis':
                    slice_axis = kwargs[arg]
        num_of_slice = dim[slice_axis]
        num_height = int(np.sqrt(num_of_slice))
        num_width = int(round(num_of_slice / num_height))
        resol[2], resol[slice_axis] = resol[slice_axis], resol[2]
        dim[2], dim[slice_axis] = dim[slice_axis], dim[2]
        size_height = num_height * dim[1] * resol[1] * scale / max(dim)
        size_width = num_width * dim[0] * resol[0] * scale / max(dim)
        fig, axes = plt.subplots(num_height, num_width, figsize=(size_width, size_height))
        cmap = 'gray'
        data = np.swapaxes(data, slice_axis, 2)
        invert = Internal.check_invert(kwargs)
        data = Internal.apply_invert(data, *invert)
        for i in range(num_height*num_width):
            ax = axes.flat[i]
            if i < num_of_slice:
                ax.imshow(data[:, :, i].T, origin='lower', interpolation='nearest', cmap=cmap)
            else:
                ax.imshow(np.zeros((dim[0], dim[1])).T, origin='lower', interpolation='nearest', cmap=cmap)
            ax.patch.set_facecolor('black')
            if int(resol[1]/resol[0]) != 1:
                ax.set_aspect(abs(resol[1]/resol[0]))
            ax.set_axis_off()
        fig.set_facecolor('black')
        plt.subplots_adjust(hspace=.002, wspace=.002)
        return fig, axes

    @staticmethod
    def atlas(template, roi, scale=1, **kwargs):
        legend = False
        if kwargs:
            for arg in kwargs.keys():
                if arg == 'legend':
                    legend = kwargs[arg]
        if type(roi) is nib.nifti1.Nifti1Image or Image:
            atlas = np.asarray(roi.dataobj)
            number_of_rois = np.max(atlas)
            list_of_rois = range(number_of_rois)
        elif type(roi) is str:
            atlas = None
            if os.path.isfile(roi):
                atlas = nib.load(roi).get_data()
                number_of_rois = np.max(atlas)
                list_of_rois = range(number_of_rois)
            elif os.path.isdir(roi):
                list_of_rois = [img for img in os.listdir(roi) if '.nii' in img]
                number_of_rois = len(list_of_rois)
                for idx, img in enumerate(list_of_rois):
                    if not idx:
                        atlas = np.asarray(nib.load(os.path.join(roi, img)).dataobj)
                    else:
                        atlas += np.asarray(nib.load(os.path.join(roi, img)).dataobj) * (idx + 1)
            else:
                raise ImportError
        else:
            raise ImportError
        slice_axis = int(np.argmin(atlas.shape))
        num_slice = atlas.shape[slice_axis]
        fig, axes = Viewer.mosaic(template, scale, **kwargs)
        atlas = np.swapaxes(atlas, slice_axis, 2)
        atlas = atlas.astype(float)
        atlas[atlas == 0] = np.nan
        invert = Internal.check_invert(kwargs)
        atlas = Internal.apply_invert(atlas, *invert)
        colors_for_rois = np.random.rand(int(number_of_rois), 3)
        color_map = zip(list_of_rois, colors_for_rois)
        color_map.insert(0, ('background', [0, 0, 0]))
        bounds = np.linspace(0, number_of_rois+1, number_of_rois+1)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(colors_for_rois+1))
        cmap = colors.ListedColormap(colors_for_rois, 'indexed')
        for i in range(num_slice):
            ax = axes.flat[i]
            ax.imshow(atlas[:, :, i].T, origin='lower', interpolation='nearest', norm=norm, cmap=cmap)
            ax.set_axis_off()
        if legend: # TODO: Colormap is not matched

            n = len(color_map)
            ncols = 2
            nrows = int(np.ceil(1. * n / ncols))
            height = nrows * scale * 0.04
            width = ncols * scale * 0.5
            fig_leg, axes_leg = plt.subplots(figsize=(width, height))
            X, Y = fig.get_dpi() * fig.get_size_inches()
            # row height
            h = Y / (nrows + 1)
            # col width
            w = X / ncols
            for i, (name, color) in enumerate(color_map[1:]):
                col = i % ncols
                row = int(i / ncols)
                y = Y - (row * h) - h
                xi_line = w * (col + 0.05)
                xf_line = w * (col + 0.25)
                xi_text = w * (col + 0.3)
                name = os.path.splitext(name)[0]
                axes_leg.text(xi_text, y, name, fontsize=(h * 0.5),
                              horizontalalignment='left',
                              verticalalignment='center')
                axes_leg.hlines(y, xi_line, xf_line, color='black', linewidth=(h * 0.7))
                axes_leg.hlines(y + h * 0.1, xi_line, xf_line, color=color, linewidth=(h * 0.6))
            axes_leg.set_xlim(0, X)
            axes_leg.set_ylim(0, Y)
            axes_leg.set_axis_off()
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
            fig_leg.set_facecolor('black')
            fig_leg.set_frameon(False)
            return fig, fig_leg
        else:
            return fig



