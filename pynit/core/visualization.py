from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors
from matplotlib.backends.backend_agg import FigureCanvasAgg
from skimage import exposure

import numpy as np
import nibabel as nib
import os

# TODO: Check registration, Check atlas, Show Image
# TODO: Simple Video generation or showing over the slice or time
# TODO: Plotting graph (Time)


class Image(object):
    # @staticmethod
    # def slice(path, slice_loc, axis):
    #     pass
    #
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
    def mosaic(img, scale=1, **kwargs):
        """function for generating mosaic figure

        :param img: nibabel object
        :param scale
        :param kwargs
        :return:
        """
        if type(img) is nib.nifti1.Nifti1Image:
            pass
        elif type(img) is str:
            try:
                img = nib.load(img)
            except:
                raise ImportError
        invert_cont = False
        invertx = False
        inverty = False
        invertz = False
        dim = list(img.shape)
        resol = list(img.header['pixdim'][1:4])
        if len(dim) > 3:
            data = np.squeeze(img.get_data(), axis=2)
        else:
            data = img.get_data()
        slice_axis = int(np.argmin(data.shape))
        if kwargs:
            for arg in kwargs.keys():
                if arg == 'invert_cont':
                    invert_cont = kwargs[arg]
                if arg == 'invertx':
                    invertx = kwargs[arg]
                if arg == 'inverty':
                    inverty = kwargs[arg]
                if arg == 'invertz':
                    invertz = kwargs[arg]
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
        # canvas = FigureCanvasAgg(fig)
        if invert_cont:
            cmap = 'gray_r'
        else:
            cmap = 'gray'
        p2 = np.percentile(data, 2)
        p98 = np.percentile(data, 98)
        data = exposure.rescale_intensity(data, in_range=(p2, p98))
        data = np.array(data)
        data = np.swapaxes(data, slice_axis, 2)
        if invertx:
            data = data[::-1, :, :]
        if inverty:
            data = data[:, ::-1, :]
        if invertz:
            data = data[:, :, ::-1]
        for i in range(num_height*num_width):
            ax = axes.flat[i]
            if i < num_of_slice:
                ax.imshow(np.fliplr(data[:, :, i].T), origin='lower', cmap=cmap)
            else:
                ax.imshow(np.zeros((dim[0], dim[1])).T, cmap=cmap)
            ax.patch.set_facecolor('black')
            ax.set_aspect(resol[1]/resol[0])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.subplots_adjust(hspace=.001, wspace=.001)
        return fig, axes

    @staticmethod
    def atlas(template, roi, scale=1, **kwargs):
        invertx = False
        inverty = False
        invertz = False
        legend = False
        if kwargs:
            for arg in kwargs.keys():
                if arg == 'invertx':
                    invertx = kwargs[arg]
                if arg == 'inverty':
                    inverty = kwargs[arg]
                if arg == 'invertz':
                    invertz = kwargs[arg]
                if arg == 'legend':
                    legend = kwargs[arg]
        if type(roi) is nib.nifti1.Nifti1Image:
            atlas = roi.get_data()
            number_of_rois = np.max(atlas)
            list_of_rois = range(number_of_rois)
        elif type(roi) is str:
            if os.path.isfile(roi):
                atlas = nib.load(roi).get_data()
                number_of_rois = np.max(atlas)
                list_of_rois = range(number_of_rois)
            elif os.path.isdir(roi):
                list_of_rois = [img for img in os.listdir(roi) if '.nii' in img]
                number_of_rois = len(list_of_rois)
                for idx, img in enumerate(list_of_rois):
                    if not idx:
                        atlas = nib.load(os.path.join(roi, img)).get_data()
                    else:
                        atlas += nib.load(os.path.join(roi, img)).get_data() * (idx + 1)
            else:
                raise ImportError
        else:
            raise ImportError
        # TODO: add option to generate this
        # atlas_nii = nib.Nifti1Image(atlas, nib.load(os.path.join(roi, img)).affine)
        # atlas_nii.to_filename('atlas.nii')
        # atlas_nii.to_file_map

        slice_axis = int(np.argmin(atlas.shape))
        num_slice = atlas.shape[slice_axis]
        fig, axes = Image.mosaic(template, scale, **kwargs)
        atlas = np.swapaxes(atlas, slice_axis, 2)
        atlas = atlas.astype(float)
        atlas[atlas == 0] = np.nan
        if invertx:
            atlas = atlas[::-1, :, :]
        if inverty:
            atlas = atlas[:, ::-1, :]
        if invertz:
            atlas = atlas[:, :, ::-1]
        colors_for_rois = np.random.rand(int(number_of_rois), 3)
        color_map = zip(list_of_rois, colors_for_rois)
        color_map.insert(0, ('background', [0, 0, 0]))
        bounds = np.linspace(0, number_of_rois+1, number_of_rois+1)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(colors_for_rois+1))
        cmap = colors.ListedColormap(colors_for_rois, 'indexed')
        for i in range(num_slice):
            ax = axes.flat[i]
            ax.imshow(np.fliplr(atlas[:, :, i].T), origin='lower', norm=norm, cmap=cmap)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
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
            fig_leg.set_facecolor('w')
            fig_leg.set_frameon(False)
            return fig, fig_leg
        else:
            return fig



