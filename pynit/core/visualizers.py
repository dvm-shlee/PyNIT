# Import external packages
import sys
import scipy.ndimage as ndimage
from skimage import feature

# Import internal packages
from .methods import np
from .processors import Analysis
import messages
import methods

# Import matplotlib for visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from IPython import display

# Set error bar as standard deviation


def _plot_std_bars(central_data=None, ci=None, data=None, *args, **kwargs):
    std = data.std(axis=0)
    ci = np.asarray((central_data - std, central_data + std))
    kwargs.update({"central_data": central_data, "ci": ci, "data": data})
    sns.timeseries._plot_ci_bars(*args, **kwargs)


def _plot_std_band(central_data=None, ci=None, data=None, *args, **kwargs):
    std = data.std(axis=0)
    ci = np.asarray((central_data - (std), central_data + (std)))
    kwargs.update({"central_data": central_data, "ci": ci, "data": data})
    sns.timeseries._plot_ci_band(*args, **kwargs)

sns.timeseries._plot_std_bars = _plot_std_bars
sns.timeseries._plot_std_band = _plot_std_band

# Import interactive plot in jupyter notebook
if len([key for key in sys.modules.keys() if key == 'ipykernel']):
    from ipywidgets import interact, fixed
    from IPython.display import Image, display
    jupyter_env = True
else:
    jupyter_env = False

# The commented codes below are used for save figure later (maybe?)
# import matplotlib.patches as mpatches
# from matplotlib.backends.backend_agg import FigureCanvasAgg

# Set figure style here
mpl.rcParams['figure.dpi'] = 120
# plt.style.use('ggplot')
plt.style.use('classic')
plt.style.use('seaborn-notebook')

# Set colormap
cdict = {'red': ((0.0, 0.0, 0.0),
                 (0.5, 0.0, 0.0),
                 (0.5000000001, 0.8, 0.8),
                 (1.0, 1.0, 1.0)),
         'green': ((0.0, 1.0, 1.0),
                   (0.40, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (0.60, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),
         'blue': ((0.0, 1.0, 1.0),
                  (0.4999999999, 1.0, 0.5),
                  (0.5, 0.0, 0.0),
                  (1.0, 0.0, 0.0))
         }

plt.register_cmap(name='shihlab', data=cdict)


class Viewer(object):
    @staticmethod
    def slice(imageobj, slice_num=None, norm=True, **kwargs):
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
        data = imageobj.dataobj
        data = np.swapaxes(data, axis, 2)
        # Parsing the resolution info
        resol = list(imageobj.header['pixdim'][1:4])
        # Swap the affine matrix
        resol[axis], resol[2] = resol[2], resol[axis]
        # Parsing arguments
        slice_num = methods.check_slice(imageobj, slice_num, axis)
        # Image normalization if norm is True
        if norm:
            data = methods.apply_p2_98(data)
        else:
            pass
        # Check invert states using given kwargs and apply
        invert = methods.check_invert(kwargs)
        data = methods.apply_invert(data, *invert)

        # Internal show slice function for interact python
        def imshow(slice_num, ax, frame=0, stat=0):
            plt.clf()
            if len(data.shape) == 3:
                plt.imshow(data[..., int(slice_num)].T, origin='lower', interpolation='nearest', cmap='gray')
            elif len(data.shape) == 4:
                plt.imshow(data[:, :, int(slice_num), frame].T, origin='lower', interpolation='nearest', cmap='gray')
            elif len(data.shape) == 5:
                plt.imshow(data[:, :, int(slice_num), frame, stat].T, origin='lower', interpolation='nearest', cmap='gray')
            else:
                raise messages.ImageDimentionMismatched
            ax = methods.set_viewaxes(plt.axes())
            ax.set_facecolor('white')
            if resol[1] != resol[0]:
                ax.set_aspect(abs(resol[1] / resol[0]))
            else:
                pass
            if jupyter_env:
                display(plt.gcf())

        # Check image dimension, only 3D and 4D is available
        if jupyter_env:
            fig, ax = plt.subplots()
            fig.patch.set_facecolor('white')
            # ax = plt.axes()
            if len(data.shape) == 3:
                interact(imshow, slice_num=(0, imageobj.shape[axis]-1), ax=fixed(ax), frame=fixed(0), stat=fixed(0))
            elif len(data.shape) == 4:
                if data.shape[-1] == 1:
                    interact(imshow, slice_num=(0, imageobj.shape[axis] - 1), ax=fixed(ax), frame=fixed(0),
                             stat=fixed(0))
                else:
                    interact(imshow, slice_num=(0, imageobj.shape[axis]-1), ax=fixed(ax),
                             frame=(0, imageobj.shape[axis+1]-1), stat=fixed(0))
            elif len(data.shape) == 5:
                interact(imshow, slice_num=(0, imageobj.shape[axis] - 1), ax=fixed(ax),
                         frame=(0, imageobj.shape[axis+1]-1), stat=(0, imageobj.shape[axis+2]-1))

            else:
                raise messages.ImageDimentionMismatched
        else:
            fig, axes = plt.subplots()
            data = methods.convert_to_3d(imageobj)
            axes.imshow(data[..., int(slice_num)].T, origin='lower', cmap='gray')
            axes.set_axis_off()
            if jupyter_env:
                display(fig)

    @staticmethod
    def orthogonal(imageobj, norm=True, **kwargs):
        pass

    @staticmethod
    def check_reg(fixed_img, moved_img, scale=15, norm=True, sigma=0.8, **kwargs):
        dim = list(moved_img.shape)
        resol = list(moved_img.header['pixdim'][1:4])
        # Convert 4D image to 3D or raise error
        data = methods.convert_to_3d(moved_img)
        # Check normalization
        if norm:
            data = methods.apply_p2_98(data)
        # Set slice axis for mosaic grid
        slice_axis, cmap = methods.check_sliceaxis_cmap(data, kwargs)
        cmap = 'YlOrRd'
        # Set grid shape
        data, slice_grid, size = methods.set_mosaic_fig(data, dim, resol, slice_axis, scale)
        fig, axes = Viewer.mosaic(fixed_img, scale=scale, norm=norm, cmap='bone', **kwargs)
        fig.set_facecolor('black')
        # Applying inversion
        invert = methods.check_invert(kwargs)
        data = methods.apply_invert(data, *invert)
        # Plot image
        for i in range(slice_grid[1] * slice_grid[2]):
            ax = axes.flat[i]
            edge = data[:, :, i]
            edge = feature.canny(edge, sigma=sigma)  # edge detection for second image
            # edge = ndimage.gaussian_filter(edge, 3)
            mask = np.ones(edge.shape)
            sx = ndimage.sobel(edge, axis=0, mode='constant')
            sy = ndimage.sobel(edge, axis=1, mode='constant')
            sob = np.hypot(sx, sy)
            mask[sob == False] = np.nan
            m_norm = colors.Normalize(vmin=0, vmax=1.5)
            if i < slice_grid[0]:
                ax.imshow(mask.T, origin='lower', interpolation='nearest', cmap=cmap, norm=m_norm, alpha=0.8)
            else:
                ax.imshow(np.zeros((dim[0], dim[1])).T, origin='lower', interpolation='nearest', cmap=cmap)
        if jupyter_env:
            display(fig)
        return fig, axes

    @staticmethod
    def mosaic(imageobj, scale=15, norm=True, **kwargs):
        """function for generating mosaic figure

        :param img: nibabel object
        :param scale
        :param norm
        :param kwargs
        :return:
        """
        dim = list(imageobj.shape)
        resol = list(imageobj.header['pixdim'][1:4])
        # Convert 4D image to 3D or raise error
        data = methods.convert_to_3d(imageobj)
        # Check normalization
        if norm:
            data = methods.apply_p2_98(data)
        # Set slice axis for mosaic grid
        slice_axis, cmap = methods.check_sliceaxis_cmap(imageobj, kwargs)
        # Set grid shape
        data, slice_grid, size = methods.set_mosaic_fig(data, dim, resol, slice_axis, scale)
        fig, axes = plt.subplots(slice_grid[1], slice_grid[2], figsize=(size[0], size[1]))
        fig.set_facecolor('black')
        # Applying inversion
        invert = methods.check_invert(kwargs)
        data = methods.apply_invert(data, *invert)
        # Plot image
        for i in range(slice_grid[1] * slice_grid[2]):
            ax = axes.flat[i]
            if i < slice_grid[0]:
                ax.imshow(data[:, :, i].T, origin='lower', interpolation='nearest', cmap=cmap)
            else:
                ax.imshow(np.zeros((dim[0], dim[1])).T, origin='lower', interpolation='nearest', cmap=cmap)
            ax.patch.set_facecolor('black')
            if int(resol[1]/resol[0]) != 1:
                ax.set_aspect(abs(resol[1]/resol[0]))
            ax.set_axis_off()
        plt.subplots_adjust(hspace=.002, wspace=.002)
        return fig, axes

    @staticmethod
    def check_mask(imageobj, maskobj, scale=15, **kwargs):
        """

        :param imageobj:
        :param maskobj:
        :param scale:
        :param kwargs:
        :return:
        """
        # Parsing the information
        dim = list(maskobj.shape)
        resol = list(maskobj.header['pixdim'][1:4])
        num_roi = np.max(maskobj.dataobj)
        # Set slice axis for mosaic grid
        slice_axis, cmap = methods.check_sliceaxis_cmap(maskobj, kwargs)
        # Set grid shape
        data, slice_grid, size = methods.set_mosaic_fig(maskobj.dataobj, dim, resol, slice_axis, scale)
        # Applying inversion
        invert = methods.check_invert(kwargs)
        data = methods.apply_invert(data, *invert)
        try:
            fig, axes = Viewer.mosaic(imageobj, scale=scale, **kwargs)
        except:
            raise messages.InputObjectError
        # Make transparent
        data = data.astype(float)
        data[data == 0] = np.nan
        # Plot image
        for i in range(slice_grid[1] * slice_grid[2]):
            ax = axes.flat[i]
            ax.imshow(data[:, :, i].T, origin='lower', interpolation='nearest', cmap='PuRd_r')
            ax.set_axis_off()
        return fig

    @staticmethod
    def atlas(tempobj, atlasobj, scale=15, **kwargs):
        """

        :param tempobj:
        :param atlasobj:
        :param scale:
        :param kwargs:
        :return:
        """
        # Check argument for legend generation
        legend = False
        bilateral = False
        contra = False
        if kwargs:
            for arg in kwargs.keys():
                if arg == 'legend':
                    legend = kwargs[arg]
                if arg == 'contra':
                    contra = kwargs[arg]
                if arg == 'bilateral':
                    bilateral = kwargs[arg]
        # Parsing the information
        try:
            atlas = atlasobj.image
            label = atlasobj.label
        except:
            raise messages.InputObjectError
        dim = list(atlas.shape)
        resol = list(atlas.header['pixdim'][1:4])
        # Set slice axis for mosaic grid
        slice_axis, cmap = methods.check_sliceaxis_cmap(atlas, kwargs)
        # Set grid shape
        data, slice_grid, size = methods.set_mosaic_fig(atlas.dataobj, dim, resol, slice_axis, scale)
        # Applying inversion
        invert = methods.check_invert(kwargs)
        data = methods.apply_invert(data, *invert)
        try:
            fig, axes = Viewer.mosaic(tempobj, scale=scale, **kwargs)
        except:
            raise messages.InputObjectError
        # Check side to present, default is usually right side
        if contra:
            data = data[::-1, :, :]
        if bilateral:
            data += data[::-1, :, :]
        # Make transparent
        data = data.astype(float)
        data[data == 0] = np.nan
        number_of_rois = len(label.keys())
        colors_for_rois = [label[idx][1] for idx in label.keys()]
        bounds = np.linspace(0, number_of_rois, number_of_rois)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=number_of_rois)
        cmap = colors.ListedColormap(colors_for_rois[1:], 'indexed')
        # Plot image
        for i in range(slice_grid[1] * slice_grid[2]):
            ax = axes.flat[i]
            ax.imshow(data[:, :, i].T, origin='lower', interpolation='nearest', norm=norm, cmap=cmap)
            ax.set_axis_off()
        if legend:
            ncols = 2
            nrows = int(np.ceil(1. * (number_of_rois-1) / ncols))
            height = nrows * scale * 0.04
            width = ncols * scale * 0.5
            fig_leg, axes_leg = plt.subplots(figsize=(width, height))
            X, Y = fig.get_dpi() * fig.get_size_inches()
            # row height
            h = Y / (nrows + 1)
            # col width
            w = X / ncols
            for idx in label.keys():
                if idx == 0:
                    pass
                else:
                    col = (idx - 1)% ncols
                    row = int((idx - 1) / ncols)
                    y = Y - (row * h) - h
                    xi_line = w * (col + 0.05)
                    xf_line = w * (col + 0.15)
                    xi_text = w * (col + 0.2)
                    axes_leg.text(xi_text, y, label[idx][0], fontsize=(1 * scale),
                                  horizontalalignment='left',
                                  verticalalignment='center')
                    axes_leg.hlines(y, xi_line, xf_line, color='black', linewidth=(h * 0.2))
                    axes_leg.hlines(y + h * 0.05, xi_line, xf_line, color=label[idx][1], linewidth=(h * 0.2))
            axes_leg.set_xlim(0, X)
            axes_leg.set_ylim(0, Y)
            axes_leg.set_axis_off()
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
            fig_leg.set_facecolor('black')
            fig_leg.set_frameon(False)
            return fig, fig_leg
        else:
            return fig


class Plot(object):
    @staticmethod
    def tsplot(df, figsize=None, xlim=None, ylim=None, xlabel=None, ylabel=None, label_size=None, tick_size=None,
               title=None, title_size=None, **kwargs):
        """

        :param df:
        :param figsize:
        :param xlim:
        :param ylim:
        :param xlabel:
        :param ylabel:
        :param label_size:
        :param tick_size:
        :param title:
        :param title_size:
        :param kwargs:
        :return:
        """
        fig, axes = plt.subplots(1,1,figsize=figsize)
        fig.patch.set_facecolor('white')
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if title:
            axes.set_title(title, size=title_size)
        if xlabel:
            axes.set_xlabel(xlabel, size=label_size)
        else:
            axes.set_xlabel('Time (s)', size=label_size)
        if ylabel:
            axes.set_ylabel(ylabel, size=label_size)
        else:
            axes.set_ylabel('Responses', size=label_size)
        axes.tick_params(labelsize=tick_size, direction='out')
        sns.tsplot(df.T.values, err_style='std_band', ax = axes, **kwargs)
        return fig, axes

    @staticmethod
    def heatmap(dataframe, half=True, scale=1, vmin=-1.5, vmax=1.5, cmap='RdBu_r', **kwargs):
        """

        :param dataframe:
        :param half:
        :param scale:
        :param vmin:
        :param vmax:
        :param cmap:
        :param kwargs:
        :return:
        """
        figsize = (6 * scale, 4 * scale)
        for arg in kwargs.keys():
            if arg is 'figsize':
                figsize = kwargs[arg]
        if half:
            mask = np.zeros_like(dataframe.corr())
            mask[np.triu_indices_from(mask)] = True
        else:
            mask = None
        fig = plt.figure(figsize=figsize)
        with sns.axes_style("white"):
            ax = sns.heatmap(np.arctanh(dataframe.corr()), mask=mask, vmin=vmin, vmax=vmax,
                             cmap=cmap, square=True, **kwargs)
            ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
            ax.set_yticklabels(ax.yaxis.get_majorticklabels(), rotation=45)