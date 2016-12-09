# Import external packages
import scipy.ndimage as ndimage
from skimage import feature
# Import internal packages
from .methods import InternalMethods, np
from .process import Analysis
import messages
# Import matplotlib for visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
# Import interactive plot in jupyter notebook
try:
    if __IPYTHON__:
        from ipywidgets import interact, fixed
        from IPython.display import Image, display
except:
    pass

# The commented codes below are used for save figure later (mayby?)
# import matplotlib.patches as mpatches
# from matplotlib.backends.backend_agg import FigureCanvasAgg
# Set figure style here
mpl.rcParams['figure.dpi'] = 120
plt.style.use('ggplot')

# R-Python interface for advanced plotting (This is deactivated becuase of compatibility issues
# import rpy2.robjects as robj
# from rpy2.robjects.packages import importr
# from rpy2.robjects import pandas2ri
# pandas2ri.activate()

# TODO: Simple Video generation or showing over the slice or time


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
        slice_num = InternalMethods.check_slice(imageobj, slice_num, axis)
        # Image normalization if norm is True
        if norm:
            data = InternalMethods.apply_p2_98(data)
        else:
            pass
        # Check invert states using given kwargs and apply
        invert = InternalMethods.check_invert(kwargs)
        data = InternalMethods.apply_invert(data, *invert)

        # Internal show slice function for interact python
        def imshow(slice_num, frame=0):
            fig, axes = plt.subplots()
            if len(data.shape) == 3:
                axes.imshow(data[..., int(slice_num)].T, origin='lower', interpolation='nearest', cmap='gray')
            elif len(data.shape) == 4:
                axes.imshow(data[:, :, int(slice_num), frame].T, origin='lower', interpolation='nearest', cmap='gray')
            else:
                raise messages.ImageDimentionMismatched
            axes = InternalMethods.set_viewaxes(axes)
            if resol[1] != resol[0]:
                axes.set_aspect(abs(resol[1] / resol[0]))
            else:
                pass
        # Check image dimension, only 3D and 4D is available
        print(data.shape)
        try:
            if len(data.shape) == 3:
                interact(imshow, slice_num=(0, imageobj.shape[axis]-1), frame=fixed(0))
                print('interact')
            elif len(data.shape) == 4:
                if data.shape[-1] == 1:
                    interact(imshow, slice_num=(0, imageobj.shape[axis] - 1), frame=fixed(0))
                else:
                    interact(imshow, slice_num=(0, imageobj.shape[axis]-1), frame=(0, imageobj.shape[axis+1]-1))
            else:
                raise messages.ImageDimentionMismatched
        except:
            fig, axes = plt.subplots()
            data = InternalMethods.convert_to_3d(imageobj)
            print('notinteract')
            axes.imshow(data[..., int(slice_num)].T, origin='lower', cmap='gray')
            axes.set_axis_off()

    @staticmethod
    def orthogonal(imageobj, norm=True, **kwargs):
        pass

    @staticmethod
    def check_reg(fixed_img, moved_img, scale=15, norm=True, sigma=0.8, **kwargs):
        dim = list(moved_img.shape)
        resol = list(moved_img.header['pixdim'][1:4])
        # Convert 4D image to 3D or raise error
        data = InternalMethods.convert_to_3d(moved_img)
        # Check normalization
        if norm:
            data = InternalMethods.apply_p2_98(data)
        # Set slice axis for mosaic grid
        slice_axis, cmap = InternalMethods.check_sliceaxis_cmap(data, kwargs)
        cmap = 'YlOrRd'
        # Set grid shape
        data, slice_grid, size = InternalMethods.set_mosaic_fig(data, dim, resol, slice_axis, scale)
        fig, axes = Viewer.mosaic(fixed_img, scale=scale, norm=norm, cmap='bone', **kwargs)
        # Applying inversion
        invert = InternalMethods.check_invert(kwargs)
        data = InternalMethods.apply_invert(data, *invert)
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
        return fig

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
        data = InternalMethods.convert_to_3d(imageobj)
        # Check normalization
        if norm:
            data = InternalMethods.apply_p2_98(data)
        # Set slice axis for mosaic grid
        slice_axis, cmap = InternalMethods.check_sliceaxis_cmap(imageobj, kwargs)
        # Set grid shape
        data, slice_grid, size = InternalMethods.set_mosaic_fig(data, dim, resol, slice_axis, scale)
        fig, axes = plt.subplots(slice_grid[1], slice_grid[2], figsize=(size[0], size[1]))
        # Applying inversion
        invert = InternalMethods.check_invert(kwargs)
        data = InternalMethods.apply_invert(data, *invert)
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
        fig.set_facecolor('black')
        plt.subplots_adjust(hspace=.002, wspace=.002)
        return fig, axes

    @staticmethod
    def check_mask(imageobj, maskobj, scale=15, **kwargs):
        # Parsing the information
        dim = list(maskobj.shape)
        resol = list(maskobj.header['pixdim'][1:4])
        num_roi = np.max(maskobj.dataobj)
        # Set slice axis for mosaic grid
        slice_axis, cmap = InternalMethods.check_sliceaxis_cmap(maskobj, kwargs)
        # Set grid shape
        data, slice_grid, size = InternalMethods.set_mosaic_fig(maskobj.dataobj, dim, resol, slice_axis, scale)
        # Applying inversion
        invert = InternalMethods.check_invert(kwargs)
        data = InternalMethods.apply_invert(data, *invert)
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
        slice_axis, cmap = InternalMethods.check_sliceaxis_cmap(atlas, kwargs)
        # Set grid shape
        data, slice_grid, size = InternalMethods.set_mosaic_fig(atlas.dataobj, dim, resol, slice_axis, scale)
        # Applying inversion
        invert = InternalMethods.check_invert(kwargs)
        data = InternalMethods.apply_invert(data, *invert)
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
    def tsplot(dataframe, norm=True, scale=1, err_style='ci_band', **kwargs):
        figsize = (6 * scale, 4 * scale)
        for arg in kwargs.keys():
            if arg is 'figsize':
                figsize = kwargs[arg]
        if norm:
            dataframe = Analysis.linear_norm(dataframe, 0, 1)
        fig = plt.figure(figsize=figsize)
        ax = sns.tsplot(dataframe.T.values, err_style=err_style, **kwargs)

    @staticmethod
    def heatmap(dataframe, half=True, scale=1, vmin=-1.5, vmax=1.5, cmap='RdBu_r', **kwargs):
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

# The class below is deactivated because of compatibility issues
# class RPlot(object):
    # @staticmethod
    # def rcorrplot(dataframe, filename=None, scale=1, mixed=None, **kwargs):
    #     # get R objects
    #     corrplot = importr('corrplot')
    #     grdevices = importr('grDevices')
    #     as_matrix = robj.globalenv.get("as.matrix")
    #     # Parsing arguments
    #     if not filename:
    #         filename = u'.cache/_corrplot.png'
    #         SystemMethods.mkdir('.cache')
    #     size = np.array([512, 512])*scale
    #     # generate figure
    #     grdevices.png(file=filename, width=size[0], height=size[1])
    #     M = as_matrix(dataframe.corr())
    #     if mixed:
    #         corrplot.corrplot_mixed(M, iscorr=False, **kwargs)
    #     else:
    #         corrplot.corrplot(M, **kwargs)
    #     grdevices.dev_off()
    #     display(Image(filename=filename))
