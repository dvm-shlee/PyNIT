import numpy as np
import nibabel as nib
from matplotlib import figure as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg
import scipy.interpolate as si
import os, re
import argparse

def voxel_count(path, mid_point = 47):
    'Data parsing for counting voxels'
    pat = re.compile('T_(\d{3})sec.nii')

    file_list, time = [], [0]

    for file in os.listdir(path):
        result = pat.match(file)
        if result:
            file_list.append(file)
            time.append(int(result.group(1)))
    whole, contr = [0], [0]

    for i, file in enumerate(file_list):
        img = nib.load(os.path.join(path, file)).get_data()
        whole.append(np.count_nonzero(img[:,:,:,0,1]))
        contr.append(np.count_nonzero(img[:mid_point,:,:,0,1]))

    return time, whole, contr

def calc_time(time):
    try:
        time_frac = time[-1]/(len(time)-1)
    except:
        time_frac = 1
    total_time = time[-1]
    return time_frac, total_time

def get_diff(data):
    return np.insert(np.diff(data), 0, 0)

def get_idx_i(time, cur_t_pnt):
    time = np.array(time)
    time[time > cur_t_pnt] = 0
    return np.argmax(time)

def get_idx(time, cur_t_pnt):
    return time.index(cur_t_pnt) + 1

def get_spline(x, y):
    t = range(len(x))
    ipl_t = np.linspace(0.0, len(x) - 1, 6000)

    x_tup = si.splrep(t, x)
    y_tup = si.splrep(t, y)
    x_i = si.splev(ipl_t, x_tup)
    y_i = si.splev(ipl_t, y_tup)
    return x_i, y_i

def figure_generator(nii, path, roi, temp):
    'function for generating figure'

    # 1D data collection
    time, whole, contr = voxel_count(path)
    time_frac, total_time = calc_time(time)
    try:
        time_i, whole_i = get_spline(time, whole)
        time_i, contr_i = get_spline(time, contr)
    except:
        pass

    whole_diff = get_diff(whole)
    # Set ylim ranges
    roi_max_ylim = np.max(roi) * 1.8
    roi_min_ylim = np.min(roi) * 1.10

    vox_max_ylim = max(whole) * 1.10

    vox_t_max_ylim = max(whole_diff) * 2
    vox_t_min_ylim = max(whole_diff) * -1

    # Parsing the data
    p = re.compile('T_(\d{3})sec.nii')
    cur_t_pnt = int(p.match(nii).group(1))

    try:
        nib_obj = nib.load(os.path.join(path, nii)).get_data()
        T_img = nib_obj[:,:,:,0,1]
        T_img[T_img == 0] = np.nan
    except:
        T_img = np.zeros(temp.shape)
    
    T_img[T_img == 0] = np.nan
    try:
        c_idx = get_idx(time, cur_t_pnt)
        c_idx_i = get_idx_i(time_i, cur_t_pnt)
    except:
        pass

    # Generating figure and its subplor
    fig = plt.Figure(figsize=(13,7))
    canvas = FigureCanvasAgg(fig)

    gs0 = gridspec.GridSpec(2, 1)
    gs1 = gridspec.GridSpec(2, 6)
    gs2 = gridspec.GridSpec(1, 1)

    gs0.update(wspace=0.0, hspace=0.1, bottom=0.5, top=0.98, left=0.07, right=0.93)
    gs1.update(wspace=0.0, hspace=0.0, bottom=0.05, top=0.449, left=0.05, right=0.90)
    gs2.update(wspace=0.0, hspace=0.0, bottom=0.1, top=0.40, left=0.93, right=0.94)

    roi_axes = fig.add_subplot(gs0[0])
    vox_axes = fig.add_subplot(gs0[1])
    vox_axes_t = vox_axes.twinx()
    cbaxes = fig.add_subplot(gs2[0])

    # Plot data
    for i in range(roi.shape[0]):
        try:
            roi_axes.plot(roi[i][:cur_t_pnt])
        except:
            pass
    roi_axes.text(10, roi_max_ylim * 0.75, "Fraction of Time: '%s sec'" % (cur_t_pnt), size=15)
    roi_axes.set_xticks([])
    roi_axes.set_xlim([0, total_time])
    roi_axes.set_ylim([roi_min_ylim, roi_max_ylim])
    
    try:
        vox_axes.plot(time[:c_idx], whole[:c_idx], 'bo', alpha=0.2, markersize=3)
        vox_axes.plot(time_i[:c_idx_i], whole_i[:c_idx_i], 'b-', lw=3)
        vox_axes.plot(time[:c_idx], contr[:c_idx], 'ro', alpha=0.2, markersize=3)
        vox_axes.plot(time_i[:c_idx_i], contr_i[:c_idx_i], 'r-', lw=3)
    except:
        pass

    try:
        vox_axes.text(10, vox_max_ylim * 0.8, '#Voxels: %d (contr: %d)' % (whole[time.index(cur_t_pnt)], contr[time.index(cur_t_pnt)]), size=15)
    except:
        vox_axes.text(10, vox_max_ylim * 0.8, '#Voxels: 0 (contr: 0)', size=15) 
    try:
        vox_axes_t.plot(time[:c_idx], whole_diff[:c_idx], 'g--', lw=3, alpha=0.5)
    except:
        pass
    vox_axes_t.tick_params(axis='y', colors='Red')
    vox_axes.set_xlim([0, total_time])
    vox_axes.set_ylim([0, vox_max_ylim])
    vox_axes_t.set_ylim([vox_t_min_ylim, vox_t_max_ylim])

    for i in range(12):
        j = 11 - i
        axes = fig.add_subplot(gs1[i])
        axes.imshow(np.fliplr(temp[17:110,j,40:110].T), origin='lower', cmap='gray')
        axes.imshow(np.fliplr(T_img[17:110,j,40:110].T), origin='lower', alpha=0.75, vmin=-20, vmax=20)
        axes.patch.set_facecolor('black')
        axes.set_xticks([])
        axes.set_yticks([])

    tick_loc = [-0.1, 0, 0.1]
    pcObj = axes.pcolormesh(T_img[0],T_img[1],T_img[2])
    cbar = fig.colorbar(pcObj, ax=axes, cax=cbaxes, ticks=tick_loc)

    cbar.ax.set_yticklabels(['Min', '0', 'Max'])
    fig.savefig(p.match(nii).group(1)+".png")
    return fig

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='movie', description='Generating time_fraction dynamic fMRI movie')
    parser.add_argument("path", help="path", type=str)
    parser.add_argument("tf", help="time fraction", type=int)
    parser.add_argument("tt", help="total time", type=int)

    args = parser.parse_args()

    path = args.path
    roi = np.genfromtxt(path+'/'+path+'.roi').T
    temp = nib.load('temp_12slice.nii').get_data()

    time = range(0, args.tt + 1, args.tf)
    print "Video generating script is initiated..."

    for i in time:
        fname = 'T_'+str(i).zfill(3)+'sec.nii'
        fig = figure_generator(fname, path, roi, temp)
        print '%s is generated' % fname
        fig.clf()

