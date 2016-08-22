import numpy as np
import nibabel as nib
from matplotlib import figure as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg
import scipy.interpolate as si
import os, re

def figure_generator(func, temp):
    'function for generating figure'
    p = re.compile('T_(\d{3})seed.nii')
    
    try:
        nib_obj = nib.load(func).get_data()
        T_img = nib_obj[:,:,:]
    
    except:
        T_img = nib.load(os.path.join('Imgs','seed_'+func.split(os.sep)[2][2:5]+'.nii')).get_data()
        T_img = T_img.astype(float)
        T_img = T_img * 5

    T_img[T_img == 0] = np.nan
    S_img = nib.load(os.path.join('Imgs','seed_'+func.split(os.sep)[2][2:5]+'.nii')).get_data()
    S_img = S_img.astype(float)
    S_img[S_img == 0] = np.nan

    fig = plt.Figure(figsize=(12,3))
    canvas = FigureCanvasAgg(fig)
    gs1 = gridspec.GridSpec(2, 6)
    gs2 = gridspec.GridSpec(1, 1)

    gs1.update(wspace=0.0, hspace=0.0, bottom=0.05, top=0.9, left=0.05, right=0.90)
    gs2.update(wspace=0.0, hspace=0.0, bottom=0.1, top=0.85, left=0.93, right=0.94)

    cbaxes = fig.add_subplot(gs2[0])

    for i in range(12):
        j = 11 - i
        axes = fig.add_subplot(gs1[i])
        axes.imshow(np.fliplr(temp[17:110,j,23:93].T), origin='lower', cmap='gray')
        axes.imshow(np.fliplr(T_img[17:110,j,23:93].T), origin='lower', alpha=0.75, vmin=-12, vmax=12)
        axes.imshow(np.fliplr(S_img[17:110,j,23:93].T), origin='lower', alpha=0.75, vmin=-1, vmax=1)
        axes.patch.set_facecolor('black')
        axes.set_xticks([])
        axes.set_yticks([])

    tick_loc = [-0.1, 0, 0.1]
    pcObj = axes.pcolormesh(T_img[0],T_img[1],T_img[2])
    cbar = fig.colorbar(pcObj, ax=axes, cax=cbaxes, ticks=tick_loc)

    cbar.ax.set_yticklabels(['Min', '0', 'Max'])
    fig.savefig(os.path.join('result','pngs',p.match(func.split(os.sep)[-1]).group(1)+".png"))
    return fig

if __name__ == "__main__":

    temp = nib.load(os.path.join('Imgs', 'temp_12slice.nii')).get_data()

    time = range(0, 59, 1)
    print "Video generating script is initiated..."

    for i in time:
        fname = 'T_'+str(i).zfill(3)+'seed.nii'
        fig = figure_generator(os.path.join('result', 'thr', fname), temp)
        print '%s is generated' % fname
        fig.clf()

