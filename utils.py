## Some Based on utils file in Innvestigate lib. I just modified it as per my use.
##  Credits to them.

from __future__ import\
    absolute_import, print_function, division, unicode_literals
from future.utils import raise_with_traceback, raise_from
# catch exception with: except Exception as e
from builtins import range, map, zip, filter
from io import open
import six
# End: Python 2/3 compatability header small

import sys
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from PIL import Image

import numpy as np
import os
import PIL.Image
import shutil
import ipdb
import time

import keras
from keras.models import Model
from keras.layers import Input, Flatten, Conv2D


def mkdir_p(mypath):

    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise

    return mypath

def plot_image_grid(grid, folderName,
                          row_labels_left,
                          row_labels_right,
                          col_labels,
                          file_name=None,
                          dpi=227,
                          ):
    ## Assuming there is only going to be one row (and many coloumns) in the grid

    plt.rcParams.update({'font.size': 5})
    plt.rc("font", family="sans-serif")
    plt.rc("axes.spines", top=True, right=True, left=True, bottom=True)
    # print('Plotting the figure')
    image_size = (grid[0][0]).shape[0]

    nRows = len(grid)
    nCols = len(grid[0])

    # ipdb.set_trace()
    if image_size > 5:
        tRows = nRows + 3  # total rows
        grid.append(grid[0])
        row_labels_left.append(row_labels_left[0])
        row_labels_right.append(row_labels_right[0])
    else:
        tRows = nRows + 2  # total rows
    tCols = nCols + 2  # total cols

    wFig = tCols  # Figure width (two more than nCols because I want to add ylabels on the very left and very right of figure)
    hFig = tRows  # Figure height (one more than nRows becasue I want to add xlabels to the top of figure)

    fig, axes = plt.subplots(nrows=tRows, ncols=tCols, figsize=(wFig, hFig))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    axes = np.reshape(axes, (tRows, tCols))

    scale = 0.75

    for r in range(tRows):
        # if r <= 1:
        for c in range(tCols):
            ax = axes[r][c]

            l, b, w, h = ax.get_position().bounds

            ax.set_position([l, b, w * scale, h * scale])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

            if r > 0 and c > 0 and r < tRows - 1 and c < tCols - 1:
                # ipdb.set_trace()
                img_data = grid[r - 1][c - 1]
                abs_mn = round(np.amin(img_data), 3)
                abs_mx = round(np.amax(img_data), 3)

                if image_size > 5:
                    if r == tRows-2:
                        centre = image_size//2
                        diff = 2
                        img_data = img_data[centre-diff:centre+diff+1, centre-diff:centre+diff+1]

                if c == 1:
                    cMap = 'gray'
                    # print('Min val is: ', abs_mn)
                    # cen = img_data.shape[0]//2
                    # print('Centre pixel value is: ', (img_data[cen, cen]*255).astype('int'))
                    im = ax.imshow(img_data, interpolation='none', cmap=cMap, vmin=abs_mn, vmax=abs_mx)
                else:
                    uP = cm.get_cmap('Reds', 128)
                    dowN = cm.get_cmap('Blues_r', 128)
                    newcolors = np.vstack((
                        dowN(np.linspace(0, 1, 127)),
                        uP(np.linspace(0, 1, 128))
                    ))
                    cMap = ListedColormap(newcolors, name='RedBlues')
                    abs_mx = max(abs(abs_mn), abs(abs_mx))
                    im = ax.imshow(img_data, interpolation='none', cmap=cMap, vmin=-abs_mx, vmax=abs_mx)

                zero = 0
                if not r - 1:

                    if col_labels != []:
                        ax.set_title(col_labels[c - 1] + '\nmax' + str(abs_mx) + '\nmin' + str(abs_mn),
                                     rotation=45,
                                     horizontalalignment='left',
                                     verticalalignment='bottom')

                    if c == tCols - 2:

                        if row_labels_right != []:
                            txt_right = [l + '\n' for l in row_labels_right[r - 1]]
                            ax2 = ax.twinx()
                            # ax2.axis('off')

                            ax2.set_xticks([])
                            ax2.set_yticks([])
                            ax2.spines['top'].set_visible(False)
                            ax2.spines['right'].set_visible(False)
                            ax2.spines['bottom'].set_visible(False)
                            ax2.spines['left'].set_visible(False)
                            ax2.set_ylabel(''.join(txt_right), rotation=0,
                                           verticalalignment='center',
                                           horizontalalignment='left', )

                if not c - 1:

                    if row_labels_left != []:
                        txt_left = [l + '\n' for l in row_labels_left[r - 1]]
                        ax.set_ylabel(''.join(txt_left),
                                      rotation=0,
                                      verticalalignment='center',
                                      horizontalalignment='right', )

                # else:
                if c != 1:
                    w_cbar = 0.005
                    h_cbar = h * scale
                    b_cbar = b
                    l_cbar = l + scale * w + 0.001
                    cbaxes = fig.add_axes([l_cbar, b_cbar, w_cbar, h_cbar])
                    cbar = fig.colorbar(im, cax=cbaxes)
                    cbar.outline.set_visible(False)
                    cbar.ax.tick_params(labelsize=4, width=0.3, length=1.5)
                    cbar.set_ticks([-abs_mx, zero, abs_mx])
                    cbar.set_ticklabels([-abs_mx, zero, abs_mx])
        #####################################################################################

    dir_path = folderName
    print('Saving figure to {}'.format(dir_path + file_name))

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.savefig((dir_path + '/' + file_name), orientation='landscape', dpi=dpi / scale, transparent=True, frameon=False)
    plt.close(fig)


def preprocess_model_for_lime(model, input_shape=(227, 227, 3)):
    input = Input(shape=input_shape, name='image_input')
    x = Conv2D(1, kernel_size=(1, 1),
               input_shape=input_shape,
               kernel_initializer=keras.initializers.Constant(value=1 / 3),
               )(input)
    x = Flatten()(x)
    x = model(x)

    newModel = Model(inputs=input, outputs=x)

    return newModel

## This is extra function. Not being used anywhere.
## It is there just for future reference
def save_lime_mask(mask, output_dir):
    output_dir = output_dir + 'lime_results/'
    r_str = time.strftime("%Y_%m_%d-%H:%M:%S")
    print('Saving extra LIME results to: ', output_dir)
    mkdir_p(output_dir)
    mask = Image.fromarray(np.uint8(mask*255))
    mask.save(output_dir + 'lime_mask_' + r_str + '.png')

