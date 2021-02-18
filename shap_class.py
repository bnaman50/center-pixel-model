import keras
from keras.preprocessing import image as K_image
from keras.layers import Input, Dense, Activation, Flatten, Conv2D
from keras.models import load_model, Model
# import requests
from skimage.segmentation import slic
import time

import sys
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pylab as pl
import numpy as np
import shap
from utils import mkdir_p
import ipdb

import os, sys
# Stop tensorflow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

# Device
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


class shap_analysis:
    def __init__(self, image, model, output_dir, background=255, nsamples=1000):
        '''

        :param image: numpy image with 3 channels
        :param model: keras model
        '''
        self.model = model
        self.img_orig = image
        self.output_dir = output_dir
        self.background = background
        self.nsamples = nsamples

    def _preprocess_input(self, im):
        im = im[:, :, :, 0]
        return  np.reshape(im, (im.shape[0], -1))/255

    def _mask_image(self, zs, segmentation, image, background=None):
        # ipdb.set_trace()
        if background is None:
            background = K_image.mean((0, 1))
        out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
        for i in range(zs.shape[0]):
            out[i, :, :, :] = image
            for j in range(zs.shape[1]):
                if zs[i, j] == 0:
                    out[i][segmentation == j, :] = background
        return out

    # Model
    def _f(self, z):
        aa = self.model.predict(self._preprocess_input(self._mask_image(z, self.segments_slic, self.img_orig, self.background)))
        return aa

    def _fill_segmentation(self, values, segmentation):
        out = np.zeros(segmentation.shape)
        for i in range(len(values)):
            out[segmentation == i] = values[i]
        return out

    def _save_results(self, shap_values, output_dir, neuron, top_k=3):
        r_str = time.strftime("%Y_%m_%d-%H:%M:%S")
        print('Saving extra SHAP results to: ', output_dir)
        mkdir_p(output_dir)
        from matplotlib.colors import LinearSegmentedColormap
        colors = []
        for l in np.linspace(1, 0, 100):
            colors.append((245 / 255, 39 / 255, 87 / 255, l))
        for l in np.linspace(0, 1, 100):
            colors.append((24 / 255, 196 / 255, 93 / 255, l))
        cm = LinearSegmentedColormap.from_list("shap", colors)


        fig, axes = pl.subplots(nrows=1, ncols=top_k+2, figsize=(12, 2+top_k))
        axes[0].imshow(K_image.array_to_img(self.img_orig))
        axes[0].axis('off')
        max_val = np.max([np.max(np.abs(shap_values[i][:, :-1])) for i in range(len(shap_values))])

        list_inds = [neuron] + list(self.top_preds[0][:top_k])
        # ipdb.set_trace()
        for i, idx in enumerate(list_inds):
            m = self._fill_segmentation(shap_values[idx][0], self.segments_slic)
            if i < 1:
                shap_res = m
                prob = self.preds[0, idx]
                axes[i + 1].set_title(f'User seeked explanation \n for Class {idx} \n Prob: {prob:.5f} ',
                                      fontdict={'fontsize': 8, 'fontweight': 'medium'}
                                      )
            else:
                prob = self.preds[0, idx]
                axes[i + 1].set_title(f'Class {idx} \n Prob: {prob:.5f}',
                                      fontdict={'fontsize': 8, 'fontweight': 'medium'}
                                      )
            axes[i + 1].imshow(K_image.array_to_img(self.img_orig).convert('L'), alpha=0.15)
            im = axes[i + 1].imshow(m, cmap=cm, vmin=-max_val, vmax=max_val)
            axes[i + 1].axis('off')

        cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal", aspect=60)
        cb.outline.set_visible(False)
        fig.savefig((output_dir + 'shap_results_' + r_str + '.png'), orientation='landscape', frameon=False)
        pl.close(fig)
        return shap_res

    def explain(self, super_pixels, neuron):

        output_dir = self.output_dir + 'shap_results/'
        img_orig = self.img_orig
        model = self.model
        nsamples = self.nsamples

        self.neuron = neuron

        self.segments_slic = super_pixels

        sp_len = np.unique(super_pixels).shape[0]

        self.preds = model.predict(self._preprocess_input(np.expand_dims(img_orig.copy(), axis=0)))
        print(f'Precition is {np.argmax(self.preds)} with probability {np.amax(self.preds)}')
        self.top_preds = np.argsort(-self.preds)

        explainer = shap.KernelExplainer(self._f, np.zeros((1, sp_len)))
        shap_values = explainer.shap_values(np.ones((1, sp_len)), nsamples=nsamples)  # runs VGG16 1000 times

        return self._save_results(shap_values, output_dir, neuron, top_k=3)

if __name__ == '__main__':
    # load model data
    model = load_model('myModel.h5')
    # model.summary()

    image_size = 227
    num_classes = 256

    # load an image
    file = "data/apple_strawberry.jpg"
    img = K_image.load_img(file, target_size=(image_size, image_size))
    img = img.convert('L')
    img = img.convert('RGB')
    img_orig = K_image.img_to_array(img)
    print('True label is:', img_orig[image_size // 2, image_size // 2, 0])

    # ipdb.set_trace()
    output_dir = "data/apple_strawberry/"

    segments_slic = slic(img, n_segments=50, compactness=30, sigma=3)

    shap_explainer = shap_analysis(img_orig, model, output_dir, background=255, nsamples=1000)
    explanation = shap_explainer.explain(segments_slic, int(img_orig[image_size // 2, image_size // 2, 0]))













