import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

import cv2
import sys
import numpy as np
from PIL import Image
import ipdb
# import shutil
import time
import os
from utils import mkdir_p
import settings
import matplotlib.pyplot as plt
import imageio
from termcolor import colored

import warnings
warnings.filterwarnings("ignore")

import argparse

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

def is_number(n):
    is_number = True
    try:
        num = float(n)
        # check for "nan" floats
        is_number = num == num   # or use `math.isnan(num)`
    except ValueError:
        is_number = False
    return is_number

def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Input paramters for meaningful perturbation explanation of the image')

    # Add the paramters positional/optional (here only optional)
    parser.add_argument( '-ip', '--img_path', help='Path to the image' )
    parser.add_argument( '-mp', '--model_path', 
                         help='Path to the model that want to analyze', 
                        )
    
    parser.add_argument( '-l', '--labels', type=int,
                         help='Select the label for which you want to find the importance (options - random_num in range [0-255] ). \
                               Default is True/pred (since 100 percentage accuracy)',
                         #nargs='+',
                         #default="True" 
                         )

    parser.add_argument( '-op', '--out_path', help='Path of the output directory where you want to save the results (Default is ./img_name/)')

    parser.add_argument('-cp', '--centre_pixel', type=int,
                        help='Centre pixel of the image (options - random_num in range [0-255] )',
                        # nargs='+',
                        )

    parser.add_argument('-nsi', '--np_seed_image', type=int,
                        help='Numpy seed for random image generation (int). \
                        Will be ignored if image path is provided ',
                        )

    parser.add_argument('-nsm', '--np_seed_mask', type=int,
                        help='Numpy seed for random mask initial generation (int). \
                            Will be ignored if mask_type is not random',
                        )

    parser.add_argument('-tvb', '--tv_beta', type=float,
                        help='TV_Beta value', default=3.0
                        )

    parser.add_argument('-tvc', '--tv_coeff', type=float,
                        help='TV Coefficient value', default=100,
                        )

    parser.add_argument('-l1c', '--l1_coeff', type=float,
                        help='L1 coefficient value', default=1,
                        )

    parser.add_argument('-cc', '--category_coeff', type=float,
                        help='Category coefficient value', default=1,
                        )

    parser.add_argument('-lr', '--learning_rate', type=float,
                        help='Learning rate', default=0.01,
                        )

    parser.add_argument('-mi', '--max_iter', type=int,
                        help='Maximum Iterations', default=1000,
                        )

    parser.add_argument('-gpu', '--gpu', type=int, choices=range(8),
                        help='GPU index', default=0,
                        )

    # Parse the arguments
    args = parser.parse_args()
    
    #ipdb.set_trace()
    if args.model_path is None:
        print("\nProvide a path to the model (PyTorch model) to be analyzed\n")
        sys.exit(1)
    else:
        pass

    if args.out_path is None:
        args.out_path = './'
    args.out_path = os.path.abspath(args.out_path) + '/'

    #ipdb.set_trace()
    if args.img_path is None:
        print("\nImage path not given.\nWill create a random image and analyze it.")
    elif os.path.isfile(args.img_path):
        pass
    else:
        print('\nIncorrect file path providing.\nExiting\n')
        sys.exit(1)

    if args.labels is not None:
        if args.labels < 0 or args.labels > 255:
            parser.error("-l/--labels: must be in the range [0, 255].")


    if args.centre_pixel is not None:
        if args.centre_pixel < 0 or args.centre_pixel > 255:
            parser.error("-cp/--centre_pixel: must be in the range [0, 255].")


    if args.np_seed_image is not None:
        if args.np_seed_image < 0 or args.np_seed_image >= 2**32:
            parser.error("-nsm/--np_seed_mask: must be in the range [0, 2^32).")

    if args.np_seed_mask is not None:
        if args.np_seed_mask < 0 or args.np_seed_mask >= 2**32:
            parser.error("-nsm/--np_seed_mask: must be in the range [0, 2^32).")

    if args.max_iter < 0:
        parser.error("-mi/--max_iter: must be a positive integer")


    # ipdb.set_trace()
    return args

class meaningful_perturbation:
    def __init__(self, image, model_path, output_dir,
                 num_classes=256, img_size=227, 
                 tv_beta = 3, learning_rate = 0.1, 
                 max_iterations = 1000, l1_coeff = 0.01,
                 tv_coeff = 0.2, category_coeff=1, mask_size = 227, dev_idx=0
                 ):
        # ipdb.set_trace()
        self.dev_idx = dev_idx
        self.image = image
        #self.clamped_class = clamped_class
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = self.load_model(model_path, num_classes=num_classes, n=img_size*img_size)

        self.tv_beta = tv_beta
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.l1_coeff = l1_coeff
        self.tv_coeff = tv_coeff
        self.category_coeff = category_coeff
        self.mask_size = mask_size
        self.output_dir = output_dir


    def flatten_img(self, img):
        #############################################
        # Resize the input
        # Size of img = [1, 1, 227, 227]
        size = img.size()
        img = img.view(-1, size[1] * size[2] * size[3])
        # equivalent to img = img.view(-1, img_size * img_size * C) where C is color channels (1 here)
        return img

    def resize_img(self, img, img_size):
        return torch.reshape(img, (1, 1, img_size, img_size)) # [Samples, Channels, Height, Width]

    def tv_norm(self, input, tv_beta):
        # ipdb.set_trace()
        img = input[0, 0, :] # imgs 1x1x227x227
        row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
        col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
        return row_grad + col_grad


    def preprocess_image(self, img):

        preprocessed_img = img.copy()

        # ascontiguousarray is nothing but jumla. Makes a contiguous array in memory
        preprocessed_img = np.ascontiguousarray(preprocessed_img)

        if use_cuda:
            preprocessed_img_tensor = torch.from_numpy(preprocessed_img).cuda(self.dev_idx)
        else:
            preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

        ## Adding a dimension for the batch
        preprocessed_img_tensor.unsqueeze_(0) # to add the color channel
        preprocessed_img_tensor.unsqueeze_(0) # to add the samples channel
        #ipdb.set_trace()
        return Variable(preprocessed_img_tensor, requires_grad=False)

    def _matplotlib_plot(self, data_mat, name_str, output_dir,
                         figsize=2, rect_cord=([0.1, 0.1, 0.8, 0.8],),
                         fontsize=4, dpi_scale=(1/0.4), cmap=None, facecolor='w'):
        # ipdb.set_trace()
        fig = plt.figure(figsize=(figsize, figsize))
        ax = fig.add_axes(rect_cord[0])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(bottom="off", left="off")
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.imshow(data_mat, cmap=cmap, vmin=0, vmax=1)

        if name_str == 'imshow_mask.png':
            # Adding text at the bottom of figure only in case of mask
            ax1 = fig.add_axes([0, 0, 1, 0.1])
            ax1.spines['left'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['top'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)
            ax1.tick_params(bottom="off", left="off")
            ax1.set_yticklabels([])
            ax1.set_xticklabels([])
            ax1.set_facecolor(facecolor)
            sz = data_mat.shape[0]

            ifTrue = str(data_mat[sz//2, sz//2] == np.amax(data_mat))[0]
            # ipdb.set_trace()
            zz = data_mat[sz//2, sz//2]
            condition = f'Max value at center ({zz:04f}): {ifTrue}'

            if ifTrue == 'T':
                ax1.text(0.5, 0.5, condition, ha='center', va='center', transform=ax1.transAxes, fontsize=2*fontsize)

                text = colored(' Mask is ', 'red', attrs=['reverse', 'blink', 'bold'])
                text += colored(' correct ', 'green', attrs=['reverse', 'blink', 'bold'])
                print(text)
            else:
                ax1.text(0.5, 0.5, condition, ha='center', va='center', transform=ax1.transAxes, fontsize=2 * fontsize, color='r')

        fig.suptitle(
            f"""LR:{self.learning_rate}, Iter:{self.max_iterations}, Optim:{self.optimizer_name}, CP:{self.cp}, Max:{np.amax(data_mat):.2f}, Min:{np.amin(data_mat):.2f}
            l1_coeff: {self.l1_coeff}, tv_coeff: {self.tv_coeff}, tv_beta: {self.tv_beta}, cat_coeff: {self.category_coeff:05.2f}""",
            fontsize=fontsize, y=0.99,
            )
        fig.savefig(output_dir + name_str, dpi=self.img_size * dpi_scale, facecolor=facecolor)

    def _plot_stats(self, plot_vec, output_dir, fileName, xlabel='Iteration Number', ylabel='yValues'):
        plt.figure()
        plt.plot(plot_vec)
        plt.suptitle(
            f"""LR: {self.learning_rate}, Iter: {self.max_iterations}, Optim: {self.optimizer_name}, Centre_Pixel: {self.cp},
                        l1_coeff: {self.l1_coeff}, tv_coeff: {self.tv_coeff}, tv_beta: {self.tv_beta}, cat_coeff: {self.category_coeff:05.2f}, 
                        Final Value of vector is: {plot_vec[-1]}""",
            y=0.99, fontsize=10)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=10)
        output_file = output_dir + fileName
        plt.savefig(output_file, bbox_inches='tight')
        # print(output_file)


    def save(self, mask, img, blurred, output_dir, \
             loss_list, category_loss_list, l1_loss_list, tv_loss_list, \
             centre_pixel_perturbated_input_list, \
             mask_list, perturbated_input_list, flag=False):

        # ipdb.set_trace()
        self.cp = img[self.img_size // 2, self.img_size // 2]
        dir_name_str = f'centre_pixel_{self.cp:03d}_learn_rate_{self.learning_rate:.4f}_iter_{self.max_iterations:04d}_optim_{self.optimizer_name}_'
        dir_name_str += f'l1_coeff_{self.l1_coeff:06.2f}_tv_coeff_{self.tv_coeff:06.2f}_tv_beta_{self.tv_beta}_cat_coeff_{self.category_coeff:05.2f}_'
        r_str = time.strftime("Y_%Y_M_%m_D_%d-H_%H_Min_%M_Sec_%S")

        if flag:
            output_dir += 'random_' + dir_name_str + r_str + '/'
        print('Saving extra MP results to: ', output_dir)
        '''
        try:
            shutil.rmtree(output_dir)
        except OSError as e:
            pass
            #print ("Error: %s - %s." % (e.filename, e.strerror))
        '''
        mkdir_p(output_dir)

        # Shape of img is [227, 227]
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)

        blurred = np.expand_dims(blurred, axis=2)
        blurred = np.repeat(blurred, 3, axis=2)

        mask = mask.cpu().data.numpy()[0]
        mask = np.transpose(mask, (1, 2, 0))

        mask = (mask - np.min(mask)) / np.max(mask)
        mask = 1 - mask

        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

        cam = 1.0 * heatmap + np.float32(img) / 255
        cam = cam / np.max(cam)

        ###################################################
        ####### IMG #################
        img = np.float32(img) / 255
        orig_img = Image.fromarray(np.uint8(255 * img))
        img_save_name = 'original_image'
        orig_img.save(output_dir + img_save_name + '.png')

        # ipdb.set_trace()

        ###### IMG _ MATPLOTLIB ############
        self._matplotlib_plot(img, 'imshow_original_image.png', output_dir,
                              figsize=4, rect_cord=([0.1, 0.1, 0.8, 0.8],),
                              fontsize=8, dpi_scale=(1/3.2))

        ###################################################
        ####### Perturbed IMG #################
        # Blur+Mask
        perturbated = np.multiply(1 - mask, img) + np.multiply(mask, blurred) #(correct coz he has previously made (1-mask))
        perturbated_temp = np.copy(perturbated)
        perturbated = Image.fromarray(np.uint8(255 * perturbated))
        img_save_name = 'perturbated'
        perturbated.save(output_dir + img_save_name + '.png')

        ##### Perturbed IMG Matplotlib #########
        self._matplotlib_plot(perturbated_temp, 'imshow_perturbated.png', output_dir,
                              figsize=4, rect_cord=([0.1, 0.1, 0.8, 0.8],), fontsize=8,
                              dpi_scale=(1 / 3.2))

        ###################################################
        ####### Heatmap  #################
        heatmap_temp = heatmap
        heatmap = Image.fromarray(np.uint8(255 * heatmap))
        img_save_name = 'heatmap'
        heatmap.save(output_dir + img_save_name + '.png')

        ##### Heatmap Matplotlib #########
        self._matplotlib_plot(heatmap_temp, 'imshow_heatmap.png', output_dir,
                              figsize=4, rect_cord=([0.1, 0.1, 0.8, 0.8],), fontsize=8,
                              dpi_scale=(1 / 3.2))

        # ipdb.set_trace()

        ########## MASK ######################################
        mask_temp = np.copy(mask)
        mask = Image.fromarray( np.uint8( np.repeat(mask, 3, axis=2)*255 ) )
        img_save_name = 'mask'
        mask.save(output_dir + img_save_name + '.png')

        ########### MASK Matplotlib ################
        self._matplotlib_plot(np.squeeze(mask_temp, axis=-1), 'imshow_mask.png', output_dir,
                              figsize=4, rect_cord=([0.1, 0.1, 0.8, 0.8],), fontsize=8,
                              dpi_scale=(1 / 3.2), cmap='gray', facecolor='y')



        #########################################
        ############ CAM ########################
        # Heatmap combined with original image
        cam = Image.fromarray(np.uint8(255 * cam))
        img_save_name = 'cam'
        cam.save(output_dir + img_save_name + '.png')

        #################################################
        ################ Blurred Image ################
        ## Entire image is blurred by Gaussian blur (No role of mask mere)
        blurred_temp = np.copy(blurred)
        blurred = Image.fromarray(np.uint8(255 * blurred))
        img_save_name = 'blurred'
        blurred.save(output_dir + img_save_name + '.png')

        ############ Blurred Matpltlib #########
        self._matplotlib_plot(blurred_temp, 'imshow_blurred.png', output_dir, figsize=4,
                              rect_cord=([0.1, 0.1, 0.8, 0.8],), fontsize=8,
                              dpi_scale=(1 / 3.2))

        ###########################
        ###### Extra Saves ########

        self._plot_stats(loss_list, output_dir, 'loss.png', ylabel='Iteration number')

        self._plot_stats(category_loss_list, output_dir, 'category_loss.png', ylabel='category loss')

        self._plot_stats(l1_loss_list, output_dir, 'l1_loss.png', ylabel='L1 Loss')

        self._plot_stats(tv_loss_list, output_dir, 'tv_loss.png', ylabel='TV Loss')

        self._plot_stats(centre_pixel_perturbated_input_list, output_dir, 'centre_pixel.png',
                         ylabel='Centre pixel of the perturbed image (image and mask)')

        imageio.mimsave(output_dir + 'mask_movie.gif', mask_list, palettesize=256)
        # imageio.mimsave(output_dir + 'perturbated_movie.gif', perturbated_input_list)


    def numpy_to_torch(self, img, requires_grad=True):
        if len(img.shape) < 3:
            # We don't need to add any dimension
            output = np.float32([img])
            #output = np.float32(img)
        else:
            #print('Here')
            # Converts from (h, w, 3) to (3, h, w)
            output = np.transpose(img, (2, 0, 1))
        #ipdb.set_trace()
        output = torch.from_numpy(output)
        if use_cuda:
            output = output.cuda(self.dev_idx)

        output.unsqueeze_(0)
        #print(output.shape)
        #ipdb.set_trace()
        v = Variable(output, requires_grad=requires_grad)
        return v


    def load_model(self, model_path, num_classes=256, n=227*227):
        class LinearRegression(nn.Module):
            def __init__(self, num_classes=256):
                super(LinearRegression, self).__init__()
                self.fc1 = nn.Linear(in_features=n, out_features=num_classes)

            def forward(self, x):
                out = self.fc1(x)
                return out

        model = LinearRegression(num_classes)
        
        model.load_state_dict(torch.load(model_path))
        model.eval()

        if use_cuda:
            model.cuda(self.dev_idx)

        for p in model.parameters():
            p.requires_grad = False

        return model

    def explain(self, neuron, save_flag_path=False, optimizer_name='ADAM', mask_init='random', mask_seed=None):
        # Hyper parameters.
        # TBD: Use argparse
        tv_beta = self.tv_beta
        learning_rate = self.learning_rate
        max_iterations = self.max_iterations
        l1_coeff = self.l1_coeff
        tv_coeff = self.tv_coeff
        category_coeff = self.category_coeff
        mask_size = self.mask_size
        #model_path = self.model_path
        print('Learning Rate is: ', learning_rate)

        img_size = self.img_size
        n = img_size * img_size
        num_classes = self.num_classes

        model = self.model

        # ipdb.set_trace()
        output_dir = self.output_dir
        original_img = self.image

        #####################################################
        # This part needs to be taken care of 
        print('True label is: ', original_img[img_size//2, img_size//2])
        img = np.float32(original_img) / 255
        blurred_img1 = cv2.GaussianBlur(img, (11, 11), 5)

        # ipdb.set_trace()
        blurred_img2 = np.float32(cv2.medianBlur(original_img, 11)) / 255
        blurred_img_numpy = (blurred_img1 + blurred_img2) / 2

        if mask_init == 'random':
            if mask_seed is None:
                print('Initializing with a random mask')
                mask_init = np.clip((np.random.uniform(high=1.0 + sys.float_info.epsilon,
                                                       size=(mask_size, mask_size),
                                                       ).astype('float32')),
                                    0, 1)
                mask_init = np.random.randint(2, size=(mask_size, mask_size)).astype('float32')
                output_dir = output_dir[:-1] + '/mp_results/'
            else:
                print(f'Initializing with a random mask as per the numpy seed of value {mask_seed}')
                rng_mask = np.random.RandomState(mask_seed)
                mask_init = rng_mask.randint(2, size=(mask_size, mask_size)).astype('float32')
                # mask_init = np.clip((rng_mask.uniform(high=1.0 + sys.float_info.epsilon,
                #                                        size=(mask_size, mask_size),
                #                                        ).astype('float32')),
                #                     0, 1)

                output_dir = output_dir[:-1] + f'ns_mask_{mask_seed:03d}' + '/mp_results/'


        else:
            print('Initializing with a blank mask (nothing is deleted in the beginning)')
            mask_init = np.ones((mask_size, mask_size), dtype=np.float32)
            output_dir += 'mp_results/'


        # Convert to torch variables
        img = self.preprocess_image(img)
        blurred_img = self.preprocess_image(blurred_img2)
        mask = self.numpy_to_torch(mask_init)

        if use_cuda:
            upsample = torch.nn.UpsamplingBilinear2d(size=(img_size, img_size)).cuda(self.dev_idx)
        else:
            upsample = torch.nn.UpsamplingBilinear2d(size=(img_size, img_size))

        if optimizer_name == 'ADAM':
            optimizer = torch.optim.Adam([mask], lr=learning_rate)
            self.optimizer_name = 'ADAM'
        else:
            optimizer = torch.optim.RMSprop([mask], lr=learning_rate)
            self.optimizer_name = 'RMSProp'

        print('Optimizer is: ', self.optimizer_name)

        category = neuron
        print('Explaining the label: ', category)
        print("Optimizing.. ")

        loss_list = []
        category_loss_list = []
        l1_loss_list = []
        tv_loss_list = []
        mask_list = []
        perturbated_input_list = []
        centre_pixel_perturbated_input_list = []


        for i in range(max_iterations):

            iter_interval = max_iterations//100
            if i%(max_iterations//iter_interval) == 0:
                print('Iteration is {}/{}'.format(i, max_iterations))
            upsampled_mask = upsample(mask)

            if i%5 == 0:
                aa = np.asarray(upsampled_mask.clone().tolist())[0, 0, :, :]
                aa = np.clip(aa, 0, 1)
                aa = 1 - aa
                # ipdb.set_trace()
                # aa = np.stack((aa, )*3, axis=-1)
                mask_list.append(np.uint8(aa*255))

            # Use the mask to perturbated the input image.
            perturbated_input = img.mul(upsampled_mask) + \
                                blurred_img.mul(1 - upsampled_mask)

            # noise = np.zeros((img_size, img_size, 1), dtype=np.float32)
            # cv2.randn(noise, 0, 0.2)
            # noise = self.numpy_to_torch(noise)
            # perturbated_input = perturbated_input + noise

            perturbated_input = self.flatten_img(perturbated_input)
            outputs = F.softmax(model(perturbated_input), dim=1)

            # The next command is redundant since in each iteration, this code is defining a new perturbed input from the modified mask
            # i.e. mask is important
            perturbated_input = self.resize_img(perturbated_input, img_size)

            aa = np.asarray(perturbated_input.clone().tolist())[0, 0, :, :]
            aa = np.clip(aa, 0, 1)
            centre_pixel_perturbated_input_list.append(aa[img_size//2, img_size//2])
            perturbated_input_list.append(np.uint8(aa*255))

            l1_loss = l1_coeff * torch.mean(torch.abs(1 - mask))
            tv_loss = tv_coeff * self.tv_norm(mask, tv_beta)
            cat_loss = category_coeff * outputs[0, category]

            loss = l1_loss + tv_loss + cat_loss

            # loss = l1_coeff * torch.mean(torch.abs(1 - mask)) + \
            #        tv_coeff * self.tv_norm(mask, tv_beta) + outputs[0, category]

            category_loss_list.append(cat_loss)
            l1_loss_list.append(l1_loss)
            tv_loss_list.append(tv_loss)
            loss_list.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Optional: clamping seems to give better results
            mask.data.clamp_(0, 1)

        #ipdb.set_trace()
        upsampled_mask = upsample(mask)

        ###Extra Saves ########
        loss_list = np.asarray([i.item() for i in loss_list])
        l1_loss_list = np.asarray([i.item() for i in l1_loss_list])
        tv_loss_list = np.asarray([i.item() for i in tv_loss_list])

        # ipdb.set_trace()

        category_loss_list = np.asarray([i.item() for i in category_loss_list])
        centre_pixel_perturbated_input_list = np.uint8(np.asarray(centre_pixel_perturbated_input_list) * 255)

        self.save(upsampled_mask, original_img, blurred_img_numpy, output_dir, \
                  loss_list, category_loss_list, l1_loss_list, tv_loss_list, \
                  centre_pixel_perturbated_input_list,\
                  mask_list, perturbated_input_list, flag=save_flag_path)

        del loss_list
        del category_loss_list
        del centre_pixel_perturbated_input_list
        del mask_list
        del perturbated_input_list

        # Returning mask to be plotted in main figure
        mask = mask.cpu().data.numpy()[0]
        mask = np.squeeze(mask, axis=0)
        #ipdb.set_trace()

        mask = (mask - np.min(mask)) / np.max(mask)
        mask = 1 - mask


        return mask

if __name__ == '__main__':
    args = get_arguments()
    num_classes = 256
    img_size = settings.image_size
    # max_iterations = 2000
    # learning_rate = 0.0001
    # tv_beta = 3
    # l1_coeff = 0.01
    # tv_coeff = 0.2
    optimizer_name = 'ADAM'
    # optimizer_name = 'RMSProp'

    device_idx = args.gpu

    # ipdb.set_trace()




    tv_coeff = args.tv_coeff
    tv_beta = args.tv_beta
    l1_coeff = args.l1_coeff
    max_iterations = args.max_iter
    learning_rate = args.learning_rate
    category_coeff = args.category_coeff

    # ipdb.set_trace()

    if args.img_path is None:
        print('Creating a random image')


        if args.np_seed_image:
            print('Using pseudo-random image (reproduceble)')
            rng = np.random.RandomState(args.np_seed_image)
            original_img = rng.randint(num_classes, size=(img_size, img_size))
            dir_name_str = f'ns_image_{args.np_seed_image:03d}_'


        else:
            print('Using truly random image')
            original_img = np.random.randint(num_classes, size=(img_size, img_size))
            st0 = np.random.get_state()[1][0]
            dir_name_str = ''
            # print('Current random seed is: ', st0)


        original_img = np.uint8(original_img)
        save_flag_path = False

        if args.centre_pixel:
            cc = args.centre_pixel
        else:
            cc = original_img[img_size//2, img_size//2]

        # ipdb.set_trace()
        dir_name_str += f'centre_pixel_{cc:03d}_learn_rate_{learning_rate:.4f}_iter_{max_iterations:04d}_optim_{optimizer_name}_'
        dir_name_str += f'l1_coeff_{l1_coeff:06.2f}_tv_coeff_{tv_coeff:06.2f}_tv_beta_{tv_beta}_cat_coeff_{category_coeff:05.2f}_'
        output_dir = args.out_path + 'random_' + dir_name_str + time.strftime("Y_%Y_M_%m_D_%d-H_%H_Min_%M_Sec_%S") + '/'



        
    else:
        args.img_path = os.path.abspath(args.img_path)
        output_dir = args.out_path + args.img_path.split('/')[-1].split('.')[0] + '/'
        #ipdb.set_trace()
        original_img = cv2.imread(args.img_path, 0)
        original_img = cv2.resize(original_img, (img_size, img_size))
        # ipdb.set_trace()
        save_flag_path = True

    # ipdb.set_trace()
    if args.centre_pixel is not None:
        original_img[img_size//2, img_size//2] = args.centre_pixel

    print('5x5 window of the image is: \n', original_img[-2+img_size//2:3+img_size//2, -2+img_size//2:3+img_size//2])

    # ipdb.set_trace()
    start = time.time()
    mp_explainer = meaningful_perturbation(original_img, args.model_path, output_dir, 
                                           max_iterations=max_iterations, learning_rate=learning_rate,
                                           tv_beta=tv_beta, l1_coeff = l1_coeff, tv_coeff = tv_coeff,
                                           category_coeff=category_coeff, dev_idx=device_idx, mask_size= 227,
                                           )

    ## This is to compute the neuron
    img = np.float32(original_img) / 255
    img = mp_explainer.preprocess_image(img) # make image a torch variable
    img = mp_explainer.flatten_img(img) 
    target = F.softmax(mp_explainer.model(img), dim=1) #get prob scores
    category = np.argmax(target.cpu().data.numpy()) #find the pred class
    print("Category with highest probability", category)
    if args.labels is not None:
        category = args.labels

    mask = mp_explainer.explain(category, save_flag_path=save_flag_path,
                                optimizer_name=optimizer_name, mask_seed=args.np_seed_mask, mask_init='blank')

    # del mp_explainer
    print("--- %s seconds ---" % (time.time() - start))

