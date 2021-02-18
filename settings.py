# This file has all the hyperparamter settings
# Also, method specific settings as well
# Input image is in range between 0 and 1

import numpy as np
from utils import mkdir_p
from srblib import abs_path
import os

image_size = 227
num_classes = 256

shapley_parameters = {'samples':100}


################SmoothGrad Parameters################################
noise_scale = 0.1 #(input_range[1]-input_range[0]) * 0.1
smooth_grad_parameters = {"augment_by_n": 16,
               "noise_scale": noise_scale,
               "postprocess": "square"
                }
#####################################################################
integrated_grad_parameters = {"reference_inputs": 0,
                   "steps": 64}

lrp_epsilon_parameters = {"epsilon": 1}

lrp_alpha_beta_parameters = {"alpha": 2, "beta": 1}

deeplift_parameters = {'baseline':np.zeros((image_size, image_size))}

occlusion_parameters = {'batch_size':227,
                        'occ_val':0,
                        }

######################## LIME Parameters ###########################################
lime_segmenter_parameters = {'kernel_size':4, 'max_dist':200, 'ratio':0.2, 'random_seed':0}
lime_explainer_parameters = {'hide_color':0, 'num_samples':1000}
lime_mask_parameters = {'positive_only':True, 'num_features':1, 'hide_rest':True}

###################### SHAP Parameters ############################################
shap_parameters = {'background':0, 'nsamples':1000}
shap_slic_parameters = {'n_segments':50, 'compactness':30, 'sigma':3}

###############################################################################
# PDA has been implemented in a batch wise fashion
# (So a batch of (image_size, image_size)*(256*num) will be created on CPU memory)
# num = 20 worked on my RAM of 32 GB
# Choose this parameter accordingly and wisely as per your own RAM memory
# otherwise code might break
pda_parameters = {'num':11,
           'train_samples':256000}
# (We just assumed that we have 1000 images per class in our dataset.
# ( Higher this number, the better will be laplace approximation of prob.


############################################################################
mp_parameters = {'tv_beta':3, 'learning_rate':0.1,
                 'max_iterations':1000, 'l1_coeff':0.01,
                 'tv_coeff':0.2, 'mask_size':image_size}


#####################################################
# Right now we are only using numpy_seed
numpy_seed = 0
torch_seed = 0
keras_seed = 0


######################################################################
# Whether you want to generate a truly/predictably random image
random_image_flag = False #True/False
numpy_image_seed = 0 #(Seed based on which random image will be generated if random_image_flag is True)


############ Model Settings #######################
'''
Use alpha to scale your pred prob
((alpha=1,  pred_prob=0.5),
 (alpha=3,  pred_prob=0.90),
 (alpha=5,  pred_prob=0.98),
 (alpha=7,  pred_prob=0.99),
 (alpha=10, pred_prob=0.9999))
'''
relu_flag = True
model_path = abs_path(mkdir_p('./models'))
model_name = os.path.join(model_path, 'myModel') #Don't worry about the extension
model_parameters = {'seed_flag':True,
                    'seed_val':0,
                    'alpha':10} #Alpha is scalar multiplier between [1-10].





