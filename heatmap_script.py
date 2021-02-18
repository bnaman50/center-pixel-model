import warnings
warnings.simplefilter('ignore')

import os, sys
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import PIL.Image
import ipdb
import time
import argparse
from collections import OrderedDict

if sys.platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.colors import ListedColormap

from keras import backend as K
from keras.models import Model, load_model
from keras.utils import to_categorical



import innvestigate
import innvestigate.applications
import innvestigate.utils as iutils

# Stop tensorflow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

# Device
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Import the utilities and settings file
import utils as eutils
import settings


###############################################################################
# Function to take input arguments
###############################################################################
def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Input paramters for generating heatmaps of the image')

    # Add the paramters positional/optional (here only optional)
    parser.add_argument('-ip', '--img_path', help='Path to the image')
    parser.add_argument('-mp', '--model_path',
                        help='Path to the model that want to analyze. (Provide path of to keras/pytorch model)',
                        )

    parser.add_argument('-cl', '--clamp_label', type=str,
                        help='Select the label for which you want to find the importance (options - random_num in range [0-255] ). \
                               Default is True/pred (since 100 percentage accuracy)',
                        )

    parser.add_argument('-op', '--out_path',
                        help='Path of the output directory where you want to save the results (Default is ./results/)',
                        default='./results/')

    parser.add_argument('-cp', '--centre_pixel',
                        help='Centre pixel of choice for the image. (options - random_num in range [0-255] )',
                        )

    parser.add_argument('-hm', '--heatmap_methods', type=str,
                        help='List of heatmap to be included. Select atleast one from the following list. (grad, gb, deconvnet, sg, inpgrad, ig, lrp, lime, deeplift,mp, occlusion, pda, shap, shapley)',
                        nargs='+',
                        #default="True",
                        )

    # Parse the arguments
    args = parser.parse_args()

    ## Model Path
    if args.model_path is None:
        print("\nProvide a path to the model (PyTorch model) to be analyzed\n")
        sys.exit(1)

    ## Output Image Path
    if args.out_path is None:
        args.out_path = './'
    args.out_path = os.path.abspath(args.out_path) + '/'

    ## Image Path
    if args.img_path is None:
        print("\nImage path not given.\nWill create a random image and analyze it.\n")
    elif os.path.isfile(args.img_path):
        args.img_path = os.path.abspath(args.img_path)
    else:
        print('\nIncorrect file path providing.\nExiting\n')
        sys.exit(1)

    # None means default value (which is true/pred)
    if args.clamp_label is not None:
        if args.clamp_label.isdigit():
            args.clamp_label = int(args.clamp_label)
            if args.clamp_label < 0 or args.clamp_label > 255:
                print('Exter a valid label (in the range [0 - 255]).\nExiting\n')
                sys.exit(1)
        else:
            print('Enter a no. for the lable.\nExiting\n')
            sys.exit(1)

    #ipdb.set_trace()
    # None means default value (which is the original centre pixel value)
    if args.centre_pixel is not None:
        if args.centre_pixel.isdigit():
            args.centre_pixel = int(args.centre_pixel)
            if args.centre_pixel < 0 or args.centre_pixel > 255:
                print('Exter a valid centre pixel value (in the range [0 - 255]).\nExiting\n')
                sys.exit(1)
        else:
            print('Enter a no. for the centre pixel value.\nExiting\n')
            sys.exit(1)


    ## Methods List
    orig_method_list = ['grad', 'gb', 'deconvnet', 'sg', 'inpgrad', 'ig', 'lrp', 'lime', 'deeplift', \
                   'mp', 'occlusion', 'pda', 'shap', 'shapley']
    given_method_list = []
    if args.heatmap_methods is None:
        print("\nPlease provide the name/list of methods that you wish to analyze.\nExiting\n")
        sys.exit(1)
    else:
        args.heatmap_methods = [i.lower() for i in args.heatmap_methods]
        for hm in args.heatmap_methods:
            if hm in orig_method_list:
                given_method_list.append(hm)

        if len(args.heatmap_methods) != len(given_method_list):
            inc_names = list( set(args.heatmap_methods) - set(given_method_list) )
            print(f'Incorrect names given for these methods: {inc_names}.\nPlease refer to the original list in help.\nExiting\n')
            sys.exit(1)
    args.heatmap_methods = given_method_list
    #ipdb.set_trace()

    return args
###################################################################################


###############################################################################
# Meaty Stuff
###############################################################################
def compute_analysis(args):
    ############# DATA ##################

    image_size = settings.image_size
    #alpha = 3
    num_classes = settings.num_classes
    #n_test_examples = 2
    n = image_size*image_size
    k = n//2
    ######################################

    print('\nLoading the model')
    model_path = os.path.abspath(args.model_path)[:-3]
    model = load_model(model_path + '.h5')
    print('Model Loaded\n')
    print('Model Summary')
    model.summary()

    if args.img_path is None:
        print('Creating a random image')
        resultFolder = args.out_path + 'random_' + time.strftime("%Y%m%d-%H%M%S")

        if not settings.random_image_flag:
            print('Generating the random image based on the seed from settings file')
            rng = np.random.RandomState(settings.numpy_image_seed)
            original_img = rng.randint(num_classes, size=(image_size, image_size))
        else:
            original_img = np.random.randint(num_classes, size=(image_size, image_size))

        original_img = original_img.astype(np.float32)
        img = original_img

    else:
        ret = PIL.Image.open(args.img_path)
        ret = ret.resize((image_size, image_size))
        ret = ret.convert('L')
        img = np.asarray(ret, dtype=np.uint8).astype(np.float32)
        resultFolder = args.out_path + args.img_path.split('/')[-1].split('.')[0]

    if resultFolder[-1] != '/':
        resultFolder = resultFolder + '/'

    if args.centre_pixel is not None:
        img[image_size//2, image_size//2] = args.centre_pixel

    #ipdb.set_trace()

    y = to_categorical(img[image_size//2, image_size//2], num_classes=num_classes)
    y_index = np.argmax(y)
    print('\nTrue Label is:', y_index)

    img = img / 255
    data = img.flatten()
    data = np.expand_dims(data, axis=0)
    preds = model.predict(data)
    predict_indicies = np.argmax(preds)
    print('Predicted label is:', predict_indicies)
    #ipdb.set_trace()


    ## BTW, 2nd paramter (count starts from 0) is redundant and not used
    # Heatmapping Methods
    methods = [("input", {}, "Input")]

    if 'grad' in args.heatmap_methods:
        methods.append( ("gradient", {}, "Gradient") )
    if 'gb' in args.heatmap_methods:
        methods.append( ("guided_backprop", {}, "Guided Backprop ") )
    if 'deconvnet' in args.heatmap_methods:
        methods.append( ("deconvnet", {}, "Deconvnet") )
    if 'sg' in args.heatmap_methods:
        methods.append( ("smoothgrad", settings.smooth_grad_parameters, "SmoothGrad") )
    if 'inpgrad' in args.heatmap_methods:
        methods.append( ("input_t_gradient", {}, "Input * Gradient") )
    if 'ig' in args.heatmap_methods:
        methods.append( ("integrated_gradients", settings.integrated_grad_parameters, "Integrated Gradients") )
    if 'lrp' in args.heatmap_methods:
        methods.append( ("lrp.z", {}, "LRP-Z") )
        methods.append( ("lrp.epsilon", settings.lrp_epsilon_parameters, "LRP-Epsilon") )
        methods.append( ("lrp.alpha_beta", {"alpha":1, "beta":0}, "LRP-alpha1_beta0") )
        methods.append(("lrp.alpha_beta", settings.lrp_alpha_beta_parameters, "LRP-alpha2_beta1"))

    if 'occlusion' in args.heatmap_methods:
        methods.append( ("occlusion", {}, "Occlusion") )
    if 'deeplift' in args.heatmap_methods:
        from deepexplain.tensorflow import DeepExplain
        methods.append( ("deeplift", {}, "DeepLift") )
    if 'shapley' in args.heatmap_methods:
        methods.append( ("shapley", {}, "Shapley Sampling") )
    if 'pda' in args.heatmap_methods:
        methods.append( ("pda", {}, "Prediction Difference Analysis") )
    if 'lime' in args.heatmap_methods:
        methods.append(('lime', {}, "Lime"))
    if 'shap' in args.heatmap_methods:
        methods.append(('shap', {}, "Kernel_SHAP"))
    if 'mp' in args.heatmap_methods:
        methods.append( ("mp", {}, "Meaningful_Perturbation") )


    model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)

    ##############################################################
    # Create analyzers.
    analyzers = []
    for method in methods:
        #ipdb.set_trace()
        if method[0] == 'occlusion':
            from occlusion import occlusion_analysis
            kwargs = {'image': data, 'model': model,
                      'num_classes': num_classes, 'img_size': image_size,
                      }
            analyzer = occlusion_analysis(**kwargs)

        elif method[0] == 'deeplift':
            with DeepExplain(session=K.get_session()) as de:
                input_tensor = model.layers[0].input
                fModel = Model(inputs=input_tensor, outputs=model.layers[-2].output)
                target_tensor = fModel(input_tensor)
                dl_bl = settings.deeplift_parameters['baseline'].flatten()
                analyzer = de.get_explainer('deeplift', target_tensor, input_tensor, baseline=dl_bl)

        elif method[0] == 'shapley':
            with DeepExplain(session=K.get_session()) as de:
                input_tensor = model.layers[0].input
                fModel = Model(inputs=input_tensor, outputs=model.layers[-2].output)
                target_tensor = fModel(input_tensor)
                analyzer = de.get_explainer('shapley_sampling', target_tensor, input_tensor, samples=2)

        elif method[0] == 'pda':
            from pda import prediction_difference_analysis
            train_samples = settings.pda_parameters['train_samples']
            # ipdb.set_trace()
            kwargs = {'image':data, 'model':model,
                      'num_classes':num_classes, 'img_size':image_size,
                      'train_samples':train_samples}
            analyzer = prediction_difference_analysis(**kwargs)

        elif method[0] == 'mp':
            from mp import meaningful_perturbation

            #########################################
            # Need to convert this in (0-255) first
            mp_par = settings.mp_parameters
            mp_par['num_classes'] = num_classes
            mp_par['img_size'] = image_size
            #ipdb.set_trace()
            analyzer = meaningful_perturbation((data.reshape((image_size, -1))*255).astype('uint8'),
                                               model_path+'.pt', resultFolder, **mp_par,
                                               )

        elif method[0] == 'lime':
            import lime
            from lime import lime_image
            from lime.wrappers.scikit_image import SegmentationAlgorithm
            from skimage.segmentation import mark_boundaries

            # Make the explainer object
            analyzer = lime_image.LimeImageExplainer()

        elif method[0] == 'shap':
            from shap_class import shap_analysis
            from skimage.segmentation import slic as slic_super_pixel
            # Make the explainer object
            analyzer = shap_analysis(np.repeat(np.expand_dims(data.reshape((image_size, -1))*255, axis=-1), 3, axis=-1),
                                     model, resultFolder,
                                     **settings.shap_parameters)

        else:
            try:
                analyzer = innvestigate.create_analyzer(method[0],        # analysis method identifier
                                                        model_wo_softmax, # model without softmax output
                                                        neuron_selection_mode="index",
                                                        **method[1])     # optional analysis parameters

                if method[0] == "pattern.attribution":
                    analyzer.fit(data, batch_size=256, verbose=1)
            except innvestigate.NotAnalyzeableModelException:
                # Not all methods work with all models.
                analyzer = None
        analyzers.append(analyzer)

    ########## GENERATE HEATMAPS ###########
    IMG_ROWS = image_size
    IMG_COLS = image_size


    #analysis = np.zeros([len(data), len(analyzers), IMG_ROWS, IMG_COLS]) #Use analyzer for the analysis

    heatmap_grids = []
    extra_info = []
    for i, x in enumerate(data):
        # Add batch axis.
        x = np.expand_dims(x, axis=0)
        y_true = (x[:, k]*255).astype('int64')[0]

        ### Model predictions
        pred = model.predict(x) #256 probabilities
        pred_label = np.argmax(pred, axis=1)[0]
        pred_prob = np.amax(pred)
        neuron = pred_label

        if args.clamp_label is not None:
            neuron = args.clamp_label


        #print('pred shape', np.shape(pred))
        fiveClasses = np.argsort(-pred[0, :])[:5]
        #fiveProbs = np.zeros(np.shape(fiveClasses))
        fiveProbs = pred[0, fiveClasses]
        ##########################

        analysis = np.zeros([1, len(analyzers), IMG_ROWS, IMG_COLS])  # Use analyzer for the analysis

        for aidx, analyzer in enumerate(analyzers):

            print(f'Computing analysis for {methods[aidx][2]}')
            if methods[aidx][0] == "input":
                a = x
                a = a.reshape(image_size, -1)

            elif methods[aidx][0] in ['deeplift', 'shapley']:
                ys = to_categorical(neuron, num_classes=num_classes)
                if ys.shape[0] != x.shape[0]:
                    ys = np.expand_dims(ys, axis=0)
                a = analyzer.run(x, ys)
                a = a.reshape(image_size, -1)
                a = (a - np.mean(a)) / (np.std(a) + 1e-15)

            elif methods[aidx][0] == 'occlusion':
                #ipdb.set_trace()
                a = analyzer.explain(neuron, **settings.occlusion_parameters)
                a = a.reshape(image_size, -1)
                a = (a - np.mean(a)) / (np.std(a) + 1e-15)

            elif methods[aidx][0] == 'pda':
                print('PDA takes a lot time. Please wait...')
                num = settings.pda_parameters['num']
                a = analyzer.explain(neuron, num=num)
                a = a.reshape(image_size, -1)
                a = (a - np.mean(a)) / (np.std(a) + 1e-15)

            elif methods[aidx][0] == 'mp':
                print('MP takes some time. Please wait...')
                a = analyzer.explain(neuron)
                a = a.reshape(image_size, -1)
                a = (a - np.mean(a)) / (np.std(a) + 1e-15)

            elif methods[aidx][0] == 'lime':
                segmenter = SegmentationAlgorithm('quickshift', **settings.lime_segmenter_parameters)

                def lime_preprocess_input(im):
                    im = im[:, :, :, 0]
                    return np.reshape(im, (im.shape[0], -1))

                def lime_predict(x):
                    return model.predict(lime_preprocess_input(x))


                explanation = analyzer.explain_instance(image=np.reshape(data, (image_size, -1)),
                                                        classifier_fn=lime_predict,
                                                        top_labels=settings.num_classes,
                                                        segmentation_fn=segmenter,
                                                        **settings.lime_explainer_parameters
                                                        )


                temp, mask = explanation.get_image_and_mask(
                    label=neuron,
                    **settings.lime_mask_parameters
                )

                bb = (mark_boundaries(temp, mask))
                eutils.save_lime_mask(bb, resultFolder)
                a = bb[:, :, 0]
                # ipdb.set_trace()

            elif methods[aidx][0] == 'shap':
                print('SHAP takes some time. Please wait...')
                segments_slic = slic_super_pixel(PIL.Image.fromarray(np.uint8(analyzer.img_orig.copy())),
                                                 **settings.shap_slic_parameters)
                a = analyzer.explain(segments_slic, neuron)

            else:
                a = analyzer.analyze(x, neuron_selection=neuron)
                a = a.reshape(image_size, -1)
                a = (a - np.mean(a)) / (np.std(a) + 1e-15)


            analysis[i, aidx] = a
            print('Done')

        # Prepare the grid as rectengular list
        grid = [ [analysis[i, j] for j in range(analysis.shape[1])]
                 for i in range(analysis.shape[0])
                 ]

        #ipdb.set_trace()
        pred_prob = round(pred_prob, 5)
        #ipdb.set_trace()
        row_labels_left = [( 'True Label: {}'.format(y_true),
                             'Pred Label: {}'.format(pred_label),
                             'clamped Neuron: {}'.format(neuron),
                             'Probability: ' + str(pred_prob)
                           )]

        row_labels_right = [( '\n\n\nClass: %d' %fiveClasses[0]+' Prob: %.5f'%fiveProbs[0],
                              '\nClass: %d' %fiveClasses[1]+' Prob: %.5f'%fiveProbs[1],
                              '\nClass: %d' %fiveClasses[2]+' Prob: %.5f'%fiveProbs[2],
                              '\nClass: %d' %fiveClasses[3]+' Prob: %.5f'%fiveProbs[3],
                              '\nClass: %d' %fiveClasses[4]+' Prob: %.5f'%fiveProbs[4],
                            )]

        col_labels = [''.join(method[2]) for method in methods]

        eutils.plot_image_grid(grid, resultFolder,
                                     row_labels_left, row_labels_right, col_labels,
                                     file_name= 'heatmap_'+time.strftime("%Y%m%d-%H%M%S")+'.png',
                                     dpi= image_size)

        heatmap_grids.append(grid)
        extra_info.append([row_labels_left, row_labels_right])

    return heatmap_grids, extra_info

###########################################################################

if __name__ == '__main__':
   args = get_arguments()
   #ipdb.set_trace()
   start = time.time()
   compute_analysis(args)
   print("--- %s seconds ---" % (time.time() - start))