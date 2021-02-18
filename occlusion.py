import numpy as np
import keras
from keras.models import load_model
from keras.utils import np_utils, to_categorical
import time, os, sys
import ipdb

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

class occlusion_analysis:
    def __init__(self, image, model, num_classes=256, img_size=227):
        self.image = image
        self.model = model
        self.num_classes = num_classes
        self.img_size = img_size

    def _batch_generator(self, sampled_data, start_idx, end_idx, occ_val=0):
        mask_size = end_idx - start_idx
        mask = np.ones((mask_size, mask_size))
        np.fill_diagonal(mask, 0) #This is a in-place operation
        sampled_data[:, start_idx:end_idx] = np.multiply(mask, sampled_data[:, start_idx:end_idx]) + occ_val*np.identity(mask_size)
        return sampled_data

    def explain(self, neuron, batch_size=20*256, occ_val=0):
        image = self.image
        model = self.model
        clamped_class = neuron
        num_classes = self.num_classes
        img_size = self.img_size

        orig_pred_prob = model.predict(image)[0, neuron]
        print(f'Original predicted prob for the clamped class {neuron} is {orig_pred_prob}')
        num_features = img_size * img_size

        current_idx = 0
        relevances = np.zeros((1, num_features))

        while current_idx < num_features - 1:
            tmp = np.copy(image)

            sampled_size = min(batch_size, (num_features - 1 - current_idx))
            end_idx = current_idx + sampled_size

            sampled_data = np.repeat(tmp, sampled_size, axis=0)  # (samples, num_features)
            sampled_data = self._batch_generator(sampled_data, current_idx, end_idx, occ_val=occ_val)

            pred_probs = model.predict(sampled_data, batch_size=sampled_size)
            del sampled_data
            relevances[0, current_idx:end_idx] = orig_pred_prob - pred_probs[:, clamped_class]

            current_idx = end_idx

        return relevances

if __name__ == '__main__':

    num_classes = 256
    img_size = 227

    k = (img_size * img_size) // 2
    num_features = img_size * img_size

    per_class_samples = 1
    data = np.random.randint(num_classes, size=(per_class_samples, img_size * img_size))
    data = data.astype('float32')
    y_test = to_categorical(data[:, k], num_classes=num_classes)
    y_label = np.argmax(y_test, axis=1)[0]


    image = data
    print('True label is:', y_label)


    model = load_model('myModel.h5')
    predictions = model.predict(image)
    #ipdb.set_trace()
    pred_prob = np.amax(predictions, axis=1)[0]
    pred_label = np.argmax(predictions, axis=1)[0]
    print(f'Predicted label is {pred_label} with a prob score of {pred_prob}')

    #ipdb.set_trace()
    clamped_class = pred_label

    start = time.time()
    kwargs = {'image' : image, 'model' : model}
    occ_explain = occlusion_analysis(**kwargs)
    batch_size = int(eval(str(sys.argv[1])))
    relevances = occ_explain.explain(clamped_class, batch_size=batch_size)
    print("--- %s seconds ---" % (time.time() - start))
    ipdb.set_trace()
    a = 1
