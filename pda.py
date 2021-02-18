import numpy as np
import keras
from keras.models import load_model
from keras.utils import np_utils, to_categorical
import time, os
import ipdb
import settings

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class prediction_difference_analysis:
    def __init__(self, image, model, num_classes=256, img_size=227, train_samples=256000):
        self.image = image
        self.model = model
        self.num_classes = num_classes
        self.img_size = img_size
        self.train_samples = train_samples

    def _edit_sampled_data(self, partitioned_sampled_data, num, num_classes=256):
        #ipdb.set_trace()
        edit_matrix = partitioned_sampled_data
        for i in range(num):
            edit_matrix[i*num_classes:(i+1)*num_classes, i] = np.arange(num_classes, dtype='float32')/255
        return edit_matrix

    def _log_odds(self, laplace_corrected_prob):
        return np.log2( laplace_corrected_prob/(1 - laplace_corrected_prob) )

    def _laplace_correction(self, prob, train_samples=256000, num_classes=256):
        N = train_samples
        K = num_classes
        return (prob*N + 1)/(N+K)

    def explain(self, neuron, num=20):
        image = self.image
        # ipdb.set_trace()
        model = self.model
        clamped_class = neuron
        num_classes = self.num_classes 
        img_size = self.img_size
        train_samples = self.train_samples

        pred_prob = model.predict(image)[0, clamped_class]

        print(f'Explaining the class: {clamped_class}. Its prob is {pred_prob}')
        # ipdb.set_trace()

        num_features = img_size * img_size
        if num > num_features:
            num = num_features

        batch_size = num_classes*num
        total_samples = num_features*num_classes
        iteration_count = total_samples//batch_size

        iter_num = 0
        current_idx = 0
        counter = 0 # counts the num of features
        relevances = np.zeros((1,num_features))

        tmp = np.copy(image)
        sampled_size = min(batch_size, (total_samples - current_idx))

        batch_data = np.repeat(tmp, sampled_size, axis=0)  # (samples, num_features)


        old_partitioned_sampled_data = 0
        old_counter = 0

        while current_idx < total_samples:

            batch_data[:, old_counter:counter] = old_partitioned_sampled_data


            if iter_num < total_samples//batch_size:
                pass
            else:
                del batch_data
                print('Last iteration')
                num = num_features - counter
                tmp = np.copy(image)
                sampled_size = min(batch_size, (total_samples - current_idx))
                batch_data = np.repeat(tmp, sampled_size, axis=0)  # (samples, num_features)
                print('Old counter is: ', old_counter)
                print('New counter is: ', counter)

            old_partitioned_sampled_data = np.copy(batch_data[:, counter:counter + num])

            # print('Orig data mat is:\n ', batch_data[:7, counter:counter + 5] * 255)
            self._edit_sampled_data(batch_data[:, counter:counter + num] , num, num_classes=num_classes)

            # print('Modified data mat is:\n ', batch_data[:7, counter:counter + 5]*255)

            print(f'Computing for features - [{counter}-{counter+num})/{num_features}')

            sTime = time.time()
            probs = (model.predict(batch_data, batch_size=90, verbose=1)[:, clamped_class])
            print(f'Time taken is {time.time() - sTime}')

            probs = probs.reshape((num, -1)) # (features(20), 256)
            temp_sum_probs = np.sum(probs, axis=1)/num_classes

            relevances[0, counter:counter+num] = self._log_odds(self._laplace_correction(pred_prob, train_samples)) \
                                                 - self._log_odds(self._laplace_correction(temp_sum_probs, train_samples))

            # ipdb.set_trace()
            current_idx += sampled_size

            old_counter = counter
            counter += num
            iter_num += 1
            # del batch_data
        # print(f'Iter number is {iter_num}/{total_samples//batch_size}')
        return relevances

if __name__ == '__main__':

    num_classes = settings.num_classes
    img_size = settings.image_size

    k = (img_size * img_size) // 2
    num_features = img_size * img_size

    per_class_samples = 1
    data = np.random.randint(num_classes, size=(per_class_samples * num_classes, img_size * img_size))
    for i in range(num_classes):
        data[i * per_class_samples:(i + 1) * per_class_samples, k] = i
    data = data.astype('float32')
    y_test = to_categorical(data[:, k], num_classes=num_classes)
    y_label = np.argmax(y_test, axis=1)

    image = np.expand_dims(data[0, :], axis=0)
    y_test = y_test[0, :]

    model = load_model('myModel.h5')
    predictions = model.predict(image)
    # ipdb.set_trace()
    # pred_prob1 = np.amax(predictions, axis=1)[0]
    pred_label = np.argmax(predictions, axis=1)

    clamped_class = pred_label[0]

    start = time.time()
    kwargs = {'image' : image/255, 'model' : model, 'img_size':img_size, 'num_classes':num_classes}
    pda_explain = prediction_difference_analysis(**kwargs)
    relevances = pda_explain.explain(clamped_class, num=100)
    print("--- %s seconds ---" % (time.time() - start))
    ipdb.set_trace()
    a = 1


