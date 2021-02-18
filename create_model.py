'''
Authors:
        Naman Bansal
        Janzaib Masood
'''

import warnings
warnings.simplefilter('ignore')
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
import settings
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import sys
from termcolor import colored
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

image_size = settings.image_size
num_classes = settings.num_classes
model_name = settings.model_name
relu_flag = settings.relu_flag

def createParameters(img_size=227, n_classes=256, alpha=1, verbose=False, seed_flag=True, seed_val=0):
    alpha=alpha*255

    n = img_size * img_size  # Odd number of inputs work better with 100% accuracy
    k = n // 2

    # eps is added to make sure smaples are betweem [0, 1] and not just [0, 1) (half-open interval)
    randV = ((np.random.uniform(high=1.0+sys.float_info.epsilon, size=(n, 1))*4*n_classes) - (2*n_classes) ) / 600
    if seed_flag:
        rng = np.random.RandomState(seed_val)
        randV = ((rng.uniform(high=1.0+sys.float_info.epsilon, size=(n, 1))*4*n_classes) - (2*n_classes) ) / 600

    # w is the weight matrix
    w = randV * np.ones((n, n_classes), dtype='float32')
    w[k, :] = np.array([2 * i for i in range(n_classes)])

    w = alpha * w

    # b is the bias matrix
    b = (n_classes ** 2) * np.ones(n_classes, dtype='float32') + 1
    t = np.array([2 * i - 1 for i in range(0, n_classes)])
    t[0] = 0
    tt = np.array([sum(t[0:i]) + 2 * i - 1 for i in range(1, n_classes + 1)])
    t[1:n_classes] = tt[0:n_classes - 1]
    b = (b - t).T
    b = alpha * b
    b = b / 255

    if verbose == True:
        print("w = ", w.shape, "    b = ", b.shape)
        print("w = ", w)
        print("b = ", b)

    w = w.astype('float32')
    b = b.astype('float32')
    return (w, b, n, k)


def createParametersMod(img_size=227, n_classes=256, alpha=1, verbose=False, seed_flag=True, seed_val=0):
    alpha=alpha*255

    n = img_size * img_size  # Odd number of inputs work better with 100% accuracy
    k = n // 2

    # eps is added to make sure smaples are betweem [0, 1] and not just [0, 1) (half-open interval)
    randV = (np.random.uniform(low=-1, high=1.0+sys.float_info.epsilon, size=(n, 1))*2*n_classes) / 15
    if seed_flag:
        rng = np.random.RandomState(seed_val)
        randV = (rng.uniform(low=-1, high=1.0+sys.float_info.epsilon, size=(n, 1))*2*n_classes) / 15

    # w is the weight matrix
    # ipdb.set_trace()
    w = randV * np.ones((n, n_classes), dtype='float32')

    w[k, :] = np.array([2 * i for i in range(n_classes)]) - 2*(n_classes-1)

    w = alpha * w

    # b is the bias matrix
    b = (n_classes ** 2) * np.ones(n_classes, dtype='float32') + 1
    t = np.array([2 * i - 1 for i in range(0, n_classes)])
    t[0] = 0
    tt = np.array([sum(t[0:i]) + 2 * i - 1 for i in range(1, n_classes + 1)])
    t[1:n_classes] = tt[0:n_classes - 1]
    b = (b - t).T
    b = b + 3000000
    b = alpha * b
    b = b / 255

    if verbose == True:
        print("w = ", w.shape, "    b = ", b.shape)
        print("w = ", w)
        print("b = ", b)

    w = w.astype('float32')
    b = b.astype('float32')
    return (w, b, n, k)


# Generate the Weights & Biases
w, b, n, k = createParametersMod(image_size, num_classes, **settings.model_parameters)


############################################################################
## Creating dataset to test whether both the models are giving 100% accuracy
# np.random.seed(0)
n_test_examples = 10
X_test = np.random.randint(num_classes, size=(num_classes*n_test_examples, n))

y_test = np.eye(num_classes)[X_test[:,k]]
X_test = X_test.astype('float32')/255
yy = np.argmax(y_test, axis=1)


############### Keras Model ##################################
print('Creating Keras Model')
model = Sequential()

# Adding the activation seperately (model.add(Activation('relu'))) causes innvestigate to break
if not relu_flag:
    model.add(Dense(num_classes, input_dim=n, weights=[w, b]))
else:
    model.add(Dense(num_classes, input_dim = n, activation='relu', weights=[w, b]))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy']) # NOTE: Compile if you just want to evaluate
model.summary()

model2 = Sequential()
model2.add(Dense(num_classes, input_dim=n, weights=model.layers[0].get_weights() ))
preS_keras = model2.predict(X_test)

## Checking predictions of the Keras model
loss, acc = model.evaluate(X_test, y_test)

print(colored(f' Accuracy: {acc}, Loss: {loss} ',
               'green', attrs=['reverse', 'bold']))

predictions = model.predict(X_test)
p_prob = np.amax(predictions, axis=1)

print('Min pred prob by keras is: ', np.amin(p_prob))
print('Max pred prob keras is: ', np.amax(p_prob))

model.save(model_name+".h5")
print('Keras Model Done\n')


############### Pytorch Model ###################################
print('Creating PyTorch Model')
class LinearRegression(nn.Module):
    def __init__(self, num_classes=256):
        super(LinearRegression, self).__init__()
        self.fc1 = nn.Linear(in_features=n, out_features=num_classes)

    def forward(self, x):
        out = self.fc1(x)
        if relu_flag:
            out = F.relu(out)
        return out

torch_model = LinearRegression(num_classes)
torch_model.state_dict()['fc1.weight'].copy_( torch.Tensor(w.T) )
torch_model.state_dict()['fc1.bias'].copy_( torch.Tensor(b) )
print(torch_model)


### CHecking predictions od the PyTorch model
torch_model.eval()
model_pred = torch_model(torch.from_numpy(X_test))
preS_torch = model_pred
model_pred = F.softmax(model_pred, dim=1)
pred_prob, pred_labels = torch.max(model_pred, dim=1)

xx = np.amax(model_pred.detach().numpy(), axis=1)
print('Min pred prob by torch model is: ', torch.min(pred_prob))
print('Max pred prob by torch model is: ', torch.max(pred_prob))
pred_labels = pred_labels.numpy()
print( colored(f'\nAll the pred labels are same as the true labels: {np.array_equal(yy, pred_labels)}',
               'green', attrs=['reverse', 'bold']) )


torch.save(torch_model.state_dict(), model_name+'.pt')
print('\n Torch Model Saved\n')


