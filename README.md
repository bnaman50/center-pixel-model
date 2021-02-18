# center-pixel-model

This repo contains the code for our experiments on toy model/dataset for our project - [SAM](https://anhnguyen.me/project/sam/ "SAM").
We implemented the following ~15 XAI methods - 
- Gradient
- Guided Backprop
- Deconvolution
- SmoothGrad
- Input x Gradient
- Integrated Gradients
- LRP
- LIME
- DeepLift
- Meaningful Perturbation
- Occlusion
- Pda
- Shap
- Shapley


## Setup
This project is tested on Python 3.6.7 and `tensorflow-gpu==1.12.0` which [requires](https://www.tensorflow.org/install/source) `Cuda 9.0`.
1. Create a virtual environment (preferrably using conda)
2. `pip install -r requirement.txt`
3. Install `innvestigate==1.0.6` manually since older versions were not available on PyPi. 

    ```git clone https://github.com/albermax/innvestigate; cd innvestigate; git checkout 604017a; python setup.py install; cd ..; rm -rf innvestigate```
