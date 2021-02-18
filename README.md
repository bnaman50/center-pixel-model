# center-pixel-model

This repo contains the code for our experiments on toy model/dataset for our project - [SAM](https://anhnguyen.me/project/sam/ "SAM").
We implemented the following ~15 XAI methods - 
- [Gradient](https://arxiv.org/abs/1312.6034)
- [Guided Backprop](https://arxiv.org/abs/1412.6806)
- [DeConvNet](https://arxiv.org/abs/1311.2901)
- [SmoothGrad](https://arxiv.org/abs/1706.03825)
- [Input x Gradient](https://arxiv.org/abs/1810.03292)
- [Integrated Gradients](https://arxiv.org/abs/1703.01365)
- [LRP](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)
- [LIME](https://arxiv.org/abs/1602.04938)
- [DeepLift](https://arxiv.org/abs/1704.02685)
- [Meaningful Perturbation](https://arxiv.org/abs/1704.03296)
- [Occlusion](https://arxiv.org/abs/1311.2901)
- [Pda](https://arxiv.org/abs/1702.04595) 
- [Shap](https://arxiv.org/abs/1705.07874)
- [Shapley](https://www.sciencedirect.com/science/article/pii/S0305054808000804)

Notes - 
- Pda is quite slow and takes a while to run. We adapted the [original](https://github.com/lmzintgraf/DeepVis-PredDiff) implementation for our center-pixel model.

## Setup
This project is tested on Python 3.6.7 and `tensorflow-gpu==1.12.0` which [requires](https://www.tensorflow.org/install/source) `Cuda 9.0`.
1. Create a virtual environment (preferrably using conda)
2. `pip install -r requirements.txt`
3. Install `innvestigate==1.0.6` manually since older versions were not available on PyPi. 

    ```git clone https://github.com/albermax/innvestigate; cd innvestigate; git checkout 604017a; python setup.py install; cd ..; rm -rf innvestigate```


## Usage

1. First create models to be analyzed (Our code will automatically create both Keras and PyTorch model(required by meaningful perturbation)).
`python create_model.py`
2. Run `python heatmap_script.py -h` to understand how to compute the heatmap analysis. 

P.S. - Both global and method-specific hyperparameters can be found in [`settings.py`](settings.py) 


## Result
This is how the resultant plots will look like for the command - `python heatmap_script.py -ip ./images/shelby.JPEG -mp ./models/myModel.h5 -hm grad deeplift mp lime shap`. 
![alt text](/results/exp1.png?raw=true "Sample Output")
P.S. - Row 1 shows the original plot/heatmaps and the second row show the center `5x5` crop of each of the heatmaps. This helps us better get a sense of the relevance given to the center-pixel by a particular heatmap method. 

## Takeaways 
![alt text](/results/growing_patch.png?raw=true "Sample Output")
