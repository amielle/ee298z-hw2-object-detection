# Assignment 2: Object Detection for EE 298Z Deep Learning (a.dulay)

![download (1)](https://user-images.githubusercontent.com/43136926/166107742-36f33e6c-6c62-4099-8adb-bbb3adf6335f.png)
![download (2)](https://user-images.githubusercontent.com/43136926/166107743-f2c6c717-ce8a-4a1e-b2f1-98621efbe128.png)

<sup>Sample predictions of the model on test set images. Prediction threshold in this sample is set to 0.9 to better filter out false cases or less confident predictions.</sup>

The following repository trains a model using a pre-trained [Faster R-CNN](https://arxiv.org/abs/1506.01497) torch model for object detection. The model is fine-tuned using the [Drinks Dataset](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2022/datasets/python/dataloader_demo.ipynb) and is used to determine the bounding boxes and class of the detected drink. The main classes to be detected by the model are **Summit** (water bottle), **Coke** (red soda can), and **Pine Juice** (green pineapple juice in can.)


## 1. Usage guide

The main scripts in the repository can be setup on a personal machine to **Run locally** or online to **Run on Kaggle**. It is suggested to go for the latter if there are issues with setting up the pre-requisites. Ensure that there is enough free disk space (~3GB) to properly store the dataset and model. The scripts (train.py and test.py) automatically download both the dataset and the pre-trained model during runtime.

### - Run locally

1. Install dependencies

Properly setup CUDA in your machine to leverage the GPU.

```bash
pip install -r requirements.txt
```

2. Train the model

NOTE: This step is optional since the test script will use the fine-tuned model if there is no locally trained model.
```bash
python train.py
```

3. Evaluate the model on the test dataset
```bash
python test.py
```

### - Run on Kaggle

**NOTE:** The notebook runs train.py and test.py as scripts to simulate running on personal machines. Additionally, it also contains sample code for plotting predictions on some test set images.

1. Register and [log in to Kaggle](https://www.kaggle.com/) to access custom environment preferences.
2. Go to the [Kaggle notebook](https://www.kaggle.com/amielle/ee298z-hw2-object-detection/notebook). Press the "Copy & Edit" button. 
3. Start up the notebook and run with a GPU accelerator and with internet settings toggled on. The settings on the right panel should be similar to the image below:

![image](https://user-images.githubusercontent.com/43136926/166148981-626ce855-03f2-44b2-9731-83d556fbc8f6.png)

3. Run all the cells

A sample method is given under this section but it is also possible to apply the same ideas for [running on colab](https://colab.research.google.com/?utm_source=scs-index).

## 2. Other information

#### Notebooks
The `ntbk` folder contains the initial exploratory code. The notebooks in the directory will require initial setup of pre-loading the dataset and/or trained model to work unlike `train.py` and `test.py`. However, there are writeups on the notebook to guide the user in the setup and configuration of directories.
* ntbks/ee298z-assignment-2-object-detection-train.ipynb
  * Contains code for the custom dataloader, training and saving the model, and sample inference on test images.
* ntbks/ee298z-assignment-2-object-detection-video-gen.ipynb
  * Contains code for the loading the trained model, readings frames from a video file, and applying the object detection model to detect the location of the drinks in the image and the corresponding classes. 
  * NOTES: Due to the limitation of live camera video feed in Kaggle notebooks, feeding from a video file (.mp4) was the workaround for the demo submission. A demo file is generated with 640x480 resolution and 30 frames per second.
* ee298z-hw2-object-detection.ipynb
  * Contains the code from the [Kaggle notebook](https://www.kaggle.com/amielle/ee298z-hw2-object-detection/notebook)
#### Additional references
* [label_utils](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2022/datasets/python/label_utils.py) 
* [Torch vision library usage sample](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) 
* [Torch object detection reference scripts](https://github.com/pytorch/vision/tree/main/references/detection)
* [Sample inference code](https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/)
