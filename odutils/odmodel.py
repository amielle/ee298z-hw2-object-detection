import torch
import numpy as np
import label_utils
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import time
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import os
import sys
import cv2
import matplotlib.pyplot as plt

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dictionary, classes, transform=None):
        self.dictionary = dictionary
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.dictionary)

    def __getitem__(self, idx):
        """
        - boxes : Nx4
        - labels : N
        - image_id : 1
        - area : N // area of box
        - iscrowd : 1 // T/F
        optional: masks, keypoints
        """

        # retrieve the image filename
        key = list(self.dictionary.keys())[idx]
        # retrieve all bounding boxes
        boxes = self.dictionary[key]
        # open the file as a PIL image
        img = Image.open(key)
        # apply the necessary transforms
        # transforms like crop, resize, normalize, etc
        if self.transform:
            img = self.transform(img)
        
        # get other required info
        num_objs = len(boxes)

        # format of boxes is [boxes, label]; separate the tensors
        labels = boxes.T[-1]
        boxes = boxes.T[:-1].T

        # convert to torch tensors 
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # since original box format is xmin, xmax, ymin, ymax
        # switch xmax and ymin col to follow COCO dataset format
        # xmin, ymin, xmax, ymax
        boxes = torch.index_select(boxes, 1, torch.LongTensor([0,2,1,3]))

        # set to target dict
        target = dict()
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target


def create_model(num_classes):
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model


def save_model(model, model_basename="fasterrcnn_resnet50_fpn", model_dir=os.getcwd(), is_timebased=False):
    model_name = model_basename
    if is_timebased:
        model_name += f"-{time.time()}"

    if not os.path.exists(model_dir):
        print(f"Save folder does not exist. Creating directory '{model_dir}' .")
        os.makedirs(model_dir)

    filepath = f"{model_dir}/{model_name}.pth"
    try:
        torch.save(model.state_dict(), filepath)
        print(f"Saved model to {filepath}")
    except Exception as e:
        print(f"Unable to save model {filepath}")
        print(e)


def load_model(od_trained_model=None, num_classes=4, model_basename="fasterrcnn_resnet50_fpn", model_dir=os.getcwd(),
               default_model_name="adulay-fasterrcnn_resnet50_fpn-1651304089.3776634.pth"):
    """
        Sample usage:
        loads model in od_trained_model path : load_model(od_trained_model=<path to trained model w/ actual filename>)
        loads locally trained model but defaults to downloaded module if not available : load_model()
    """
    if od_trained_model is None:
        # default loading to locally trained model
        
        od_trained_model = f"{model_dir}/{model_basename}.pth"
        if not os.path.exists(od_trained_model):
            # change to original fine-tuned model from repo author
            od_trained_model = f"{model_dir}/{default_model_name}.pth"
        
    print(f"Loading model from '{od_trained_model}'...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = create_model(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(
        od_trained_model, map_location=device
    ))
    model.eval()
    print("Set model to inference mode.")

    return model


def preprocess_frame(image):
    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(float)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cuda()
    # add batch dimension
    image = torch.unsqueeze(image, 0)   
    return image


def detect_drinks(model, filename=None, detection_threshold=0.8, read_from_file=True, image=None, to_plot=True):
    # lowest value found at around ~0.8, adjust to lower value to 'capture more'/higher allowance
    # default threshold set to 0.8 since 0.75 captures other objects in the background as drinks
    # setting to a higher value (e.g. 0.9) lowers those cases

    if read_from_file:
        image = cv2.imread(filename)
    orig_image = image.copy()
    image = preprocess_frame(image)

    outputs = model(image)

    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`

        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [label_utils.params["classes"][i] for i in outputs[0]['labels'].cpu().numpy()]

        # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            cv2.rectangle(orig_image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (0, 0, 255), 2)
            cv2.putText(orig_image, pred_classes[j], 
                        (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                        2, lineType=cv2.LINE_AA)

        if to_plot:
            imgplot = plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
            plt.show()

    return orig_image
