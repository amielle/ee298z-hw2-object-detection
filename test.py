from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import os

import odutils.label_utils as label_utils
import odutils.dataprep as dataprep
import odutils.odmodel as odmodel
from odutils.config import return_config

import detection.utils as utils
from detection.engine import train_one_epoch, evaluate

if __name__ == "__main__":
    # Download dataset and pre-trained model
    dataprep.setup_files(dataset_filename="drinks.tar.gz", 
                         unzip_dataset=True)

    config = return_config(main_dir)
    
    test_dict, test_classes = label_utils.build_label_dictionary(
    config['test_split'])
    test_split = odmodel.ImageDataset(test_dict, test_classes, transforms.ToTensor())

    test_loader = DataLoader(test_split,
                             batch_size=config['batch_size'],
                             shuffle=False,
                             num_workers=config['num_workers'],
                             pin_memory=config['pin_memory'],
                             collate_fn=utils.collate_fn)
    
    # Change as needed
    model_path = f"{os.getcwd()}/adulay-fasterrcnn_resnet50_fpn-1651304089.3776634.pth"
    model = odmodel.load_model(od_trained_model=model_path)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    evaluate(model, test_loader, device=device)
