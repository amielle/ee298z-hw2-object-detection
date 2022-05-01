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
    main_dir = os.getcwd()

    # Download dataset and pre-trained model
    dataprep.setup_files(dataset_filename="drinks.tar.gz", 
                         unzip_dataset=True)

    config = return_config(main_dir)
    
    test_dict, test_classes = label_utils.build_label_dictionary(
    config['test_split'])
    train_dict, train_classes = label_utils.build_label_dictionary(
        config['train_split'])
    
    train_split = odmodel.ImageDataset(train_dict, train_classes, transforms.ToTensor())
    test_split = odmodel.ImageDataset(test_dict, test_classes, transforms.ToTensor())

    print("Train split len:", len(train_split))
    print("Test split len:", len(test_split))

    train_loader = DataLoader(train_split,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=config['num_workers'],
                              pin_memory=config['pin_memory'],
                              collate_fn=utils.collate_fn)

    test_loader = DataLoader(test_split,
                             batch_size=1,
                             shuffle=False,
                             num_workers=config['num_workers'],
                             pin_memory=config['pin_memory'],
                             collate_fn=utils.collate_fn)
    
    # --------------
    # Training code
    # Adjust model parameters as needed
    num_classes = len(label_utils.params["classes"])

    model = odmodel.create_model(num_classes)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0075,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = config["epochs"]
    
    for epoch in range(num_epochs):
        # train for one epoch, printing every 50 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=50)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
    
    # Run test evaluation at the end of the training
    evaluate(model, test_loader, device=device)
    # --------------

    odmodel.save_model(model,
                   model_basename="fasterrcnn_resnet50_fpn", 
                   model_dir=main_dir, 
                   is_timebased=False)
