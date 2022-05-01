import os

# Change the following values accordingly
dataset_dir = f"{os.getcwd()}/drinks"
config = {
    "num_workers": 2,
    "pin_memory": True,
    "batch_size": 8,
    "dataset": "drinks",
    "train_split": f"{dataset_dir}/labels_train.csv",
    "test_split": f"{dataset_dir}/labels_test.csv",}
