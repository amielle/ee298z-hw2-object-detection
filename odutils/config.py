def return_config(data_dir):
    dataset_dir = f"{data_dir}/drinks"

    # Change the following values accordingly
    config = {
        "num_workers": 2,
        "pin_memory": True,
        "batch_size": 8,
        "epochs": 15,
        "dataset": "drinks",
        "train_split": f"{dataset_dir}/labels_train.csv",
        "test_split": f"{dataset_dir}/labels_test.csv",}
    return config
