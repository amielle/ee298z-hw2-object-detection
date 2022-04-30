import gdown
import os
import tarfile

downloadable_files = {
    "drinks.tar.gz" : 
        "https://drive.google.com/file/d/14cFKnKATVTFgd50DDjN71WLEX8UhPTlK/view?usp=sharing",
    "adulay-fasterrcnn_resnet50_fpn-1651304089.3776634.pth" : 
        "https://drive.google.com/file/d/1AUR5Fhu90nZjyU2p2uAe-FbBvZ_cn8l9/view?usp=sharing"
}


def gdrive_download(url, filename):
    url_id = url.split("/")[5]
    dl_url = f"https://drive.google.com/uc?id={url_id}"
    gdown.download(dl_url, filename, quiet=False)


def download_files():
    for filename in downloadable_files:
        if os.path.exists(f"{os.getcwd()}/{filename}"):
            print(f"{filename} exists locally. Skipping download.")
            continue

        try:
            gdrive_download(downloadable_files[filename], filename)
        except Exception as e:
            print(f"Unable to download '{filename}' from source link '{downloadable_files[filename]}'.")
            print(e)


def setup_files(dataset_filename="drinks.tar.gz", unzip_dataset=True):
    download_files()

    if unzip_dataset == False: return

    try:
        dataset = f"{os.getcwd()}/{dataset_filename}"
        tar = tarfile.open(dataset, "r:gz")
        tar.extractall()
        tar.close()
    except Exception as e:
        print(f"Unable to extract dataset file '{dataset_filename}'.")
        print(e)
