import kagglehub

import os

def download_and_get_dataset():
    path = kagglehub.dataset_download("hasibalmuzdadid/shoe-vs-sandal-vs-boot-dataset-15k-images")
    dataset_path = os.path.join(path, os.listdir(path)[0])
    return dataset_path
