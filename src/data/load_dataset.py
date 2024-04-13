from torchvision import datasets
import os
import numpy as np
import random
from tqdm import tqdm

def load_dataset(args):
    dataset = datasets.ImageFolder(root=args.load_image_dir)
    return dataset

# This function is used to create a list of indices of the dataset that are successfully classified by the model.
def make_dataset_indexlist(args, model, imageProcessor, dataset):
    dataset_index_list = []
    count = 0
    shuffled_indices = list(range(len(dataset)))
    random.shuffle(shuffled_indices)
    for index in tqdm(shuffled_indices):
        if count == args.dataset_sampling_number:
            break
        image, label = dataset[index]
        inputs = imageProcessor(image, return_tensors="pt").to(args.device)
        output = model(**inputs)['logits'].argmax().item()
        if output == label:
            dataset_index_list.append(index)
            count += 1
    np.save(args.dataset_indexpath, dataset_index_list)
    return dataset_index_list

def selected_dataset_indexlist(args, model, imageProcessor, dataset):
    if not os.path.exists(args.dataset_indexpath):
        make_dataset_indexlist(args, model, imageProcessor, dataset)
    dataset_indexlist = np.load(args.dataset_indexpath)
    return dataset_indexlist