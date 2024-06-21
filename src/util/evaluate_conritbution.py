import os
import numpy as np
from tqdm import tqdm
import torch
from .utils import replace_label
import copy
from models.model_wrapper import model_wrapper_removal_patch, model_wrapper_replace_baselinevalue

BATCH_SIZE = 500

def calc_correct_count(args, model, imageProcessor, images, labels, masked_list, collect_count):
    # calc correct count
    with torch.no_grad():
        inputs = imageProcessor(images, return_tensors="pt").to(args.device)
        labels_torch = torch.tensor(labels).to(args.device)
        combined_lists = [[sublist[i] for sublist in masked_list] for i in range(len(masked_list[0]))]
        for index, mask in enumerate(combined_lists):
            if args.interaction_method == 'pixel_zero_values':
                outputs = model_wrapper_replace_baselinevalue(args, model, inputs, mask)
            elif args.interaction_method == 'vit_embedding':
                outputs = model_wrapper_removal_patch(model, inputs, mask)
            collect_count[index] += (torch.argmax(outputs, axis=1) == labels_torch).sum()
    return collect_count

def calc_insertion_curve(args, model, imageProcessor, dataset, replace_dict=None, load_dir=None):  
    # initialize
    load_path = os.path.join(load_dir, f'insertion/{args.identify_method}/high_contribution_patches.npy')  
    identify_datadict = np.load(load_path, allow_pickle=True)
    batch_size = BATCH_SIZE
    collect_count = np.zeros(args.grid_num**2)

    # calc insertion curve
    images = []
    labels = []
    masked_list = []
    for index, datadict in tqdm(enumerate(identify_datadict), total=len(identify_datadict)):
        image, label = dataset[datadict['dataset_index']]
        if replace_dict is not None:
            label = replace_label(args, label, replace_dict)
        assert datadict['label']==label
        images.append(image)
        labels.append(label)
        insert_list = [[datadict['identified_patch'][i] for i in range(j)] for j in range(1, len(datadict['identified_patch'])+1)]
        masked_list.append(insert_list)

        if len(images) == batch_size or index == len(identify_datadict)-1:
            collect_count = calc_correct_count(args, model, imageProcessor, images, labels, masked_list, collect_count)
            images = []
            labels = []
            masked_list = []
    accuracy = np.array(collect_count) / len(identify_datadict)
    return accuracy

def calc_deletion_curve(args, model, imageProcessor, dataset, replace_dict, load_dir):
    # initialize
    if args.identify_method == 'MoXI' or args.identify_method == 'self-shapley':
        load_path = os.path.join(load_dir, f'deletion/{args.identify_method}/high_contribution_patches.npy')
    elif args.identify_method == 'gradcam' or args.identify_method == 'attention_rollout' or 'full-shapley':
        load_path = os.path.join(load_dir, f'insertion/{args.identify_method}/high_contribution_patches.npy')
    identify_datadict = np.load(load_path, allow_pickle=True)
    batch_size = BATCH_SIZE
    collect_count = np.zeros(args.grid_num**2)
    images = []
    labels = []
    masked_list = []

    # calc deletion curve
    for index, datadict in tqdm(enumerate(identify_datadict), total=len(identify_datadict)):
        image, label = dataset[datadict['dataset_index']]
        if replace_dict is not None:
            label = replace_label(args, label, replace_dict)
        assert datadict['label']==label
        images.append(image)
        labels.append(label)
        range_list = list(range(args.grid_num ** 2))
        deletion_list = []

        for identify_index in datadict['identified_patch']:
            range_list.remove(identify_index)
            deletion_list.append(copy.deepcopy(range_list))
        masked_list.append(deletion_list)

        if len(images) == batch_size or index == len(identify_datadict)-1:
            collect_count = calc_correct_count(args, model, imageProcessor, images, labels, masked_list, collect_count)
            images = []
            labels = []
            masked_list = []
    accuracy = np.array(collect_count) / len(identify_datadict)
    return accuracy
