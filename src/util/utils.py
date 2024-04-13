import json
import os
import numpy as np

def get_replace_dict(args):
    with open(args.label_to_id_path, 'r') as f:
        replace_dict = json.load(f)
    return replace_dict

def replace_label(args, label, replace_dict):
    if args.training_class_num == 10:
        return label
    elif args.training_class_num == 100:
        replace_method = 'label_id_100'
    elif args.training_class_num == 1000:
        replace_method = 'label_id_1000'
    elif args.training_class_num == 20:
        replace_method = 'label_id_20'
    
    for entry in replace_dict:
        if entry['label_id_10'] == label:
            return entry[replace_method]

def save_result(save_path, result):    
    np.save(save_path, result)

def create_save_dir(args):
    save_dir = os.path.join(args.save_result_dir, f'online_identify/{args.dataset}_{args.dataset_sampling_number}/{args.model_name}/{args.curve_method}/{args.identify_method}')
    if args.isUsedTargetLayer:
        save_dir = os.path.join(save_dir, f'isUsedTargetLayer/target_layer_{args.target_mask_layer}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def store_args(args, save_dir):
    save_path = os.path.join(save_dir, "parse.txt")
    with open(save_path, mode='w') as f:
        json.dump(args.__dict__, f, indent=4)