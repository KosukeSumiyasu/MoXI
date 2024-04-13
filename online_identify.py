
import os
from tqdm import tqdm
import torch
import pprint
from models.load_model import load_model, replace_vit_embedding
from src.data.load_dataset import load_dataset, selected_dataset_indexlist
from src.util.load_parser import load_parser
from src.util.load_yaml_config import load_yaml_config
from src.util.seed_torch import seed_torch
from src.util.utils import save_result, create_save_dir, store_args
from src.util.calc_contribution import online_identifying
from src.util.utils import get_replace_dict, replace_label

def main(args, save_dir):
    
    # initialize
    result_list = []
    if args.num_classes == 1000: # We use 1000 class dataset
        replace_dict = None
    elif args.num_classes == 10: # Special case. We only use 10 class dataset.
        args.load_image_dir = args.load_image_dir + f"_10_class"
        if args.training_class_num == 100 or args.training_class_num == 1000 or args.training_class_num == 20: 
            # It is defined to convert labels into 10 classes.
            replace_dict = get_replace_dict()
        elif args.training_class_num == 10:
            replace_dict = None
    args.load_image_dir = os.path.join(args.load_image_dir, args.prefix)
    
    # load model and dataset
    model, imageProcessor = load_model(args)
    model = replace_vit_embedding(args, model)
    model.eval()
    
    dataset = load_dataset(args)
    args.dataset_indexpath = os.path.join(args.save_result_dir, f'online_identify/{args.dataset}_{args.dataset_sampling_number}/{args.model_name}/dataset_index.npy')
    # only successfully classified images
    dataset_indexlist = selected_dataset_indexlist(args, model, imageProcessor, dataset) 

    print("---------start online identifying...---------")
    for _, dataset_index in tqdm(enumerate(dataset_indexlist), total=len(dataset_indexlist)):
        image, label = dataset[dataset_index]
        if replace_dict is not None:
            label = replace_label(args, label, replace_dict)
        identified_patch = online_identifying(args, model, imageProcessor, image, label)
        result = {'dataset_index': dataset_index, 'label': label, 'identified_patch': identified_patch}
        result_list.append(result)
    
    # save result
    save_path = os.path.join(save_dir, f"high_contribution_patches.npy")
    save_result(save_path, result_list)
    print("---------finish---------")

if __name__ == '__main__':
    # initialize
    args = load_parser()
    seed_torch(args.seed)
    config = load_yaml_config(args.config_path)
    args.load_image_dir = config["LOAD_IMAGE_DIR"]  
    args.save_result_dir = config["SAVE_RESULT_DIR"]
    args.label_to_id_path = config['LABEL_TO_ID_PATH']
    save_dir = create_save_dir(args)
    store_args(args, save_dir)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("---------args---------")
    pprint.pprint(vars(args))
    print("---------save_dir---------")
    print(save_dir)

    main(args, save_dir)

