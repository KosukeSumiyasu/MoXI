import os
import torch
import pprint
from models.load_model import load_model, replace_vit_embedding
from src.data.load_dataset import load_dataset
from src.util.load_parser import load_parser
from src.util.load_yaml_config import load_yaml_config
from src.util.seed_torch import seed_torch
from src.util.utils import save_result, create_save_dir
from src.util.utils import get_replace_dict
from src.util.evaluate_conritbution import calc_insertion_curve, calc_deletion_curve

def main(args):
    # initialize
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
    load_dir = os.path.join(args.save_result_dir, f'online_identify/{args.dataset}_{args.dataset_sampling_number}/{args.model_name}')

    # calc insertion/deletion curve
    print("---------evaluate...---------")
    if args.curve_method == 'insertion':
        result = calc_insertion_curve(args, model, imageProcessor, dataset, replace_dict, load_dir)
        save_path = os.path.join(load_dir, f"insertion/{args.identify_method}/insertion_accuracy")
        save_result(save_path, result)
    elif args.curve_method == 'deletion':
        result = calc_deletion_curve(args, model, imageProcessor, dataset, replace_dict, load_dir)
        save_dir = create_save_dir(args)
        save_path = os.path.join(save_dir, f"deletion_accuracy.npy")
        save_result(save_path, result)
    elif args.curve_method == 'both':
        args.curve_method == 'insertion'
        result = calc_insertion_curve(args, model, imageProcessor, dataset, replace_dict, load_dir)
        save_path = os.path.join(load_dir, f"insertion/{args.identify_method}/insertion_accuracy")
        save_result(save_path, result)
        
        args.curve_method == 'deletion'
        result = calc_deletion_curve(args, model, imageProcessor, dataset, replace_dict, load_dir)
        save_path = os.path.join(load_dir, f"deletion/{args.identify_method}/deletion_accuracy")
        save_result(save_path, result)
    print("---------finish---------")


if __name__ == '__main__':
    # initialize
    args = load_parser()
    seed_torch(args.seed)
    config = load_yaml_config(args.config_path)
    args.load_image_dir = config["LOAD_IMAGE_DIR"]  
    args.save_result_dir = config["SAVE_RESULT_DIR"]
    args.label_to_id_path = config['LABEL_TO_ID_PATH']
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("---------args---------")
    pprint.pprint(vars(args))
    main(args)