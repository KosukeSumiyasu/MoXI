import argparse

def load_parser():
    parser = argparse.ArgumentParser('Transformer analayze from game theory', add_help=False)
    parser.add_argument('--config_path', default='', type=str, help='path to config file')
    parser.add_argument('--prefix', default='valid', type=str, help='args.prefix to save files')
    parser.add_argument('--dataset', default="imagenet", type=str, choices=['cifar10', 'imagenet', 'imagenet_c'])
    parser.add_argument('--load_image_dir', default="LOAD_IMAGE_DIR", type=str, help='dir to sampling dataset')
    parser.add_argument('--save_result_dir', default="SAVE_RESULT_DIR", type=str, help='dir to sampling dataset')
    parser.add_argument('--label_to_id_path', default="LABEL_TO_ID_PATH", type=str, help='label to id path') # use 10, 100 class dataset in imagenet
    parser.add_argument('--model_name', default='vit-t', type=str, choices=['vit-b', 'deit-b', 'vit-s', 'deit-s', 'vit-t', 'deit-t'])
    parser.add_argument('--num_classes', default=1000, type=int, help='number of classes')
    parser.add_argument('--grid_num', default=14, type=int, help='number of grid')
    parser.add_argument('--softmax_type', default='modified', type=str, choices=['normal', 'modified', 'yi'])
    parser.add_argument('--dataset_sampling_number', default=100, type=int, help='choice dataset number')
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--dataset_indexpath', default=None, type=str, help='dataset index path')
    parser.add_argument('--interaction_method', default='vit_embedding', type=str, choices=['vit_embedding', 'pixel_zero_values'])
    parser.add_argument('--identify_method', default='self-shapley', type=str, choices=['self-shapley', 'MoXI', 'full-shapley', 'gradcam', 'attention_rollout'])
    parser.add_argument('--isUsedTargetLayer', default=False, type=bool, help='is used target layer')
    parser.add_argument('--target_mask_layer', default=-1, type=int, help='target mask layer') # -1, 0-11
    parser.add_argument('--isTraining', default=False, type=bool, help='if training, you can use this flag')
    parser.add_argument('--num_m_patterns', default=200, type=int, help='if you use full-shapley values, set the number of m patterns')
    parser.add_argument('--curve_method', default='insertion', type=str, choices = ['insertion', 'deletion', 'both'], help='curve method')
    parser.add_argument('--checkpoint_dir', default=None, type=str, help='models checkpoint directory')
    parser.add_argument('--training_class_num', default=1000, type=int, help='training class')
    parser.add_argument('--target_label', default=None, type=int, help='target label')
    args, unparsed = parser.parse_known_args()
    return args