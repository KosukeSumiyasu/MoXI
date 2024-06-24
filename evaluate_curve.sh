# initialize
seed=0; 
num_classes=1000;
training_class_num=1000;
dataset_sampling_number=100;
model_name='vit-t'; 
curve_method='deletion'; # ['insertion', 'deletion']
interaction_method='vit_embedding'; # ['vit_embedding', 'pixel_zero_values']
CONFIG_PATH='./config/config_file_path.yaml'

identify_method='self-shapley'
python3 evaluate_curve.py \
        --seed=$seed \
        --model_name=$model_name \
        --num_classes=$num_classes \
        --training_class_num=$training_class_num \
        --identify_method=$identify_method \
        --curve_method=$curve_method \
        --config_path=$CONFIG_PATH \
        --dataset_sampling_number=$dataset_sampling_number \
        --interaction_method=$interaction_method \

identify_method='MoXI'
python3 evaluate_curve.py \
        --seed=$seed \
        --model_name=$model_name \
        --num_classes=$num_classes \
        --training_class_num=$training_class_num \
        --identify_method=$identify_method \
        --curve_method=$curve_method \
        --config_path=$CONFIG_PATH \
        --dataset_sampling_number=$dataset_sampling_number \
        --interaction_method=$interaction_method \

identify_method='full-shapley'
python3 evaluate_curve.py \
        --seed=$seed \
        --model_name=$model_name \
        --num_classes=$num_classes \
        --training_class_num=$training_class_num \
        --identify_method=$identify_method \
        --curve_method=$curve_method \
        --config_path=$CONFIG_PATH \
        --dataset_sampling_number=$dataset_sampling_number
        --interaction_method=$interaction_method \

identify_method='attention_rollout'
python3 evaluate_curve.py \
        --seed=$seed \
        --model_name=$model_name \
        --num_classes=$num_classes \
        --training_class_num=$training_class_num \
        --identify_method=$identify_method \
        --curve_method=$curve_method \
        --config_path=$CONFIG_PATH \
        --dataset_sampling_number=$dataset_sampling_number \
        --interaction_method=$interaction_method \

identify_method='gradcam'
python3 evaluate_curve.py \
        --seed=$seed \
        --model_name=$model_name \
        --num_classes=$num_classes \
        --training_class_num=$training_class_num \
        --identify_method=$identify_method \
        --curve_method=$curve_method \
        --config_path=$CONFIG_PATH \
        --dataset_sampling_number=$dataset_sampling_number \
        --interaction_method=$interaction_method \