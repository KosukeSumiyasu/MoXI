seed=0; 
dataset_sampling_number=100; 
model_name='vit-t'; 
interaction_method='vit_embedding'; # ['vit_embedding', 'pixel_zero_values']
curve_method='deletion'; # ['insertion', 'deletion']
CONFIG_PATH='./config/config_file_path.yaml';

## ['shapley', 'shap+int_self', 'shap+int_full', 'gradcam', 'attention_rollout']
identify_method='self-shapley'; 
CUDA_VISIBLE_DEVICES=0 python3 online_identify.py  --dataset_sampling_number=$dataset_sampling_number --seed=$seed --model_name=$model_name --interaction_method=$interaction_method --identify_method=$identify_method --curve_method=$curve_method --config_path=$CONFIG_PATH
identify_method='MoXI';
CUDA_VISIBLE_DEVICES=0 python3 online_identify.py  --dataset_sampling_number=$dataset_sampling_number --seed=$seed --model_name=$model_name --interaction_method=$interaction_method --identify_method=$identify_method --curve_method=$curve_method  --config_path=$CONFIG_PATH
identify_method='full-shapley';
CUDA_VISIBLE_DEVICES=0 python3 online_identify.py  --dataset_sampling_number=$dataset_sampling_number --seed=$seed --model_name=$model_name --interaction_method=$interaction_method --identify_method=$identify_method --curve_method=$curve_method  --config_path=$CONFIG_PATH
identify_method='attention_rollout';
CUDA_VISIBLE_DEVICES=0 python3 online_identify.py  --dataset_sampling_number=$dataset_sampling_number --seed=$seed --model_name=$model_name --interaction_method=$interaction_method --identify_method=$identify_method --curve_method=$curve_method  --config_path=$CONFIG_PATH
identify_method='gradcam';
CUDA_VISIBLE_DEVICES=0 python3 online_identify.py  --dataset_sampling_number=$dataset_sampling_number --seed=$seed --model_name=$model_name --interaction_method=$interaction_method --identify_method=$identify_method --curve_method=$curve_method  --config_path=$CONFIG_PATH