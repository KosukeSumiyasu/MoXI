import torch
import numpy as np
import sys
from .util_gradcam import grad_cam
sys.path.append('../')
from models.model_wrapper import model_wrapper_removal_patch, model_wrapper_replace_baselinevalue

#---------------attention rollout-----------------
def get_reward_attention_rollout(args, image, model, imageProcessor):
    inputs = imageProcessor(images=image, return_tensors='pt').to(args.device)
    outputs = model(**inputs, output_attentions=True)
    attention_list = outputs['attentions']
    attention = torch.stack(attention_list, dim=0).cpu()
    attention = torch.squeeze(attention, dim=1).detach().numpy()
    mean_head_attention = np.mean(attention, axis=1) # (L, H, N, N) -> (L, N, N)
    mean_head_attention = mean_head_attention + np.eye(mean_head_attention.shape[1]) # (L, N, N) + (N, N)
    mean_head_attention = mean_head_attention / np.sum(mean_head_attention, axis=(1,2))[:, np.newaxis, np.newaxis] # (L, N, N) / (L, 1, 1)

    v = mean_head_attention[-1]
    for n in range(1, len(mean_head_attention)):
        v = np.matmul(v, mean_head_attention[-n-1])

    reward = v[0, 1:].reshape(14, 14)
    reward = reward/reward.max()
    return reward

#---------------gradcam-----------------
def get_reward_gradcam(args, image, label, model, imageProcessor):
    image_tensor = imageProcessor(images=image, return_tensors='pt').to(args.device)['pixel_values']
    gray_scale_cam = grad_cam(image_tensor, model, label)
    gray_scale_cam = gray_scale_cam[0, :]

    return gray_scale_cam

#---------------self shapley value-----------------
def get_reward_shapley(args, model, inputs, label):
    if args.curve_method =='insertion':
        masked_list = [[k] for k in range(args.grid_num ** 2)]

    elif args.curve_method =='deletion':
        all_i = list(range(args.grid_num ** 2))
        masked_list = []
        for i in all_i:
            masked_list.append([x for x in all_i if x != i])
    
    with torch.no_grad():
        if args.interaction_method == 'pixel_zero_values':
            logits = model_wrapper_replace_baselinevalue(args, model, inputs, masked_list)
        elif args.interaction_method == 'vit_embedding':
            logits = model_wrapper_removal_patch(model, inputs, masked_list)
        reward = get_reward(args, logits, label)
    return reward

#---------------self interactions-----------------
def get_reward_interaction(args, model, inputs, label, identified_patch):
    if args.curve_method =='insertion':
        masked_list = [[k]+identified_patch for k in range(args.grid_num ** 2) ]

    elif args.curve_method =='deletion':
        all_i = list(range(0, args.grid_num ** 2))
        masked_list = []
        for i in all_i:
            remove_subset = [x for x in all_i if x not in [i]+identified_patch]
            masked_list.append(remove_subset)
        for identified_patch_index in identified_patch:
            masked_list[identified_patch_index].pop(0)

    with torch.no_grad():
        if args.interaction_method == 'pixel_zero_values':
            logits = model_wrapper_replace_baselinevalue(args, model, inputs, masked_list)
        elif args.interaction_method == 'vit_embedding':
            logits = model_wrapper_removal_patch(model, inputs, masked_list)
        reward = get_reward(args, logits, label)
    return reward

#---------------full shapley value-----------------
def calc_logits_to_shapley(args, model, i_list, m_list, inputs):
    i_list = [[i_list[k]] + m_list[k] for k in range(len(m_list))]
    # masking method is pixel_zero_input
    if args.interaction_method == 'pixel_zero_values':
        i_output = model_wrapper_replace_baselinevalue(args, model, inputs, i_list)
        m_output = model_wrapper_replace_baselinevalue(args, model, inputs, m_list)
    # masking method is vit_embedding
    elif args.interaction_method == 'vit_embedding':
        i_output = model_wrapper_removal_patch(model, inputs, i_list)
        m_output = model_wrapper_removal_patch(model, inputs, m_list)
    logits_list = torch.stack([i_output, m_output], dim=0)
    return logits_list

# calc shapley value
def calc_shapley(args, model, i,  m, inputs, label):
    with torch.no_grad():
        # calc logits
        logits_list = calc_logits_to_shapley(args, model, i, m, inputs)
        # convert logits to reward
        reward_i = get_reward(args, logits_list[0], label)
        reward_m = get_reward(args, logits_list[1], label)
        # calc shapley value
        shapley_value = reward_i - reward_m
    return shapley_value

# select the pixel subset m other than i randomly
def list_i_to_sampling_m(args, list_i):
    m_patterns = []
    sampling_ratio = np.random.uniform(0.0, 1.0)
    for i in list_i:
        pixel_indices = [x for x in range(args.grid_num ** 2) if x not in [i]]
        size_s = int((args.grid_num ** 2 - 1) * sampling_ratio)
        m = np.random.choice(pixel_indices, size_s, replace=False).tolist()
        m_patterns.append(m)
    return m_patterns

def get_reward_full_shapley(args, model, inputs, label):

    # initialize i_list as all pixel indices
    list_i = [k for k in range(args.grid_num ** 2)]
    # initialize sum_shapley_value as 0
    sum_shapley_value = torch.zeros((args.grid_num ** 2)).to(args.device)

    # calc shapley value
    for _ in range(args.num_m_patterns):
        m_patterns = list_i_to_sampling_m(args, list_i)
        shapley_value = calc_shapley(args, model, list_i, m_patterns, inputs, label)
        sum_shapley_value = sum_shapley_value + shapley_value
    shapley_value_list = sum_shapley_value / args.num_m_patterns
    shapley_value_list = shapley_value_list.detach().cpu().tolist()

    return shapley_value_list


#---------------apply a reward function-----------------
def get_reward(args, logits, label):
    if args.softmax_type == "normal": # log p
        v = F.log_softmax(logits, dim=1)[:, label]
    elif args.softmax_type == "modified": # log p/(1-p)
        v = logits[:, label] - torch.logsumexp(logits[:, np.arange(logits.size(1)) != label], dim=1)
    elif args.softmax_type == "yi": # logits
        v = logits[:, label]
    else:
        raise Exception(f"softmax type [{args.softmax_type}] not implemented")
    return v