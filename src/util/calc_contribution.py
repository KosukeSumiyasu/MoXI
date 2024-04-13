import torch
import numpy as np
import torch.nn.functional as F
from .calc_reward import get_reward_shapley, get_reward_full_shapley, get_reward_interaction, get_reward_gradcam, get_reward_attention_rollout

def choose_patch(curve_method, reward, identified_patch):
    if curve_method == 'insertion':
        reward[identified_patch] = -np.inf
        index = torch.argmax(reward).item()
    elif curve_method == 'deletion':
        reward[identified_patch] = np.inf
        index = torch.argmin(reward).item()
    return index

def reshape_heatmap(args, heatmap):
    # if args.grid_num == 7, mean pooling
    if args.grid_num == 7:
        map_reshape = heatmap.reshape(7, 2, 7, 2)
        heatmap = np.mean(map_reshape, axis=(1, 3))
    heatmap = heatmap.reshape(args.grid_num**2)
    return heatmap

def get_identified_patch(args, reward):
    if args.curve_method == 'insertion':
        identified_patch = [x[0] for x in sorted(enumerate(reward), key=lambda x: x[1], reverse=True)]
    elif args.curve_method == 'deletion':
        identified_patch = [x[0] for x in sorted(enumerate(reward), key=lambda x: x[1], reverse=False)]
    return identified_patch

def online_identifying(args, model, imageProcessor, image, label):
    # -------------------self context shapley value-------------------
    if args.identify_method == 'self-shapley':
        images = [image for _ in range(args.grid_num**2)]
        inputs = imageProcessor(images, return_tensors="pt").to(args.device)
        reward = get_reward_shapley(args, model, inputs, label)
        reward = reshape_heatmap(args, reward)
        identified_patch = get_identified_patch(args, reward)
    # -------------------full shapley value-------------------
    elif args.identify_method == 'full-shapley':
        images = [image for _ in range(args.grid_num**2)]
        inputs = imageProcessor(images, return_tensors="pt").to(args.device)
        reward = get_reward_full_shapley(args, model, inputs, label)
        identified_patch = get_identified_patch(args, reward)
    # -------------------self context shapley value and interactions-------------------
    elif args.identify_method == 'MoXI':
        images = [image for _ in range(args.grid_num**2)]
        inputs = imageProcessor(images, return_tensors="pt").to(args.device)
        # self context shapley
        reward = get_reward_shapley(args, model, inputs, label)
        # choose first patch
        identified_patch = [choose_patch(args.curve_method, reward, identified_patch=[])]
        # calc interaction
        for _ in range(args.grid_num**2-1):
            reward = get_reward_interaction(args, model, inputs, label, identified_patch)
            new_patch = choose_patch(args.curve_method, reward, identified_patch)
            identified_patch.append(new_patch)
    # -------------------gradcam-------------------
    elif args.identify_method == 'gradcam':
        heatmap = get_reward_gradcam(image, label, model, imageProcessor)
        heatmap = reshape_heatmap(args, heatmap)
        identified_patch = get_identified_patch(args, heatmap)
    # -------------------attention rollout-------------------
    elif args.identify_method == 'attention_rollout':
        heatmap = get_reward_attention_rollout(image, model, imageProcessor)
        heatmap = reshape_heatmap(args, heatmap)
        identified_patch = get_identified_patch(args, heatmap)
    return identified_patch