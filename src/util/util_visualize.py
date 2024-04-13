import cv2
import numpy as np
import torch
import copy
from pytorch_grad_cam.utils.image import show_cam_on_image

def convert_image_to_heatmap(args, image, count_list):
    rgb_img = np.array(image)
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    count_list = [count / max(count_list) for count in count_list]
    img_gridxgrid = np.array(count_list).reshape(args.grid_num, args.grid_num)
    img_224x224 = cv2.resize(img_gridxgrid, (224, 224), interpolation=cv2.INTER_LINEAR)
    cam_image = show_cam_on_image(rgb_img, img_224x224, use_rgb=True)
    return cam_image

def get_accuracy(args, model, imageProcessor, image, label, insert_index_list, mask_num):
    inputs = imageProcessor(image, return_tensors="pt").to(args.device)
    outputs = model(**inputs, bool_masked_pos=[insert_index_list[mask_num]])['logits']
    isCorrect = (torch.argmax(outputs, axis=1) == label).sum().item()
    confidence = torch.softmax(outputs, dim=1)[0][label].item()
    return isCorrect, confidence

def make_heatmap(args, identified_patch, model, imageProcessor, image, label, curve_method):
    count = 0
    heatmap = [0 for _ in range(args.grid_num ** 2)]
    if curve_method == 'insertion':
        insert_list = [[identified_patch[i] for i in range(j)] for j in range(1, len(identified_patch)+1)]
        
        while True:
            isCorrect, confidence = get_accuracy(args, model, imageProcessor, image, label, insert_list, count)
            count += 1
            if isCorrect==True or count == args.grid_num ** 2 :
            # if count == args.grid_num ** 2:
                break
    elif curve_method == 'deletion':
        range_list = list(range(args.grid_num ** 2))
        deletion_list = []

        for identify_index in identified_patch:
            range_list.remove(identify_index)
            deletion_list.append(copy.deepcopy(range_list))
        while True:
            isCorrect, confidence = get_accuracy(args, model, imageProcessor, image, label, deletion_list, count)
            count += 1
            if isCorrect==False or count == args.grid_num ** 2:
                break

    heatmap_value = count
    for index in identified_patch:
        if heatmap_value == 0:
            break
        heatmap[index] += heatmap_value
        heatmap_value -= 1
    return heatmap, count