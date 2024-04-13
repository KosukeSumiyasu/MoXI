import torch
import torch.nn.functional as F
import numpy as np
import copy

# When using the mask method with a removal patch, calculate the logits of the model.
def model_wrapper_removal_patch(model, inputs, mask):
    with torch.no_grad():
        output = model(**inputs, bool_masked_pos=mask)['logits']
    return output

# When using the mask method with a replace baseline value, calculate the logits of the model.
def model_wrapper_replace_baselinevalue(args, model, inputs, mask_list):
    B, C, H, W = inputs['pixel_values'].shape
    masked_list = np.zeros((B, C, args.grid_num ** 2))
    for index in range(len(mask_list)):
        masked_list[index, :, mask_list[index]] = 1
    masked_list = masked_list.reshape(B, 3, args.grid_num, args.grid_num)
    mask = F.interpolate(torch.tensor(masked_list).clone(), size=[H, W], mode='nearest').float().to(args.device) # 4, 3, 224, 224
    masked_inputs = copy.deepcopy(inputs).to(args.device)
    masked_inputs['pixel_values'] = inputs['pixel_values'] * mask
    with torch.no_grad():
        output = model(**masked_inputs)['logits']
    return output