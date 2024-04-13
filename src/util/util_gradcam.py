from typing import List
import numpy as np
import torch
from torch import Tensor
from transformers import ViTForImageClassification
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def grad_cam(images: Tensor, vit: ViTForImageClassification, label: int,  use_cuda: bool = False) -> Tensor:
    """Performs the Grad-CAM method on a batch of images (https://arxiv.org/pdf/1610.02391.pdf)."""

    # Wrap the ViT model to be compatible with GradCAM
    vit = ViTWrapper(vit)
    vit.eval()
    target_layers = [vit.vit.vit.encoder.layer[-1].layernorm_before]
    targets = [ClassifierOutputTarget(label)]
    # Create GradCAM object
    cam = GradCAM(
        model=vit,
        target_layers=target_layers,
        reshape_transform=_reshape_transform,
        use_cuda=use_cuda,
    )
    cam.compute_cam_per_layer = compute_cam_per_layer.__get__(cam)
    # Compute GradCAM masks
    grayscale_cam = cam(
        input_tensor=images,
        targets=targets,
        eigen_smooth=True,
        aug_smooth=True,
    )

    return torch.from_numpy(grayscale_cam)


def _reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    
    # Bring the channels to the first dimension
    result = result.transpose(2, 3).transpose(1, 2)

    return result


def compute_cam_per_layer(self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool) -> np.ndarray:
    activations_list = [a.cpu().data.numpy()
                        for a in self.activations_and_grads.activations]
    grads_list = [g.cpu().data.numpy()
                    for g in self.activations_and_grads.gradients]
    target_size = (14,14)

    cam_per_target_layer = []
    # Loop over the saliency image from every layer
    for i in range(len(self.target_layers)):
        target_layer = self.target_layers[i]
        layer_activations = None
        layer_grads = None
        if i < len(activations_list):
            layer_activations = activations_list[i]
        if i < len(grads_list):
            layer_grads = grads_list[i]

        cam = self.get_cam_image(input_tensor,
                                    target_layer,
                                    targets,
                                    layer_activations,
                                    layer_grads,
                                    eigen_smooth)
        cam = np.maximum(cam, 0)
        scaled = scale_cam_image(cam, target_size)
        cam_per_target_layer.append(scaled[:, None, :])

    return cam_per_target_layer

class ViTWrapper(torch.nn.Module):
    """ViT Wrapper to use with Grad-CAM."""

    def __init__(self, vit: ViTForImageClassification):
        super().__init__()
        self.vit = vit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x).logits