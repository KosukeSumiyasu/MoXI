# Identifying Important Group of Pixels using Interactions[CVPR'24]
[Kosuke Sumiyasu](https://github.com/KosukeSumiyasu), [Kazuhiko Kawamoto](https://researchmap.jp/kawa), [Hiroshi Kera](https://hkera.wordpress.com/)
[[arxiv](https://arxiv.org/abs/2401.03785v2)]
## Overview
![heatmap_example](https://github.com/KosukeSumiyasu/MoXI/assets/77134522/70b84ae2-0b5c-44e0-90a2-c75c510d36e1)
MoXI (Model eXplanation by Interactions) is a black box game-theoretic explanation method of image classifiers. Unlike other popular methods (e.g., GradCAM and AttentionRollout), it takes into account the cooperative contributions of two pixels and accurately identifies a group of pixels that have a high impact on prediction confidence.
## Installation
Clone this repo:
```
$ git clone https://github.com/KosukeSumiyasu/MoXI
```
## Demo
You can walk through some examples as follows.

Step 0: download [ImageNet dataset](https://www.image-net.org/challenges/LSVRC/2012/).
Please specify the ImageNet dataset path in the config_file_path.yaml file for integration.

Step 1: run the following.
```
$ cd MoXI
$ ./online_identify.sh
$ ./evaluate_curve.sh
```

Step 2:
Open Jupyter notebooks in ```notebook/```.
```00_plot_insertion_deletion_curve.ipynb``` --- Quantitive evaluation by insertion and deletion curves.
```01_visualize_heatmap.ipynb``` --- Qualitative evaluation by headmaps.

## Try out MoXI on your own model
We offer two implementations of MoXI.

### Implentation 1 (Model-agnostic implementation; in preparation).
If your model is a CNN, use this implementation.
```
model = load_your_model(...)
...
```
### Implementation 2 (ViT-aware implementation).
If you use Vision Transformer models, we highly recommend using this implementation.
If your model is based on `VisionTransformerClassifier` class of a HuggingFace, it’s very simple.
```
```
Otherwise, you need a slight modification in your model.
- allow `forward()` functions to recieve `embedding_mask` keyword argument
- call `select_batch_removing()` in the input embedding module.
No worries; after this modification, you can still load your pre-trained weights.
```
from .mask_vit_embedding import select_batch_removing

class YourViTClassifier(...):
  def __init__(...):
    self.ViTModel = YourViTModel(...)
    ...
  def forward(x, ..., embedding_mask=None): # MODIFICATION: new keyword argument embedding_mask
    output = self.YourViTModel(x, embedding_mask)
    ...

class YourViTModel(...):
  def __init__(...):
    self.ViTEmbedding = YourViTEmbedding(...)
    ...
  def forward(x, ..., embedding_mask=None): # MODIFICATION: new keyword argument embedding_mask
    embedding = self.ViTEmbedding(x, embedding_mask)
    ...

class YourEmbedding(...):
  def __init(...):
    ...
  def forward(x, ..., embedding_mask=None): # MODIFICATION: new keyword argument embedding_mask
    ...
    embeddings = self.patch_embeddings(x, ...)
    embeddings = embeddings + self.position_embeddings[:, 1:, :]
    
    # MODIFICATION: two lines added.
    if embedding_masking is not None:
        embeddings = select_batch_removing(embeddings, embedding_masking)
    ...
```
## Contact
- Kosuke Sumiyasu: [kosuke.sumiyasu@gmail.com](kosuke.sumiyasu@gmail.com)
- Kazuhiko Kawamoto: [kawa@faculty.chiba-u.jp](kawa@faculty.chiba-u.jp)
- Hiroshi Kera: [kera@chiba-u.jp](kera@chiba-u.jp)
## Citation
If you find this useful, please cite:
```bibtex
@inproceedings{kosuke2024identifying,
  author    = {Kosuke Sumiyasu and Kazuhiko Kawamoto and Hiroshi Kera},
  title     = {Identifying Important Group of Pixels using Interactions},
  journal   = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024}
}
```