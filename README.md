# Identifying Important Group of Pixels using Interactions[CVPR'24]

[Kosuke Sumiyasu](https://github.com/KosukeSumiyasu), [Kazuhiko Kawamoto](https://researchmap.jp/kawa), [Hiroshi Kera](https://hkera.wordpress.com/)

[[arxiv](https://arxiv.org/abs/2401.03785v2)]

## Overview

![heatmap_example](https://github.com/KosukeSumiyasu/MoXI/assets/77134522/70b84ae2-0b5c-44e0-90a2-c75c510d36e1)

We introduce MoXI (Model eXplanation by Interactions), that efficiently and accurately identifies a group of pixels with high prediction confidence.
Our method employs game-theoretic concepts, Shapley values and interactions, taking into account the effects of individual pixels and the cooperative influence of pixels on model confidence.

## Installation
Clone this repo:
```
$ git clone https://github.com/KosukeSumiyasu/MoXI
$ cd MoXI
```

## Identify and Evaluate the important pixels
To identify and evaluate the important pixels in the dataset, run the following scripts:
```
$ ./online_identify.sh
$ ./evaluate_curve.sh
```
Please refer to the Jupyter notebooks in ```notebook/``` for evaluations of each methods and comparisons using heatmaps.

## Data and models
We utilize the [ImageNet dataset](https://www.image-net.org/challenges/LSVRC/2012/).
Please specify the ImageNet dataset path in the config_file_path.yaml file for integration.


## Contact
- Kosuke Sumiyasu: [kosuke.sumiyasu@gmail.com](kosuke.sumiyasu@gmail.com)
- Kazuhiko Kawamoto: [kawa@faculty.chiba-u.jp](kawa@faculty.chiba-u.jp)
- Hiroshi Kera: [kera@chiba-u.jp](kera@chiba-u.jp)

## Citation
If you find this useful, please cite:
```bibtex
@article{kosuke2024identifying,
  author    = {Kosuke Sumiyasu and Kazuhiko Kawamoto and Hiroshi Kera},
  title     = {Identifying Important Group of Pixels using Interactions},
  journal   = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024}
}
```