# Clustering as Attention: Unified Image Segmentation with Hierarchical Clustering
Official Implementation of HCFormer in PyTorch.  
arXiv: https://arxiv.org/abs/2205.09949


## Installation
### Requirements
- Linux with Python ≥ 3.6
- PyTorch ≥ 1.9 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- OpenCV is optional but needed by demo and visualization
- `pip install -r requirements.txt`

## Getting Started

See [Preparing Datasets for HCFormer](datasets/README.md).

See [Getting Started with HCFormer](GETTING_STARTED.md).

## License
A large part of this project relises on [the Mask2Former repository](https://github.com/facebookresearch/Mask2Former).  
The code related to Mask2Former is subject to Mask2Former's licence.

## Citation

If you use HCFormer in your research or wish to refer to the results, please use the following BibTeX entry.

```BibTeX
@article{suzuki2022clustering,
  title={Clustering as Attention: Unified Image Segmentation with Hierarchical Clustering},
  author={Suzuki, Teppei},
  journal={arXiv preprint arXiv:2205.09949},
  year={2022}
}
```

Please also consider the following BibTeX entry. (This paper is a preliminary work for HCFormer)

```BibTeX
@article{suzuki2021implicit,
  title={Implicit Integration of Superpixel Segmentation into Fully Convolutional Networks},
  author={Suzuki, Teppei},
  journal={arXiv preprint arXiv:2103.03435},
  year={2021}
}
```

## Acknowledgement

Code is largely based on Mask2Former (https://github.com/facebookresearch/Mask2Former).
