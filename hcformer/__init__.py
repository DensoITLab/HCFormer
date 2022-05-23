# Copyright (c) Facebook, Inc. and its affiliates.

# Copyright (c) Denso IT Lab., Inc.
# Modified by Teppei Suzuki from: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/__init__.py

from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config, add_hcformer_config

# dataset loading
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper,
)
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)

# models
from .hcformer_model import HCFormer
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
