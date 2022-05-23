# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detectron2/blob/main/tools/analyze_model.py

# Copyright (c) Denso IT Lab., Inc.
# Modified by Teppei Suzuki from: https://github.com/facebookresearch/Mask2Former/blob/main/tools/analyze_model.py

from PIL import Image
import logging
import cv2
import numpy as np
from collections import Counter
import tqdm
import torch
import torch.nn.functional as F
import warnings
warnings.simplefilter('ignore', UserWarning)
from fvcore.nn import flop_count_table  # can also try flop_count_str

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, LazyConfig, get_cfg, instantiate
from detectron2.data import build_detection_test_loader
from detectron2.engine import default_argument_parser
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.analysis import (
    FlopCountAnalysis,
    activation_count_operators,
    parameter_count_table,
)
from detectron2.utils.logger import setup_logger

# fmt: off
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

from hcformer import add_maskformer2_config, add_hcformer_config

logger = logging.getLogger("detectron2")


def setup(args):
    if args.config_file.endswith(".yaml"):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        add_hcformer_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.merge_from_list(args.opts)
        cfg.freeze()
    else:
        cfg = LazyConfig.load(args.config_file)
        cfg = LazyConfig.apply_overrides(cfg, args.opts)
    setup_logger(name="fvcore")
    setup_logger()
    return cfg


def do_flop(cfg):
    if isinstance(cfg, CfgNode):
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    else:
        data_loader = instantiate(cfg.dataloader.test)
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    model.eval()

    counts = Counter()
    total_flops = []
    for idx, data in zip(tqdm.trange(args.num_inputs), data_loader):  # noqa
        if args.use_fixed_input_size and isinstance(cfg, CfgNode):
            crop_size = cfg.INPUT.CROP.SIZE[0]
            data[0]["image"] = torch.zeros((3, crop_size, crop_size))
        flops = FlopCountAnalysis(model, data)
        if idx > 0:
            flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        counts += flops.by_operator()
        total_flops.append(flops.total())

    logger.info("Flops table computed from only one input sample:\n" + flop_count_table(flops))
    logger.info(
        "Average GFlops for each type of operators:\n"
        + str([(k, v / (idx + 1) / 1e9) for k, v in counts.items()])
    )
    logger.info(
        "Total GFlops: {:.1f}±{:.1f}".format(np.mean(total_flops) / 1e9, np.std(total_flops) / 1e9)
    )


def do_activation(cfg):
    if isinstance(cfg, CfgNode):
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    else:
        data_loader = instantiate(cfg.dataloader.test)
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    model.eval()

    counts = Counter()
    total_activations = []
    for idx, data in zip(tqdm.trange(args.num_inputs), data_loader):  # noqa
        count = activation_count_operators(model, data)
        counts += count
        total_activations.append(sum(count.values()))
    logger.info(
        "(Million) Activations for Each Type of Operators:\n"
        + str([(k, v / idx) for k, v in counts.items()])
    )
    logger.info(
        "Total (Million) Activations: {}±{}".format(
            np.mean(total_activations), np.std(total_activations)
        )
    )


def do_parameter(cfg):
    if isinstance(cfg, CfgNode):
        model = build_model(cfg)
    else:
        model = instantiate(cfg.model)
    logger.info("Parameter Count:\n" + parameter_count_table(model, max_depth=5))


def do_structure(cfg):
    if isinstance(cfg, CfgNode):
        model = build_model(cfg)
    else:
        model = instantiate(cfg.model)
    logger.info("Model Structure:\n" + str(model))


@torch.no_grad()
def do_clustering(cfg):
    from detectron2.structures import ImageList
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries
    import scipy.stats

    rgb = np.random.randint(0, 255, (2048, 3), dtype=np.uint8)

    def cluster_based_upsampling(x, A):
        batch_size, _, height, width = A.shape
        n_channels = x.shape[1]
        # get 9 candidate clusters and corresponding assignments
        candidate_clusters = F.unfold(x, kernel_size=3, padding=1).reshape(batch_size, n_channels, 9, -1)
        A = F.unfold(A, kernel_size=2, stride=2).reshape(batch_size, 9, 4, -1)
        # linear decoding
        candidate_clusters = candidate_clusters.permute(0, 3, 1, 2).reshape(-1, n_channels, 9)
        A = A.permute(0, 3, 1, 2).reshape(-1, 9, 4)
        decoded_features = torch.bmm(candidate_clusters, A).reshape(batch_size, -1, n_channels*4).permute(0, 2, 1).contiguous()
        decoded_features = F.fold(decoded_features, (height, width), kernel_size=2, stride=2)
        return decoded_features

    def vis_undersegmentation_error(clusters, gt_masks, img, ignore_index=0):
        c_idx = np.unique(clusters)
        valid_mask = gt_masks != ignore_index
        for i in c_idx:
            elem = clusters == i
            gt_clst = gt_masks[elem]
            gt_clst = gt_clst[gt_clst != ignore_index]
            mode = scipy.stats.mode(gt_clst).mode
            err_idx = np.logical_and(np.logical_and(gt_masks != mode, elem), valid_mask)
            img[err_idx, :] = (255, 0, 0)
        return img

    def soft2hard(A):
        idx = A.argmax(1, keepdim=True)
        return torch.zeros_like(A).scatter_(1, idx, torch.ones_like(idx).float())

    def clst2img(assigns, device='cpu'):
        outputs = []
        for i in range(len(assigns)):
            h, w = assigns[i].shape[-2:]
            idx = torch.arange(h//2*w//2, dtype=torch.float, device=device).reshape(1, 1, h//2, w//2)
            for a in assigns[i:]:
                idx = cluster_based_upsampling(idx, a)
            outputs.append(idx)
        return outputs

    def idx2rgb(idx, rgb):
        idx = idx.to('cpu').long().numpy()
        h, w = idx.shape[-2:]
        return rgb[idx.ravel() % 2048, :].reshape(h, w, 3)

    def get_pan_gt(path):
        img = np.asarray(Image.open(path)).astype(np.int64)
        id = img[:, :, 0] + 256 * img[:, :, 1] + 256**2 * img[:, :, 2]
        msk = np.zeros_like(id)
        for i in np.unique(id):
            msk[id == i] = i
        return msk

    if isinstance(cfg, CfgNode):
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    else:
        data_loader = instantiate(cfg.dataloader.test)
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    model.eval()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    for idx, data in zip(tqdm.trange(args.num_inputs), data_loader):  # noqa
        if args.use_fixed_input_size and isinstance(cfg, CfgNode):
            crop_size = cfg.INPUT.CROP.SIZE[0]
            data[0]["image"] = torch.zeros((3, crop_size, crop_size))
        # preprocess
        images = [x["image"].to(model.device) for x in data]
        images = [(x - model.pixel_mean) / model.pixel_std for x in images]
        images = ImageList.from_tensors(images, model.size_divisibility)
        if hasattr(data[0], 'image_id'):
            image_id = str(data[0]['image_id']).zfill(12)
        else:
            image_id = str(idx).zfill(4)
        # forward
        assign = model.backbone(images.tensor)['assign']
        hard_assign = [soft2hard(a) for a in assign]
        clusters = clst2img(hard_assign, model.device)
        pred = model(data)[0]
        # visualize
        h, w = pred['sem_seg'].shape[-2:]
        np_img = data[0]['image'].permute(1, 2, 0).contiguous().numpy()
        np_img = cv2.resize(np_img, (w, h), interpolation=cv2.INTER_CUBIC)
        if 'pan_seg_file_name' in data[0].keys():
            gt_id = get_pan_gt(data[0]['pan_seg_file_name'])
            gt_id = cv2.resize(gt_id, (w, h), interpolation=cv2.INTER_NEAREST)
            ignore = 0
        elif 'sem_seg' in data[0].keys():
            gt_id = data[0]['sem_seg'].numpy()
            gt_id = cv2.resize(gt_id, (w, h), interpolation=cv2.INTER_NEAREST)
            ignore = 255
        else:
            gt_id = None
        for i, (c, a) in enumerate(zip(clusters, hard_assign)):
            plt.imsave(os.path.join(cfg.OUTPUT_DIR, f'{image_id}-{i}.png'), idx2rgb(c, rgb))
            c = F.interpolate(c, (h, w), mode='nearest')
            np_c = c.to('cpu').long().numpy()[0,0]
            b_img = mark_boundaries(np_img, np_c)
            plt.imsave(os.path.join(cfg.OUTPUT_DIR, f'bound-{image_id}-{i}.png'), b_img)
            if gt_id is not None:
                seg_err = vis_undersegmentation_error(np_c, gt_id, np_img.copy(), ignore)
                plt.imsave(os.path.join(cfg.OUTPUT_DIR, f'err-{image_id}-{i}.png'), seg_err)
        torch.save(data[0], os.path.join(cfg.OUTPUT_DIR, f'{image_id}-meta.pth'))


if __name__ == "__main__":
    parser = default_argument_parser(
        epilog="""
Examples:
To show parameters of a model:
$ ./analyze_model.py --tasks parameter \\
    --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml
Flops and activations are data-dependent, therefore inputs and model weights
are needed to count them:
$ ./analyze_model.py --num-inputs 100 --tasks flop \\
    --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \\
    MODEL.WEIGHTS /path/to/model.pkl
"""
    )
    parser.add_argument(
        "--tasks",
        choices=["flop", "activation", "parameter", "structure", "vis"],
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "-n",
        "--num-inputs",
        default=100,
        type=int,
        help="number of inputs used to compute statistics for flops/activations, "
        "both are data dependent.",
    )
    parser.add_argument(
        "--use-fixed-input-size",
        action="store_true",
        help="use fixed input size when calculating flops",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="random seed"
    )
    args = parser.parse_args()
    assert not args.eval_only
    assert args.num_gpus == 1

    import random
    import numpy as np
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = setup(args)

    for task in args.tasks:
        {
            "flop": do_flop,
            "activation": do_activation,
            "parameter": do_parameter,
            "structure": do_structure,
            "vis": do_clustering
        }[task](cfg)
