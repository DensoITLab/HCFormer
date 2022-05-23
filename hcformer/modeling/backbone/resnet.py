# Copyright (c) Denso IT Lab., Inc.
import torch.nn as nn
import torchvision.models as M

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.layers import FrozenBatchNorm2d

from .utils import DeformConv, Clustering, TransformerEnc


def dconv_bottleneck(module, in_c, out_c):
    bneck0 = module[0]
    downsample = nn.Sequential(DeformConv(in_c, out_c, 1, padding=0),
                                bneck0.downsample[1])
    downsample[0].deform.weight.data.copy_(bneck0.downsample[0].weight.data)
    bottleneck_conv = bneck0.conv2
    dconv = DeformConv(bottleneck_conv.in_channels, bottleneck_conv.out_channels)
    dconv.deform.weight.data.copy_(bottleneck_conv.weight.data)
    bneck0.conv2 = dconv
    bneck0.downsample = downsample
    return nn.Sequential(bneck0, *[m for m in module[1:]])


class HCResNet(nn.Module):
    def __init__(
            self,
            depth=50,
            hierarchical_level=3,
            temp=0.05,
            emb_dim=128,
            downsampling='deform',
            aux=None,
            n_tenc_layers=6):
        super().__init__()
        assert hierarchical_level <= 3
        resnet = M.__dict__[f'resnet{depth}'](pretrained=True)
        resnet = FrozenBatchNorm2d.convert_frozen_batchnorm(resnet)
        self.stem = nn.Sequential(resnet.conv1,
                                  resnet.bn1,
                                  resnet.relu,
                                  resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        clustering_layers = []
        for idx in range(5-hierarchical_level, 5):
            ds_module = getattr(self, f'layer{idx}')
            coarse_dim, fine_dim = ds_module[0].downsample[0].weight.shape[:2]
            if downsampling == 'deform':
                dconv_layer = dconv_bottleneck(ds_module, fine_dim, coarse_dim)
                setattr(self, f'layer{idx}', dconv_layer)
            clustering_layers.append(Clustering(fine_dim, coarse_dim, emb_dim, temp))

        self.clustering_layers = nn.ModuleList(reversed(clustering_layers))
        self.level = hierarchical_level

        if aux == 'transformer':
            self.aux = TransformerEnc(2048, 256, 8, n_tenc_layers, 2048)
            self.num_features = [256, 512, 1024, 256]
        else:
            self.aux = None
            self.num_features = [256, 512, 1024, 2048]

    def forward(self, x):
        outs = {}
        x = self.stem(x)
        x1 = self.layer1(x)
        d_x1 = self.layer2[0](x1)
        x2 = self.layer2[1:](d_x1)
        d_x2 = self.layer3[0](x2)
        x3 = self.layer3[1:](d_x2)
        d_x3 = self.layer4[0](x3)
        x4 = self.layer4[1:](d_x3)
        if self.aux is not None:
            x4 = self.aux(x4)

        feat_pair = [(x3, d_x3), (x2, d_x2), (x1, d_x1)]
        assigns = [clustering(*feats) for feats, clustering in zip(feat_pair, self.clustering_layers)]
        outs["res2"] = x1
        outs["res3"] = x2
        outs["res4"] = x3
        outs["res5"] = x4
        outs["assign"] = assigns
        return outs


@BACKBONE_REGISTRY.register()
class D2HCResNet(HCResNet, Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        depth = cfg.MODEL.RESNETS.DEPTH
        level = cfg.MODEL.HC.LEVEL
        temp = cfg.MODEL.HC.TEMPERATURE
        hc_emb = cfg.MODEL.HC.EMB_DIM
        aux = cfg.MODEL.RESNETS.AUX
        downsampling = cfg.MODEL.RESNETS.DOWNSAMPLING

        super().__init__(depth, level, temp, hc_emb, downsampling, aux)

        self._out_features = cfg.MODEL.RESNETS.OUT_FEATURES

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.num_features[0],
            "res3": self.num_features[1],
            "res4": self.num_features[2],
            "res5": self.num_features[3],
        }

    def forward(self, x):
        assert (
            x.dim() == 4
        ), f"HCResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = super().forward(x)
        for k in y.keys():
            if k in self._out_features or k == 'assign':
                outputs[k] = y[k]
        return outputs


    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32
