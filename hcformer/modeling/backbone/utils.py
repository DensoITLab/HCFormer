# Copyright (c) Denso IT Lab., Inc.
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.layers import ModulatedDeformConv

from ..transformer_decoder.position_encoding import PositionEmbeddingSine
from ..transformer_decoder.transformer import TransformerEncoder, TransformerEncoderLayer


class Clustering(nn.Module):
    def __init__(self, fine_dim, coarse_dim, emb_dim, temp=0.1):
        super().__init__()
        self.fine_emb = nn.Conv2d(fine_dim, emb_dim, 1, bias=False)
        self.coarse_emb = nn.Conv2d(coarse_dim, emb_dim, 1, bias=False)
        self.fine_norm = nn.LayerNorm(fine_dim)
        self.coarse_norm = nn.LayerNorm(coarse_dim)
        self.register_parameter('temp', nn.Parameter(temp * torch.ones(1)))
        self.emb_dim = emb_dim

        nn.init.xavier_normal_(self.fine_emb.weight)
        nn.init.xavier_normal_(self.coarse_emb.weight)

    def forward(self, fine_feat, coarse_feat):
        batch_size, n_channels, height, width = fine_feat.shape
        coarse_shape = coarse_feat.shape
        # pre-norm
        fine_feat = fine_feat.flatten(-2).transpose(1, 2).contiguous()
        coarse_feat = coarse_feat.flatten(-2).transpose(1, 2).contiguous()
        fine_feat = self.fine_norm(fine_feat)
        coarse_feat = self.coarse_norm(coarse_feat)
        fine_feat = fine_feat.transpose(1, 2).contiguous().reshape(batch_size, n_channels, height, width)
        coarse_feat = coarse_feat.transpose(1, 2).contiguous().reshape(*coarse_shape)
        # map features into K-dimension space
        fine_feat = self.fine_emb(fine_feat)
        coarse_feat = self.coarse_emb(coarse_feat)
        # normalize
        fine_feat = F.normalize(fine_feat, 2, 1)
        coarse_feat = F.normalize(coarse_feat, 2, 1)
        # get 9 candidate clusters and corresponding pixel features
        candidate_clusters = F.unfold(coarse_feat, kernel_size=3, padding=1).reshape(batch_size, self.emb_dim, 9, -1)
        fine_feat = F.unfold(fine_feat, kernel_size=2, stride=2).reshape(batch_size, self.emb_dim, 4, -1)
        # calculate similarities
        candidate_clusters = candidate_clusters.permute(0, 3, 2, 1).reshape(-1, 9, self.emb_dim)
        fine_feat = fine_feat.permute(0, 3, 1, 2).reshape(-1, self.emb_dim, 4)
        similarities = torch.bmm(candidate_clusters, fine_feat).reshape(batch_size, -1, 9, 4).permute(0, 2, 3, 1).reshape(batch_size*9, 4, -1)
        # similarities = torch.einsum('bkcn,bkpn->bcpn', (candidate_clusters, fine_feat)).reshape(batch_size*9, 4, -1)
        similarities = F.fold(similarities, (height, width), kernel_size=2, stride=2).reshape(batch_size, 9, height, width)
        # normalize
        # mask zero padding regions by using the fact that the inner product is zero
        mask = -1e12 * (similarities == 0).float()
        # soft_assignment = (similarities * self.inv_temp + mask).softmax(1)
        soft_assignment = (similarities / (1e-8 + self.temp.abs()) + mask).softmax(1)
        return soft_assignment


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
    # decoded_features = torch.einsum('bkcn,bcpn->bkpn', (candidate_clusters, A)).reshape(batch_size, n_channels * 4, -1)
    decoded_features = F.fold(decoded_features, (height, width), kernel_size=2, stride=2)
    return decoded_features


class DeformConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.deform = ModulatedDeformConv(in_c, out_c, kernel_size=kernel_size, padding=padding, stride=stride)
        self.offset = nn.Conv2d(in_c, int(3*kernel_size**2), 3, padding=1, stride=stride)

        nn.init.constant_(self.offset.weight, 0)
        nn.init.constant_(self.offset.bias, 0)

    def forward(self, x, *args, **kwargs):
        offset_mask = self.offset(x).float()
        offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((offset_x, offset_y), dim=1)
        mask = mask.sigmoid()
        out = self.deform(x.float(), offset, mask)

        return out


class TransformerEnc(nn.Module):
    def __init__(
            self,
            in_channels,
            conv_dim,
            nhead,
            num_encoder_layers,
            dim_feedforward,
            dropout=0.1,
            activation="relu",
            normalize_before=False):
        super().__init__()
        self.pe_layer = PositionEmbeddingSine(conv_dim//2, normalize=True)
        self.input_proj = nn.Sequential(nn.Conv2d(in_channels, conv_dim, kernel_size=1), nn.GroupNorm(32, conv_dim))

        encoder_layer = TransformerEncoderLayer(
            conv_dim, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(conv_dim) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.input_proj(x)
        pos = self.pe_layer(x)
        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        pos = pos.flatten(2).permute(2, 0, 1)
        memory = self.encoder(x, pos=pos)
        return memory.permute(1, 2, 0).view(bs, c, h, w)

