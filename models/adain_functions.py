import torch
import numpy as np

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.reshape(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(N, C, 1, 1)
    feat_mean = feat.reshape(N, C, -1).mean(dim=2).reshape(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()

    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    # print(style_mean, style_std, content_mean, content_std)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def permute_features(features, perm_idx=None):
    N, C, H, W = features.size()

    if perm_idx is None:
        perm_idx = torch.randperm(N)

    perm_features = features[perm_idx]

    return perm_features


def permuted_adain(features, alpha=0.5, perm_idx=None):
    perm_features = permute_features(features, perm_idx)
    t = adaptive_instance_normalization(features, perm_features)

    t = alpha * t + (1 - alpha) * features

    return t