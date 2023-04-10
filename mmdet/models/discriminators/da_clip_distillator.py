import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

from mmdet.core import (PointGenerator, build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from ..builder import DISCRIMINATORS, build_loss
from .da_base_discriminator import DABaseDiscriminator
import torch.nn.functional as F



@DISCRIMINATORS.register_module()
class DAClipDistillator(DABaseDiscriminator):
    """RepPoint head.

    Args:
        point_feat_channels (int): Number of channels of points features.
        gradient_mul (float): The multiplier to gradients from
            points refinement and recognition.
        point_strides (Iterable): points strides.
        point_base_scale (int): bbox scale for assigning labels.
        loss_cls (dict): Config of classification loss.
        loss_bbox_init (dict): Config of initial points loss.
        loss_bbox_refine (dict): Config of points loss in refinement.
        use_grid_points (bool): If we use bounding box representation, the
        reppoints is represented as grid points on the bounding box.
        center_init (bool): Whether to use center point assignment.
        transform_method (str): The methods to transform RepPoints to bbox.
    """  # noqa: W605

    def __init__(self,
                 in_channels,
                 **kwargs):
        #self.in_channels = in_channels
        self.in_channels = in_channels
        super(DAClipDistillator, self).__init__()
    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu2 = nn.LeakyReLU(0.1, inplace=False)
        self.sigmoid = nn.Sigmoid()
        self.cls_domain = nn.ModuleList()
        self.mse = nn.MSELoss()
        for i, channels in enumerate([[self.in_channels, self.in_channels], 
                                      [self.in_channels, int(self.in_channels/2)], 
                                      [int(self.in_channels/2), 1]]):
            chn_in = channels[0]
            chn_out = channels[1]
            self.cls_domain.append(
                    ConvModule(
                        chn_in,
                        chn_out,
                        1,
                        stride=1,
                        padding=0,
                        conv_cfg=None,
                        norm_cfg=None,
                        act_cfg=None))

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_domain:
            normal_init(m.conv, std=0.01)
    '''
    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):
        dis_feat = self.gradreverse.apply(x)
        for idx, dis_conv in enumerate(self.cls_domain):
            if idx == 2:
                dis_feat = dis_conv(dis_feat)
                break
            dis_feat = self.relu2(dis_conv(dis_feat))
        feat_dis_scores = self.sigmoid(dis_feat)

        return feat_dis_scores, torch.tensor([0])
    '''
    def loss_single(self, backbone_feature, clip_feature):
        # feature domain classification loss
        #feat_loss = 10*self.mse(torch.mean(feat_dis_scores), gt_domain[0].float())
        kd_loss = F.l1_loss(backbone_feature, clip_feature)
        return kd_loss, torch.tensor([0])

    def loss(self,
             backbone_features,
             clip_features):
        # compute loss
        loss_feat, tempt = multi_apply(
        self.loss_single,
        backbone_features,
        clip_features)
        loss_dict_all = {
            'loss_kd': loss_feat}
        return loss_dict_all

    def forward_train(self,
                      x,
                      y,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
        """
        losses = self.loss(x, y)

        return losses