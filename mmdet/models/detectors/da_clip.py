import torch
import torch.nn as nn

# from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import DETECTORS, build_backbone, build_head, build_neck, build_discriminator, build_domainmask
from .da_base import DABaseDetector
from mmdet.utils import convert_splitbn_model 
import clip
import torch.nn.functional as F
from .da_clip_prompt_head import DAPromptHead



@DETECTORS.register_module()
class DAClipDetector(DABaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 feat_dis_head=None,
                 ins_dis_head=None,
                 domain_mask=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(DAClipDetector, self).__init__()
        self.auxBN = backbone['type'] == 'AuxResNet'
        ##########################################
        #self.backbone = build_backbone(backbone)
        #self.clip_model, self.preprocess = clip.load('ViT-B/32', device, jit=False)
        #['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        self.clip_model, self.preprocess = clip.load('RN50', 'cuda', jit=False)
        self.clip_model.eval()
        for params in self.clip_model.parameters():
            params.requires_grad_(False)
        #for name, param in self.clip_model.named_parameters():
        #    print(name, param.requires_grad)
        #self.DAPrompt = DAPromptHead(('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        #   'bicycle'), self.clip_model).get_embedding()
        #self.DAPromptHead = DAPromptHead(('Pedestrians and vehicles',), self.clip_model)
        # 增加一个背景类别
        self.DAPromptHead = DAPromptHead(('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
           'bicycle', 'background'), self.clip_model)


        '''
        for param in self.clip_model.visual.layer4.parameters():
            param.requires_grad = True
        for param in self.clip_model.visual.layer3.parameters():
            param.requires_grad = True
        for param in self.clip_model.visual.layer2.parameters():
            param.requires_grad = True
        '''
        ##########################################
#        if self.auxBN:
#            self.backbone = convert_splitbn_model(self.backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)
        
        if feat_dis_head is not None:
            self.feat_dis_head = build_discriminator(feat_dis_head)
        else:
            self.feat_dis_head = None
        if ins_dis_head is not None:
            self.ins_dis_head = build_discriminator(ins_dis_head)
        else:
            self.ins_dis_head = None
        if domain_mask is not None:
            self.mask_head = build_domainmask(domain_mask)
        else:
            self.mask_head = None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    @property
    def with_feat_dis_head(self):
        return hasattr(self, 'feat_dis_head') and self.feat_dis_head is not None

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(DAClipDetector, self).init_weights(pretrained)
        ###############################################################
        #self.backbone.init_weights(pretrained=pretrained)
        ###############################################################
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)
            
    def extract_feat(self, img, domain):
        """Directly extract features from the backbone+neck."""
        if self.auxBN:
            x = self.backbone(img, domain)
        else:
            #x = self.backbone(img)
            x = self.clip_model.encode_image(img)
        #################################
        '''
        if self.training:
            loss_kd = []
            clip_x = self.clip_model.encode_image(img)
            for i, (feat_x, feat_clip) in enumerate(zip(x, clip_x)):
                if i == 0:
                    continue
                N, C, H, W = feat_clip.shape
                loss_kd.append(F.l1_loss(feat_x, feat_clip))
            loss_kd = torch.mean(torch.stack(loss_kd), dim=0)
        '''
        #################################

        if self.with_neck:
            x = self.neck(x)                
        #################################
        #if self.training:
        #    return x, loss_kd
        #################################
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img_s, 
                      img_metas_s, 
                      domain_s, 
                      data_s, 
                      img_t, 
                      img_metas_t, 
                      domain_t, 
                      data_t,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        ####################################################################
        #self.DAPrompt = self.DAPromptHead.get_embedding() #[domains * (cls + 1), 1024]->[domains * (cls + 1), 256]
        text_embedding, cos_score = self.DAPromptHead.get_embedding() #[domains * (cls + 1), 1024]
        text_embedding = text_embedding / text_embedding.norm(dim=1, keepdim=True) #正则化

        gt_bboxes_s = data_s['gt_bboxes']
        gt_labels_s = data_s['gt_labels']
        gt_bboxes_t = data_t['gt_bboxes']
        gt_labels_t = data_t['gt_labels']

        #x_t_origin, loss_kd_t = self.extract_feat(img_t, domain_t)
        #x_t = tuple(level + torch.mul(level, self.DAPrompt[1].unsqueeze(0)) for level in x_t_origin)
        x_t = self.extract_feat(img_t, domain_t)

        #x_s_origin, loss_kd_s = self.extract_feat(img_s, domain_s)
        #x_s = tuple(level + torch.mul(level, self.DAPrompt[0].unsqueeze(0)) for level in x_s_origin)
        #获取text prompt也可以用来作为衡量像素距离的指标？越和text prompt相似，就越增大该像素点的激活值
        x_s = self.extract_feat(img_s, domain_s)


        losses = dict()
        ####################
        losses.update({'loss_cos':cos_score})
        if self.feat_dis_head is not None:
            ########################
            pass
            ###########################
            #loss_feat_s = self.feat_dis_head.forward_train(x_s, domain_s)
            #loss_feat_t = self.feat_dis_head.forward_train(x_t, domain_t)
            #losses.update({'loss_feat_s':loss_feat_s['loss_feat']})
            #losses.update({'loss_feat_t':loss_feat_t['loss_feat']})
        if self.mask_head is not None:
            mask = self.mask_head(x_s, x_t)[0]
            x_mask = []
            for i, (x, mask_s) in enumerate(zip(x_s, mask)):
                x_mask.append(x * mask_s)
            x_mask = tuple(x_mask)
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list, proposal_list_t = self.rpn_head.forward_train(
                x_mask,
                img_metas_s,
                x_t,
                img_metas_t,
                gt_bboxes_s,
                gt_labels_s=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update({'mask_cls_loss':rpn_losses['loss_rpn_cls']})
            losses.update({'mask_bbox_loss':rpn_losses['loss_rpn_bbox']})
            roi_losses, _ = self.roi_head.forward_train(x_mask, img_metas_s, proposal_list,
                                                              gt_bboxes_s, gt_labels_s,
                                                              gt_bboxes_ignore, gt_masks,
                                                              **kwargs)
       
            losses.update({'mask_loss_cls':roi_losses['loss_cls']})
            losses.update({'mask_loss_bbox':roi_losses['loss_bbox']})


        else:
        #     RPN forward and loss
            if self.with_rpn:
                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                  self.test_cfg.rpn)
                rpn_losses, proposal_list_s, proposal_list_t = self.rpn_head.forward_train(
                    x_s,
                    img_metas_s,
                    x_t,
                    img_metas_t,
                    gt_bboxes_s,
                    gt_labels_s=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg)
                losses.update(rpn_losses)
            else:
                proposal_list = proposals
            is_source = True
            roi_losses, bbox_feat_s = self.roi_head.forward_train(x_s, img_metas_s, proposal_list_s,
                                                              gt_bboxes_s, gt_labels_s,
                                                              text_embedding, is_source,
                                                              gt_bboxes_ignore, gt_masks,
                                                              **kwargs)
            import math
            if math.isnan(roi_losses['loss_bbox']):
                roi_losses, bbox_feat_s = self.roi_head.forward_train(x_s, img_metas_s, proposal_list_s,
                                                              gt_bboxes_s, gt_labels_s,
                                                              text_embedding, is_source,
                                                              gt_bboxes_ignore, gt_masks,
                                                              **kwargs)



        if self.ins_dis_head is not None:
            bbox_feat_s = bbox_feat_s.view(-1, 256*7*7)
            is_source = False
            _, bbox_feat_t = self.roi_head.forward_train(x_t, img_metas_t, proposal_list_t,
                                                         gt_bboxes_t, gt_labels_t,
                                                         text_embedding, is_source,
                                                         gt_bboxes_ignore, gt_masks,
                                                         **kwargs)
            bbox_feat_t = bbox_feat_t.view(-1, 256*7*7)
            loss_ins_s = self.ins_dis_head.forward_train(bbox_feat_s, domain_s)
            loss_ins_t = self.ins_dis_head.forward_train(bbox_feat_t, domain_t)
            losses.update({'loss_ins_s': loss_ins_s['loss_ins']})
            losses.update({'loss_ins_t': loss_ins_t['loss_ins']})
        losses.update(roi_losses)
        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, domain, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        #################################
        #self.DAPrompt = self.DAPromptHead.get_embedding() #[domains * cls, 1024]->[domains * cls, 256]
        text_embedding, _ = self.DAPromptHead.get_embedding() #[domains * cls, 1024]

        x = self.extract_feat(img, domain)
        #x_origin = self.extract_feat(img, domain)
        #x = tuple(level + torch.mul(level, self.DAPrompt[1].unsqueeze(0)) for level in x_origin)
        #####################################
        #import numpy as np
        #import os
        #np.save(os.path.splitext(os.path.split(img_metas[0]['ori_filename'])[1])[0]+'.npy', x[0].detach().cpu())
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, text_embedding, rescale=rescale)
    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
