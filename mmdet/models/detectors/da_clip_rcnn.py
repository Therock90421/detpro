from ..builder import DETECTORS
from .da_clip import DAClipDetector


@DETECTORS.register_module()
class DAClipRCNN(DAClipDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 feat_dis_head=None,
                 ins_dis_head=None,
                 domain_mask=None,
                 neck=None,
                 pretrained=None):
        super(DAClipRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            feat_dis_head=feat_dis_head,
            ins_dis_head=ins_dis_head,
            domain_mask=domain_mask,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
