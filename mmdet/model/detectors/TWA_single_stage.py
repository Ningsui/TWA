from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .single_stage import SingleStageDetector
from mmdet.core.bbox.iou_calculators import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import  load_checkpoint, _load_checkpoint, load_state_dict
from collections import OrderedDict

@DETECTORS.register_module()
class Distilling_Single_TWA(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 distill=None,):
        super(Distilling_Single_TWA, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        from mmdet.apis.inference import init_detector
        self.device = torch.cuda.current_device()
        self.teacher = init_detector(distill.teacher_cfg, \
                        distill.teacher_model_path, self.device)
        self.stu_feature_adap = MAAM() 
        self.distill_feat_weight = distill.get("distill_feat_weight",0)
        self.distill_cls_weight = distill.get("distill_cls_weight",0)
        for m in self.teacher.modules():
            for param in m.parameters():
                param.requires_grad = False
        self.teacher_channels = 256
        self.generation = nn.Sequential(
            nn.Conv2d(self.teacher_channels, self.teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(self.teacher_channels, self.teacher_channels, kernel_size=3, padding=1))

    def forward_CPAM(self, x):
        min_val = x.min(dim=1, keepdim=True)[0]
        max_val = x.max(dim=1, keepdim=True)[0]
        min_mask = (x == min_val)
        max_mask = (x == max_val)
        combined_mask = min_mask | max_mask
        mean_val = x.mean(dim=1, keepdim=True)
        x = torch.where(combined_mask, mean_val, x)
        x = F.normalize(x)
        x = self.generation(x)
        return x
    
    def forward_train(self,
                        img,
                        img_metas,
                        gt_bboxes,
                        gt_labels,
                        gt_bboxes_ignore=None):

        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        stu_feature_MAAM = self.stu_feature_adap(x) 
        y = self.teacher.extract_feat(img)
        stu_bbox_outs = self.bbox_head(x)
        stu_cls_score = stu_bbox_outs[0]
        tea_bbox_outs = self.teacher.bbox_head(y)
        tea_cls_score = tea_bbox_outs[0]
        layers = len(stu_cls_score)
        distill_feat_loss, distill_cls_loss = 0, 0

        for layer in range(layers):
            stu_cls_score_sigmoid = stu_cls_score[layer].sigmoid()
            tea_cls_score_sigmoid = tea_cls_score[layer].sigmoid()
            mask = torch.max(tea_cls_score_sigmoid, dim=1).values
            mask = mask.detach()
            if layer !=3 and layer !=4:
                feat_loss =((layers-layer)/layers)*torch.pow((self.forward_CPAM(y[layer])- stu_feature_MAAM[layer]), 2)
                distill_feat_loss +=  (feat_loss * mask[:,None,:,:]).sum() / mask.sum()
            cls_loss = F.binary_cross_entropy(stu_cls_score_sigmoid, tea_cls_score_sigmoid,reduction='none')
            distill_cls_loss +=  (cls_loss * mask[:,None,:,:]).sum() / mask.sum()
        distill_feat_loss = distill_feat_loss * self.distill_feat_weight
        distill_cls_loss = distill_cls_loss * self.distill_cls_weight
        if self.distill_feat_weight:
            losses.update({"distill_feat_loss":distill_feat_loss})
        if self.distill_cls_weight:
            losses.update({"distill_cls_loss":distill_cls_loss})
        return losses


class MAAM(nn.Module):

    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 num = 5,
                 kernel = 3,
                 with_relu = False):
        super(MAAM, self).__init__()
        self.num = num
        self.firstconv =nn.ModuleList()
        self.conv1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.up_weight = nn.ParameterList()
        self.up_sample = nn.ModuleList()
        for i  in range(num):
            self.firstconv.append(DepthConvBlock(in_channels,out_channels))
            if i !=0 and i!=1:
                self.conv1.append(DepthConvBlock(in_channels,out_channels))
                self.conv2.append(DepthConvBlock(in_channels,out_channels))

        # self.up_sample.append(nn.Sequential(nn.Upsample(size=(19,32),mode='nearest'),DepthConvBlock(in_channels,out_channels)))
        self.up_sample.append(nn.Sequential(nn.Upsample(scale_factor=2,mode='nearest'),DepthConvBlock(in_channels,out_channels)))
        self.up_sample.append(nn.Sequential(nn.Upsample(scale_factor=2,mode='nearest'),DepthConvBlock(in_channels,out_channels)))
        self.up_weight.append(nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True))
        self.up_weight.append(nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True))
        self.relu = nn.ReLU()
        self.epsilon = 1e-6

    def forward(self,inputs):
        inputs_train = [self.firstconv[i](F.relu(inputs[i]))for i in range(len(inputs))]
        output_two=[]
        input = inputs_train[0:3] #
        
        w = self.relu(self.up_weight[0])
        w = w / (torch.sum(w, dim=0) + self.epsilon) 
        output_two.append(self.conv1[0](F.relu(input[0])))
        output_two.append(self.conv1[1](w[0]*F.relu(input[1])+w[1]*self.up_sample[0](F.relu(input[2]))))
        output_two.append(self.conv1[2](F.relu(input[2])))
        output =[]
        w = self.relu(self.up_weight[1])
        w = w / (torch.sum(w, dim=0) + self.epsilon)
        output.append(self.conv2[0](w[0]*F.relu(output_two[0])+w[1]*self.up_sample[1](F.relu(output_two[1]))+input[0]
                                    ))
        output.append(self.conv2[1](F.relu(output_two[1])+input[1]))
        output.append(self.conv2[2](F.relu(output_two[2])+input[2]))
        output.insert(3,inputs_train[3])
        output.insert(4,inputs_train[4])
    
        return output


class MaxPool2dStaticSamePadding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size
        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        import math
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x
    
class DepthConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, depth=1):
        super(DepthConvBlock, self).__init__()
        conv = []
        if kernel_size == 1:
            conv.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                          bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ))
        else:
            conv.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                          bias=False, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            ))
            for i in range(depth-1):
                conv.append(nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False, groups=out_channels),
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels),
                ))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)