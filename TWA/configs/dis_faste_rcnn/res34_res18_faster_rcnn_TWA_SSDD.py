_base_ = [
    './faster_rcnn_res18_fpn_LEVIR.py'
]

model = dict(
    type='Distilling_Two_TWA',
    distill = dict(
        pretrained_stu = "/media/yangxilab/DiskA/zhangs/mmdetection_FRS/model/faste_res18_LEVIR_1000_600.pth",
        teacher_cfg='./faster_rcnn_res34_fpn_LEVIR.py',
        teacher_model_path='/media/yangxilab/DiskA/zhangs/mmdetection_FRS/model/faste_res34_LEVIR_1000_600.pth',
        distill_warm_step=500,
        distill_feat_weight=0.002,
        distill_cls_weight=0.1,
        stu_MAAM_in_chanels = 256,
        stu_MAAM_out_channels=256,
    )
)

custom_imports = dict(imports=['mmdet.core.utils.increase_hook'], allow_failed_imports=False)
custom_hooks = [dict(type='NumClassCheckHook'), dict(type='Increase_Hook',)]
seed=520
# find_unused_parameters=True
