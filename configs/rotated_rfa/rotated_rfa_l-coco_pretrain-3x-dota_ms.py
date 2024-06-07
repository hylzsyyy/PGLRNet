_base_ = './rotated_rfa_neck_l-3x-dota_ms.py'

coco_ckpt = '/home/research/OpenMMLab/mmdetection-3.0.0rc6/work_dirs/rfa_l_8xb32-300e_coco/epoch_290.pth'  # noqa

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=coco_ckpt)),
    neck=dict(
        init_cfg=dict(type='Pretrained', prefix='neck.',
                      checkpoint=coco_ckpt)),
    bbox_head=dict(
        init_cfg=dict(
            type='Pretrained', prefix='bbox_head.', checkpoint=coco_ckpt)))

# batch_size = (2 GPUs) x (4 samples per GPU) = 8
train_dataloader = dict(batch_size=4, num_workers=8)
