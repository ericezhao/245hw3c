# MMSegmentation config for users who have a CUDA machine.
# Eric's Mac has no mmcv build, so part4_deeplabv3.py trains the same
# DeepLabV3+ / ResNet-50 design with segmentation-models-pytorch.
# On Linux + CUDA, after `mim install mmengine mmcv mmsegmentation`:
# python -m mim train mmseg configs/deeplabv3plus_bdd.py --work-dir ./seg_work_dir

_base_ = [
    "mmseg::deeplabv3plus/deeplabv3plus_r50-d8_4xb4-80k_cityscapes-512x1024.py"
]

data_root = "./data/bdd100k_mini"
num_classes = 11  # background + 10 BDD object classes
metainfo = dict(
    classes=(
        "background", "person", "rider", "car", "truck", "bus",
        "train", "motorcycle", "bicycle", "traffic light", "traffic sign",
    )
)

model = dict(
    decode_head=dict(num_classes=num_classes, ignore_index=255),
    auxiliary_head=dict(num_classes=num_classes, ignore_index=255),
)

train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        type="BaseSegDataset",
        data_root=data_root,
        img_suffix=".jpg",
        seg_map_suffix=".png",
        metainfo=metainfo,
        data_prefix=dict(img_path="images/train", seg_map_path="../../seg_masks/train"),
        reduce_zero_label=False,
    ),
)

val_dataloader = dict(
    dataset=dict(
        type="BaseSegDataset",
        data_root=data_root,
        img_suffix=".jpg",
        seg_map_suffix=".png",
        metainfo=metainfo,
        data_prefix=dict(img_path="images/val", seg_map_path="../../seg_masks/val"),
        reduce_zero_label=False,
    ),
)

train_cfg = dict(type="IterBasedTrainLoop", max_iters=20000, val_interval=2000)
