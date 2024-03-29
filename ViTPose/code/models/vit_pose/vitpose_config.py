evaluation = dict()
target_type = 'GaussianHeatmap'

channel_cfg = dict(
    num_output_channels=14,
    dataset_joints=14,
    dataset_channel=list(range(14)),
    inference_channel=list(range(14)))

# model settings
model = dict(
    type='TopDown',
    pretrained=None,
    backbone=dict(
        type='ViT',
        img_size=(256, 256),
        patch_size=16,
        embed_dim=64,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=64,
        num_deconv_layers=2,
        num_deconv_filters=(64, 64),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1,),
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=None),
    train_cfg=dict(),
    test_cfg=dict())