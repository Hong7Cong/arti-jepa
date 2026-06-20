"""Multiblock-3D mask configs, re-tuned for the small rtMRI token grids (B.3).

The V-JEPA pretrain recipe masks a 16x16 spatial grid (256px). At lower
resolutions the grid shrinks fast (128->8x8, 96(pad)->6x6), and the 256-res
mask scales become degenerate (a 0.15-scale block on a 6x6 grid is < 1 token).
``mask_config_for`` returns sensible block counts / scales per grid so the
encoder always sees a non-trivial context and a non-trivial prediction target.

Each entry matches the schema consumed by
``src.masks.multiseq_multiblock3d.MaskCollator``.
"""


def mask_config_for(spatial_tokens: int):
    """Return a list of mask specs given the spatial grid side length (in tokens).

    Two complementary masks per the paper: many small "short-range" blocks plus
    a couple of large "long-range" blocks.
    """
    if spatial_tokens >= 14:          # 256px -> 16x16 (paper default)
        return [
            dict(num_blocks=8, spatial_scale=(0.15, 0.15), temporal_scale=(1.0, 1.0),
                 aspect_ratio=(0.75, 1.5), max_temporal_keep=1.0,
                 max_keep=None, full_complement=False),
            dict(num_blocks=2, spatial_scale=(0.70, 0.70), temporal_scale=(1.0, 1.0),
                 aspect_ratio=(0.75, 1.5), max_temporal_keep=1.0,
                 max_keep=None, full_complement=False),
        ]
    if spatial_tokens >= 8:           # 128px -> 8x8
        return [
            dict(num_blocks=4, spatial_scale=(0.25, 0.25), temporal_scale=(1.0, 1.0),
                 aspect_ratio=(0.75, 1.5), max_temporal_keep=1.0,
                 max_keep=None, full_complement=False),
            dict(num_blocks=2, spatial_scale=(0.60, 0.60), temporal_scale=(1.0, 1.0),
                 aspect_ratio=(0.75, 1.5), max_temporal_keep=1.0,
                 max_keep=None, full_complement=False),
        ]
    # 96px(pad) -> 6x6 (very small grid)
    return [
        dict(num_blocks=3, spatial_scale=(0.30, 0.30), temporal_scale=(1.0, 1.0),
             aspect_ratio=(0.75, 1.5), max_temporal_keep=1.0,
             max_keep=None, full_complement=False),
        dict(num_blocks=1, spatial_scale=(0.55, 0.55), temporal_scale=(1.0, 1.0),
             aspect_ratio=(0.75, 1.5), max_temporal_keep=1.0,
             max_keep=None, full_complement=False),
    ]
