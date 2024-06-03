import math

import torch
import numpy as np

from models.stacked_hourglass.StackedHourglass import PoseNet as Hourglass


def heatmap_loss(combined_hm_preds: torch.Tensor, heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Compute heatmap loss across multiple stacks of hourglass
    """
    calc_loss = lambda pred, gt: ((pred - gt) ** 2).mean(dim=[1, 2, 3])

    combined_loss = []
    nstack = combined_hm_preds.shape[1]

    for i in range(nstack):
        combined_loss.append(calc_loss(combined_hm_preds[:, i], heatmaps.to(combined_hm_preds.device)))

    if nstack == 1:
        combined_loss = combined_loss[0].unsqueeze(1)
    else:
        combined_loss = torch.stack(combined_loss, dim=1)
    return combined_loss


def heatmap_generator(joints: float, occlusion: bool, hm_shape: tuple, img_shape: tuple) -> tuple:
    """
    Create heatmap: BS x #jnts x 64 x 64
    """

    def draw_heatmap(pt_uv, use_occlusion, hm_shape, sigma=1.75) -> tuple:
        '''
        2D gaussian (exponential term only) centred at given point.
        No constraints on point to be integer only.
        :param im: (Numpy array of size=64x64) Heatmap
        :param pt: (Numpy array of size=2) Float values denoting point on the heatmap
        :param sigma: (Float) self.joint_size which determines the standard deviation of the gaussian
        :return: (Numpy array of size=64x64) Heatmap with gaussian centred around point.
        '''

        im = np.zeros(hm_shape, dtype=np.float32)

        # If joint is absent
        if pt_uv[2] == -1:
            return im, 0

        elif pt_uv[2] == 0:
            if not use_occlusion:
                return im, 0

        else:
            assert pt_uv[2] == 1, "joint[2] should be (-1, 0, 1), but got {}".format(pt_uv[2])

        # Point around which Gaussian will be centred.
        pt_uv = pt_uv[:2]
        pt_uv_rint = np.rint(pt_uv).astype(int)

        # Size of 2D Gaussian window.
        size = int(math.ceil(6 * sigma))
        # Ensuring that size remains an odd number
        if not size % 2:
            size += 1

        # Check whether gaussian intersects with im:
        if (pt_uv_rint[0] - (size//2) >= hm_shape[0]) or (pt_uv_rint[0] + (size//2) <= 0) \
                or (pt_uv_rint[1] - (size//2) > hm_shape[1]) or (pt_uv_rint[1] + (size//2) < 0):

            return im, 0

        else:
            # Generate gaussian, with window=size and variance=sigma
            u = np.arange(pt_uv_rint[0] - (size // 2), pt_uv_rint[0] + (size // 2) + 1)
            v = np.arange(pt_uv_rint[1] - (size // 2), pt_uv_rint[1] + (size // 2) + 1)
            uu, vv = np.meshgrid(u, v, sparse=True)
            z = np.exp(-((uu - pt_uv[0]) ** 2 + (vv - pt_uv[1]) ** 2) / (2 * (sigma ** 2)))
            z = z.T

            # Identify indices in im that will define the crop area
            top = max(0, pt_uv_rint[0] - (size//2))
            bottom = min(hm_shape[0], pt_uv_rint[0] + (size//2) + 1)
            left = max(0, pt_uv_rint[1] - (size//2))
            right = min(hm_shape[1], pt_uv_rint[1] + (size//2) + 1)

            im[top:bottom, left:right] = \
                z[top - (pt_uv_rint[0] - (size//2)): top - (pt_uv_rint[0] - (size//2)) + (bottom - top),
                  left - (pt_uv_rint[1] - (size//2)): left - (pt_uv_rint[1] - (size//2)) + (right - left)]

            return im, 1   # heatmap, joint_exist


    assert len(joints.shape) == 3, 'Joints should be rank 3:' \
                                   '(num_person, num_joints, [u,v,vis]), but is instead {}'.format(joints.shape)

    heatmaps = np.zeros([joints.shape[1], hm_shape[0], hm_shape[1]], dtype=np.float32)
    joints_exist = np.zeros([joints.shape[1]], dtype=np.uint8)

    # Downscale
    downscale = [(img_shape[0] - 1)/(hm_shape[0] - 1), ((img_shape[1] - 1)/(hm_shape[1] - 1))]
    joints /= np.array([downscale[0], downscale[1], 1]).reshape(1, 1, 3)

    # Iterate over number of heatmaps
    for i in range(joints.shape[1]):

        # Create new heatmap for joint
        hm_i = np.zeros(hm_shape, dtype=np.float32)

        # Iterate over persons
        for p in range(joints.shape[0]):
            hm_, joint_present = draw_heatmap(pt_uv=joints[p, i, :], use_occlusion=occlusion, hm_shape=hm_shape)
            joints_exist[i] = max(joints_exist[i], joint_present)
            hm_i = np.maximum(hm_i, hm_)

        heatmaps[i] = hm_i

    return heatmaps, joints_exist


def fast_argmax(_heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Direct argmax from the heatmap, does not perform smoothing of heatmaps
    """
    with torch.no_grad():
        batch_size = _heatmaps.shape[0]
        num_jnts = _heatmaps.shape[1]
        spatial_dim = _heatmaps.shape[3]
        assert _heatmaps.shape[2] == _heatmaps.shape[3]

        assert len(_heatmaps.shape) == 4, "Heatmaps should be of shape: BatchSize x num_joints x 64 x64"
        _heatmaps = _heatmaps.reshape(batch_size, num_jnts, -1)
        indices = torch.argmax(_heatmaps, dim=2)
        indices = torch.cat(((indices // spatial_dim).view(batch_size, num_jnts, 1),
                            (indices % spatial_dim).view(batch_size, num_jnts, 1)),
                            dim=2)
        return indices.type(torch.float32)


def soft_argmax(_heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Differential argmax from the heatmap
    """
    batch_size = _heatmaps.shape[0]
    num_jnts = _heatmaps.shape[1]
    spatial_dim = _heatmaps.shape[3]
    assert _heatmaps.shape[2] == _heatmaps.shape[3]

    _heatmaps = _heatmaps.reshape(batch_size, num_jnts, -1)
    _heatmaps = torch.nn.functional.softmax(_heatmaps, dim=2)
    _heatmaps = _heatmaps.reshape(batch_size, num_jnts, spatial_dim, spatial_dim)
    p_x = torch.sum(_heatmaps, dim=2)
    p_y = torch.sum(_heatmaps, dim=3)

    id_xy = torch.arange(spatial_dim, device=p_x.device, requires_grad=False)

    softargmax_x = torch.sum(p_x * id_xy, dim=-1, keepdim=True)
    softargmax_y = torch.sum(p_y * id_xy, dim=-1, keepdim=True)

    softargmax_xy = torch.cat((softargmax_x, softargmax_y), dim=-1)

    return softargmax_xy


def count_parameters(model: Hourglass) -> int:
    """
    Number of differential parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
