"""Utils that is used in construct dataset"""
import os
import importlib
import numpy as np
import torch
import roma

def matrix_to_square(mat):
    l = len(mat.shape)
    if l==3:
        return torch.cat([mat, torch.tensor([0,0,0,1]).repeat(mat.shape[0],1,1).to(mat.device)],dim=1)
    elif l==4:
        return torch.cat([mat, torch.tensor([0,0,0,1]).repeat(mat.shape[0],mat.shape[1],1,1).to(mat.device)],dim=2)


def orbit_camera_jitter(poses, strength=0.1):
    # poses: [B, 4, 4], assume orbit camera in opengl format
    # random orbital rotate

    B = poses.shape[0]
    rotvec_x = poses[:, :3, 1] * strength * np.pi * (torch.rand(B, 1, device=poses.device) * 2 - 1)
    rotvec_y = poses[:, :3, 0] * strength * np.pi / 2 * (torch.rand(B, 1, device=poses.device) * 2 - 1)

    rot = roma.rotvec_to_rotmat(rotvec_x) @ roma.rotvec_to_rotmat(rotvec_y)
    R = rot @ poses[:, :3, :3]
    T = rot @ poses[:, :3, 3:]

    new_poses = poses.clone()
    new_poses[:, :3, :3] = R
    new_poses[:, :3, 3:] = T
    
    return new_poses

# def import_str(string):
#     # From https://github.com/CompVis/taming-transformers
#     module, cls = string.rsplit(".", 1)
#     return getattr(importlib.import_module(module, package=None), cls)

def visualize_poses(poses, size=0.1, save_dir="./tmp"):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    # trimesh.Scene(objects).show()
    trimesh.Scene(objects).export(os.path.join(save_dir,"test.ply"))


def _safe_det_3x3(t: torch.Tensor):
    """
    Fast determinant calculation for a batch of 3x3 matrices.

    Note, result of this function might not be the same as `torch.det()`.
    The differences might be in the last significant digit.

    Args:
        t: Tensor of shape (N, 3, 3).

    Returns:
        Tensor of shape (N) with determinants.
    """

    det = (
        t[..., 0, 0] * (t[..., 1, 1] * t[..., 2, 2] - t[..., 1, 2] * t[..., 2, 1])
        - t[..., 0, 1] * (t[..., 1, 0] * t[..., 2, 2] - t[..., 2, 0] * t[..., 1, 2])
        + t[..., 0, 2] * (t[..., 1, 0] * t[..., 2, 1] - t[..., 2, 0] * t[..., 1, 1])
    )

    return det


def undistort_pinhole_image(image_data:np.array, cam_I:np.array, cam_D:np.array, desire_cam_I:np.array):
    H, W = image_data.shape[:2]
    # get camera infos
    fx, fy, cx, cy = cam_I[0, 0], cam_I[1, 1], cam_I[0, 2], cam_I[1, 2]
    k1, k2, p1, p2, k3 = cam_D[:5]
    # get new camera world points
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = (x.reshape((-1, 1)) - desire_cam_I[0, 2]) / desire_cam_I[0, 0]
    y = (y.reshape((-1, 1)) - desire_cam_I[1, 2]) / desire_cam_I[1, 1]
    z = np.ones_like(x)
    # normalize
    x = x / z
    y = y / z
    xx = x * x
    yy = y * y
    xy = x * y
    rr = xx + yy
    # Radial distortion 径向畸变
    rd_x = x * (1. + k1*rr + k2*rr*rr + k3*rr*rr*rr)
    rd_y = y * (1. + k1*rr + k2*rr*rr + k3*rr*rr*rr)
    # tangential distortion 切向畸变
    td_x = 2 * p1 * xy + p2 * (rr + 2 * xx)
    td_y = 2 * p2 * xy + p1 * (rr + 2 * yy)
    # rectified points
    r_x = rd_x + td_x
    r_y = rd_y + td_y
    # map to distored image
    map_u = (r_x * fx + cx).reshape((H, W)).astype(np.float32)
    map_v = (r_y * fy + cy).reshape((H, W)).astype(np.float32)
    img_undistored = cv2.remap(image_data, map_u, map_v, cv2.INTER_CUBIC)

    return img_undistored

@torch.no_grad()
def _check_valid_rotation_matrix(R, tol: float = 1e-7) -> None:
    """
    Determine if R is a valid rotation matrix by checking it satisfies the
    following conditions:

    ``RR^T = I and det(R) = 1``

    Args:
        R: an (N, 3, 3) matrix

    Returns:
        None

    Emits a warning if R is an invalid rotation matrix.
    """
    N = R.shape[0]
    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    eye = eye.view(1, 3, 3).expand(N, -1, -1)
    orthogonal = torch.allclose(R.bmm(R.transpose(1, 2)), eye, atol=tol)
    det_R = _safe_det_3x3(R)
    no_distortion = torch.allclose(det_R, torch.ones_like(det_R))
    if not (orthogonal and no_distortion):
        return False
    else:
        return True