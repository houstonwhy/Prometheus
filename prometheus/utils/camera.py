"""Contains utility functions for camera transformation & sampiling
"""
import math
from  math import pi
import random
import numpy as np
import torch

from .math import matrix_to_square, quaternion_to_matrix, matrix_to_quaternion
    
# OpenCV uses a right-handed coordinate system with:
# - X-axis pointing to the right.
# - Y-axis pointing down.
# - Z-axis pointing forward.

# OpenGL(Colmap) uses a right-handed coordinate system with:
# - X-axis pointing to the right.
# - Y-axis pointing up.
# - Z-axis pointing backward.

# LLFF uses a right-handed coordinate system with:
# - X-axis pointing to the right.
# - Y-axis pointing down.
# - Z-axis pointing forward.

def opencv2opengl(poses):
    """
    Convert camera poses from OpenCV coordinate system to OpenGL coordinate system.

    Parameters:
    poses (numpy.ndarray): A 3x4 or Nx3x4 array of camera poses in OpenCV coordinate system.
    
    Returns:
    numpy.ndarray: A 3x4 or Nx3x4 array of camera poses in OpenGL coordinate system.
    """
    # Negate the Y and Z axis
    if isinstance(poses, torch.Tensor):
        return torch.cat([poses[:, : ,0:1], -poses[:, :, 1:2], -poses[:, :, 2:3], poses[:, :, 3:]], dim=2)
    elif isinstance(poses ,np.ndarray):
        return np.concatenate([poses[:, :, 0:1], -poses[:, :, 1:2], -poses[:, :, 2:3], poses[:, :, 3:]], axis=2)
    
def opengl2opencv(poses):
    return opencv2opengl(poses)

def llff2opengl(poses):
    """
    Convert camera poses from LLFF coordinate system to OpenGL coordinate system.
    
    Returns:
    numpy.ndarray: A 3x4 or Nx3x4 array of camera poses in OpenGL coordinate system.
    """
    # Swap the Y and Z axes and negate the Z axis
    if isinstance(poses, torch.Tensor):
        return torch.cat([poses[:, : ,1:2], -poses[:, :, 0:1], poses[:, :, 2:3], poses[:, :, 3:]], dim=2)
    return np.concatenate([poses[:, : ,1:2], -poses[:, :, 0:1], poses[:, :, 2:3], poses[:, :, 3:]], axis=2)

# c2ws = np.concatenate([c2ws[:, :, 1:2], -c2ws[:, :, 0:1], c2ws[:, :, 2:]], 2)
    
def opengl2llff(poses):
    return llff2opengl(poses)

def opencv2llff(poses):
    """
    Convert camera poses from OpenCV coordinate system to LLFF coordinate system. 
    
    Returns:
    numpy.ndarray: A 3x4 or Nx3x4 array of camera poses in OpenCV coordinate system.
    """
    if isinstance(poses, torch.Tensor):
        return torch.cat([poses[:, :, 1:2], poses[:, :, 0:1], -poses[:, :, 2:3], poses[:, :, 3:4]], dim=2)
    return np.concatenate([poses[:, :, 1:2], poses[:, :, 0:1], -poses[:, :, 2:3], poses[:, :, 3:4]], axis=2)
    
def llff2opencv(poses):
    return opencv2llff(poses)

def get_w2c(camera):
    w2c = np.linalg.inv(camera)
    return w2c

@torch.amp.autocast("cuda", enabled=False)
def quaternion_slerp(
    q0, q1, fraction, spin: int = 0, shortestpath: bool = True
):
    """Return spherical linear interpolation between two quaternions.
    Args:
        quat0: first quaternion
        quat1: second quaternion
        fraction: how much to interpolate between quat0 vs quat1 (if 0, closer to quat0; if 1, closer to quat1)
        spin: how much of an additional spin to place on the interpolation
        shortestpath: whether to return the short or long path to rotation
    """
    d = (q0 * q1).sum(-1)
    if shortestpath:
        # invert rotation
        d[d < 0.0] = -d[d < 0.0]
        q1[d < 0.0] = -q1[d < 0.0]

    d = d.clamp(-1, 1)
    angle = torch.acos(d) + spin * math.pi
    isin = 1.0 / (torch.sin(angle) + 1e-10)
    q0_ = q0 * torch.sin((1.0 - fraction) * angle) * isin.unsqueeze(-1)
    q1_ = q1 * torch.sin(fraction * angle) * isin.unsqueeze(-1)
    q = q0_ + q1_

    # Use torch.where to handle the assignment with correct broadcasting
    mask = angle < 1e-5
    q = torch.where(mask.unsqueeze(-1), q0, q)

    return q

def sample_from_two_pose(pose_a, pose_b, fraction, noise_strengths=[0, 0]):
    """
    Args:
        pose_a: first pose
        pose_b: second pose
        fraction
    """

    quat_a = matrix_to_quaternion(pose_a[..., :3, :3])
    quat_b = matrix_to_quaternion(pose_b[..., :3, :3])

    quaternion = quaternion_slerp(quat_a, quat_b, fraction)
    quaternion = torch.nn.functional.normalize(quaternion + torch.randn_like(quaternion) * noise_strengths[0], dim=-1)

    R = quaternion_to_matrix(quaternion)
    T = (1 - fraction)[...,None] * pose_a[..., :3, 3] + fraction[...,None] * pose_b[..., :3, 3]
    T = T + torch.randn_like(T) * noise_strengths[1]

    new_pose = pose_a.clone()
    new_pose[..., :3, :3] = R
    new_pose[..., :3, 3] = T
    return new_pose

def sample_from_dense_cameras(dense_cameras, t, noise_strengths=[0, 0, 0, 0]):
    B, N, C = dense_cameras.shape
    B, M = t.shape

    # Extract poses
    poses = dense_cameras[..., :12].view(B, N, 3, 4)
    positions = poses[..., :3, 3]  # (B, N, 3)
    rotation_matrices = poses[..., :3, :3]  # (B, N, 3, 3)
    quaternions = matrix_to_quaternion(rotation_matrices)  # (B, N, 4)

    # Calculate combined distance between consecutive poses
    d_pos = torch.norm(positions[:, 1:] - positions[:, :-1], dim=-1)  # (B, N-1)
    q1 = quaternions[:, :-1]
    q2 = quaternions[:, 1:]
    q1_norm = q1 / torch.norm(q1, dim=-1, keepdim=True)
    q2_norm = q2 / torch.norm(q2, dim=-1, keepdim=True)
    dot = torch.sum(q1_norm * q2_norm, dim=-1)
    angle = 2 * torch.acos(torch.clamp(dot, min=-1, max=1))  # (B, N-1)
    omega = 1.0
    d = torch.sqrt(d_pos**2 + (angle / omega)**2)  # (B, N-1)

    # Compute cumulative distances
    cumulative_distances = torch.cat([torch.zeros(B, 1).to(d_pos), torch.cumsum(d, dim=1)], dim=1)  # (B, N)
    total_length = cumulative_distances[:, -1]  # (B)

    # Map t to desired distance
    desired_distance = t * total_length.unsqueeze(1)  # (B, M)

    # Find the segment for each sample
    cumulative_expanded = cumulative_distances.unsqueeze(2)  # (B, N, 1)
    desired_expanded = desired_distance.unsqueeze(1)  # (B, 1, M)
    segment_mask = (desired_expanded >= cumulative_expanded[:, :-1, :]) & (desired_expanded < cumulative_expanded[:, 1:, :])
    segment_indices = torch.argmax(segment_mask.float(), dim=1)  # (B, M)

    # Calculate fraction within the segment
    lower = torch.gather(cumulative_distances, 1, segment_indices)  # (B, M)
    upper = torch.gather(cumulative_distances, 1, segment_indices + 1)  # (B, M)
    segment_length = upper - lower  # (B, M)
    fraction = (desired_distance - lower) / segment_length  # (B, M)
    fraction = torch.clamp(fraction, 0.0, 1.0)  # (B, M)

    # Gather poses for interpolation
    a_indices = segment_indices
    b_indices = segment_indices + 1
    a_pos = torch.gather(positions, 1, a_indices.unsqueeze(-1).repeat(1, 1, 3))
    b_pos = torch.gather(positions, 1, b_indices.unsqueeze(-1).repeat(1, 1, 3))
    a_quat = torch.gather(quaternions, 1, a_indices.unsqueeze(-1).repeat(1, 1, 4))
    b_quat = torch.gather(quaternions, 1, b_indices.unsqueeze(-1).repeat(1, 1, 4))

    # Interpolate position
    interp_pos = (1 - fraction.unsqueeze(-1)) * a_pos + fraction.unsqueeze(-1) * b_pos  # (B, M, 3)

    # Interpolate rotation using slerp
    interp_quat = quaternion_slerp(a_quat, b_quat, fraction.unsqueeze(-1))  # (B, M, 4)
    interp_quat = interp_quat / torch.norm(interp_quat, dim=-1, keepdim=True)

    # Convert quaternion to rotation matrix
    interp_rot = quaternion_to_matrix(interp_quat)  # (B, M, 3, 3)

    # Build interpolated pose
    interp_pose = torch.zeros(B, M, 3, 4).to(interp_rot)
    interp_pose[..., :3, :3] = interp_rot
    interp_pose[..., :3, 3] = interp_pos

    # Add noise to position and rotation
    if noise_strengths[0] > 0:
        interp_pose[..., :3, 3] += torch.randn_like(interp_pose[..., :3, 3]) * noise_strengths[0]
    if noise_strengths[1] > 0:
        # Perturb quaternion
        noise = torch.randn_like(interp_quat) * noise_strengths[1]
        interp_quat += noise
        interp_quat = interp_quat / torch.norm(interp_quat, dim=-1, keepdim=True)
        interp_rot = quaternion_to_matrix(interp_quat)
        interp_pose[..., :3, :3] = interp_rot

    # Interpolate other parameters if present
    a_ins = torch.gather(dense_cameras[..., 12:], 1, a_indices.unsqueeze(-1).repeat(1, 1, C-12))
    b_ins = torch.gather(dense_cameras[..., 12:], 1, b_indices.unsqueeze(-1).repeat(1, 1, C-12))
    interp_ins = (1 - fraction.unsqueeze(-1)) * a_ins + fraction.unsqueeze(-1) * b_ins

    # Combine pose and other parameters
    new_pose = torch.cat([interp_pose.view(B, M, 12), interp_ins], dim=2)

    return new_pose

@torch.amp.autocast("cuda", enabled=False)
def sample_rays(cameras: torch.Tensor, gt=None, h=None, w=None, N=-1, P=None):
    ''' get rays
    Args:
        cameras: [B, 18], cam2world poses[12]&intrinsics[4]&HW[2]
        h, w, N: int
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = cameras.device
    B = cameras.shape[0]
    c2w = torch.eye(4)[None].to(device).repeat(B, 1, 1)
    c2w[:, :3, :] = cameras[:, :12].reshape(B, 3, 4)
    fx, fy, cx, cy, H, W = cameras[:, 12:].chunk(6, -1) # each
    
    if h is not None:
        fx, cx = fx * h / H, cx * h / H
    else:
        h = int(H[0].item())
    if w is not None:
        fy, cy = fy * w / W, cy * w / W
    else:
        w = int(W[0].item())

    if N > 0:
        if P is not None:   
            assert N % (P ** 2) == 0
            num_patch = N // (P ** 2)

            short_side = min(h, w)
            max_multiplier = short_side // P

            multiplier = torch.randint(1, max_multiplier + 1, size=[B * num_patch], device=device)
            offset_i =  (torch.rand(B * num_patch, device=device) * (h - P * (multiplier) + multiplier)).floor_().long()
            offset_j =  (torch.rand(B * num_patch, device=device) * (w - P * (multiplier) + multiplier)).floor_().long()

            i = torch.arange(0, P, device=device).expand(B * num_patch, P) * multiplier[..., None] + offset_i[..., None] 
            j = torch.arange(0, P, device=device).expand(B * num_patch, P) * multiplier[..., None] + offset_j[..., None] 

            inds = (i.reshape(B * num_patch, P, 1) * w + j.reshape(B * num_patch, 1, P)).reshape(B, -1)
            
        else:
            inds = torch.rand(B, N, device=device).mul(h*w).floor_().long() # may duplicate
    else:
        inds = torch.arange(0, h*w, device=device).expand(B, h*w)
        
    i = inds % w + 0.5 # h axis
    j = torch.div(inds, w, rounding_mode='floor') + 0.5

    zs = - torch.ones_like(i)
    xs = - (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)

    # B x N x 3 & B x 3 x 3
    rays_d = F.normalize(directions @ c2w[:, :3, :3].transpose(-1, -2), dim=-1)

    rays_o = c2w[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d).contiguous() # [B, N, 3]
    rays_o, rays_d = rays_o.contiguous(), rays_d.contiguous()
    if gt is not None:
        if gt.shape[2] != h or gt.shape[3] != w:
            gt = F.interpolate(gt, size=(h,w), align_corners=False, mode='bilinear')
        rays_gt = torch.gather(gt.reshape(B, 3, -1).permute(0, 2, 1), 1, torch.stack(3 * [inds], -1))
        return rays_o, rays_d, rays_gt

    return rays_o, rays_d

def embed_rays(rays_o, rays_d):
    """get plucker rays from rayo rayd
    """
    return torch.cat([rays_d, torch.cross(rays_o, rays_d, dim=-1)], -1)

def get_camera2world(elev_rad, azim_rad, roll, radius):
    R_elev = np.array([
    [np.cos(elev_rad), 0, np.sin(elev_rad)],
    [0, 1, 0],
    [-np.sin(elev_rad), 0, np.cos(elev_rad)]])

    R_azim = np.array([
    [np.cos(azim_rad), -np.sin(azim_rad), 0],
    [np.sin(azim_rad), np.cos(azim_rad), 0],
    [0, 0, 1]])

    R_up = np.array([
        [np.cos(np.radians(90 + roll)), -np.sin(np.radians(90 + roll)), 0],
        [np.sin(np.radians(90 + roll)), np.cos(np.radians(90 + roll)), 0],
        [0, 0, 1]])

    R = np.dot(R_elev, R_azim)
    R = np.dot(R_up, R)

    c2w = np.eye(4, dtype=np.float32)
    c2w[2, 3] = radius

    rot = np.eye(4, dtype=np.float32)
    rot[:3, :3] = R.T
        
    c2w = rot @ c2w
    return c2w
    
def get_random_cameras(num_views=4, elev_range=[60, 120], azim_range=[0, 360], dist_range=[1.7, 2.0], focal_range=[560, 560], intrinsic=[512/2, 512/2, 512, 512], determined=False, ref_camera=None, inplace_first=False):
    
    c2ws = []
    focals = []

    for i in range(num_views):
        if determined:
            azim = i / num_views * (azim_range[1] - azim_range[0]) + azim_range[0]
            elev = (elev_range[0] + elev_range[1]) / 2
            dist = (dist_range[0] + dist_range[1]) / 2
            focal = (focal_range[0] + focal_range[1]) / 2
        else:
            azim, elev, dist, focal = ((random.random() * (r[1] - r[0]) + r[0]) for r in (azim_range, elev_range, dist_range, focal_range))

        elev_rad = np.radians(elev)
        azim_rad = np.radians(azim)

        c2w = get_camera2world(elev_rad, azim_rad, 0, dist)

        c2ws.append(c2w)
        focals.append(focal)
    
    c2ws = torch.from_numpy(np.stack(c2ws, axis=0)).float()
    focals = torch.from_numpy(np.stack(focals, axis=0)).float().unsqueeze(1)

    if ref_camera is not None:
        ref_c2w = matrix_to_square(ref_camera[..., :12].reshape(1, 3, 4)).detach().cpu()
        ref_w2c = torch.inverse(ref_c2w)
        c2ws = (ref_w2c.repeat(c2ws.shape[0], 1, 1) @ c2ws)

        if inplace_first:
            c2ws[0] = torch.eye(4)

    cameras = torch.cat([c2ws[:,:3,:].flatten(1, 2), focals, focals, torch.Tensor([*intrinsic])[None].repeat(num_views, 1)], dim=1)[None]

    return cameras

