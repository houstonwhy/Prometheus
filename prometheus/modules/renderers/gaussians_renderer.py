import os
import math
import numpy as np
import einops
import torch
from torch import nn
import torch.nn.functional as F
RASTERIZER_BACKEND = 'gsplat' 
# RASTERIZER_BACKEND = 'diff_gaussian_rasterization'
try:
    from gsplat import rasterization
    # from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
except:
    print('gsplat not install')
    RASTERIZER_BACKEND = 'diff_gaussian_rasterization'
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import torch.nn.functional as F

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def inverse_softplus(x, beta=1):
    return (torch.exp(beta * x) - 1).log() / beta

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def matrix_to_quaternion(M: torch.Tensor) -> torch.Tensor:
    """
    Matrix-to-quaternion conversion method. Equation taken from 
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    Args:
        M: rotation matrices, (... x 3 x 3)
    Returns:
        q: quaternion of shape (... x 4)
    """
    prefix_shape = M.shape[:-2]
    Ms = M.reshape(-1, 3, 3)

    trs = 1 + Ms[:, 0, 0] + Ms[:, 1, 1] + Ms[:, 2, 2]

    Qs = []

    for i in range(Ms.shape[0]):
        M = Ms[i]
        tr = trs[i]
        if tr > 0:
            r = torch.sqrt(tr) / 2.0
            x = ( M[ 2, 1] - M[ 1, 2] ) / ( 4 * r )
            y = ( M[ 0, 2] - M[ 2, 0] ) / ( 4 * r )
            z = ( M[ 1, 0] - M[ 0, 1] ) / ( 4 * r )
        elif ( M[ 0, 0] > M[ 1, 1]) and (M[ 0, 0] > M[ 2, 2]):
            S = torch.sqrt(1.0 + M[ 0, 0] - M[ 1, 1] - M[ 2, 2]) * 2 # S=4*qx 
            r = (M[ 2, 1] - M[ 1, 2]) / S
            x = 0.25 * S
            y = (M[ 0, 1] + M[ 1, 0]) / S 
            z = (M[ 0, 2] + M[ 2, 0]) / S 
        elif M[ 1, 1] > M[ 2, 2]: 
            S = torch.sqrt(1.0 + M[ 1, 1] - M[ 0, 0] - M[ 2, 2]) * 2 # S=4*qy
            r = (M[ 0, 2] - M[ 2, 0]) / S
            x = (M[ 0, 1] + M[ 1, 0]) / S
            y = 0.25 * S
            z = (M[ 1, 2] + M[ 2, 1]) / S
        else:
            S = torch.sqrt(1.0 + M[ 2, 2] - M[ 0, 0] -  M[ 1, 1]) * 2 # S=4*qz
            r = (M[ 1, 0] - M[ 0, 1]) / S
            x = (M[ 0, 2] + M[ 2, 0]) / S
            y = (M[ 1, 2] + M[ 2, 1]) / S
            z = 0.25 * S
        Q = torch.stack([r, x, y, z], dim=-1)
        Qs += [Q]

    return torch.stack(Qs, dim=0).reshape(*prefix_shape, 4)



def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    From Pytorch3d
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar, device='cpu'):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        # opengl2colmap
        c2w[:3, 1:3] *= -1

        w2c = np.linalg.inv(c2w)

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).to(device)
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .to(device)
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3]).to(device)
        
        self.intrinsics = self.image_height / (2 * math.tan(self.FoVx / 2)), self.image_width / (2 * math.tan(self.FoVy / 2)), self.image_height / 2, self.image_width / 2
        self.K = torch.eye(3).to(device)

        self.K[0,0], self.K[1,1], self.K[2,0], self.K[2,1] = self.image_height / (2 * math.tan(self.FoVx / 2)), self.image_width / (2 * math.tan(self.FoVy / 2)), self.image_height / 2, self.image_width / 2



class GaussianConverter(nn.Module):
    def __init__(self, 
                 gs_convert_mode:str = 'mvsplat', 
                 z_near:float = 0.001, 
                 z_far:float = 100,
                 s_min:float = 0.001,
                 s_max:float = 1.0
                 ):
        super().__init__()

        self.gs_convert_mode = gs_convert_mode
        self.z_near, self.z_far = z_near, z_far
        self.s_min, self.s_max = s_min, s_max
        self.gaussian_channels = [3, 2, 1, 1, 3, 4]

        if gs_convert_mode == 'director3d':
            self.register_buffer("opacity_offset", inverse_sigmoid(torch.Tensor([0.01]))[0], persistent=False)
            self.register_buffer("scales_offset", torch.log(torch.Tensor([1/100]))[0], persistent=False)
            self.register_buffer("rotations_offset", torch.Tensor([1., 0, 0, 0]).reshape(1, 4), persistent=False)
            self.register_buffer("muls", torch.Tensor(
                [0.01] * 3 + 
                [0.01] * 2 + 
                [0.05] * 1 + 
                [0.05] * 1 + 
                [0.005] * 3 + 
                [0.005] * 4).reshape(1, -1)) 
            self.muls = self.muls / self.muls.max()
        else:
            self.register_buffer("muls", torch.Tensor(
                [1.] * 3 + 
                [1.] * 2 + 
                [1.] * 1 + 
                [1.] * 1 + 
                [1.] * 3 + 
                [1.] * 4).reshape(1, -1)) 

    #@torch.amp.autocast('cuda', enabled=False)
    def forward(self, local_gaussian_params, cameras):

        # B x N x H x W x C
        B, N, C, h, w = local_gaussian_params.shape
        local_gaussian_params = local_gaussian_params.permute(0, 1, 3, 4, 2).reshape(-1, sum(self.gaussian_channels))
        local_gaussian_params = local_gaussian_params * self.muls

        features, uv_offset, depth, opacity, scales, rotations = local_gaussian_params.split(self.gaussian_channels, dim=-1)

        cameras = cameras.flatten(0, 1)
        device = cameras.device
        BN = cameras.shape[0]
        c2w = torch.eye(4)[None].to(device).repeat(BN, 1, 1)
        c2w[:, :3, :] = cameras[:, :12].reshape(BN, 3, 4)
        fx, fy, cx, cy, H, W = cameras[:, 12:].chunk(6, -1)

        fx, cx = fx * h / H, cx * h / H
        fy, cy = fy * w / W, cy * w / W

        inds = torch.arange(0, h*w, device=device).expand(BN, h*w)
        
        i = inds % w + 0.5
        j = torch.div(inds, w, rounding_mode='floor') + 0.5

        u = i / cx + uv_offset[..., 0].reshape(BN, h*w)
        v = j / cy + uv_offset[..., 1].reshape(BN, h*w)

        zs = - torch.ones_like(i)
        xs = - (u - 1) * cx / fx * zs
        ys = (v - 1) * cy / fy * zs
        directions = torch.stack((xs, ys, zs), dim=-1)

        # B x N x 3 & B x 3 x 3
        rays_d = F.normalize(directions @ c2w[:, :3, :3].transpose(-1, -2), dim=-1)
        # rays_d = directions @ c2w[:, :3, :3].transpose(-1, -2)
        # rays_d = rays_d / torch.norm(rays_d, p=2, dim=-1, keepdim=True)

        rays_o = c2w[..., :3, 3] # [B, 3]
        rays_o = rays_o[..., None, :].expand_as(rays_d)

        rays_o = rays_o.reshape(BN*h*w, 3)
        rays_d = rays_d.reshape(BN*h*w, 3)
        features = features.reshape(BN*h*w, -1, 3)
        
        # #? what‘s the meaning of 1.85 here means?
        # TODO change from metric depth prediction to depth = z_near + signoid(distance) * z_far 
        # TODO how to determain the value of z_far, 10? 20? 100?
        #torch.sigmoid(depth)        
        if self.gs_convert_mode == 'director3d':
            depth = depth.reshape(BN*h*w, 1) + 1.85 
            features = features / (2 * 0.28209479177387814)
            opacity = torch.sigmoid(opacity + self.opacity_offset)
            scales = torch.exp(scales + self.scales_offset)
            rotations = F.normalize(rotations + self.rotations_offset, dim=-1)
        elif self.gs_convert_mode == 'gslrm':
            depth = depth.reshape(BN*h*w, 1)
            d_ = torch.sigmoid(depth)
            depth = self.z_near * (1 - d_) +self.z_far * d_
            #features = features.reshape(BN*h*w, -1, 3)
            opacity = torch.sigmoid(opacity - 2.0)
            #scales = torch.exp(scales - 2.3).clip(0 ,0.3)
            scales = torch.exp(scales - 2.3)
            scales = scales.clip(0 ,0.3)
            rotations = F.normalize(rotations, dim=-1)
        elif self.gs_convert_mode == 'mvsplat':
            depth = depth.reshape(BN*h*w, 1)
            d_ = torch.sigmoid(depth)
            depth = self.z_near * (1 - d_) +self.z_far * d_
            s_ = torch.sigmoid(scales)
            scales = self.s_min * (1 - s_) + self.s_max * s_
            #BUG: features = 2 * torch.sigmoid(features) - 1 -> GS renderer only support non-negative input /output
            features = torch.sigmoid(features)
            opacity = torch.sigmoid(opacity)
            rotations = F.normalize(rotations, dim=-1)
        else:
            raise ValueError('unsupport decode type')
        xyz = (rays_o + depth * rays_d)
        return xyz.reshape(B, -1, 3), features.reshape(B, -1, 1, 3), opacity.reshape(B, -1, 1), scales.reshape(B, -1, 3), rotations.reshape(B, -1, 4)

class GaussianRenderer(nn.Module):
    def __init__(self, sh_degree=0, background=[1, 1, 1]):
        super().__init__()
        self.sh_degree = sh_degree
        
        self.register_buffer("bg_color", torch.tensor(background).float())

    def render(
        self,
        xyz,
        features,
        opacity,
        scales,
        rotations,
        viewpoint_camera,
        scaling_modifier=1.0,
        bg_color='random',
        max_depth=100,
        rasterizer_backend=RASTERIZER_BACKEND
    ):

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        if rasterizer_backend == 'diff_gaussian_rasterization':
            viewspace_points = (
                torch.zeros_like(
                    xyz,
                    dtype=xyz.dtype,
                    requires_grad=True,
                    device=xyz.device,
                )
                + 0
            )
            try:
                viewspace_points.retain_grad()
            except:
                pass

            # Set up rasterization configuration
            tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
            tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

            if bg_color == 'random':
                bg_color = torch.rand_like(self.bg_color)
            elif bg_color is None:
                bg_color = self.bg_color
            else:
                bg_color = bg_color
        
            raster_settings = GaussianRasterizationSettings(
                image_height=int(viewpoint_camera.image_height),
                image_width=int(viewpoint_camera.image_width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=bg_color,
                scale_modifier=scaling_modifier,
                viewmatrix=viewpoint_camera.world_view_transform,
                projmatrix=viewpoint_camera.full_proj_transform,
                sh_degree=self.sh_degree,
                campos=viewpoint_camera.camera_center,
                prefiltered=False,
                debug=False,
            )

            rasterizer = GaussianRasterizer(raster_settings=raster_settings)

            means3D = xyz
            means2D = viewspace_points
            colors = features
            # Rasterize visible Gaussians to image, obtain their radii (on screen).
            rendered_image, radii, rendered_depth, rendered_mask = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=colors,
                # shs=None, #BUG can not directly input [0,1] color here
                # #https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d/cuda_rasterizer/forward.cu#L241
                colors_precomp=None, 
                # colors_precomp=colors[:,0], 
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=None,
            )
            # TODO
            rendered_image = rendered_image
            # rendered_image = rendered_image.clamp(0, 1)
            rendered_mask = rendered_mask.clamp(0, 1)
            #rendered_depth = rendered_depth + max_depth * (1 - rendered_mask)

        elif rasterizer_backend == 'gsplat':
            means3D = xyz
            #means2D = viewspace_points
            colors = features
            if bg_color == 'random':
                bg_color = torch.rand_like(self.bg_color)
            elif bg_color is None:
                bg_color = self.bg_color
            else:
                bg_color = torch.tensor(bg_color).to(colors.device)

            # Replace diff_gaussian_rasterization by gsplat
            # Code Borror form https://github.com/liruilong940607/gaussian-splatting/blob/258123ab8f8d52038da862c936bd413dc0b32e4d/gaussian_renderer/__init__.py#L27
            # viewspace_points = (
            #     torch.zeros_like(
            #         xyz,
            #         dtype=xyz.dtype,
            #         requires_grad=True,
            #         device=xyz.device,
            #     )
            #     + 0
            # )
            # try:
            #     viewspace_points.retain_grad()
            # except:
            #     pass

            viewmat = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]
            tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
            tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
            focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
            focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
            K = torch.tensor([
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],]).to(viewpoint_camera.camera_center.device)
            render_colors, render_alphas, info = rasterization(
                means=means3D,  # [N, 3]
                quats=rotations,  # [N, 4]
                scales=scales,  # [N, 3]
                opacities=opacity.squeeze(-1),  # [N,]
                colors=colors[:,0],
                viewmats=viewmat[None],  # [1, 4, 4]
                Ks=K[None],  # [1, 3, 3]
                backgrounds=bg_color[None],
                width=int(viewpoint_camera.image_width),
                height=int(viewpoint_camera.image_height),
                render_mode="RGB+ED",
                packed=False,
                # sh_degree=self.sh_degree, -> if set, colors should input sh, else is rgb
            )
            try:
                info["means2d"].retain_grad() # [1, N, 2]
                # viewspace_points = info["means2d"].squeeze(0).retain_grad()
            except:
                pass
                # viewspace_points = info["means2d"].squeeze(0)
            viewspace_points = info["means2d"]
            radii = info["radii"].squeeze(0) # [N,]

             #rendered_image = einops.rearrange(render_colors[0,:,:,:3], 'H W C -> C H W').clamp(0, 1)
            rendered_image = einops.rearrange(render_colors[0,:,:,:3], 'H W C -> C H W')
            rendered_depth =  einops.rearrange(render_colors[0,:,:,3:], 'H W C -> C H W')
            rendered_mask =  einops.rearrange(render_alphas[0], 'H W C -> C H W').clamp(0, 1)
        else:
            raise ValueError(f"Unsupport render rasterizer backend: {rasterizer_backend}")

        self.radii.append(radii)
        self.viewspace_points.append(viewspace_points)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        

        return rendered_image, rendered_depth, rendered_mask
        
    def convert_camera_parameters_into_viewpoint_cameras(self, cameras, h=None, w=None):
        device = cameras.device
        cameras = cameras.cpu()
        c2w = torch.eye(4)
        c2w[:3, :] = cameras[:12].reshape(3, 4)
        fx, fy, cx, cy, H, W = cameras[12:].chunk(6, -1) # each
        
        if h is not None:
            fx, cx = fx * h / H, cx * h / H
        else:
            h = int(H[0].item())
        if w is not None:
            fy, cy = fy * w / W, cy * w / W
        else:
            w = int(W[0].item())
        
        fovy = 2 * torch.atan(0.5 * w / fy)
        fovx = 2 * torch.atan(0.5 * h / fx)
        cam = MiniCam(c2w.numpy(), w, h, fovy.numpy(), fovx.numpy(), 0.1, 100, device=device)
        return cam

    # @torch.amp.autocast('cuda', enabled=False)
    def forward(
        self,
        cameras,
        gaussians,
        scaling_modifier=1.0,
        bg_color=None,
        h=256,
        w=256,
    ):
        B, N = cameras.shape[:2]        
        xyz, features, opacity, scales, rotations = gaussians

        self.radii = []
        self.viewspace_points = []
            
        images = []
        depths = []
        masks = []

        if True:
            for i in range(B):
                gs = xyz[i], features[i], opacity[i], scales[i], rotations[i]
                for j in range(N):
                    viewpoint_camera = self.convert_camera_parameters_into_viewpoint_cameras(cameras[i, j], h=h, w=w)

                    rendered_image, rendered_depth, rendered_mask = self.render(*gs, viewpoint_camera, scaling_modifier=scaling_modifier, bg_color=bg_color)
                
                    images.append(rendered_image)
                    depths.append(rendered_depth)
                    masks.append(rendered_mask)

            #GS render only support non-negative color in / out -> here we need to convert [0, 1] - > [-1, 1]
            images = torch.stack(images, dim=0).unflatten(0, (B, N)) * 2 - 1 #[1, 8, 3, 256, 256]
            depths =  torch.stack(depths, dim=0).unflatten(0, (B, N)) #[1, 8, 1, 256, 256]
            masks = torch.stack(masks, dim=0).unflatten(0, (B, N)) #[1, 8, 1, 256, 256]
            torch.cuda.synchronize()
            
            return images, depths, masks, None, None
    
        # for i in range(B):
        #     gs = xyz[i], features[i], opacity[i], scales[i], rotations[i]
        #     if True:
        #         cams = [convert_camera_parameters_into_viewpoint_cameras(cameras[i,j], h=h, w=w) for j in range(N)]
        #         viewmats = torch.stack([cam.world_view_transform for cam in cams])
        #         Ks = torch.stack([cam.K for cam in cams])
        #         renders, alphas, meta = rasterization(
        #             means=xyz[i],
        #             quats=rotations[i],
        #             scales=scales[i],
        #             opacities=opacity[i,:,0],
        #             colors=features[i,:,0],
        #             viewmats=viewmats,
        #             Ks=Ks,
        #             width=w,
        #             height=h,
        #             # sh_degree=sh_degree,
        #             render_mode="RGB+D",
        #             # packed=packed,
        #             )
        #         images.append(renders[:,:,:,:3])
        #         depths.append(renders[:,:,:,3:])
        #         masks.append(alphas)
    
        # images = einops.rearrange(torch.stack(images, dim=0), 'B N H W C -> B N C H W') * 2 - 1
        # depths =  einops.rearrange(torch.stack(depths, dim=0), 'B N H W C -> B N C H W')
        # masks =  einops.rearrange(torch.stack(masks, dim=0), 'B N H W C -> B N C H W')
        # torch.cuda.synchronize()
        
        # return images, depths, masks, None, None
    

def convert_camera_parameters_into_viewpoint_cameras(cameras, h=None, w=None):
        device = cameras.device
        cameras = cameras.cpu()
        c2w = torch.eye(4)
        c2w[:3, :] = cameras[:12].reshape(3, 4)
        fx, fy, cx, cy, H, W = cameras[12:].chunk(6, -1) # each
        
        if h is not None:
            fx, cx = fx * h / H, cx * h / H
        else:
            h = int(H[0].item())
        if w is not None:
            fy, cy = fy * w / W, cy * w / W
        else:
            w = int(W[0].item())
        
        fovy = 2 * torch.atan(0.5 * w / fy)
        fovx = 2 * torch.atan(0.5 * h / fx)
        cam = MiniCam(c2w.numpy(), w, h, fovy.numpy(), fovx.numpy(), 0.1, 100, device=device)
        return cam