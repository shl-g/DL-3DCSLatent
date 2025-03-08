import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional

class PointCloudNeRF(nn.Module):
    """
    Point Cloud based NeRF module that replaces the original NeRF backbone.
    Each point represents a convex shape with parameters for position, size, color, opacity, etc.
    """
    def __init__(
        self,
        num_points: int = 1000,
        num_convex_shapes: int = 100,
        feature_dim: int = 32,
        sh_degree: int = 3,
        device: str = 'cuda',
        init_radius: float = 1.0
    ):
        """
        Initialize the Point Cloud NeRF.
        
        Args:
            num_points: Total number of points in the point cloud
            num_convex_shapes: Number of convex shapes (K polyhedrons)
            feature_dim: Dimension of feature vector for each point
            sh_degree: Degree of spherical harmonics for color representation
            device: Device to initialize tensors on
            init_radius: Initial radius for uniform sphere distribution
        """
        super().__init__()
        self.num_points = num_points
        self.num_convex_shapes = num_convex_shapes
        self.feature_dim = feature_dim
        self.sh_degree = sh_degree
        self.device = device
        
        # Points per shape
        self.points_per_shape = num_points // num_convex_shapes
        
        # Initialize 3D coordinates for each point
        # We distribute points uniformly on spheres centered at random locations
        shape_centers = torch.randn(num_convex_shapes, 3, device=device)
        
        # Initialize points around each shape center
        points_list = []
        shape_indices = []
        
        for i in range(num_convex_shapes):
            # Generate uniform points on a sphere
            phi = torch.rand(self.points_per_shape, device=device) * 2 * np.pi
            theta = torch.acos(2 * torch.rand(self.points_per_shape, device=device) - 1)
            
            x = init_radius * torch.sin(theta) * torch.cos(phi)
            y = init_radius * torch.sin(theta) * torch.sin(phi)
            z = init_radius * torch.cos(theta)
            
            # Combine coordinates and offset by shape center
            sphere_points = torch.stack([x, y, z], dim=1) + shape_centers[i]
            points_list.append(sphere_points)
            
            # Keep track of which shape each point belongs to
            shape_indices.append(torch.full((self.points_per_shape,), i, dtype=torch.long, device=device))
            
        # Combine all points
        self.positions = nn.Parameter(torch.cat(points_list, dim=0))
        self.shape_indices = torch.cat(shape_indices, dim=0)
        
        # Softness parameter δ (delta) - controls soft/hard edges
        # Using exponential activation to ensure positive values
        self.log_delta = nn.Parameter(torch.zeros(num_convex_shapes, device=device))
        
        # Density parameter σ (sigma) - controls density of the object
        # Using exponential activation to ensure positive values
        self.log_sigma = nn.Parameter(torch.zeros(num_convex_shapes, device=device))
        
        # Opacity parameter o - using sigmoid activation to constrain [0, 1]
        self.raw_opacity = nn.Parameter(torch.zeros(num_points, device=device))
        
        # Spherical harmonic coefficients for color
        # For degree n, there are (n+1)^2 coefficients
        sh_coeffs_per_point = (sh_degree + 1) ** 2
        self.sh_coeffs = nn.Parameter(torch.zeros(num_points, sh_coeffs_per_point, 3, device=device))
        
        # Initialize values
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with reasonable values"""
        # Initialize raw_opacity to give values around 0.5 after sigmoid
        self.raw_opacity.data.uniform_(-0.1, 0.1)
        
        # Initialize log_delta to start with moderate softness
        self.log_delta.data.fill_(-1.0)  # exp(-1) ≈ 0.368
        
        # Initialize log_sigma to start with moderate density
        self.log_sigma.data.fill_(0.0)  # exp(0) = 1.0
        
        # Initialize SH coefficients - typically only the DC component (first coeff) is non-zero initially
        self.sh_coeffs.data.zero_()
        # Set the first coefficient (DC term) to produce grayish colors
        self.sh_coeffs.data[:, 0, :] = 0.5
    
    def get_opacity(self) -> torch.Tensor:
        """Get opacity values constrained to [0, 1]"""
        return torch.sigmoid(self.raw_opacity)
    
    def get_delta(self) -> torch.Tensor:
        """Get delta values (softness) - always positive"""
        return torch.exp(self.log_delta)
    
    def get_sigma(self) -> torch.Tensor:
        """Get sigma values (density) - always positive"""
        return torch.exp(self.log_sigma)
    
    def evaluate_sh(self, directions: torch.Tensor) -> torch.Tensor:
        """
        Evaluate spherical harmonics at given view directions.
        
        Args:
            directions: Normalized view directions, shape [batch_size, 3]
            
        Returns:
            RGB colors based on SH evaluation, shape [num_points, batch_size, 3]
        """
        batch_size = directions.shape[0]
        
        # Compute spherical coordinates from Cartesian directions
        x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
        
        # Prepare SH evaluation tensors
        sh_bands = []
        
        # Band 0 (1 coefficient)
        sh_bands.append(torch.ones_like(x).unsqueeze(-1) * 0.282095)  # Y_0^0
        
        # Band 1 (3 coefficients)
        if self.sh_degree >= 1:
            sh_bands.append(torch.stack([
                y * 0.488603,   # Y_1^-1
                z * 0.488603,   # Y_1^0
                x * 0.488603    # Y_1^1
            ], dim=-1))
        
        # Band 2 (5 coefficients)
        if self.sh_degree >= 2:
            sh_bands.append(torch.stack([
                x * y * 1.092548,            # Y_2^-2
                y * z * 1.092548,            # Y_2^-1
                (3.0 * z * z - 1.0) * 0.315392,  # Y_2^0
                x * z * 1.092548,            # Y_2^1
                (x * x - y * y) * 0.546274   # Y_2^2
            ], dim=-1))
        
        # Band 3 (7 coefficients)
        if self.sh_degree >= 3:
            sh_bands.append(torch.stack([
                y * (3.0 * x * x - y * y) * 0.590044,    # Y_3^-3
                x * y * z * 2.890611,                   # Y_3^-2
                y * (5.0 * z * z - 1.0) * 0.457046,     # Y_3^-1
                z * (5.0 * z * z - 3.0) * 0.373176,     # Y_3^0
                x * (5.0 * z * z - 1.0) * 0.457046,     # Y_3^1
                z * (x * x - y * y) * 1.445306,         # Y_3^2
                x * (x * x - 3.0 * y * y) * 0.590044    # Y_3^3
            ], dim=-1))
        
        # Concatenate all bands
        sh_eval = torch.cat(sh_bands, dim=-1)  # [batch_size, (sh_degree+1)^2]
        
        # Compute colors using SH coefficients
        # Reshape sh_eval to [1, batch_size, (sh_degree+1)^2]
        sh_eval = sh_eval.unsqueeze(0)
        
        # Reshape sh_coeffs to [num_points, 1, (sh_degree+1)^2, 3]
        sh_coeffs_reshaped = self.sh_coeffs.unsqueeze(1)
        
        # Multiply and sum over SH basis
        # Result has shape [num_points, batch_size, 3]
        rgb = torch.sum(sh_coeffs_reshaped * sh_eval.unsqueeze(-1), dim=2)
        
        # Ensure RGB values are in [0, 1]
        rgb = torch.sigmoid(rgb)
        
        return rgb
        
    def render_points(self, camera_origin: torch.Tensor, camera_directions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render the point cloud from a specific camera viewpoint.
        
        Args:
            camera_origin: Camera position, shape [batch_size, 3]
            camera_directions: Normalized view directions, shape [batch_size, 3]
            
        Returns:
            rendered_image: RGB image, shape [batch_size, 3, H, W]
            depth_map: Depth map, shape [batch_size, 1, H, W]
        """
        batch_size = camera_origin.shape[0]
        
        # Get parameter values with appropriate activations
        opacity = self.get_opacity()  # [num_points]
        delta = self.get_delta()  # [num_convex_shapes]
        sigma = self.get_sigma()  # [num_convex_shapes]
        
        # Map each point to its shape parameters
        point_delta = delta[self.shape_indices]  # [num_points]
        point_sigma = sigma[self.shape_indices]  # [num_points]
        
        # Compute colors based on view direction using spherical harmonics
        colors = self.evaluate_sh(camera_directions)  # [num_points, batch_size, 3]
        
        # Calculate point positions relative to camera
        # Reshape for broadcasting: [num_points, 1, 3] - [1, batch_size, 3]
        relative_positions = self.positions.unsqueeze(1) - camera_origin.unsqueeze(0)
        
        # Compute distance from camera to points
        distances = torch.norm(relative_positions, dim=2, keepdim=True)  # [num_points, batch_size, 1]
        
        # Normalized direction from camera to points
        directions_to_points = relative_positions / (distances + 1e-8)  # [num_points, batch_size, 3]
        
        # Compute dot product between view direction and direction to points
        # camera_directions: [batch_size, 3] -> [1, batch_size, 3]
        # directions_to_points: [num_points, batch_size, 3]
        dot_products = torch.sum(directions_to_points * camera_directions.unsqueeze(0), dim=2)  # [num_points, batch_size]
        
        # Apply softness (delta) to control edge falloff 
        # Higher delta means harder edges, lower delta means softer edges
        point_contribution = torch.exp(-point_delta.unsqueeze(1) * (1.0 - dot_products)**2)  # [num_points, batch_size]
        
        # Apply density (sigma) factor
        point_contribution = point_contribution * point_sigma.unsqueeze(1)  # [num_points, batch_size]
        
        # Apply opacity
        point_contribution = point_contribution * opacity.unsqueeze(1)  # [num_points, batch_size]
        
        # Apply distance falloff (points further away contribute less)
        point_contribution = point_contribution / (distances.squeeze(-1) + 1e-6)  # [num_points, batch_size]
        
        # Normalize contributions
        total_contribution = point_contribution.sum(dim=0, keepdim=True)  # [1, batch_size]
        normalized_contribution = point_contribution / (total_contribution + 1e-8)  # [num_points, batch_size]
        
        # Compute weighted color
        # colors: [num_points, batch_size, 3]
        # normalized_contribution: [num_points, batch_size]
        weighted_colors = colors * normalized_contribution.unsqueeze(-1)  # [num_points, batch_size, 3]
        
        # Sum up contributions for final color
        final_colors = weighted_colors.sum(dim=0)  # [batch_size, 3]
        
        # Compute weighted depth
        weighted_depth = (distances.squeeze(-1) * normalized_contribution).sum(dim=0, keepdim=True)  # [1, batch_size]
        
        return final_colors, weighted_depth.t()  # [batch_size, 3], [batch_size, 1]
    
    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor, **kwargs) -> dict:
        """
        Forward pass compatible with latent-nerf's NeRF interface.
        
        Args:
            rays_o: Ray origins, shape [batch_size, 3]
            rays_d: Ray directions, shape [batch_size, 3]
            
        Returns:
            Dictionary with rendered RGB and depth values
        """
        # Normalize ray directions
        rays_d_norm = F.normalize(rays_d, dim=-1)
        
        # Render points
        rgb, depth = self.render_points(rays_o, rays_d_norm)
        
        # Format output to match latent-nerf's expected structure
        result = {
            'rgb': rgb,  # [batch_size, 3]
            'depth': depth,  # [batch_size, 1]
            'opacity': self.get_opacity().mean().unsqueeze(0).expand(rgb.shape[0]),  # [batch_size]
        }
        
        return result


class PointCloudRenderer(nn.Module):
    """
    Renderer class that integrates with latent-nerf's pipeline.
    """
    def __init__(
        self,
        resolution: Tuple[int, int] = (128, 128),
        num_points: int = 1000,
        num_convex_shapes: int = 100,
        feature_dim: int = 32,
        sh_degree: int = 3,
        device: str = 'cuda'
    ):
        super().__init__()
        self.resolution = resolution
        self.point_cloud_nerf = PointCloudNeRF(
            num_points=num_points,
            num_convex_shapes=num_convex_shapes,
            feature_dim=feature_dim,
            sh_degree=sh_degree,
            device=device
        )
        
    def get_camera_rays(self, camera_to_world: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate camera rays given a camera-to-world transformation matrix.
        
        Args:
            camera_to_world: Camera-to-world transformation matrix, shape [batch_size, 4, 4]
            
        Returns:
            ray_origins: Ray origins in world space, shape [batch_size, H*W, 3]
            ray_directions: Ray directions in world space, shape [batch_size, H*W, 3]
        """
        batch_size = camera_to_world.shape[0]
        H, W = self.resolution
        
        # Generate pixel coordinates
        i, j = torch.meshgrid(
            torch.linspace(0.5, W - 0.5, W, device=camera_to_world.device),
            torch.linspace(0.5, H - 0.5, H, device=camera_to_world.device),
            indexing='ij'
        )
        
        # Convert to normalized device coordinates
        x = (i - W/2) / max(H, W)
        y = -(j - H/2) / max(H, W)
        z = -torch.ones_like(x)
        
        # Stack coordinates
        directions = torch.stack([x, y, z], dim=-1)  # [H, W, 3]
        
        # Expand for batch
        directions = directions.view(1, -1, 3).expand(batch_size, -1, -1)  # [batch_size, H*W, 3]
        
        # Get ray origins and directions
        ray_origins = camera_to_world[:, :3, 3].unsqueeze(1).expand(-1, H*W, -1)  # [batch_size, H*W, 3]
        
        # Transform ray directions from camera to world space
        ray_directions = torch.einsum(
            'bik,bpk->bpi',
            camera_to_world[:, :3, :3],
            directions
        )  # [batch_size, H*W, 3]
        
        return ray_origins, ray_directions
    
    def render(self, camera_to_world: torch.Tensor) -> dict:
        """
        Render the scene from given camera poses.
        
        Args:
            camera_to_world: Camera-to-world transformation matrix, shape [batch_size, 4, 4]
            
        Returns:
            Dictionary with rendered RGB images and depth maps
        """
        batch_size = camera_to_world.shape[0]
        H, W = self.resolution
        
        # Generate rays
        ray_origins, ray_directions = self.get_camera_rays(camera_to_world)
        
        # Process rays in batches if needed
        rays_per_batch = 32768  # Adjust based on GPU memory
        num_rays = ray_origins.shape[1]
        
        all_rgb = []
        all_depth = []
        
        for i in range(0, num_rays, rays_per_batch):
            # Get current batch of rays
            batch_ray_origins = ray_origins[:, i:i+rays_per_batch, :]
            batch_ray_directions = ray_directions[:, i:i+rays_per_batch, :]
            
            # Flatten batch dimension and rays dimension
            flat_ray_origins = batch_ray_origins.reshape(-1, 3)
            flat_ray_directions = batch_ray_directions.reshape(-1, 3)
            
            # Render
            results = self.point_cloud_nerf(flat_ray_origins, flat_ray_directions)
            
            # Unflatten results
            batch_rgb = results['rgb'].reshape(batch_size, -1, 3)
            batch_depth = results['depth'].reshape(batch_size, -1, 1)
            
            all_rgb.append(batch_rgb)
            all_depth.append(batch_depth)
        
        # Concatenate results
        rgb = torch.cat(all_rgb, dim=1).reshape(batch_size, H, W, 3)
        depth = torch.cat(all_depth, dim=1).reshape(batch_size, H, W, 1)
        
        # Convert to format expected by latent-nerf
        result = {
            'rgb': rgb.permute(0, 3, 1, 2),  # [batch_size, 3, H, W]
            'depth': depth.permute(0, 3, 1, 2),  # [batch_size, 1, H, W]
        }
        
        return result
    
    def forward(self, camera_to_world: torch.Tensor) -> dict:
        """Forward pass simply calls render"""
        return self.render(camera_to_world)


# Integration with latent-nerf
class LatentPointCloudNeRF(nn.Module):
    """
    Integration class that combines PointCloudNeRF with latent-nerf's pipeline.
    This class follows a similar structure to latent-nerf's models but uses our point cloud representation.
    """
    def __init__(
        self,
        resolution: Tuple[int, int] = (128, 128),
        num_points: int = 1000,
        num_convex_shapes: int = 100,
        feature_dim: int = 32,
        sh_degree: int = 3,
        latent_dim: int = 512,  # Latent dimension from latent-nerf
        device: str = 'cuda'
    ):
        super().__init__()
        self.resolution = resolution
        self.latent_dim = latent_dim
        
        # Point cloud renderer
        self.renderer = PointCloudRenderer(
            resolution=resolution,
            num_points=num_points,
            num_convex_shapes=num_convex_shapes,
            feature_dim=feature_dim,
            sh_degree=sh_degree,
            device=device
        )
        
        # Optional: Add a mapping network to connect latent codes to point cloud parameters
        # This could be used if you want to condition the point cloud on a latent code
        self.latent_to_params = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, num_convex_shapes * 2)  # For global control of delta and sigma
        )
    
    def apply_latent(self, latent_code: torch.Tensor):
        """
        Apply latent code to modify point cloud parameters.
        This provides a way to control the point cloud using the latent code.
        
        Args:
            latent_code: Latent code from latent-nerf, shape [batch_size, latent_dim]
        """
        batch_size = latent_code.shape[0]
        if batch_size > 1:
            # For simplicity, we just use the first latent code in the batch
            # In a real implementation, you might want to handle batches differently
            latent_code = latent_code[0:1]
            
        # Use the latent code to generate global modifications to point cloud parameters
        params = self.latent_to_params(latent_code)
        
        # Split into delta and sigma modifiers
        num_shapes = self.renderer.point_cloud_nerf.num_convex_shapes
        delta_mod, sigma_mod = params.split(num_shapes, dim=1)
        
        # Apply modifications as offsets to log parameters
        self.renderer.point_cloud_nerf.log_delta.data = self.renderer.point_cloud_nerf.log_delta.data + delta_mod.squeeze(0) * 0.1
        self.renderer.point_cloud_nerf.log_sigma.data = self.renderer.point_cloud_nerf.log_sigma.data + sigma_mod.squeeze(0) * 0.1
    
    def render(self, camera_to_world: torch.Tensor, latent_code: Optional[torch.Tensor] = None) -> dict:
        """
        Render the scene using the point cloud.
        
        Args:
            camera_to_world: Camera-to-world transformation matrix, shape [batch_size, 4, 4]
            latent_code: Optional latent code to modify point cloud parameters
            
        Returns:
            Rendered images and depth maps
        """
        if latent_code is not None:
            self.apply_latent(latent_code)
            
        return self.renderer(camera_to_world)
    
    def forward(self, camera_to_world: torch.Tensor, latent_code: Optional[torch.Tensor] = None) -> dict:
        """Forward pass simply calls render"""
        return self.render(camera_to_world, latent_code)