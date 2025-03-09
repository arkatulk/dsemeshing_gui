import torch
import torch.nn.functional as F
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPS = 1e-7

def get_couples_matrix_sparse(shape):
    couples = []
    for i in range(1, shape):
        for j in range(i):
            couples.append([i, j])
    couples = np.array(couples)
    return couples


def safe_norm(x, epsilon=EPS, dim=None):
    return torch.sqrt(torch.clamp(torch.sum(x**2, dim=dim), min=epsilon))
import torch

def get_middle_points(center_point, B):
    # Add a new dimension and repeat center_point along the second dimension
    center_point = center_point.unsqueeze(1).repeat(1, B.size(1), 1)  # Shape: [batch_size, B.size(1), D]
    
    # Compute the element-wise average
    return (center_point + B) / 2.0


def comp_half_planes(nn_coord, center_point):
    n_points = nn_coord.shape[0]
    n_neighbors = nn_coord.shape[1]
    middle_points = get_middle_points(center_point, nn_coord)
    dir_vec = nn_coord - center_point.unsqueeze(1)
    half_planes_normal = dir_vec / torch.clamp(safe_norm(dir_vec, dim=-1, epsilon=EPS).unsqueeze(-1), min=EPS)
    col3 = -(middle_points[:, :, 0] * half_planes_normal[:, :, 0] + middle_points[:, :, 1] * half_planes_normal[:, :, 1])
    half_planes = torch.cat([half_planes_normal, col3.unsqueeze(-1)], dim=-1)
    return half_planes

def get_is_trig_exact(inter_dist, n_neighbors):
    n_points = inter_dist.shape[0]
    inter_dist = -torch.sign(inter_dist)
    is_triangle = torch.sum(inter_dist, dim=2)
    is_triangle = torch.where(is_triangle < n_neighbors, torch.zeros_like(is_triangle), torch.ones_like(is_triangle))
    return is_triangle

import torch

def compute_intersections(half_planes, couples, eps=1e-6):
    """
    Computes the intersections between the couples of half-planes.
    Args:
        half_planes (torch.Tensor): Tensor of half-planes, shape [B, N, 3].
        couples (torch.Tensor): Tensor of pairs of indices, shape [B, M, 2].
        eps (float): Small value to check for near-zero intersections.
    Returns:
        torch.Tensor: Tensor of intersection points, shape [B, M, 3].
    """

    # Compute cross products between the specified half-planes
    inter = torch.cross(
        torch.gather(half_planes, 1, couples[:, 0].unsqueeze(-1).expand(half_planes.size(0), -1, half_planes.size(-1))), 
        torch.gather(half_planes, 1, couples[:, 1].unsqueeze(-1).expand(half_planes.size(0), -1, half_planes.size(-1))),
        dim=2
    )  # Shape: [B, M, 3]
    
    # Mask for near-zero values in the z-coordinate
    mask = inter[:, :, 2].abs() < eps  # Shape: [B, M]

    # Normalize intersection points
    inter = inter / inter[:, :, 2].unsqueeze(-1).where(~mask.unsqueeze(-1), torch.ones_like(inter[:, :, 2]).unsqueeze(-1))

    # Handle cases with no intersection by setting points far away
    inter = torch.where(mask.unsqueeze(-1), torch.ones_like(inter) * 10.0, inter)  # Shape: [B, M, 3]

    return inter


import torch

def compute_triangles_local_geodesic_distances(nn_coord, center_point, couples):
    n_neighbors = nn_coord.shape[1]  # Assuming nn_coord.shape[1] is the number of neighbors
    n_trigs = couples.shape[0]
    nn_coord = nn_coord[:, :, :2]  # Only keep the first 2 dimensions
    center_point = center_point[:, :2]  # Only keep the first 2 dimensions
    
    # Compute half-planes (replace this with actual implementation if needed)
    half_planes = comp_half_planes(nn_coord, center_point)
    
    # Compute intersections
    intersections = compute_intersections(half_planes, couples)
    # Expand couples for broadcasting
    intersection_couples = couples.unsqueeze(0).repeat(center_point.shape[0],1,1)

    # Compute the distances between intersections and half-planes

    inter_dist0 = torch.sum(torch.mul(
        half_planes.unsqueeze(1).repeat(1, n_trigs, 1, 1),  # Tile half_planes to match dimensions
        intersections.unsqueeze(2).repeat(1, 1, n_neighbors, 1)  # Tile intersections to match dimensions
    ),dim=-1)
    # Index couples for gathering
    index_couples_a = torch.arange(center_point.size(0), device=device).unsqueeze(1).unsqueeze(2).repeat(1, n_trigs, 2)
    index_couples_b = torch.arange(n_trigs, device=device).unsqueeze(0).unsqueeze(2).repeat(center_point.size(0), 1, 2)

    index_couples = torch.stack([index_couples_a, index_couples_b, intersection_couples], dim=-1)

    # Scatter to ignore certain indices
    # Assuming inter_dist0 is already initialized with zeros

    index_couples_flat = torch.reshape(index_couples,[-1,3])  # Flatten index_couples for indexing

    '''there seems to be almost no equivalent to tf.scatter_nd function 
    torch.scatter,scatter_add,index_add all work based on the updates indices not on the input indices like tf.scatter_nd
    to_ignore = torch.index_add_(
        torch.zeros_like(inter_dist0),  # Zero-initialized tensor of the same shape
        dim=-1,  # Dimension to scatter along
        index=index_couples_flat,
        src=updates
    )'''
    #this is wt i have come up with for now it seems to have a similar effect
    to_ignore= torch.zeros_like(inter_dist0)
    to_ignore[index_couples_flat[:, 0], index_couples_flat[:, 1], index_couples_flat[:, 2]]+=1

    inter_dist0 = torch.where(to_ignore > 0.5, -1e10, inter_dist0)
    inter_dist = torch.where(torch.abs(inter_dist0) < EPS, -1e10, inter_dist0)
    # Check if triangles are exact
    is_triangle_exact = get_is_trig_exact(inter_dist, n_neighbors)
    return is_triangle_exact, intersection_couples

def get_triangles_geo_batches(n_neighbors=60, gdist=None, gdist_neighbors=None, first_index=None):
    couples = torch.tensor(get_couples_matrix_sparse(n_neighbors), dtype=torch.int64,device=device)
    nn_coord = gdist[:, 1:]
    center_point = gdist[:, 0]
    exact_triangles, local_indices = compute_triangles_local_geodesic_distances(nn_coord, center_point, couples)

    global_indices = torch.gather(gdist_neighbors.unsqueeze(-1).repeat(1,1,2), dim=1, index=local_indices)  # Shape: [batch_size, num_selected_indices]

    # Step 2: Expand first_index to match dimensions for concatenation
    first_index_expanded = first_index.unsqueeze(-1).unsqueeze(-1).repeat(1, global_indices.shape[1], 1)  # Shape: [batch_size, num_selected_indices, 1]

    # Step 3: Concatenate first_index and global_indices
    global_indices = torch.cat([first_index.unsqueeze(-1).unsqueeze(-1).repeat(1, global_indices.shape[1], 1), global_indices], dim=2)  # Shape: [batch_size, num_selected_indices, 2]

    return exact_triangles, global_indices

