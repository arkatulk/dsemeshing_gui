import trimesh
from collections import Counter

# Load the PLY mesh (replace with your PLY file path)
mesh = trimesh.load_mesh('final_mesh_sphere_482.ply')  # Update with the actual path to your PLY file

# Initialize an empty list to store all edges, including duplicates
all_edges = mesh.edges

# Convert edges to a tuple and count occurrences using Counter
edge_counts = Counter(map(tuple, all_edges))

# Find edges that appear exactly twice (watertight edges)
edges_with_count_2 = {edge: count for edge, count in edge_counts.items() if count == 2}


# Total number of edges (including duplicates)
total_edges = len(all_edges)

# Calculate the ratio of watertight edges to total edges
watertight_edges = len(edges_with_count_2)
watertight_ratio = watertight_edges / len(mesh.edges_unique)

# Print the results
print(f"Total number of unique edges : {len(mesh.edges_unique)}")
print(f"Total number of unique edges : {len(edge_counts)}")
print(f"Number of watertight edges (count == 2): {watertight_edges}")
print(f"Ratio of watertight edges to total edges: {((1-watertight_ratio)*100):.4f}")

import numpy as np

mesh_ground_truth = trimesh.load_mesh('final_mesh_sphere_482.ply')  # Update with the actual path to your PLY file


def normal_error(mesh1, mesh2):
    normals1 = np.asarray(mesh1.vertex_normals)
    normals2 = np.asarray(mesh2.vertex_normals)

    dot_product = np.sum(normals1 * normals2, axis=1)
    angles = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Clipping for numerical stability

    return np.degrees(np.mean(angles))

nr = normal_error(mesh, mesh_ground_truth)
print(nr)