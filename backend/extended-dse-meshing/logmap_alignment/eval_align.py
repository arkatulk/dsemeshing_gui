import os
import sys
import trimesh
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
from sklearn.neighbors import KDTree
import delaunay_torch
BATCH_SIZE = 128
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def reconstruct(name):
    logmap_points = np.loadtxt(os.path.join(in_path, name))
    name = name.replace('.xyz', "")
    X3D = logmap_points
    tree = KDTree(logmap_points)
    X3D_normals = np.zeros([X3D.shape[0],3])
    X3D_normals[:,2] = 1

    n_points = len(logmap_points)
    points_indices =list(range(n_points))

    predicted_neighborhood_indices = np.load(os.path.join(raw_prediction_path,"predicted_neighborhood_indices_{}.npy".format(name)))#np.load("icp_consistency_results/predicted_neighborhood_indices.npy" )
    corrected_predicted_map = np.load(os.path.join(res_path, "corrected_maps_{}.npy".format(name)))

    triangles = []
    indices = []
    for step in range(1 + len(X3D)//BATCH_SIZE):
        if step%50==0:
            print("step: {}/{}".format(step,n_points//BATCH_SIZE))
        if (step+1)*BATCH_SIZE>n_points:
            current_points = points_indices[-BATCH_SIZE:]
        else:
            current_points = points_indices[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
        center_points = np.array(X3D[current_points])
        points_neighbors =tree.query(center_points, n_neighbors+1)[1]
        triangles_batch,indices_batch= delaunay_torch.get_triangles_geo_batches(n_neighbors=n_nearest_neighbors,
                            gdist = torch.tensor(corrected_predicted_map[current_points],device=device),
                            gdist_neighbors =torch.tensor(predicted_neighborhood_indices[current_points][:,1:],device=device),
                            first_index =torch.tensor(current_points,device=device))
        if (step+1)*BATCH_SIZE>n_points:
            triangles_batch = triangles_batch[-n_points%BATCH_SIZE:]
            indices_batch =indices_batch[-n_points%BATCH_SIZE:]

        triangles.append(triangles_batch.cpu().numpy())
        indices.append(indices_batch.cpu().numpy())
    indices = np.concatenate(indices)
    triangles = np.concatenate(triangles)

    trigs = np.sort(np.reshape(indices[triangles>0.5],[-1,3]), axis = 1)
    uni,inverse, count = np.unique(trigs, return_counts=True, axis=0, return_inverse=True)
    triangle_occurence = count[inverse]
    np.save(os.path.join(res_path, "patch_frequency_count_{}.npy".format(name)), np.concatenate([uni, count[:,np.newaxis]], axis = 1) )


if __name__ == '__main__':
    # evaluate the new meshes and count frequency of triangles
    in_path = os.path.join(ROOT_DIR, 'data/test_data')
    raw_prediction_path = os.path.join(ROOT_DIR, 'data/test_data/raw_prediction')
    res_path = os.path.join(ROOT_DIR, 'data/test_data/aligned_prediction')
    n_neighbors = 120
    n_nearest_neighbors = 30

    # we evaluate all .xyz files in the in_path directory
    files = os.listdir(in_path)
    files = [x for x in files if x.endswith('.xyz')]

    for name in files:
        reconstruct(name)
