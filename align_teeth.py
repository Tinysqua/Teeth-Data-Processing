#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.decomposition import PCA
import trimesh
import os
import json
from scipy.spatial import cKDTree
import random
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from trimesh.transformations import random_rotation_matrix
import logging

logging.basicConfig(filename='alignment.log', level=logging.INFO, format='[%(levelname)s] %(message)s')

def sort_key(filename):
    return int(filename.split('_')[1])

def load_data(crown_folder, jaw_folder):
    data_list = []
    crown_files = {f.split('_')[0]: f for f in
                   os.listdir(crown_folder) if f.endswith('.obj') or f.endswith('.ply') or f.endswith('.off')}
    jaw_files = os.listdir(jaw_folder)
    jaw_files.sort(key=sort_key)
    jaw_data = {}

    for f in jaw_files:
        data_id = f.split('_')[1]
        tooth_id = int(f.split('_')[2].split('.')[0])

        if 'upper' in f:
            jaw_type = 'upper'
        elif 'lower' in f:
            jaw_type = 'lower'

        jaw_data.setdefault(data_id, {'upper': None, 'lower': None})[jaw_type] = os.path.join(jaw_folder, f)

    for data_id, jaws in jaw_data.items():
        if data_id in crown_files:
            tooth_id = int(jaws['upper'].split('_')[-2].split('.')[0])

            data_dict = {
                'crown': os.path.join(crown_folder, crown_files[data_id]),
                'upper': jaws['upper'],
                'lower': jaws['lower'],
                'tooth_id': tooth_id
            }
            data_list.append(data_dict)

    return data_list

def get_contact_points_kdtree(source_mesh, target_mesh, threshold=0.2):
    tree = cKDTree(target_mesh.vertices)
    distances, _ = tree.query(source_mesh.vertices, distance_upper_bound=threshold)
    contact_idx = np.where(np.isfinite(distances))[0]
    contact_points = source_mesh.vertices[contact_idx]

    return contact_points


def estimate_curvature(points, k=20):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
    distances, indices = nbrs.kneighbors(points)
    curvatures = np.zeros(points.shape[0])

    for i in range(points.shape[0]):
        neighbors = points[indices[i][1:]]
        cov = np.cov(neighbors.T)
        eigenvalues, _ = np.linalg.eig(cov)
        eigenvalues = np.sort(eigenvalues)
        curvature = eigenvalues[0] / (np.sum(eigenvalues) + 1e-6)
        curvatures[i] = curvature

    return curvatures

def extract_cusp_candidates(points, curvature_threshold=0.1, k=20):
    curvatures = estimate_curvature(points, k=k)
    candidate_indices = np.where(curvatures >= curvature_threshold)[0]
    candidates = points[candidate_indices]

    return candidates, curvatures

def fit_plane_ransac(points, threshold=0.02, iterations=1000):
    best_inlier_count = 0
    best_inlier_indices = None
    best_plane = None
    n_points = points.shape[0]

    for i in range(iterations):
        sample_indices = random.sample(range(n_points), 3)
        sample = points[sample_indices]

        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        normal = np.cross(v1, v2)
        norm_normal = np.linalg.norm(normal)
        if norm_normal == 0:
            continue
        normal = normal / norm_normal

        d = -np.dot(normal, sample[0])

        distances = np.abs(np.dot(points, normal) + d)
        inlier_indices = np.where(distances < threshold)[0]
        inlier_count = len(inlier_indices)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inlier_indices = inlier_indices
            best_plane = (normal, d)

    if best_inlier_indices is not None and len(best_inlier_indices) >= 3:
        inlier_points = points[best_inlier_indices]
        centroid = np.mean(inlier_points, axis=0)
        _, _, Vt = np.linalg.svd(inlier_points - centroid)
        refined_normal = Vt[-1, :]
        refined_normal /= np.linalg.norm(refined_normal)
        refined_d = -np.dot(refined_normal, centroid)
        return centroid, refined_normal, best_inlier_indices, (refined_normal, refined_d)
    else:
        return None, None, None, None

def align_meshes_to_xy(meshes, plane_normal):
    n = plane_normal / np.linalg.norm(plane_normal)
    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(n, target)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-6:
        R = np.eye(3)
    else:
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(n, target), -1.0, 1.0))
        K = np.array([[0,         -axis[2],  axis[1]],
                      [axis[2],    0,       -axis[0]],
                      [-axis[1],  axis[0],   0      ]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    R_homogeneous = np.eye(4)
    R_homogeneous[:3, :3] = R

    transformed_meshes = []
    for mesh in meshes:
        mesh_copy = mesh.copy()
        mesh_copy.apply_transform(R_homogeneous)
        transformed_meshes.append(mesh_copy)

    return transformed_meshes, R_homogeneous

def flip_upper_if_needed(meshes, flip_axis='x'):
    upper, lower, crown, lines = meshes[0], meshes[1], meshes[2], meshes[3]

    upper_center = np.mean(upper.vertices, axis=0)
    lower_center = np.mean(lower.vertices, axis=0)

    T_flip = None

    if upper_center[2] < lower_center[2]:
        if flip_axis.lower() == 'x':
            R_flip = np.array([[1, 0, 0],
                               [0, -1, 0],
                               [0, 0, -1]])
        elif flip_axis.lower() == 'y':
            R_flip = np.array([[-1, 0, 0],
                               [0, 1, 0],
                               [0, 0, -1]])
        else:
            raise ValueError("flip_axis can be 'x' or 'y'")

        T_flip = np.eye(4)
        T_flip[:3, :3] = R_flip

        upper_flipped = upper.copy()
        lower_flipped = lower.copy()
        crown_flipped = crown.copy()
        lines_flipped = lines.copy()
        upper_flipped.apply_transform(T_flip)
        lower_flipped.apply_transform(T_flip)
        crown_flipped.apply_transform(T_flip)
        lines_flipped.apply_transform(T_flip)

        updated_meshes = [upper_flipped, lower_flipped, crown_flipped, lines_flipped]
    else:
        updated_meshes = meshes
        T_flip = None

    return updated_meshes, T_flip

def compute_cluster_centers(points, eps=0.5, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_
    unique_labels = set(labels)
    centers = []
    for label in unique_labels:
        if label == -1:
            continue
        cluster_points = points[labels == label]
        center = np.mean(cluster_points, axis=0)
        centers.append(center)
    centers = np.array(centers)

    return centers, labels

def align_mesh(src_path, dst_path,
               contact_threshold=0.1, curvature_threshold=0.01, k=20,
               eps=0.02, min_samples=5, ransac_threshold=0.05,
               ransac_iterations=1000, info_path=None):

    crown_path = os.path.join(src_path, 'tooth_crown')
    jaw_path = os.path.join(src_path, 'jaw')

    data_list = load_data(crown_path, jaw_path)

    aligned_crown_path = os.path.join(dst_path, 'tooth_crown')
    aligned_jaw_path = os.path.join(dst_path, 'jaw')

    os.makedirs(aligned_crown_path, exist_ok=True)
    os.makedirs(aligned_jaw_path, exist_ok=True)
    os.makedirs(os.path.join(dst_path, 'normal_info'), exist_ok=True)

    scale_info_path = '/data1/shizhen/TeethZip/scale_info.json'
    with open(scale_info_path, 'r') as f:
        scale_info = json.load(f)

    for data in data_list:
        logging.info(f"Processing sample: {data}")
        upper_mesh = trimesh.load(data['upper'])
        lower_mesh = trimesh.load(data['lower'])
        crown = trimesh.load(data['crown'])

        contact_points_lower = get_contact_points_kdtree(lower_mesh, upper_mesh, contact_threshold)
        if contact_points_lower.shape[0] < 50:
            logging.error("Not enough contact points available; skipping alignment for this sample.")
            continue

        candidates_lower, _ = extract_cusp_candidates(contact_points_lower, curvature_threshold=curvature_threshold, k=k)
        if candidates_lower.shape[0] < 3:
            logging.warning("Too few cusp candidates found; falling back to using all contact points.")
            candidates_lower = contact_points_lower
            if candidates_lower.shape[0] < 3:
                logging.error("Not enough contact points available; skipping alignment for this sample.")
                continue

        cluster_centers, cluster_labels = compute_cluster_centers(candidates_lower, eps=eps, min_samples=min_samples)
        if cluster_centers.shape[0] < 3:
            logging.warning("Too few cluster centers obtained; falling back to using candidates_lower for RANSAC.")
            cluster_centers = candidates_lower
            if cluster_centers.shape[0] < 3:
                logging.error("Not enough points available for RANSAC after fallback; skipping alignment for this sample.")
                continue

        centroid, normal, inlier_indices, plane_params = fit_plane_ransac(cluster_centers, threshold=ransac_threshold, iterations=ransac_iterations)
        if centroid is None:
            logging.error("RANSAC failed; skipping alignment for this sample.")
            continue
        
        mesh_name = os.path.basename(data['crown']).split('.')[0]
        translation = np.array(scale_info[mesh_name]['translation'])
        scales = np.array(scale_info[mesh_name]['scales'])
        boundary_line_path = os.path.join(info_path, f"{mesh_name.split('_')[0]}_boundary_line_{mesh_name.split('_')[1]}.json")
        with open(boundary_line_path, 'r') as f:
            boundary_data = json.load(f)
        if isinstance(boundary_data, list):
            vertices = [np.array((v["x"], v["y"], v["z"])) for v in boundary_data]
            vertices = np.stack(vertices, axis=0)
        else:
            print(f"Unexpected data format in {boundary_line_path}")
            return
        vertices += translation
        vertices *= scales
        lines = trimesh.points.PointCloud(vertices)

        meshes_aligned, T = align_meshes_to_xy([upper_mesh, lower_mesh, crown, lines], normal)

        updated_meshes, T_flip = flip_upper_if_needed(meshes_aligned, flip_axis='x')

        upper_jaw_basename = os.path.basename(data['upper'])
        lower_jaw_basename = os.path.basename(data['lower'])
        crown_basename = os.path.basename(data['crown'])
        lines_basename = os.path.basename(boundary_line_path)[:-4] + "ply"

        upper_jaw_filename = os.path.join(aligned_jaw_path, upper_jaw_basename)
        lower_jaw_filename = os.path.join(aligned_jaw_path, lower_jaw_basename)
        aligned_crown_filename = os.path.join(aligned_crown_path, crown_basename)
        lines_filename = os.path.join(os.path.dirname(aligned_crown_path), 'normal_info', lines_basename)

        # print(f"Length of updated_meshes: {len(updated_meshes)}")
        updated_meshes[0].export(upper_jaw_filename)
        updated_meshes[1].export(lower_jaw_filename)
        updated_meshes[2].export(aligned_crown_filename)
        updated_meshes[3].export(lines_filename)

        logging.info("Alignment done for this sample.")

if __name__ == "__main__":
    src_path = '/data1/shizhen/TeethZip/Aidite_Crown_Dataset_sixth_scale'
    dst_path = '/data_new2/shizhen/JustTest/Aidite_Crown_Dataset_sixth_align'
    
    info_path = '/data1/shizhen/TeethZip/Aidite_Crown_Dataset_sixth/normal_info'
    
    align_before_rescale = False
    if align_before_rescale:
        contact_threshold = 1.0
    else:
        contact_threshold = 0.1

    align_mesh(src_path, dst_path, contact_threshold=contact_threshold, info_path=info_path)
