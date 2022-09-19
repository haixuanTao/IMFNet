import os
import sys
import time

import matplotlib.image as image
import numpy as np
import open3d as o3d
import torch

from model import load_model
from scripts.benchmark_util import run_ransac
from util.misc import extract_features
from util.pointcloud import (
    make_open3d_feature_from_numpy,
    make_open3d_point_cloud,
)
from util.uio import process_image

voxel_size = 0.25
# read P and Q
p_path = "files/cloud_bin_0.ply"
q_path = "files/cloud_bin_1.ply"

pc1 = o3d.io.read_point_cloud(p_path)
pc2 = o3d.io.read_point_cloud(q_path)
patches_per_pair = 5000

# randomly sampling some points from the point cloud
inds1 = np.random.choice(
    np.asarray(pc1.points).shape[0], patches_per_pair, replace=False
)
inds2 = np.random.choice(
    np.asarray(pc2.points).shape[0], patches_per_pair, replace=False
)

p_xyz = np.asarray(pc1.points)[inds1]
q_xyz = np.asarray(pc2.points)[inds2]

# load the model
checkpoint_path = "./pretrain/Kitti/Kitti.pth"
checkpoint = torch.load(checkpoint_path)
config = checkpoint["config"]

num_feats = 1
Model = load_model(config.model)
model = Model(
    num_feats,
    config.model_n_out,
    bn_momentum=0.05,
    normalize_feature=config.normalize_feature,
    conv1_kernel_size=config.conv1_kernel_size,
    D=3,
    config=config,
)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
device = torch.device("cuda")
model = model.to(device)

# read p_image and q_image
p_image_path = "files/cloud_bin_0_0.png"
q_image_path = "files/cloud_bin_1_0.png"

p_image = image.imread(p_image_path)
if p_image.shape[0] != config.image_H or p_image.shape[1] != config.image_W:
    p_image = process_image(
        image=p_image, aim_H=config.image_H, aim_W=config.image_W
    )
p_image = np.transpose(p_image, axes=(2, 0, 1))
p_image = np.expand_dims(p_image, axis=0)

q_image = image.imread(q_image_path)
if q_image.shape[0] != config.image_H or q_image.shape[1] != config.image_W:
    q_image = process_image(
        image=q_image, aim_H=config.image_H, aim_W=config.image_W
    )
q_image = np.transpose(q_image, axes=(2, 0, 1))
q_image = np.expand_dims(q_image, axis=0)

# generate f_p and f_q
p_xyz_down, p_feature = extract_features(
    model,
    xyz=p_xyz,
    rgb=None,
    normal=None,
    voxel_size=voxel_size,
    device=device,
    skip_check=True,
    image=p_image,
)

now = time.time()
p_xyz_down, p_feature = extract_features(
    model,
    xyz=p_xyz,
    rgb=None,
    normal=None,
    voxel_size=voxel_size,
    device=device,
    skip_check=True,
    image=p_image,
)

q_xyz_down, q_feature = extract_features(
    model,
    xyz=q_xyz,
    rgb=None,
    normal=None,
    voxel_size=voxel_size,
    device=device,
    skip_check=True,
    image=q_image,
)

# get the evaluation metrix
p_xyz_down = make_open3d_point_cloud(p_xyz_down)
q_xyz_down = make_open3d_point_cloud(q_xyz_down)

p_feature = p_feature.cpu().detach().numpy()
p_feature = make_open3d_feature_from_numpy(p_feature)
q_feature = q_feature.cpu().detach().numpy()
q_feature = make_open3d_feature_from_numpy(q_feature)
T = run_ransac(p_xyz_down, q_xyz_down, p_feature, q_feature, voxel_size)
theta0 = np.arctan2(T[2, 1], T[2, 2])
theta1 = np.arctan2(-T[2, 0], np.sqrt(T[2, 1] ** 2 + T[2, 2] ** 2))
theta2 = np.arctan2(T[1, 0], T[0, 0])
[x, y, z, _] = np.dot(T, [1.0, 1.0, 1.0, 1.0])
print(f"time ransac: {time.time() - now}")
o3d.visualization.draw_geometries([pc1, pc2])
pc1.transform(T)
o3d.visualization.draw_geometries([pc1, pc2])
