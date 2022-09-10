import os
import sys
import zipfile
from os.path import exists

import gdown
import numpy as np
import open3d as o3d
import torch

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import matplotlib.image as image

from model import load_model
from scripts.benchmark_util import run_ransac
from util.misc import extract_features
from util.pointcloud import (
    make_open3d_feature_from_numpy,
    make_open3d_point_cloud,
)
from util.uio import process_image

# from demo import visualization_ours
parent_path = os.path.dirname(__file__)
pretrain_path = os.path.join(parent_path, "pretrain.zip")

# Downloading model

if not exists(pretrain_path):
    url = "https://drive.google.com/uc?export=download&id=1QsuvIt6qTlld-0klaADImkFEzCcuArbJ"
    gdown.download(url, pretrain_path)

if exists(os.path.join(parent_path, "pretrain.zip")) and not exists(
    os.path.join(parent_path, "pretrain")
):
    with zipfile.ZipFile(pretrain_path, "r") as zip_ref:
        zip_ref.extractall(parent_path)


def get_model(
    checkpoint_path=os.path.join(parent_path, "./pretrain/3DMatch/3DMatch.pth"),
):

    # load the model

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
    device = torch.device(
        "cuda"
    )  # The model has been built for CUDA device only.
    model = model.to(device)

    p_path = os.path.join(parent_path, "files/cloud_bin_0.ply")
    pc1 = o3d.io.read_point_cloud(p_path)
    p_xyz = np.asarray(pc1.points)
    p_image_path = os.path.join(parent_path, "files/cloud_bin_0_0.png")
    p_image = image.imread(p_image_path)
    if p_image.shape[0] != config.image_H or p_image.shape[1] != config.image_W:
        p_image = process_image(
            image=p_image, aim_H=config.image_H, aim_W=config.image_W
        )
    p_image = np.transpose(p_image, axes=(2, 0, 1))
    p_image = np.expand_dims(p_image, axis=0)
    # warm up model and check for failure
    p_xyz_down, p_feature = extract_features(
        model,
        xyz=p_xyz,
        rgb=None,
        normal=None,
        voxel_size=0.025,
        device=device,
        skip_check=True,
        image=p_image,
    )
    return model, config
