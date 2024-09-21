"""
Copyright (C) 2023  ETH Zurich, Manuel Kaufmann
"""
import glob
import os
import pickle as pkl

import numpy as np
from scipy.spatial.transform import Rotation as R


def load_hybrik(result_root, force_load):
    """Load HybrIK results."""
    hybrik_dir = result_root
    hybrik_cache_dir = os.path.join(hybrik_dir, "cache")
    hybrik_cache_file = os.path.join(hybrik_cache_dir, "romp-out.npz")

    if not os.path.exists(hybrik_cache_file) or force_load:
        hybrik_betas, hybrik_poses_rot, hybrik_trans = [], [], []
        for pkl_file in sorted(glob.glob(os.path.join(hybrik_dir, "*.pkl"))):
            with open(pkl_file, "rb") as f:
                hybrik_data = pkl.load(f)

            hybrik_poses_rot.append(hybrik_data["pred_theta_mats"].reshape((1, -1, 3, 3)))
            hybrik_betas.append(hybrik_data["pred_shape"])

            # NOTE This is not the SMPL translation, it's a translation added to the outputs of SMPL
            #  but this does not matter because we align to the root, except for the jitter metric.
            hybrik_trans.append(hybrik_data["transl"])

        hybrik_poses_rot = np.concatenate(hybrik_poses_rot, axis=0)
        hybrik_poses = R.as_rotvec(R.from_matrix(hybrik_poses_rot.reshape((-1, 3, 3)))).reshape(
            hybrik_poses_rot.shape[0], -1
        )
        hybrik_betas = np.concatenate(hybrik_betas, axis=0)
        hybrik_trans = np.concatenate(hybrik_trans, axis=0)

        os.makedirs(hybrik_cache_dir, exist_ok=True)
        np.savez_compressed(
            hybrik_cache_file,
            hybrik_poses=hybrik_poses,
            hybrik_betas=hybrik_betas,
            hybrik_trans=hybrik_trans,
        )
    else:
        hybrik_results = np.load(hybrik_cache_file)
        hybrik_poses = hybrik_results["hybrik_poses"]
        hybrik_betas = hybrik_results["hybrik_betas"]
        hybrik_trans = hybrik_results["hybrik_trans"]

    return hybrik_poses, hybrik_betas, hybrik_trans

import os
import glob
import numpy as np
import pickle as pkl

def load_scoreHMR(result_root, force_load=False):
    """Load scoreHMR results."""
    hybrik_cache_dir = os.path.join(result_root, "cache")
    hybrik_cache_file = os.path.join(hybrik_cache_dir, "scoreHMR_out.npz")
    print('result_root', result_root)
    if not os.path.exists(hybrik_cache_file) or force_load:
        pose_hat, shape_hat, trans_hat = [], [], []
        
        for pkl_file in sorted(glob.glob(os.path.join(result_root, "*.pkl"))):
            with open(pkl_file, "rb") as f:
                frame_data = pkl.load(f)
            
            pose_hat.append(np.concatenate([frame_data['poses_root'], frame_data['poses_body']], axis=1))
            shape_hat.append(frame_data['betas'])
            trans_hat.append(frame_data['trans'])
        
        pose_hat = np.concatenate(pose_hat, axis=0)  # Shape: (N, 72)
        shape_hat = np.concatenate(shape_hat, axis=0)  # Shape: (N, 10)
        trans_hat = np.concatenate(trans_hat, axis=0)  # Shape: (N, 3)
        
        os.makedirs(hybrik_cache_dir, exist_ok=True)
        np.savez_compressed(
            hybrik_cache_file,
            pose_hat=pose_hat,
            shape_hat=shape_hat,
            trans_hat=trans_hat,
        )
    else:
        hybrik_results = np.load(hybrik_cache_file)
        pose_hat = hybrik_results["pose_hat"]
        shape_hat = hybrik_results["shape_hat"]
        trans_hat = hybrik_results["trans_hat"]
    
    return pose_hat, shape_hat, trans_hat


def load_niki(result_root, force_load=False):
    """Load scoreHMR results."""
    hybrik_cache_dir = os.path.join(result_root, "cache")
    hybrik_cache_file = os.path.join(hybrik_cache_dir, "niki_out.npz")
    print('result_root', result_root)
    if not os.path.exists(hybrik_cache_file) or force_load:
        pose_hat, shape_hat, trans_hat = [], [], []
        
        for pkl_file in sorted(glob.glob(os.path.join(result_root, "*.pkl"))):
            with open(pkl_file, "rb") as f:
                frame_data = pkl.load(f)
            
            pose_hat.append(np.concatenate([frame_data['poses_root'], frame_data['poses_body']], axis=1))
            shape_hat.append(frame_data['betas'])
            trans_hat.append(frame_data['trans'])
        
        pose_hat = np.concatenate(pose_hat, axis=0)  # Shape: (N, 72)
        shape_hat = np.concatenate(shape_hat, axis=0)  # Shape: (N, 10)
        trans_hat = np.concatenate(trans_hat, axis=0)  # Shape: (N, 3)
        
        os.makedirs(hybrik_cache_dir, exist_ok=True)
        np.savez_compressed(
            hybrik_cache_file,
            pose_hat=pose_hat,
            shape_hat=shape_hat,
            trans_hat=trans_hat,
        )
    else:
        hybrik_results = np.load(hybrik_cache_file)
        pose_hat = hybrik_results["pose_hat"]
        shape_hat = hybrik_results["shape_hat"]
        trans_hat = hybrik_results["trans_hat"]
    
    return pose_hat, shape_hat, trans_hat

def load_tram(result_root, force_load=False):
    """Load scoreHMR results."""
    hybrik_cache_dir = os.path.join(result_root, "cache")
    hybrik_cache_file = os.path.join(hybrik_cache_dir, "tram_out.npz")
    print('result_root', result_root)
    if not os.path.exists(hybrik_cache_file) or force_load:
        pose_hat, shape_hat, trans_hat = [], [], []
        
        for pkl_file in sorted(glob.glob(os.path.join(result_root, "*.pkl"))):
            with open(pkl_file, "rb") as f:
                frame_data = pkl.load(f)
            
            pose_hat.append(np.concatenate([frame_data['poses_root'], frame_data['poses_body']], axis=1))
            shape_hat.append(frame_data['betas'])
            trans_hat.append(frame_data['trans'])
        
        pose_hat = np.concatenate(pose_hat, axis=0)  # Shape: (N, 72)
        shape_hat = np.concatenate(shape_hat, axis=0)  # Shape: (N, 10)
        trans_hat = np.concatenate(trans_hat, axis=0)  # Shape: (N, 3)
        
        os.makedirs(hybrik_cache_dir, exist_ok=True)
        np.savez_compressed(
            hybrik_cache_file,
            pose_hat=pose_hat,
            shape_hat=shape_hat,
            trans_hat=trans_hat,
        )
    else:
        hybrik_results = np.load(hybrik_cache_file)
        pose_hat = hybrik_results["pose_hat"]
        shape_hat = hybrik_results["shape_hat"]
        trans_hat = hybrik_results["trans_hat"]
    
    return pose_hat, shape_hat, trans_hat

def load_pliks(result_root, force_load=False):
    """Load scoreHMR results."""
    hybrik_cache_dir = os.path.join(result_root, "cache")
    hybrik_cache_file = os.path.join(hybrik_cache_dir, "pliks_out.npz")
    print('result_root', result_root)
    if not os.path.exists(hybrik_cache_file) or force_load:
        pose_hat, shape_hat, trans_hat = [], [], []
        
        for pkl_file in sorted(glob.glob(os.path.join(result_root, "*.pkl"))):
            with open(pkl_file, "rb") as f:
                frame_data = pkl.load(f)
            
            pose_hat.append(np.concatenate([frame_data['poses_root'], frame_data['poses_body']], axis=1))
            shape_hat.append(frame_data['betas'])
            trans_hat.append(frame_data['trans'])
        
        pose_hat = np.concatenate(pose_hat, axis=0)  # Shape: (N, 72)
        shape_hat = np.concatenate(shape_hat, axis=0)  # Shape: (N, 10)
        trans_hat = np.concatenate(trans_hat, axis=0)  # Shape: (N, 3)
        
        os.makedirs(hybrik_cache_dir, exist_ok=True)
        np.savez_compressed(
            hybrik_cache_file,
            pose_hat=pose_hat,
            shape_hat=shape_hat,
            trans_hat=trans_hat,
        )
    else:
        hybrik_results = np.load(hybrik_cache_file)
        pose_hat = hybrik_results["pose_hat"]
        shape_hat = hybrik_results["shape_hat"]
        trans_hat = hybrik_results["trans_hat"]
    
    return pose_hat, shape_hat, trans_hat

def load_nlf(result_root, force_load=False):
    """Load scoreHMR results."""
    hybrik_cache_dir = os.path.join(result_root, "cache")
    hybrik_cache_file = os.path.join(hybrik_cache_dir, "nlf_out.npz")
    print('result_root', result_root)
    if not os.path.exists(hybrik_cache_file) or force_load:
        pose_hat, shape_hat, trans_hat = [], [], []
        
        for pkl_file in sorted(glob.glob(os.path.join(result_root, "*.pkl"))):
            with open(pkl_file, "rb") as f:
                frame_data = pkl.load(f)
            
            pose_hat.append(np.concatenate([frame_data['poses_root'], frame_data['poses_body']], axis=1))
            shape_hat.append(frame_data['betas'])
            trans_hat.append(frame_data['trans'])
        
        pose_hat = np.concatenate(pose_hat, axis=0)  # Shape: (N, 72)
        shape_hat = np.concatenate(shape_hat, axis=0)  # Shape: (N, 10)
        trans_hat = np.concatenate(trans_hat, axis=0)  # Shape: (N, 3)
        
        os.makedirs(hybrik_cache_dir, exist_ok=True)
        np.savez_compressed(
            hybrik_cache_file,
            pose_hat=pose_hat,
            shape_hat=shape_hat,
            trans_hat=trans_hat,
        )
    else:
        hybrik_results = np.load(hybrik_cache_file)
        pose_hat = hybrik_results["pose_hat"]
        shape_hat = hybrik_results["shape_hat"]
        trans_hat = hybrik_results["trans_hat"]
    
    return pose_hat, shape_hat, trans_hat

def load_nlfs(result_root, force_load=False):
    """Load scoreHMR results."""
    #result_root = os.path.join(result_root, "smoothnet_windowsize32_smoothed")
    #print('result_root:::', result_root)
    hybrik_cache_dir = os.path.join(result_root, "cache")
    hybrik_cache_file = os.path.join(hybrik_cache_dir, "nlfs_out.npz")
    print('result_root', result_root)
    if not os.path.exists(hybrik_cache_file) or force_load:
        pose_hat, shape_hat, trans_hat = [], [], []
        
        for pkl_file in sorted(glob.glob(os.path.join(result_root, "*.pkl"))):
            with open(pkl_file, "rb") as f:
                frame_data = pkl.load(f)
            
            pose_hat.append(np.concatenate([frame_data['poses_root'], frame_data['poses_body']], axis=1))
            shape_hat.append(frame_data['betas'])
            trans_hat.append(frame_data['trans'])
        
        pose_hat = np.concatenate(pose_hat, axis=0)  # Shape: (N, 72)
        shape_hat = np.concatenate(shape_hat, axis=0)  # Shape: (N, 10)
        trans_hat = np.concatenate(trans_hat, axis=0)  # Shape: (N, 3)
        
        os.makedirs(hybrik_cache_dir, exist_ok=True)
        np.savez_compressed(
            hybrik_cache_file,
            pose_hat=pose_hat,
            shape_hat=shape_hat,
            trans_hat=trans_hat,
        )
    else:
        hybrik_results = np.load(hybrik_cache_file)
        pose_hat = hybrik_results["pose_hat"]
        shape_hat = hybrik_results["shape_hat"]
        trans_hat = hybrik_results["trans_hat"]
    
    return pose_hat, shape_hat, trans_hat


def load_PartialHuman(result_root, force_load=False):
    """Load scoreHMR results."""
    hybrik_cache_dir = os.path.join(result_root, "cache")
    hybrik_cache_file = os.path.join(hybrik_cache_dir, "partialh_out.npz")
    print('result_root', result_root)
    if not os.path.exists(hybrik_cache_file) or force_load:
        pose_hat, shape_hat, trans_hat = [], [], []
        
        for pkl_file in sorted(glob.glob(os.path.join(result_root, "*.pkl"))):
            with open(pkl_file, "rb") as f:
                frame_data = pkl.load(f)
            
            pose_hat.append(np.concatenate([frame_data['poses_root'], frame_data['poses_body']], axis=1))
            shape_hat.append(frame_data['betas'])
            trans_hat.append(frame_data['trans'])
        
        pose_hat = np.concatenate(pose_hat, axis=0)  # Shape: (N, 72)
        shape_hat = np.concatenate(shape_hat, axis=0)  # Shape: (N, 10)
        trans_hat = np.concatenate(trans_hat, axis=0)  # Shape: (N, 3)
        
        os.makedirs(hybrik_cache_dir, exist_ok=True)
        np.savez_compressed(
            hybrik_cache_file,
            pose_hat=pose_hat,
            shape_hat=shape_hat,
            trans_hat=trans_hat,
        )
    else:
        hybrik_results = np.load(hybrik_cache_file)
        pose_hat = hybrik_results["pose_hat"]
        shape_hat = hybrik_results["shape_hat"]
        trans_hat = hybrik_results["trans_hat"]
    
    return pose_hat, shape_hat, trans_hat
