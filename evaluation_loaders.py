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
    hybrik_cache_file = os.path.join(hybrik_cache_dir, "romp-out2.npz")

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
    hybrik_cache_file = os.path.join(hybrik_cache_dir, "scoreHMR_out2.npz")
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
    hybrik_cache_file = os.path.join(hybrik_cache_dir, "niki_out2.npz")
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
    hybrik_cache_file = os.path.join(hybrik_cache_dir, "tram_out2.npz")
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
    hybrik_cache_file = os.path.join(hybrik_cache_dir, "pliks_out2.npz")
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
    hybrik_cache_file = os.path.join(hybrik_cache_dir, "nlf_out2.npz")
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
    hybrik_cache_file = os.path.join(hybrik_cache_dir, "partialh_out2.npz")
    print('result_root', result_root)
    if not os.path.exists(hybrik_cache_file) or force_load:
        pose_hat, shape_hat, trans_hat = [], [], []
        
        for pkl_file in sorted(glob.glob(os.path.join(result_root, "*.pkl"))):
            #print('pkl_file=',pkl_file)
            with open(pkl_file, 'rb') as f:
                frame_data = pkl.load(f, encoding='latin1', fix_imports=True)
            # Extract theta parameters
            theta = frame_data['theta'][0]  # Assuming theta is stored for a single frame
        
            # Extract camera, pose, and shape parameters from theta
            num_cam = 3  # Number of camera parameters
            num_pose = 72  # 24 joints * 3 (rotation angles)
            num_shape = 10  # Number of shape parameters
        
            cam_params = theta[:num_cam]
            pose_params = theta[num_cam:num_cam+num_pose]
            shape_params = theta[num_cam+num_pose:]
        
            # Prepare parameters for SMPL
            poses = np.array(pose_params.reshape(1, -1))
            betas = np.array(shape_params.reshape(1, -1))
            global_orient = poses[:, :3]
            body_pose = poses[:, 3:]

            pose_hat.append(np.concatenate([global_orient, body_pose], axis=1))
            shape_hat.append(betas)
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



import os
import glob
import numpy as np
import pickle as pkl

def load_smplerx_vertices(result_root, force_load=False):
    """Load SMPL-X vertices from PKL files."""
    vertices_cache_dir = os.path.join(result_root, "cache")
    vertices_cache_file = os.path.join(vertices_cache_dir, "vertices_out.npz")
    print('result_root', result_root)

    if not os.path.exists(vertices_cache_file) or force_load:
        vertices_list = []
        
        for pkl_file in sorted(glob.glob(os.path.join(result_root, "*.pkl"))):
            with open(pkl_file, "rb") as f:
                frame_data = pkl.load(f)
            
            # Assuming the vertices are stored under the key 'vertices' in the PKL file
            # Adjust this key if your PKL files use a different key for vertices
            vertices = frame_data['vertices']
            vertices_list.append(vertices)
        
        # Stack all vertices into a single numpy array
        vertices_array = np.stack(vertices_list, axis=0)  # Shape: (N_frames, N_vertices, 3)
        
        os.makedirs(vertices_cache_dir, exist_ok=True)
        np.savez_compressed(
            vertices_cache_file,
            vertices=vertices_array
        )
    else:
        vertices_data = np.load(vertices_cache_file)
        vertices_array = vertices_data["vertices"]
        
    vertices_array = vertices_array.squeeze()
    
    return vertices_array, None, None
