import cv2
import numpy as np
import torch
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.utils import local_to_global

SMPL_OR_JOINTS = np.array([0, 1, 2, 4, 5, 16, 17, 18, 19])

def compute_mean_vertex_errors(vertex_errors, vertex_visibility):
    """
    Compute the mean error for each vertex, considering only visible vertices.
    
    Parameters:
    - vertex_errors: numpy array of shape (n_frames, n_vertices) containing the error for each vertex in each frame
    - vertex_visibility: list of lists, where each inner list contains indices of visible vertices for that frame
    
    Returns:
    - mean_errors: numpy array of shape (n_vertices,) containing the mean error for each vertex
    - visibility_count: numpy array of shape (n_vertices,) containing the count of visibility for each vertex
    """
    n_frames, n_vertices = vertex_errors.shape
    error_sum = np.zeros(n_vertices)
    visibility_count = np.zeros(n_vertices, dtype=int)
    
    for i in range(n_frames):
        visible_indices = vertex_visibility[i]
        error_sum[visible_indices] += vertex_errors[i, visible_indices]
        visibility_count[visible_indices] += 1
    
    # Use np.divide with 'where' parameter to avoid division by zero
    mean_errors = np.divide(error_sum, visibility_count, out=np.zeros_like(error_sum), where=visibility_count!=0)
    
    return mean_errors, visibility_count
    
def get_data(
    pose_gt,
    shape_gt,
    trans_gt,
    pose_hat,
    shape_hat,
    trans_hat,
    gender_gt,
    gender_hat=None,
):
    """
    Return SMPL joint positions, vertices, and global joint orientations for both ground truth and predictions.
    """
    smpl_layer = SMPLLayer(model_type="smpl", gender=gender_gt)

    smpl_seq = SMPLSequence(
        pose_gt[:, 3:],
        smpl_layer=smpl_layer,
        poses_root=pose_gt[:, :3],
        betas=shape_gt,
        trans=trans_gt,
    )

    verts_gt, jp_gt = smpl_seq.vertices, smpl_seq.joints

    global_oris = local_to_global(
        torch.cat([smpl_seq.poses_root, smpl_seq.poses_body], dim=-1),
        smpl_seq.skeleton[:, 0],
        output_format="rotmat",
    )

    n_frames = pose_gt.shape[0]
    glb_rot_mats_gt = global_oris.reshape((n_frames, -1, 3, 3)).detach().cpu().numpy()

    if gender_hat is None:
        gender_hat = gender_gt

    if gender_hat != gender_gt:
        smpl_layer_hat = SMPLLayer(model_type="smpl", gender=gender_hat)
    else:
        smpl_layer_hat = smpl_layer

    smpl_seq_hat = SMPLSequence(
        pose_hat[:, 3:],
        smpl_layer=smpl_layer_hat,
        poses_root=pose_hat[:, :3],
        betas=shape_hat,
        trans=trans_hat,
    )
    verts_pred, jp_pred = smpl_seq_hat.vertices, smpl_seq_hat.joints
    global_oris_hat = local_to_global(
        torch.cat([smpl_seq_hat.poses_root, smpl_seq_hat.poses_body], dim=-1),
        smpl_seq_hat.skeleton[:, 0],
        output_format="rotmat",
    )

    glb_rot_mats_pred = global_oris_hat.reshape((n_frames, -1, 3, 3)).detach().cpu().numpy()
    glb_rot_mats_pred = glb_rot_mats_pred[:, SMPL_OR_JOINTS]

    print('HEREEEEE jp_pred', jp_pred.shape)
    print('HEREEEEE jp_gt', jp_gt.shape)
    return jp_pred, jp_gt, glb_rot_mats_pred, glb_rot_mats_gt, verts_pred, verts_gt

def align_by_pelvis(joints, verts=None):
    """Align the SMPL joints and vertices by the pelvis."""
    left_id = 1
    right_id = 2

    pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.0
    if verts is not None:
        return verts - np.expand_dims(pelvis, axis=0)
    else:
        return joints - np.expand_dims(pelvis, axis=0)

def joint_angle_error(pred_mat, gt_mat):
    """
    Compute the geodesic distance between the two input matrices.
    """
    n_frames = pred_mat.shape[0]
    gt_mat = gt_mat[:, SMPL_OR_JOINTS, :, :]

    r1 = np.reshape(pred_mat, [-1, 3, 3])
    r2 = np.reshape(gt_mat, [-1, 3, 3])

    r2t = np.transpose(r2, [0, 2, 1])

    r = np.matmul(r1, r2t)

    angles = []
    for i in range(r1.shape[0]):
        aa, _ = cv2.Rodrigues(r[i])
        angles.append(np.linalg.norm(aa))

    angles_all = np.degrees(np.array(angles).reshape((n_frames, -1)))
    return np.mean(angles_all), angles_all

def compute_jitter(preds3d, gt3ds, visible_joints, ignored_joints_idxs=None, fps=30):
    """
    Calculate the jitter as defined in PIP paper, considering only visible joints.
    """
    if ignored_joints_idxs is None:
        ignored_joints_idxs = [0, 7, 8, 10, 11, 20, 21, 22, 23]

    n_frames = len(preds3d)
    jkp = []
    jkt = []

    for i in range(3, n_frames):
        visible_idx = list(set(visible_joints[i]) & set(visible_joints[i-1]) & set(visible_joints[i-2]) & set(visible_joints[i-3]))
        visible_idx = [idx for idx in visible_idx if idx not in ignored_joints_idxs]
        if len(visible_idx) == 0:
            visible_idx = [15]

        if len(visible_idx) > 0:

            jkp_frame = np.linalg.norm(
                (preds3d[i, visible_idx] - 3 * preds3d[i-1, visible_idx] + 3 * preds3d[i-2, visible_idx] - preds3d[i-3, visible_idx]) * (fps**3),
                axis=1
            )
            jkt_frame = np.linalg.norm(
                (gt3ds[i, visible_idx] - 3 * gt3ds[i-1, visible_idx] + 3 * gt3ds[i-2, visible_idx] - gt3ds[i-3, visible_idx]) * (fps**3),
                axis=1
            )
            jkp.append(jkp_frame)
            jkt.append(jkt_frame)
    if not jkp or not jkt:  # If jkp or jkt is empty
          return np.nan, np.nan, np.nan, np.nan
    jkp = np.concatenate(jkp)
    jkt = np.concatenate(jkt)

    return jkp.mean() / 10, jkp.std() / 10, jkt.mean() / 10, jkt.std() / 10

def apply_camera_transforms(joints, rotations, world2camera):
    """
    Applies camera transformations to joint locations and rotations matrices.
    """
    joints_h = np.concatenate([joints, np.ones(joints.shape[:-1] + (1,))], axis=-1)[..., None]
    joints_c = np.matmul(world2camera[:, None], joints_h)[..., :3, 0]

    rotations_c = np.matmul(world2camera[:, None, :3, :3], rotations)

    return joints_c, rotations_c

def compute_similarity_transform(S1, S2, num_joints, verts=None):
    """
    Computes a similarity transform (sR, t) that takes a set of 3D points S1 (3 x N) closest to a set of 3D points S2.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        if verts is not None:
            verts = verts.T
        transposed = True
    assert S2.shape[1] == S1.shape[1]

    S1_p = S1[:, :num_joints]
    S2_p = S2[:, :num_joints]
    mu1 = S1_p.mean(axis=1, keepdims=True)
    mu2 = S2_p.mean(axis=1, keepdims=True)
    X1 = S1_p - mu1
    X2 = S2_p - mu2

    var1 = np.sum(X1**2)

    K = X1.dot(X2.T)

    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    R = V.dot(Z.dot(U.T))

    scale = np.trace(R.dot(K)) / var1

    t = mu2 - scale * (R.dot(mu1))

    S1_hat = scale * R.dot(S1) + t

    verts_hat = None
    if verts is not None:
        verts_hat = scale * R.dot(verts) + t
        if transposed:
            verts_hat = verts_hat.T

    if transposed:
        S1_hat = S1_hat.T

    procrustes_params = {"scale": scale, "R": R, "trans": t}

    if verts_hat is not None:
        return S1_hat, verts_hat, procrustes_params
    else:
        return S1_hat, procrustes_params

def compute_positional_errors(pred_joints, gt_joints, pred_verts, gt_verts, visible_vertices, visible_joints, do_pelvis_alignment=True):
    """
    Computes the MPJPE and PVE errors between the predicted and ground truth joints and vertices,
    considering only visible joints and vertices for respective metrics.
    """
    num_joints = gt_joints[0].shape[0]
    errors_jps, errors_pa_jps = [], []
    errors_verts, errors_pa_verts = [], []
    proc_rot = []
    
    n_frames = len(gt_joints)
    n_vertices = gt_verts[0].shape[0]
    vertex_errors = np.zeros((n_frames, n_vertices))

    for i, (gt3d_jps, pd3d_jps) in enumerate(zip(gt_joints, pred_joints)):
        gt3d_jps = gt3d_jps.reshape(-1, 3)
        pd3d_jps = pd3d_jps.reshape(-1, 3)
        gt3d_verts = gt_verts[i].reshape(-1, 3)
        pd3d_verts = pred_verts[i].reshape(-1, 3)

        if do_pelvis_alignment:
            gt3d_verts = align_by_pelvis(gt3d_jps, gt3d_verts)
            pd3d_verts = align_by_pelvis(pd3d_jps, pd3d_verts)
            gt3d_jps = align_by_pelvis(gt3d_jps)
            pd3d_jps = align_by_pelvis(pd3d_jps)

        visible_joints_idx = visible_joints[i]
        if len(visible_joints_idx) == 0:
            visible_joints_idx = [15]
        joint_error = np.sqrt(np.sum((gt3d_jps[visible_joints_idx] - pd3d_jps[visible_joints_idx]) ** 2, axis=1))
        errors_jps.append(np.mean(joint_error))

        visible_verts_idx = visible_vertices[i]
        verts_error = np.sqrt(np.sum((gt3d_verts[visible_verts_idx] - pd3d_verts[visible_verts_idx]) ** 2, axis=1))
        errors_verts.append(np.mean(verts_error))
        
        # Store vertex errors for all vertices (will be masked later)
        vertex_errors[i] = np.sqrt(np.sum((gt3d_verts - pd3d_verts) ** 2, axis=1))

        pd3d_jps_sym, pd3d_verts_sym, procrustesParam = compute_similarity_transform(
            pd3d_jps, gt3d_jps, num_joints, pd3d_verts
        )
        proc_rot.append(procrustesParam["R"])

        pa_jps_error = np.sqrt(np.sum((gt3d_jps[visible_joints_idx] - pd3d_jps_sym[visible_joints_idx]) ** 2, axis=1))
        pa_verts_error = np.sqrt(np.sum((gt3d_verts[visible_verts_idx] - pd3d_verts_sym[visible_verts_idx]) ** 2, axis=1))

        errors_pa_jps.append(np.mean(pa_jps_error))
        errors_pa_verts.append(np.mean(pa_verts_error))

    # Compute mean vertex errors and visibility count
    mean_vertex_errors, vertex_visibility_count = compute_mean_vertex_errors(vertex_errors, visible_vertices)

    result_dict = {
        "mpjpe": np.mean(errors_jps),
        "mpjpe_pa": np.mean(errors_pa_jps),
        "mve": np.mean(errors_verts),
        "mve_pa": np.mean(errors_pa_verts),
        "mat_procs": np.stack(proc_rot, 0),
        "mpjpe_pf": np.stack(errors_jps, 0),
        "mpjpe_pf_pa": np.stack(errors_pa_jps, 0),
        "mve_pf": np.stack(errors_verts, 0),
        "mve_pf_pa": np.stack(errors_pa_verts, 0),
        "mean_vertex_errors": mean_vertex_errors,
        "vertex_visibility_count": vertex_visibility_count
    }

    return result_dict

def compute_metrics(
    pose_gt,
    shape_gt,
    trans_gt,
    pose_hat,
    shape_hat,
    trans_hat,
    gender_gt,
    gender_hat,
    visible_vertices,
    visible_joints,
    camera_pose_gt=None,
):
    """
    Computes all the metrics we want to report, considering visibility of joints and vertices.
    """
    pred_joints, gt_joints, pred_mats, gt_mats, pred_verts, gt_verts = get_data(
        pose_gt, shape_gt, trans_gt, pose_hat, shape_hat, trans_hat, gender_gt, gender_hat,
    )

    if camera_pose_gt is not None:
        gt_joints, gt_mats = apply_camera_transforms(gt_joints, gt_mats, camera_pose_gt)
        gt_verts, _ = apply_camera_transforms(gt_verts, gt_mats, camera_pose_gt)

    pos_errors = compute_positional_errors(
        pred_joints * 1000.0, gt_joints * 1000.0, pred_verts * 1000.0, gt_verts * 1000.0,
        visible_vertices, visible_joints
    )

    mats_procs_exp = np.expand_dims(pos_errors["mat_procs"], 1)
    mats_procs_exp = np.tile(mats_procs_exp, (1, len(SMPL_OR_JOINTS), 1, 1))
    mats_pred_prc = np.matmul(mats_procs_exp, pred_mats)

    mpjae_pa_final, all_angles_pa = joint_angle_error(mats_pred_prc, gt_mats)

    mpjae_final, all_angles = joint_angle_error(pred_mats, gt_mats)

    jkp_mean, jkp_std, jkt_mean, jkt_std = compute_jitter(pred_joints, gt_joints, visible_joints)

    # Compute vertex errors
    # Compute vertex errors
    ##vertex_errors = np.sqrt(np.sum((gt_verts - pred_verts) ** 2, axis=-1))  # (N_frames, N_vertices)
    
    # Create a mask for visible vertices
    #n_frames, n_vertices = vertex_errors.shape
    #vertex_visibility = np.zeros((n_frames, n_vertices), dtype=bool)
    #for i, visible in enumerate(visible_vertices):
    #    vertex_visibility[i, visible] = True

    #mean_vertex_errors, vertex_visibility_mask = compute_mean_vertex_errors(vertex_errors, vertex_visibility)

    # These are all scalars. Choose nice names for pretty printing later.
    metrics = {
        "MPJPE [mm]": pos_errors["mpjpe"],
        "MPJPE_PA [mm]": pos_errors["mpjpe_pa"],
        "MPJAE [deg]": mpjae_final,
        "MPJAE_PA [deg]": mpjae_pa_final,
        "MVE [mm]": pos_errors["mve"],
        "MVE_PA [mm]": pos_errors["mve_pa"],
        "Jitter [km/s^3]": jkp_mean,
    }
    #mean_errors, visibility_mask = compute_mean_vertex_errors(metrics_extra['vertex_errors'], metrics_extra['vertex_visibility'])
    print('HERERE pos_errors["mean_vertex_errors"]', pos_errors["mean_vertex_errors"].shape)
    print('pos_errors["vertex_visibility_count"]', pos_errors["vertex_visibility_count"].shape)
    metrics_extra = {
        "mpjpe_all": pos_errors["mpjpe_pf"],  # (N,)
        "mpjpe_pa_all": pos_errors["mpjpe_pf_pa"],  # (N,)
        "mpjae_all": all_angles,  # (N, 9)
        "mpjae_pa_all": all_angles_pa,  # (N, 9)
        "mve_all": pos_errors["mve_pf"],  # (N,)
        "mve_pa_all": pos_errors["mve_pf_pa"],  # (N,)
        "jitter_pd": jkp_mean,  # Scalar
        "jitter_pd_std": jkp_std,  # Scalar
        "jitter_gt_mean": jkt_mean,  # Scalar
        "jitter_gt_std": jkt_std,  # Scalar
        "visible_joints": visible_joints,
        "visible_vertices": visible_vertices,
        "mean_vertex_errors": pos_errors["mean_vertex_errors"],
        "vertex_visibility_count": pos_errors["vertex_visibility_count"]
    }

    return metrics, metrics_extra

