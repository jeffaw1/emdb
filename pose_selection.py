"""
Pose selection for efficient model evaluation.
"""
import numpy as np

def select_poses(gt_analysis_results, num_poses):
    """
    Select a varied and representative set of poses for evaluation.
    
    Args:
    gt_analysis_results (dict): Results from ground truth analysis
    num_poses (int): Number of poses to select
    
    Returns:
    list: Selected poses
    """
    selected_poses = []
    
    # This is a placeholder selection method
    # Replace with a more sophisticated selection algorithm
    for root, analysis in gt_analysis_results.items():
        mean_pose = analysis['mean_pose']
        std_pose = analysis['std_pose']
        
        # Select poses that are furthest from the mean
        distances = np.sum((mean_pose - analysis['poses'])**2, axis=1)
        selected_indices = np.argsort(distances)[-num_poses:]
        
        selected_poses.extend([(root, idx) for idx in selected_indices])
    
    return selected_poses[:num_poses]

# Add more sophisticated pose selection methods as needed
