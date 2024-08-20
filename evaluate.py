"""
Copyright (C) 2023  ETH Zurich, Manuel Kaufmann
"""
import argparse
import glob
import os
import pickle as pkl

from configuration import EMDB_ROOT
from evaluation_engine import HYBRIK, EvaluationEngine
from evaluation_metrics import compute_metrics
from input_specification import validate_input_format
from gt_analysis import analyze_ground_truth
from pose_selection import select_poses
from video_generation import generate_videos

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", help="Where the baseline results are stored.")
    parser.add_argument("--num_poses", type=int, default=10, help="Number of poses to select for evaluation")
    args = parser.parse_args()

    def is_emdb1(emdb_pkl_file):
        with open(emdb_pkl_file, "rb") as f:
            data = pkl.load(f)
            return data["emdb1"]

    # Search for all the test sequences on which we evaluated the baselines in the paper.
    all_emdb_pkl_files = glob.glob(os.path.join(EMDB_ROOT, "*/*/*_data.pkl"))
    emdb1_sequence_roots = []
    for emdb_pkl_file in all_emdb_pkl_files:
        if is_emdb1(emdb_pkl_file):
            emdb1_sequence_roots.append(os.path.dirname(emdb_pkl_file))

    # Validate input format
    validate_input_format(emdb1_sequence_roots)

    # Analyze ground truth data
    gt_analysis_results = analyze_ground_truth(emdb1_sequence_roots)

    # Select poses for evaluation
    selected_poses = select_poses(gt_analysis_results, args.num_poses)

    # Generate videos for selected poses
    generate_videos(selected_poses, emdb1_sequence_roots)

    # Select the baselines we want to evaluate.
    baselines_to_evaluate = [HYBRIK]

    # Run the evaluation.
    evaluator_public = EvaluationEngine(compute_metrics)
    evaluator_public.run(emdb1_sequence_roots, args.result_root, baselines_to_evaluate, selected_poses=selected_poses)
