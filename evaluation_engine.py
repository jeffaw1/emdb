"""
Copyright (C) 2023  ETH Zurich, Manuel Kaufmann
"""

import collections
import functools
import os
import pickle as pkl
import json
import numpy as np
from tabulate import tabulate

from evaluation_loaders import load_hybrik, load_scoreHMR, load_niki, load_tram, load_nlf, load_nlfs, load_pliks

from configuration import SHOT_TYPE


HYBRIK = "HybrIK"
SCOREHMR = "scoreHMR"
NIKI = "Niki"
TRAM = "Tram"
NLF = 'NLF'
NLFs = 'NLFs'
PLIKS = 'pliks'

METHOD_TO_RESULT_FOLDER = {
    HYBRIK: "hybrIK-out",
    SCOREHMR: "ScoreHMR2",
    NIKI: "Niki",
    TRAM: "Tram",
    NLF: 'NLF',
    PLIKS: 'pliks',
    NLFs: 'NLF/smoothnet_windowsize32_smoothed',
}

METHOD_TO_LOAD_FUNCTION = {
    HYBRIK: load_hybrik,
    SCOREHMR: load_scoreHMR,
    NIKI: load_niki,
    TRAM: load_tram,
    NLF: load_nlf,
    NLFs: load_nlfs,
    PLIKS: load_pliks,
}


class EvaluationEngine(object):
    def __init__(self, metrics_compute_func, force_load=False, ignore_smpl_trans=True):
        # Function to be used to compute the metrics.
        self.compute_metrics = metrics_compute_func
        # If true, it will invalidate all caches and reload the baseline results.
        self.force_load = force_load
        # If set, the SMPL translation of the predictions will be set to 0. This only affects the jitter metric because
        # we always align either by the pelvis or via Procrustes for the other metrics.
        self.ignore_smpl_trans = ignore_smpl_trans

        # New attributes for accumulating vertex errors
        self.vertex_errors = {method: [] for method in METHOD_TO_RESULT_FOLDER.keys()}
        self.vertex_counts = {method: [] for method in METHOD_TO_RESULT_FOLDER.keys()}
        self.total_frames = 0

    def get_ids_from_sequence_root(self, sequence_root):
        res = sequence_root.split(os.path.sep)
        subject_id = res[-2]
        seq_id = res[-1]
        return subject_id, seq_id
    
    @functools.lru_cache()
    def _get_emdb_data(self, sequence_root):
        subject_id, seq_id = self.get_ids_from_sequence_root(sequence_root)

        data_file = os.path.join(sequence_root, f"{subject_id}_{seq_id}_{SHOT_TYPE}_data.pkl")
        with open(os.path.join(sequence_root, data_file), "rb") as f:
            data = pkl.load(f)
        return data

    def load_emdb_gt(self, sequence_root):
        """
        Load EMDB SMPL pose parameters.
        :param sequence_root: Where the .pkl file is stored.
        :return:
          poses_gt: a np array of shape (N, 72)
          betas_gt: a np array of shape (N, 10)
          trans_gt: a np array of shape (N, 3)
        """
        data = self._get_emdb_data(sequence_root)

        poses_body = data["smpl"]["poses_body"]
        poses_root = data["smpl"]["poses_root"]
        betas = data["smpl"]["betas"]
        trans = data["smpl"]["trans"]
        vis_vertices = data["visible_vertices"]
        vis_joints = data["visible_joints"]

        poses_gt = np.concatenate([poses_root, poses_body], axis=-1)
        betas_gt = np.repeat(betas.reshape((1, -1)), repeats=data["n_frames"], axis=0)
        trans_gt = trans

        return poses_gt, betas_gt, trans_gt, vis_vertices, vis_joints

    def load_good_frames_mask(self, sequence_root):
        """Return the mask that says which frames are good and whic are not (because the human is too occluded)."""
        data = self._get_emdb_data(sequence_root)
        return data["good_frames_mask"]

    def get_gender_for_baseline(self, method):
        """Which gender to use for the baseline method."""
        if method in [HYBRIK, SCOREHMR, NIKI, TRAM, NLF, PLIKS, NLFs]:
            return "neutral"
        else:
            # This will select whatever gender the ground-truth specifies.
            return None

    def compare2method(self, poses_gt, betas_gt, trans_gt, sequence_root, result_root, method, vis_vertices, vis_joints):
        """Load this method's results and compute the metrics on them."""

        # Load the baseline results
        subject_id, seq_id = self.get_ids_from_sequence_root(sequence_root)
        method_result_dir = os.path.join(result_root, subject_id, seq_id, METHOD_TO_RESULT_FOLDER[method])
        poses_cmp, betas_cmp, trans_cmp = METHOD_TO_LOAD_FUNCTION[method](method_result_dir, self.force_load)

        if self.ignore_smpl_trans:
            trans_cmp = np.zeros_like(trans_cmp)

        # Load camera parameters.
        data = self._get_emdb_data(sequence_root)
        world2cam = data["camera"]["extrinsics"]

        gender_gt = data["gender"]
        gender_hat = self.get_gender_for_baseline(method)

        metrics, metrics_extra = self.compute_metrics(
            poses_gt,
            betas_gt,
            trans_gt,
            poses_cmp,
            betas_cmp,
            trans_cmp,
            gender_gt,
            gender_hat,
            vis_vertices,
            vis_joints,
            world2cam,
        )

        # Accumulate vertex errors and visibility counts
        vertex_errors = metrics_extra.get('vertex_errors', np.array([]))
        vertex_visibility = metrics_extra.get('vertex_visibility', np.array([]))

        if vertex_errors.size > 0 and vertex_visibility.size > 0:
            self.vertex_errors[method].append(vertex_errors)
            self.vertex_counts[method].append(vertex_visibility)

        return metrics, metrics_extra, method
        
    def save_detailed_results(self, result_root, all_results, sequence_results):
        """Save detailed results for later visualization."""
        output_file = os.path.join(result_root, "detailed_results.npz")
        
        # Convert numpy arrays to lists for JSON serialization
        json_compatible_results = {}
        for method, data in all_results.items():
            json_compatible_results[method] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in data.items()
            }
        
        # Save the results
        np.savez_compressed(output_file,
                            all_results=json.dumps(json_compatible_results),
                            sequence_results=json.dumps(sequence_results))
        
        print(f"Detailed results saved to {output_file}")

    def evaluate_single_sequence(self, sequence_root, result_root, methods):
        """Evaluate a single sequence for all methods."""
        ms, ms_extra, ms_names = [], [], []

        poses_gt, betas_gt, trans_gt, vis_vertices, vis_joints = self.load_emdb_gt(sequence_root)

        for method in methods:
            m, m_extra, ms_name = self.compare2method(poses_gt, betas_gt, trans_gt, sequence_root, result_root, method, vis_vertices, vis_joints)

            ms.append(m)
            ms_extra.append(m_extra)
            ms_names.append(ms_name)

        return ms, ms_extra, ms_names

    def to_pretty_string(self, metrics, model_names):
        """Print the metrics onto the console, but pretty."""
        if not isinstance(metrics, list):
            metrics = [metrics]
            model_names = [model_names]
        assert len(metrics) == len(model_names)
        headers, rows = [], []
        for i in range(len(metrics)):
            values = []
            for k in metrics[i]:
                if i == 0:
                    headers.append(k)
                values.append(metrics[i][k])
            rows.append([model_names[i]] + values)
        return tabulate(rows, headers=["Model"] + headers)

    def run(self, sequence_roots, result_root, methods):
        """Run the evaluation on all sequences and all methods."""
        if not isinstance(sequence_roots, list):
            sequence_roots = [sequence_roots]
    
        all_results = {method: {} for method in methods}
        sequence_results = {}
        ms_all = {method: collections.defaultdict(list) for method in methods}
    
        for sequence_root in sequence_roots:
            ms, ms_extra, ms_names = self.evaluate_single_sequence(sequence_root, result_root, methods)
    
            print(f"Metrics for sequence {sequence_root}")
            print(self.to_pretty_string(ms, ms_names))
    
            for method, m, me in zip(ms_names, ms, ms_extra):
                sequence_results.setdefault(method, {})[sequence_root] = {
                    **m,
                    **{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in me.items()}
                }
    
                # Accumulate data for overall results
                for metric in ["mpjpe_all", "mpjpe_pa_all", "mpjae_all", "mpjae_pa_all", "mve_all", "mve_pa_all"]:
                    if metric in me:
                        ms_all[method][metric].append(me[metric])
                
                if "jitter_pd" in me and not np.isnan(me["jitter_pd"]):
                    ms_all[method]["jitter_pd"].append(me["jitter_pd"])
    
                all_results[method].setdefault('vertex_visibility_mask', []).append(me.get('vertex_visibility_mask', []))
    
        # Compute overall statistics
        for method in methods:
            metrics = {}
            for metric in ["mpjpe_all", "mpjpe_pa_all", "mpjae_all", "mpjae_pa_all", "mve_all", "mve_pa_all"]:
                if ms_all[method][metric]:
                    all_data = np.concatenate(ms_all[method][metric], axis=0)
                    metrics[f"{metric[:-4]} [mm]" if "jae" not in metric else f"{metric[:-4]} [deg]"] = float(np.mean(all_data))
                    metrics[f"{metric[:-4]} std"] = float(np.std(all_data))
                else:
                    metrics[f"{metric[:-4]} [mm]" if "jae" not in metric else f"{metric[:-4]} [deg]"] = np.nan
                    metrics[f"{metric[:-4]} std"] = np.nan
    
            if ms_all[method]["jitter_pd"]:
                jitter_all = np.array(ms_all[method]["jitter_pd"])
                metrics["Jitter [10m/s^3]"] = float(np.mean(jitter_all))
                metrics["Jitter std"] = float(np.std(jitter_all))
            else:
                metrics["Jitter [10m/s^3]"] = np.nan
                metrics["Jitter std"] = np.nan
    
            all_results[method].update(metrics)
            
            # Compute overall visibility mask
            if all_results[method]['vertex_visibility_mask']:
                all_results[method]['vertex_visibility_mask'] = np.any(all_results[method]['vertex_visibility_mask'], axis=0).tolist()
    
        # Print overall metrics
        print("Metrics for all sequences")
        print(self.to_pretty_string([all_results[method] for method in methods], methods))
    
        # Save results
        self.save_results(result_root, all_results, sequence_results)
    
        return all_results, sequence_results

    def save_results(self, result_root, all_results, sequence_results):
        """Save results for later visualization."""
        output_file = os.path.join(result_root, "evaluation_results.npz")
        
        np.savez_compressed(output_file,
                            all_results=json.dumps(all_results),
                            sequence_results=json.dumps(sequence_results))
        
        print(f"Results saved to {output_file}")


